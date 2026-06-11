"""
lumen-xarray Dashboard - Universal data explorer for xarray & tabular datasets.

Supports: NetCDF, Zarr, HDF5, GRIB (via xarray) + CSV, Parquet, JSON, Excel (via DuckDB).

Run:
    PYTHONPATH=. panel serve examples/dashboard.py --show
"""

import argparse
import io
import sys
import tempfile

import duckdb
import holoviews as hv
import hvplot.pandas  # noqa: F401
import numpy as np
import pandas as pd
import panel as pn
import xarray as xr  # noqa: F401
from bokeh.models import HoverTool

try:
    import cartopy.crs as ccrs
    import geoviews as gv
    HAS_GEO = True
except ImportError:
    HAS_GEO = False

from lumen_xarray import XArraySQLSource  # noqa: E402
from lumen_xarray.cf import (  # noqa: E402
    _LAT_NAMES,
    _LON_NAMES,
    _TIME_NAMES,
    _VERTICAL_NAMES,
    detect_coordinates,
)
from lumen_xarray.source import XARRAY_EXTENSIONS  # noqa: E402
from lumen_xarray.transforms import (  # noqa: E402
    Anomaly,
    Climatology,
    DimensionAggregate,
    LinearTrend,
    RollingWindow,
    TimeResample,
)

pn.extension("tabulator", notifications=True)
hv.extension("bokeh")
if HAS_GEO:
    gv.extension("bokeh")

# ----------------------------------------------------------------
# Tabular format support (DuckDB backend)
# ----------------------------------------------------------------
_TABULAR_EXTENSIONS = (".csv", ".tsv", ".parquet", ".json", ".jsonl", ".xlsx", ".xls")
_ALL_EXTENSIONS = tuple(XARRAY_EXTENSIONS) + _TABULAR_EXTENSIONS


class DuckDBWrapper:
    """Lightweight wrapper around an in-memory DuckDB connection for tabular files."""

    def __init__(self, path, table_name="data"):
        self.conn = duckdb.connect(":memory:")
        self.table_name = table_name
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        readers = {
            "csv": "read_csv_auto", "tsv": "read_csv_auto",
            "parquet": "read_parquet", "json": "read_json_auto",
            "jsonl": "read_json_auto", "xlsx": "st_read", "xls": "st_read",
        }
        reader = readers.get(ext, "read_csv_auto")
        if reader == "st_read":
            self.conn.execute("INSTALL spatial; LOAD spatial;")
        self.conn.execute(
            f"CREATE TABLE {table_name} AS SELECT * FROM {reader}('{path}')"
        )

    def execute(self, sql):
        return self.conn.execute(sql).fetchdf()

    def get_tables(self):
        """Return numeric column names (potential data variables)."""
        info = self.conn.execute(
            f"SELECT column_name, data_type FROM information_schema.columns "
            f"WHERE table_name = '{self.table_name}'"
        ).fetchdf()
        numeric_types = {"BIGINT", "INTEGER", "SMALLINT", "TINYINT", "DOUBLE",
                         "FLOAT", "REAL", "DECIMAL", "NUMERIC", "HUGEINT"}
        dim_cols = self._detect_dim_columns(info)
        return [
            row["column_name"] for _, row in info.iterrows()
            if row["data_type"] in numeric_types and row["column_name"] not in dim_cols
        ]

    def _detect_dim_columns(self, info):
        """Identify columns that are likely dimensions (lat, lon, time, etc.)."""
        dims = set()
        for _, row in info.iterrows():
            col_lower = row["column_name"].lower()
            if col_lower in _LAT_NAMES | _LON_NAMES | _TIME_NAMES | _VERTICAL_NAMES:
                dims.add(row["column_name"])
            elif row["data_type"] in ("TIMESTAMP", "DATE", "TIMESTAMP WITH TIME ZONE"):
                dims.add(row["column_name"])
        return dims

    def get_dimension_info(self, table=None):
        """Infer dimension info from column statistics."""
        info = self.conn.execute(
            f"SELECT column_name, data_type FROM information_schema.columns "
            f"WHERE table_name = '{self.table_name}'"
        ).fetchdf()
        dim_cols = self._detect_dim_columns(info)
        dim_info = {}
        for col in dim_cols:
            dtype = info[info["column_name"] == col]["data_type"].iloc[0]
            stats = self.conn.execute(
                f'SELECT MIN("{col}") as mn, MAX("{col}") as mx, '
                f'COUNT(DISTINCT "{col}") as cnt FROM {self.table_name}'
            ).fetchdf()
            col_lower = col.lower()
            if col_lower in _LAT_NAMES:
                role = "latitude"
            elif col_lower in _LON_NAMES:
                role = "longitude"
            elif col_lower in _TIME_NAMES or dtype in ("TIMESTAMP", "DATE", "TIMESTAMP WITH TIME ZONE"):
                role = "time"
            elif col_lower in _VERTICAL_NAMES:
                role = "vertical"
            else:
                role = "dimension"

            is_time = role == "time"
            dim_info[col] = {
                "type": "datetime" if is_time else "numeric",
                "role": role,
                "min": str(stats["mn"].iloc[0]) if is_time else float(stats["mn"].iloc[0]),
                "max": str(stats["mx"].iloc[0]) if is_time else float(stats["mx"].iloc[0]),
                "size": int(stats["cnt"].iloc[0]),
                "step": None,
            }
        return dim_info

    def estimate_size(self, table=None):
        cnt = self.conn.execute(f"SELECT COUNT(*) as c FROM {self.table_name}").fetchdf()
        rows = int(cnt["c"].iloc[0])
        ncols = len(self.conn.execute(
            f"SELECT * FROM {self.table_name} LIMIT 0"
        ).fetchdf().columns)
        return {"rows": rows, "estimated_mb": round(rows * ncols * 8 / 1e6, 1)}

    def get_metadata(self, table=None):
        return {"description": f"Tabular data from {self.table_name}"}

# Plot defaults
_PLOT_W, _PLOT_H = 720, 400


# ----------------------------------------------------------------
# State: holds the current source + derived info
# ----------------------------------------------------------------
class AppState:
    """Mutable application state - replaced when a new dataset is loaded."""

    def __init__(self, source, ds, name, backend="xarray"):
        self.source = source
        self.ds = ds  # None for tabular backend
        self.name = name
        self.backend = backend
        self.tables = source.get_tables()
        self.primary_var = self.tables[0]

        if backend == "xarray" and ds is not None:
            # Fast path: build dim_info from xarray metadata (avoids slow SQL)
            self.dim_info = self._xr_dim_info(ds, self.primary_var)
            self.size_info = self._xr_size_info(ds, self.primary_var)
        else:
            # Tabular/DuckDB: must query via SQL
            self.dim_info = source.get_dimension_info(self.primary_var)
            self.size_info = source.estimate_size(self.primary_var)

        if backend == "xarray":
            # Detect coordinate keys via CF conventions (with heuristic fallback)
            cf = detect_coordinates(ds)
            self.time_key = cf["time"] if cf["time"] in self.dim_info else None
            self.lat_key = cf["latitude"] if cf["latitude"] in self.dim_info else None
            self.lon_key = cf["longitude"] if cf["longitude"] in self.dim_info else None
            self.vertical_key = cf["vertical"] if cf["vertical"] in self.dim_info else None
        else:
            # Tabular backend: use role from dim_info (set by DuckDBWrapper)
            self.time_key = next(
                (k for k, v in self.dim_info.items() if v.get("role") == "time"), None
            )
            self.lat_key = next(
                (k for k, v in self.dim_info.items() if v.get("role") == "latitude"), None
            )
            self.lon_key = next(
                (k for k, v in self.dim_info.items() if v.get("role") == "longitude"), None
            )
            self.vertical_key = next(
                (k for k, v in self.dim_info.items() if v.get("role") == "vertical"), None
            )

        self.has_time = self.time_key is not None
        self.has_spatial = self.lat_key is not None and self.lon_key is not None
        self.has_vertical = self.vertical_key is not None

    @staticmethod
    def _xr_dim_info(ds, var_name):
        """Build dim_info from xarray metadata — instant, no SQL queries."""
        info = {}
        var = ds[var_name]
        cf = detect_coordinates(ds)
        role_map = {cf["latitude"]: "latitude", cf["longitude"]: "longitude",
                    cf["time"]: "time", cf["vertical"]: "vertical"}
        for dim in var.dims:
            coord = ds.coords.get(dim)
            if coord is None:
                continue
            entry = {"size": int(coord.size), "type": "numeric"}
            entry["role"] = role_map.get(dim)
            attrs = dict(coord.attrs)
            entry["units"] = attrs.get("units", "")
            entry["dtype"] = str(coord.dtype)
            if coord.dtype.kind == "M":  # datetime
                entry["type"] = "datetime"
                entry["min"] = str(coord.values[0])
                entry["max"] = str(coord.values[-1])
            elif coord.dtype.kind in "fiu":  # numeric
                vals = coord.values
                entry["min"] = float(vals.min())
                entry["max"] = float(vals.max())
                if len(vals) > 1:
                    entry["step"] = float(vals[1] - vals[0])
            info[dim] = entry
        return info

    @staticmethod
    def _xr_size_info(ds, var_name):
        """Estimate size from xarray shape — instant, no SQL queries."""
        var = ds[var_name]
        n_rows = 1
        for d in var.dims:
            n_rows *= ds.sizes[d]
        n_cols = len(var.dims) + 1
        est_bytes = n_rows * n_cols * 8
        return {
            "rows": n_rows,
            "columns": n_cols,
            "estimated_bytes": est_bytes,
            "estimated_mb": round(est_bytes / 1024**2, 1),
            "exceeds_warning": n_rows > 10_000_000,
        }


# Start with demo data
def _load_demo():
    ds = xr.tutorial.open_dataset("air_temperature")
    src = XArraySQLSource(_dataset=ds)
    return AppState(src, ds, "NOAA Air Temperature (demo)")


state = _load_demo()


_MAX_ROWS = 5_000_000  # Safety limit for in-memory operations
_TARGET_VIS_CELLS = 60_000  # Target max cells for spatial visualization

# ----------------------------------------------------------------
# Dataset info banner (shown above tabs)
# ----------------------------------------------------------------
info_banner = pn.pane.Markdown("", styles={"font-size": "13px"})


def _update_banner():
    s = state
    dims_str = ", ".join(f"{k}({v['size']})" for k, v in s.dim_info.items())
    vars_str = ", ".join(s.tables)
    backend_label = "xarray+DataFusion" if s.backend == "xarray" else "DuckDB"
    rows = s.size_info.get("rows", 0)
    # Grid info for spatial datasets
    grid_info = ""
    if s.has_spatial:
        lat_size = s.dim_info.get(s.lat_key, {}).get("size", 0)
        lon_size = s.dim_info.get(s.lon_key, {}).get("size", 0)
        total = lat_size * lon_size
        if total > _TARGET_VIS_CELLS:
            try:
                target = _get_target_cells()
            except NameError:
                target = _TARGET_VIS_CELLS
            if target:
                ratio = (total / target) ** 0.5
                bin_lat = max(10, int(lat_size / ratio))
                bin_lon = max(10, int(lon_size / ratio))
                grid_info = f" | Grid: {lat_size}x{lon_size} -> {bin_lat}x{bin_lon} (binned)"
            else:
                grid_info = f" | Grid: {lat_size}x{lon_size} (full resolution)"
        else:
            grid_info = f" | Grid: {lat_size}x{lon_size}"
    info_banner.object = (
        f"**{s.name}** | "
        f"Variables: `{vars_str}` | "
        f"Dims: `{dims_str}` | "
        f"{rows:,} rows (~{s.size_info['estimated_mb']} MB) | "
        f"Engine: `{backend_label}`{grid_info}"
    )


_update_banner()


# ----------------------------------------------------------------
# Sidebar: Data loader
# ----------------------------------------------------------------
file_input = pn.widgets.FileInput(
    name="Upload Dataset",
    accept=",".join(_ALL_EXTENSIONS),
    multiple=False,
    height=60,
)
path_input = pn.widgets.TextInput(
    name="File path / URL / glob",
    placeholder="/data/*.nc or s3://bucket/data.zarr",
    width=250,
)
load_btn = pn.widgets.Button(name="Load", button_type="primary", width=250)
load_status = pn.pane.Markdown("", width=250)

# Dynamic control widgets
controls_column = pn.Column(width=270)
analysis_widgets_column = pn.Column(width=270)

# Colormap selector (Feature 14)
cmap_select = pn.widgets.Select(
    name="Colormap",
    options=["RdYlBu_r", "viridis", "plasma", "inferno", "coolwarm", "RdBu_r",
             "Spectral_r", "turbo", "cividis", "magma"],
    value="RdYlBu_r", width=150,
)

# Unit conversion toggle (Feature 13)
kelvin_toggle = pn.widgets.Toggle(name="K \u2192 \u00b0C", value=False, visible=False, width=80)

# Resolution control for large datasets
resolution_mode = pn.widgets.Select(
    name="Resolution",
    options=["Auto", "Full (slow)", "Low (fast)", "Medium", "High"],
    value="Auto", width=150,
)

# Analysis widgets
resample_freq = pn.widgets.Select(
    name="Resample", options=["D", "W", "MS", "QS", "YS"], value="MS",
)
anomaly_groupby = pn.widgets.Select(
    name="Anomaly Baseline", options=["month", "season", "dayofyear", None], value="month",
)
rolling_window = pn.widgets.IntSlider(
    name="Rolling Window", start=1, end=90, value=30,
)

# Dynamic widgets - rebuilt per dataset
time_range = None
lat_range = None
lon_range = None
var_select = None
extra_dim_widgets = {}


def _build_controls():
    """Rebuild sidebar widgets from current state."""
    global time_range, lat_range, lon_range, var_select, extra_dim_widgets

    s = state
    dim_info = s.dim_info
    widget_list = []
    extra_dim_widgets = {}

    if s.time_key and dim_info[s.time_key].get("min"):
        t_min = pd.Timestamp(dim_info[s.time_key]["min"])
        t_max = pd.Timestamp(dim_info[s.time_key]["max"])
        time_range = pn.widgets.DateRangeSlider(
            name="Time Range", start=t_min, end=t_max, value=(t_min, t_max),
        )
        widget_list.append(time_range)
    else:
        time_range = None

    if s.lat_key and dim_info[s.lat_key].get("min") is not None:
        li = dim_info[s.lat_key]
        step = abs(li.get("step") or 1.0)
        lat_range = pn.widgets.RangeSlider(
            name="Latitude",
            start=float(li["min"]), end=float(li["max"]),
            value=(float(li["min"]), float(li["max"])),
            step=step,
        )
        widget_list.append(lat_range)
    else:
        lat_range = None

    if s.lon_key and dim_info[s.lon_key].get("min") is not None:
        li = dim_info[s.lon_key]
        step = abs(li.get("step") or 1.0)
        lon_range = pn.widgets.RangeSlider(
            name="Longitude",
            start=float(li["min"]), end=float(li["max"]),
            value=(float(li["min"]), float(li["max"])),
            step=step,
        )
        widget_list.append(lon_range)
    else:
        lon_range = None

    for dim_name, dinfo in dim_info.items():
        if dim_name in (s.time_key, s.lat_key, s.lon_key):
            continue
        if dinfo["type"] == "numeric":
            w = pn.widgets.RangeSlider(
                name=dim_name.capitalize(),
                start=float(dinfo["min"]), end=float(dinfo["max"]),
                value=(float(dinfo["min"]), float(dinfo["max"])),
            )
            extra_dim_widgets[dim_name] = w
            widget_list.append(w)

    if len(s.tables) > 1:
        var_select = pn.widgets.Select(
            name="Variable", options=s.tables, value=s.primary_var,
        )
        widget_list.append(var_select)
    else:
        var_select = None

    controls_column.clear()
    controls_column.extend(widget_list)

    analysis_widgets_column.clear()
    analysis_widgets_column.extend([resample_freq, anomaly_groupby, rolling_window])


_build_controls()


# ----------------------------------------------------------------
# Dataset loading logic
# ----------------------------------------------------------------
def _detect_and_load(path, name=None):
    """Route to xarray or DuckDB backend based on file extension."""
    ext = ("." + path.rsplit(".", 1)[-1].lower()) if "." in path else ""
    display_name = name or path.split("/")[-1]

    if ext in _TABULAR_EXTENSIONS:
        wrapper = DuckDBWrapper(path)
        if not wrapper.get_tables():
            raise ValueError(f"No numeric columns found in {display_name}")
        return wrapper, display_name, "duckdb"
    else:
        source = XArraySQLSource(uri=path)
        return source, display_name, "xarray"


def _get_units(var):
    """Get units string for a variable from dataset attributes."""
    s = state
    if s.backend == "xarray" and s.ds is not None and var in s.ds:
        return s.ds[var].attrs.get("units", "")
    return ""


def _apply_unit_conversion(df, var):
    """Apply K->C conversion if toggle is active and units are Kelvin."""
    if kelvin_toggle.value and _get_units(var) in ("K", "degK", "kelvin"):
        df = df.copy()
        df[var] = df[var] - 273.15
    return df


def _unit_label(var):
    """Return var name with units for plot labels."""
    units = _get_units(var)
    if kelvin_toggle.value and units in ("K", "degK", "kelvin"):
        return f"{var} [\u00b0C]"
    elif units:
        return f"{var} [{units}]"
    return var


def _finalize_load(source, name, backend="xarray"):
    """Common post-load: update state, controls, banner, tabs, and show size warnings."""
    global state
    _clear_cache()
    ds = source.dataset if backend == "xarray" else None
    state = AppState(source, ds, name, backend=backend)
    _build_controls()
    _update_banner()
    _rebuild_tabs()

    # Update K->C toggle visibility
    units = _get_units(state.primary_var)
    kelvin_toggle.visible = units in ("K", "degK", "kelvin")
    kelvin_toggle.value = False

    mb = state.size_info.get("estimated_mb", 0)
    size_note = ""
    if mb > 100:
        size_note = f" (large dataset: ~{mb} MB \u2014 filters are applied via SQL)"
    fmt_label = "xarray" if backend == "xarray" else "DuckDB"
    load_status.object = (
        f"**Loaded:** {state.name} ({len(state.tables)} vars, {fmt_label}){size_note}"
    )
    pn.state.notifications.success(f"Loaded {state.name}")


def _load_from_path(path):
    load_status.object = "*Loading...*"
    try:
        source, name, backend = _detect_and_load(path)
        _finalize_load(source, name, backend)
    except Exception as e:
        load_status.object = f"**Error:** {e}"
        pn.state.notifications.error(str(e))


def _load_from_upload(event):
    if not file_input.value:
        return
    fname = file_input.filename
    suffix = "." + fname.rsplit(".", 1)[-1] if "." in fname else ".nc"
    load_status.object = f"*Uploading {fname}...*"
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(file_input.value)
        tmp.flush()
        tmp.close()
        source, name, backend = _detect_and_load(tmp.name, fname)
        _finalize_load(source, name, backend)
    except Exception as e:
        load_status.object = f"**Error:** {e}"
        pn.state.notifications.error(str(e))


def _on_load_click(event):
    if path_input.value.strip():
        _load_from_path(path_input.value.strip())
    elif file_input.value:
        _load_from_upload(event)
    else:
        load_status.object = "Upload a file or enter a path first."


load_btn.on_click(_on_load_click)


# ----------------------------------------------------------------
# Filtering helpers
# ----------------------------------------------------------------
def _current_var():
    return var_select.value if var_select else state.primary_var


def _compute_grid_cells():
    """Total spatial grid cells from dim_info."""
    s = state
    lat_size = s.dim_info.get(s.lat_key, {}).get("size", 0) if s.lat_key else 0
    lon_size = s.dim_info.get(s.lon_key, {}).get("size", 0) if s.lon_key else 0
    return lat_size * lon_size


def _get_target_cells():
    """Get target cell count based on resolution widget."""
    mode = resolution_mode.value
    if mode == "Full (slow)":
        return None  # No binning
    if mode == "Low (fast)":
        return 10_000
    if mode == "Medium":
        return 40_000
    if mode == "High":
        return 100_000
    return _TARGET_VIS_CELLS  # Auto


def _spatial_bin_expr(coord_key):
    """SQL FLOOR expression to bin a coordinate, or raw key if small enough."""
    s = state
    info = s.dim_info.get(coord_key, {})
    dim_size = info.get("size", 0)
    total_cells = _compute_grid_cells()
    target = _get_target_cells()
    if target is None or total_cells <= target:
        return coord_key, coord_key  # (select_expr, group_expr) — no binning
    ratio = (total_cells / target) ** 0.5
    target_bins = max(10, int(dim_size / ratio))
    span = info.get("max", 1) - info.get("min", 0)
    if span <= 0:
        return coord_key, coord_key
    bw = span / target_bins
    expr = f"(FLOOR({coord_key} / {bw}) * {bw})"
    return f"{expr} as {coord_key}", expr


def _binned_spatial_select():
    """Return (lat_select, lon_select, lat_group, lon_group) for spatial queries."""
    s = state
    lat_sel, lat_grp = _spatial_bin_expr(s.lat_key)
    lon_sel, lon_grp = _spatial_bin_expr(s.lon_key)
    return lat_sel, lon_sel, lat_grp, lon_grp


# ----------------------------------------------------------------
# Query cache — avoids re-executing identical SQL when switching tabs
# ----------------------------------------------------------------
_query_cache: dict[str, pd.DataFrame] = {}
_CACHE_MAX = 20


def _cached_execute(sql):
    """Execute SQL with caching. Returns a copy to prevent mutation."""
    if sql in _query_cache:
        return _query_cache[sql].copy()
    result = state.source.execute(sql)
    if len(_query_cache) >= _CACHE_MAX:
        _query_cache.pop(next(iter(_query_cache)))
    _query_cache[sql] = result
    return result.copy()


def _clear_cache():
    """Clear the query cache (called on dataset change)."""
    _query_cache.clear()


def _filter_clauses():
    """Build shared WHERE clauses from current widget values."""
    s = state
    clauses = []
    if time_range is not None:
        t0 = pd.Timestamp(time_range.value[0]).strftime("%Y-%m-%d %H:%M:%S")
        t1 = pd.Timestamp(time_range.value[1]).strftime("%Y-%m-%d %H:%M:%S")
        clauses.append(f"{s.time_key} >= '{t0}' AND {s.time_key} <= '{t1}'")
    if lat_range is not None:
        clauses.append(
            f"{s.lat_key} >= {lat_range.value[0]} AND {s.lat_key} <= {lat_range.value[1]}"
        )
    if lon_range is not None:
        clauses.append(
            f"{s.lon_key} >= {lon_range.value[0]} AND {s.lon_key} <= {lon_range.value[1]}"
        )
    for dim_name, w in extra_dim_widgets.items():
        clauses.append(f"{dim_name} >= {w.value[0]} AND {dim_name} <= {w.value[1]}")
    return clauses


def _where_sql():
    """Return ' WHERE ...' string or '' if no filters active."""
    clauses = _filter_clauses()
    return (" WHERE " + " AND ".join(clauses)) if clauses else ""


def _table_name(var):
    """Return the SQL table name for a variable (handles DuckDB single-table)."""
    s = state
    if s.backend == "duckdb":
        return s.source.table_name
    return var


def _q(name):
    """Quote a SQL identifier to handle case-sensitive column names (e.g. ROSE)."""
    return f'"{name}"'


def _build_sql_query(var, limit=None):
    """Build a SQL query with WHERE clauses pushed down and optional LIMIT."""
    tbl = _table_name(var)
    where = _where_sql()
    sql = f"SELECT * FROM {tbl}{where}"
    if limit:
        sql += f" LIMIT {limit}"
    return sql


def _get_filtered(limit=_MAX_ROWS):
    var = _current_var()
    sql = _build_sql_query(var, limit=limit)
    return state.source.execute(sql)


def _get_time_agg_sql(var):
    """Get time series with non-time dims aggregated via SQL."""
    s = state
    tbl = _table_name(var)
    where = _where_sql()
    qv = _q(var)
    sql = (
        f"SELECT {s.time_key}, AVG({qv}) as {qv} "
        f"FROM {tbl}{where} GROUP BY {s.time_key} ORDER BY {s.time_key}"
    )
    return _cached_execute(sql)


# ----------------------------------------------------------------
# Apply button: triggers plot updates
# ----------------------------------------------------------------
apply_btn = pn.widgets.Button(name="Apply Filters", button_type="success", width=270)
_update_counter = pn.widgets.IntInput(value=0, visible=False)


def _on_apply(event):
    _clear_cache()  # Filters changed, invalidate cached queries
    _update_counter.value += 1


apply_btn.on_click(_on_apply)


# ================================================================
# EXPLORE TABS
# ================================================================

# -- Spatial Map / Heatmap --
map_mode = pn.widgets.RadioButtonGroup(
    name="Map Mode",
    options=["Geographic Map", "Heatmap"] if HAS_GEO else ["Heatmap"],
    value="Geographic Map" if HAS_GEO else "Heatmap",
    button_type="light",
)
show_coastlines = pn.widgets.Checkbox(name="Coastlines", value=True, visible=HAS_GEO)
projection_select = pn.widgets.Select(
    name="Projection",
    options=["PlateCarree", "Robinson", "Mollweide", "Orthographic"],
    value="PlateCarree",
    width=150,
    visible=HAS_GEO,
)

_PROJECTIONS = {}
if HAS_GEO:
    _PROJECTIONS = {
        "PlateCarree": ccrs.PlateCarree,
        "Robinson": ccrs.Robinson,
        "Mollweide": ccrs.Mollweide,
        "Orthographic": ccrs.Orthographic,
    }


def _get_spatial_agg_sql(var):
    """Get spatially aggregated data via SQL with adaptive spatial binning."""
    tbl = _table_name(var)
    where = _where_sql()
    qv = _q(var)
    lat_sel, lon_sel, lat_grp, lon_grp = _binned_spatial_select()
    sql = (
        f"SELECT {lat_sel}, {lon_sel}, AVG({qv}) as {qv} "
        f"FROM {tbl}{where} GROUP BY {lat_grp}, {lon_grp}"
    )
    return _cached_execute(sql)


def _make_geo_map(agg, s, var, proj_name, coastlines):
    """Render a GeoViews geographic map from aggregated lat/lon data."""
    proj_cls = _PROJECTIONS.get(proj_name, ccrs.PlateCarree)
    proj = proj_cls()
    label = _unit_label(var)
    cmap = cmap_select.value
    try:
        pivoted = agg.pivot_table(
            index=s.lat_key, columns=s.lon_key, values=var, aggfunc="mean",
        )
        lats = pivoted.index.values
        lons = pivoted.columns.values
        data = pivoted.values

        hover = HoverTool(tooltips=[
            (s.lat_key, "$y{0.2f}"), (s.lon_key, "$x{0.2f}"), (label, "@image{0.2f}"),
        ])
        img = gv.Image(
            (lons, lats, data),
            kdims=[s.lon_key, s.lat_key],
            vdims=[var],
        ).opts(
            cmap=cmap, colorbar=True, projection=proj,
            title=f"Mean {label}", width=_PLOT_W, height=_PLOT_H,
            tools=[hover],
        )
        if coastlines:
            features = gv.feature.coastline.opts(line_color="black", line_width=0.8)
            return img * features
        return img
    except Exception:
        return _make_heatmap(agg, s, var)


def _make_heatmap(agg, s, var):
    """Render a flat hvplot heatmap (no geographic projection)."""
    label = _unit_label(var)
    return agg.hvplot.heatmap(
        x=s.lon_key, y=s.lat_key, C=var, cmap=cmap_select.value,
        title=f"Mean {label}", width=_PLOT_W, height=_PLOT_H, colorbar=True,
    )


@pn.depends(_update_counter, map_mode, show_coastlines, projection_select, cmap_select, kelvin_toggle, resolution_mode)
def _spatial_view(trigger=None, mode=None, coastlines=None, proj=None, cmap=None, k2c=None, res=None):
    s = state
    var = _current_var()
    try:
        if s.has_spatial:
            agg = _get_spatial_agg_sql(var)
            if agg.empty:
                return pn.pane.Markdown("No data for selected range.")
            agg = _apply_unit_conversion(agg, var)
            if mode == "Geographic Map" and HAS_GEO:
                return _make_geo_map(agg, s, var, projection_select.value, show_coastlines.value)
            return _make_heatmap(agg, s, var)
        elif len(s.dim_info) >= 2:
            df = _get_filtered()
            if df.empty:
                return pn.pane.Markdown("No data for selected range.")
            dims = [d for d in s.dim_info if d != s.time_key][:2]
            non_plot = [d for d in s.dim_info if d not in dims and d in df.columns]
            agg = DimensionAggregate(
                dimensions=non_plot, method="mean", value_columns=[var],
            ).apply(df) if non_plot else df
            agg = _apply_unit_conversion(agg, var)
            return agg.hvplot.heatmap(
                x=dims[0], y=dims[1], C=var, cmap=cmap_select.value,
                title=f"Mean {_unit_label(var)}", width=_PLOT_W, height=_PLOT_H,
            )
        return pn.pane.Markdown("Need 2+ spatial dimensions for heatmap.")
    except Exception as e:
        return pn.pane.Markdown(f"**Error rendering spatial view:** {e}")


def _get_profile_agg_sql(var, group_dim):
    """Get profile data aggregated along one dimension via SQL."""
    tbl = _table_name(var)
    where = _where_sql()
    qv = _q(var)
    sql = (
        f"SELECT {group_dim}, AVG({qv}) as {qv}, MIN({qv}) as {_q(var + '_min')}, "
        f"MAX({qv}) as {_q(var + '_max')} FROM {tbl}{where} GROUP BY {group_dim} ORDER BY {group_dim}"
    )
    return _cached_execute(sql)


# -- Lat/Lon Profiles --
@pn.depends(_update_counter)
def _lat_profile(trigger=None):
    s = state
    var = _current_var()
    if not s.lat_key:
        return pn.pane.Markdown("No latitude dimension available.")
    try:
        agg = _get_profile_agg_sql(var, s.lat_key)
        if agg.empty:
            return pn.pane.Markdown("No data for selected range.")

        mean_plot = agg.hvplot.line(
            x=s.lat_key, y=var, color="#1976D2", label="Mean",
            title=f"{var} by Latitude (zonal mean)", width=_PLOT_W, height=_PLOT_H,
        )
        try:
            area = hv.Area(
                (agg[s.lat_key], agg[f"{var}_min"], agg[f"{var}_max"]),
                vdims=["y", "y2"],
            ).opts(alpha=0.2, color="#1976D2")
            return area * mean_plot
        except Exception:
            return mean_plot
    except Exception as e:
        return pn.pane.Markdown(f"**Error:** {e}")


@pn.depends(_update_counter)
def _lon_profile(trigger=None):
    s = state
    var = _current_var()
    if not s.lon_key:
        return pn.pane.Markdown("No longitude dimension available.")
    try:
        agg = _get_profile_agg_sql(var, s.lon_key)
        if agg.empty:
            return pn.pane.Markdown("No data for selected range.")
        return agg.hvplot.line(
            x=s.lon_key, y=var, color="#E91E63",
            title=f"{var} by Longitude (meridional mean)", width=_PLOT_W, height=_PLOT_H,
        )
    except Exception as e:
        return pn.pane.Markdown(f"**Error:** {e}")


# -- Distribution --
_HIST_SAMPLE = 500_000  # Max rows for histogram


@pn.depends(_update_counter)
def _histogram(trigger=None):
    var = _current_var()
    s = state
    try:
        # Compute stats via SQL on FULL dataset (fast aggregate, no row transfer)
        tbl = _table_name(var)
        qv = _q(var)
        where = _where_sql()
        stats = s.source.execute(
            f"SELECT COUNT({qv}) as cnt, AVG({qv}) as mean, STDDEV({qv}) as std, "
            f"MIN({qv}) as min_v, MAX({qv}) as max_v "
            f"FROM {tbl}{where}{' AND' if where else ' WHERE'} {qv} IS NOT NULL"
        )
        # Load only a sample for the histogram visualization
        sql = _build_sql_query(var, limit=_HIST_SAMPLE)
        df = s.source.execute(sql)
        if df.empty:
            return pn.pane.Markdown("No data for selected range.")

        values = df[var].dropna()

        row = stats.iloc[0]
        stats_md = (
            f"**Count:** {int(row['cnt']):,} | "
            f"**Mean:** {row['mean']:.4f} | "
            f"**Std:** {row['std']:.4f} | "
            f"**Min:** {row['min_v']:.4f} | "
            f"**Max:** {row['max_v']:.4f}"
        )

        hist = values.hvplot.hist(
            bins=50, color="#7B1FA2", alpha=0.7,
            title=f"Distribution of {var}", width=_PLOT_W, height=_PLOT_H - 50,
            xlabel=var, ylabel="Count",
        )
        kde = values.hvplot.kde(color="#FF6F00", line_width=2, ylabel="")

        return pn.Column(
            pn.pane.Markdown(stats_md, width=_PLOT_W),
            pn.pane.HoloViews(hist * kde, width=_PLOT_W, height=_PLOT_H),
        )
    except Exception as e:
        return pn.pane.Markdown(f"**Error:** {e}")


# -- Vertical Profile --
@pn.depends(_update_counter)
def _vertical_profile(trigger=None):
    s = state
    var = _current_var()
    if not s.has_vertical:
        return pn.pane.Markdown("No vertical dimension available.")
    try:
        agg = _get_profile_agg_sql(var, s.vertical_key)
        if agg.empty:
            return pn.pane.Markdown("No data for selected range.")
        vert = s.vertical_key
        agg = agg.sort_values(vert)

        plot = agg.hvplot.line(
            x=var, y=vert, color="#6A1B9A",
            title=f"{var} - Vertical Profile",
            width=_PLOT_W, height=_PLOT_H,
        ).opts(invert_yaxis=True)

        return pn.Column(
            pn.pane.Markdown(
                f"**{vert}** axis inverted (higher values = lower altitude/deeper)."
            ),
            pn.pane.HoloViews(plot, sizing_mode="stretch_width"),
        )
    except Exception as e:
        return pn.pane.Markdown(f"**Error:** {e}")


# ================================================================
# NEW EXPLORE FEATURES (Phase B)
# ================================================================

# -- B1: Time Animation Player --
anim_player = pn.widgets.DiscretePlayer(
    name="Time Step", options=["(no data)"], interval=500, width=_PLOT_W,
)


def _populate_anim_steps():
    """Load distinct time steps for the animation player."""
    s = state
    if not (s.has_time and s.has_spatial):
        return
    tbl = _table_name(s.primary_var)
    where = _where_sql()
    steps = s.source.execute(
        f"SELECT DISTINCT {s.time_key} FROM {tbl}{where} "
        f"ORDER BY {s.time_key} LIMIT 200"
    )
    vals = [str(v) for v in steps[s.time_key]]
    anim_player.options = vals if vals else ["(no data)"]
    if vals:
        anim_player.value = vals[0]


@pn.depends(anim_player, cmap_select)
def _animated_spatial_view(time_step=None, cmap=None):
    s = state
    if not (s.has_time and s.has_spatial):
        return pn.pane.Markdown("Requires time + spatial dimensions.")
    var = _current_var()
    tbl = _table_name(var)
    clauses = _filter_clauses()
    clauses.append(f"{s.time_key} = '{time_step}'")
    where = " WHERE " + " AND ".join(clauses)
    qv = _q(var)
    lat_sel, lon_sel, lat_grp, lon_grp = _binned_spatial_select()
    try:
        agg = s.source.execute(
            f"SELECT {lat_sel}, {lon_sel}, AVG({qv}) as {qv} "
            f"FROM {tbl}{where} GROUP BY {lat_grp}, {lon_grp}"
        )
        if agg.empty:
            return pn.pane.Markdown("No data for this time step.")
        agg = _apply_unit_conversion(agg, var)
        label = _unit_label(var)
        return agg.hvplot.heatmap(
            x=s.lon_key, y=s.lat_key, C=var, cmap=cmap_select.value,
            title=f"{label} at {time_step}", width=_PLOT_W, height=_PLOT_H,
            colorbar=True,
        )
    except Exception as e:
        return pn.pane.Markdown(f"**Error:** {e}")


# -- B2: Hovmoller Diagram --
hovmoller_axis = pn.widgets.RadioButtonGroup(
    name="Hovmoller Axis", options=["Latitude", "Longitude"],
    value="Latitude", button_type="light",
)


@pn.depends(_update_counter, hovmoller_axis, cmap_select)
def _hovmoller_view(trigger=None, axis=None, cmap=None):
    s = state
    if not (s.has_time and s.has_spatial):
        return pn.pane.Markdown("Requires time + spatial dimensions.")
    var = _current_var()
    tbl = _table_name(var)
    where = _where_sql()
    dim = s.lat_key if hovmoller_axis.value == "Latitude" else s.lon_key
    try:
        qv = _q(var)
        dim_sel, dim_grp = _spatial_bin_expr(dim)
        df = s.source.execute(
            f"SELECT {s.time_key}, {dim_sel}, AVG({qv}) as {qv} "
            f"FROM {tbl}{where} GROUP BY {s.time_key}, {dim_grp} "
            f"ORDER BY {s.time_key}, {dim_grp}"
        )
        if df.empty:
            return pn.pane.Markdown("No data for selected range.")
        df = _apply_unit_conversion(df, var)
        label = _unit_label(var)
        return df.hvplot.heatmap(
            x=s.time_key, y=dim, C=var, cmap=cmap_select.value,
            title=f"Hovmoller: {label} vs {dim}", width=_PLOT_W, height=_PLOT_H,
            colorbar=True, rot=45,
        )
    except Exception as e:
        return pn.pane.Markdown(f"**Error:** {e}")


# -- B3: NaN / Data Coverage Map --
@pn.depends(_update_counter)
def _nan_coverage_view(trigger=None):
    s = state
    if not s.has_spatial:
        return pn.pane.Markdown("Requires spatial dimensions.")
    var = _current_var()
    tbl = _table_name(var)
    where = _where_sql()
    try:
        qv = _q(var)
        lat_sel, lon_sel, lat_grp, lon_grp = _binned_spatial_select()
        df = s.source.execute(
            f"SELECT {lat_sel}, {lon_sel}, COUNT(*) as total, "
            f"COUNT({qv}) as valid, "
            f"CAST(COUNT({qv}) AS DOUBLE) / COUNT(*) * 100.0 as coverage_pct "
            f"FROM {tbl}{where} GROUP BY {lat_grp}, {lon_grp}"
        )
        if df.empty:
            return pn.pane.Markdown("No data for selected range.")
        return pn.Column(
            pn.pane.Markdown(
                f"**Overall:** {df['valid'].sum():,} / {df['total'].sum():,} values "
                f"({df['valid'].sum() / df['total'].sum() * 100:.1f}% coverage)"
            ),
            df.hvplot.heatmap(
                x=s.lon_key, y=s.lat_key, C="coverage_pct", cmap="YlOrRd",
                title=f"Data Coverage (%) for {var}", width=_PLOT_W, height=_PLOT_H,
                colorbar=True, clim=(0, 100),
            ),
        )
    except Exception as e:
        return pn.pane.Markdown(f"**Error:** {e}")


# -- B4: Transect / Cross-Section --
transect_start_lat = pn.widgets.FloatInput(name="Start Lat", value=-30.0, step=1.0, width=100)
transect_start_lon = pn.widgets.FloatInput(name="Start Lon", value=-60.0, step=1.0, width=100)
transect_end_lat = pn.widgets.FloatInput(name="End Lat", value=30.0, step=1.0, width=100)
transect_end_lon = pn.widgets.FloatInput(name="End Lon", value=60.0, step=1.0, width=100)
transect_npoints = pn.widgets.IntSlider(name="Points", start=10, end=200, value=50, width=200)


@pn.depends(_update_counter, transect_start_lat, transect_start_lon,
            transect_end_lat, transect_end_lon, transect_npoints)
def _transect_view(trigger=None, slat=None, slon=None, elat=None, elon=None, npts=None):
    s = state
    if not s.has_spatial:
        return pn.pane.Markdown("Requires spatial dimensions.")
    var = _current_var()
    try:
        agg = _get_spatial_agg_sql(var)
        if agg.empty:
            return pn.pane.Markdown("No data for selected range.")

        pivoted = agg.pivot_table(index=s.lat_key, columns=s.lon_key, values=var, aggfunc="mean")
        lats = pivoted.index.values
        lons = pivoted.columns.values

        from scipy.interpolate import RegularGridInterpolator
        lat_sorted = np.sort(lats)
        lon_sorted = np.sort(lons)
        data_sorted = pivoted.loc[lat_sorted, lon_sorted].values
        interp = RegularGridInterpolator(
            (lat_sorted, lon_sorted), data_sorted,
            method="linear", bounds_error=False, fill_value=np.nan,
        )

        n = transect_npoints.value
        t_lats = np.linspace(transect_start_lat.value, transect_end_lat.value, n)
        t_lons = np.linspace(transect_start_lon.value, transect_end_lon.value, n)
        points = np.column_stack([t_lats, t_lons])
        values = interp(points)

        # Distance along transect in degrees (approximate)
        dlat = t_lats - t_lats[0]
        dlon = t_lons - t_lons[0]
        distance = np.sqrt(dlat**2 + dlon**2)

        result = pd.DataFrame({"distance_deg": distance, var: values})
        result = _apply_unit_conversion(result, var)
        label = _unit_label(var)

        return pn.Column(
            pn.pane.Markdown(
                f"**Transect:** ({transect_start_lat.value}, {transect_start_lon.value}) "
                f"to ({transect_end_lat.value}, {transect_end_lon.value})"
            ),
            result.hvplot.line(
                x="distance_deg", y=var, color="#E91E63",
                title=f"{label} along transect", width=_PLOT_W, height=_PLOT_H,
                xlabel="Distance (degrees)", ylabel=label,
            ),
        )
    except Exception as e:
        return pn.pane.Markdown(f"**Error:** {e}")


# ================================================================
# TIME ANALYSIS TABS
# ================================================================

@pn.depends(_update_counter, resample_freq, kelvin_toggle)
def _time_series(trigger=None, freq=None, k2c=None):
    s = state
    var = _current_var()
    if not s.has_time:
        return pn.pane.Markdown("No time dimension available.")
    try:
        agg = _get_time_agg_sql(var)
        if agg.empty:
            return pn.pane.Markdown("No data for selected range.")
        agg = _apply_unit_conversion(agg, var)
        resampled = TimeResample(time_col=s.time_key, freq=resample_freq.value).apply(agg)
        label = _unit_label(var)
        return resampled.hvplot.line(
            x=s.time_key, y=var, color="#2196F3",
            title=f"Spatially-Averaged {label} ({resample_freq.value})",
            width=_PLOT_W, height=_PLOT_H, ylabel=label,
        )
    except Exception as e:
        return pn.pane.Markdown(f"**Error:** {e}")


@pn.depends(_update_counter, anomaly_groupby)
def _anomaly(trigger=None, groupby=None):
    s = state
    var = _current_var()
    if not s.has_time:
        return pn.pane.Markdown("No time dimension available.")
    try:
        agg = _get_time_agg_sql(var)
        if agg.empty:
            return pn.pane.Markdown("No data for selected range.")
        anom = Anomaly(
            time_col=s.time_key, value_col=var, groupby=anomaly_groupby.value,
        ).apply(agg)
        anom_col = f"{var}_anomaly"
        colors = ["#EF5350" if v >= 0 else "#42A5F5" for v in anom[anom_col]]
        return anom.hvplot.bar(
            x=s.time_key, y=anom_col, color=colors,
            title=f"{var} Anomaly (baseline: {anomaly_groupby.value or 'overall'})",
            width=_PLOT_W, height=_PLOT_H, rot=45,
        )
    except Exception as e:
        return pn.pane.Markdown(f"**Error:** {e}")


@pn.depends(_update_counter, rolling_window)
def _rolling(trigger=None, window=None):
    s = state
    var = _current_var()
    if not s.has_time:
        return pn.pane.Markdown("No time dimension available.")
    try:
        agg = _get_time_agg_sql(var)
        if agg.empty:
            return pn.pane.Markdown("No data for selected range.")
        smoothed = RollingWindow(column=var, window=rolling_window.value).apply(agg)
        rolling_col = f"{var}_rolling_mean"
        raw = smoothed.hvplot.line(
            x=s.time_key, y=var, label="Raw", color="#BDBDBD", alpha=0.5,
        )
        smooth = smoothed.hvplot.line(
            x=s.time_key, y=rolling_col, color="#FF5722",
            label=f"{rolling_window.value}-step Mean",
        )
        return (raw * smooth).opts(
            title=f"{var} - {rolling_window.value}-step Rolling Mean",
            width=_PLOT_W, height=_PLOT_H, legend_position="top_left",
        )
    except Exception as e:
        return pn.pane.Markdown(f"**Error:** {e}")


@pn.depends(_update_counter)
def _trend(trigger=None):
    s = state
    var = _current_var()
    if not s.has_time:
        return pn.pane.Markdown("No time dimension available.")
    try:
        agg = _get_time_agg_sql(var)
        if agg.empty:
            return pn.pane.Markdown("No data for selected range.")
        trended = LinearTrend(time_col=s.time_key, value_col=var).apply(agg)
        trend_col = f"{var}_trend"

        raw = trended.hvplot.line(
            x=s.time_key, y=var, label="Data", color="#90A4AE", alpha=0.6,
        )
        trend_line = trended.hvplot.line(
            x=s.time_key, y=trend_col, color="#D32F2F", line_width=2,
            label="Linear Trend",
        )
        return (raw * trend_line).opts(
            title=f"{var} - Linear Trend",
            width=_PLOT_W, height=_PLOT_H, legend_position="top_left",
        )
    except Exception as e:
        return pn.pane.Markdown(f"**Error:** {e}")


@pn.depends(_update_counter)
def _climatology_tab(trigger=None):
    s = state
    var = _current_var()
    if not s.has_time:
        return pn.pane.Markdown("No time dimension available.")
    try:
        agg = _get_time_agg_sql(var)
        if agg.empty:
            return pn.pane.Markdown("No data for selected range.")
        clim = Climatology(time_col=s.time_key, value_col=var, groupby="month").apply(agg)
        clim_col = f"{var}_climatology"

        # Extract unique monthly climatology
        ts = pd.to_datetime(clim[s.time_key])
        clim["month"] = ts.dt.month
        monthly = clim.groupby("month")[[clim_col]].mean().reset_index()

        month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                       7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
        monthly["month_name"] = monthly["month"].map(month_names)

        return monthly.hvplot.bar(
            x="month_name", y=clim_col, color="#00897B",
            title=f"{var} - Monthly Climatology",
            width=_PLOT_W, height=_PLOT_H,
            xlabel="Month", ylabel=f"Mean {var}",
        )
    except Exception as e:
        return pn.pane.Markdown(f"**Error:** {e}")


# ================================================================
# ANALYSIS TABS (Phase C)
# ================================================================

# -- C1: Difference Maps --
diff_mode = pn.widgets.RadioButtonGroup(
    name="Difference Mode", options=["Two Time Periods", "Two Levels"],
    value="Two Time Periods", button_type="light",
)
diff_period1_start = pn.widgets.DatePicker(name="Period 1 Start", width=140)
diff_period1_end = pn.widgets.DatePicker(name="Period 1 End", width=140)
diff_period2_start = pn.widgets.DatePicker(name="Period 2 Start", width=140)
diff_period2_end = pn.widgets.DatePicker(name="Period 2 End", width=140)
diff_level1 = pn.widgets.FloatInput(name="Level 1", value=0.0, width=100)
diff_level2 = pn.widgets.FloatInput(name="Level 2", value=0.0, width=100)


def _init_diff_widgets():
    """Initialize difference map date pickers from current data range."""
    s = state
    if s.has_time and s.dim_info[s.time_key].get("min"):
        t_min = pd.Timestamp(s.dim_info[s.time_key]["min"])
        t_max = pd.Timestamp(s.dim_info[s.time_key]["max"])
        mid = t_min + (t_max - t_min) / 2
        diff_period1_start.value = t_min.date()
        diff_period1_end.value = mid.date()
        diff_period2_start.value = mid.date()
        diff_period2_end.value = t_max.date()
    if s.has_vertical:
        vinfo = s.dim_info[s.vertical_key]
        diff_level1.value = float(vinfo["min"])
        diff_level2.value = float(vinfo["max"])


@pn.depends(_update_counter, diff_mode, diff_period1_start, diff_period1_end,
            diff_period2_start, diff_period2_end, diff_level1, diff_level2, cmap_select)
def _difference_map_view(trigger=None, mode=None, p1s=None, p1e=None,
                         p2s=None, p2e=None, l1=None, l2=None, cmap=None):
    s = state
    if not s.has_spatial:
        return pn.pane.Markdown("Requires spatial dimensions.")
    var = _current_var()
    tbl = _table_name(var)
    try:
        base_clauses = _filter_clauses()
        base_where = " AND ".join(base_clauses)

        if diff_mode.value == "Two Time Periods" and s.has_time:
            p1_clause = (
                f"{s.time_key} >= '{diff_period1_start.value}' AND "
                f"{s.time_key} <= '{diff_period1_end.value}'"
            )
            p2_clause = (
                f"{s.time_key} >= '{diff_period2_start.value}' AND "
                f"{s.time_key} <= '{diff_period2_end.value}'"
            )
            w1 = f" WHERE {p1_clause}" + (f" AND {base_where}" if base_where else "")
            w2 = f" WHERE {p2_clause}" + (f" AND {base_where}" if base_where else "")
            label = "Period 2 - Period 1"
        elif diff_mode.value == "Two Levels" and s.has_vertical:
            l1_clause = f"{s.vertical_key} = {diff_level1.value}"
            l2_clause = f"{s.vertical_key} = {diff_level2.value}"
            w1 = f" WHERE {l1_clause}" + (f" AND {base_where}" if base_where else "")
            w2 = f" WHERE {l2_clause}" + (f" AND {base_where}" if base_where else "")
            label = f"Level {diff_level2.value} - Level {diff_level1.value}"
        else:
            return pn.pane.Markdown("Selected mode requires time or vertical dimensions.")

        qv = _q(var)
        lat_sel, lon_sel, lat_grp, lon_grp = _binned_spatial_select()
        df1 = s.source.execute(
            f"SELECT {lat_sel}, {lon_sel}, AVG({qv}) as {qv} "
            f"FROM {tbl}{w1} GROUP BY {lat_grp}, {lon_grp}"
        )
        df2 = s.source.execute(
            f"SELECT {lat_sel}, {lon_sel}, AVG({qv}) as {qv} "
            f"FROM {tbl}{w2} GROUP BY {lat_grp}, {lon_grp}"
        )
        merged = pd.merge(df1, df2, on=[s.lat_key, s.lon_key], suffixes=("_1", "_2"))
        merged["difference"] = merged[f"{var}_2"] - merged[f"{var}_1"]

        vmax = max(abs(merged["difference"].min()), abs(merged["difference"].max()))
        return merged.hvplot.heatmap(
            x=s.lon_key, y=s.lat_key, C="difference", cmap="RdBu_r",
            title=f"Difference: {label}", width=_PLOT_W, height=_PLOT_H,
            colorbar=True, clim=(-vmax, vmax),
        )
    except Exception as e:
        return pn.pane.Markdown(f"**Error:** {e}")


# -- C2: Spectral / FFT Analysis --
@pn.depends(_update_counter)
def _spectral_view(trigger=None):
    s = state
    if not s.has_time:
        return pn.pane.Markdown("Requires time dimension.")
    var = _current_var()
    try:
        agg = _get_time_agg_sql(var)
        if agg.empty or len(agg) < 8:
            return pn.pane.Markdown("Not enough time steps for spectral analysis (need 8+).")

        values = agg[var].dropna().values
        values = values - values.mean()
        spectrum = np.fft.rfft(values)
        power = np.abs(spectrum) ** 2

        # Compute periods in time steps
        n = len(values)
        freqs = np.fft.rfftfreq(n)
        # Skip DC component (freq=0)
        freqs = freqs[1:]
        power = power[1:]

        # Try to detect time step in days
        times = pd.to_datetime(agg[s.time_key])
        if len(times) > 1:
            dt = (times.iloc[1] - times.iloc[0]).total_seconds() / 86400.0
        else:
            dt = 1.0
        periods = (1.0 / freqs) * dt  # periods in days

        spec_df = pd.DataFrame({"period_days": periods, "power": power})
        spec_df = spec_df[spec_df["period_days"] > 0]

        label = _unit_label(var)
        return pn.Column(
            pn.pane.Markdown(f"**FFT of** {label} | **Peak period:** "
                             f"{spec_df.loc[spec_df['power'].idxmax(), 'period_days']:.1f} days"),
            spec_df.hvplot.line(
                x="period_days", y="power", color="#FF5722", logx=True, logy=True,
                title=f"Power Spectrum of {label}", width=_PLOT_W, height=_PLOT_H,
                xlabel="Period (days)", ylabel="Power",
            ),
        )
    except Exception as e:
        return pn.pane.Markdown(f"**Error:** {e}")


# -- C3: Zonal/Meridional Cross-Sections --
cross_section_mode = pn.widgets.RadioButtonGroup(
    name="Cross Section", options=["Lat vs Level", "Time vs Level"],
    value="Lat vs Level", button_type="light",
)


@pn.depends(_update_counter, cross_section_mode, cmap_select)
def _cross_section_view(trigger=None, mode=None, cmap=None):
    s = state
    if not s.has_vertical:
        return pn.pane.Markdown("Requires vertical dimension.")
    var = _current_var()
    tbl = _table_name(var)
    where = _where_sql()
    try:
        qv = _q(var)
        if cross_section_mode.value == "Lat vs Level" and s.lat_key:
            lat_sel, lat_grp = _spatial_bin_expr(s.lat_key)
            df = s.source.execute(
                f"SELECT {lat_sel}, {s.vertical_key}, AVG({qv}) as {qv} "
                f"FROM {tbl}{where} GROUP BY {lat_grp}, {s.vertical_key} "
                f"ORDER BY {lat_grp}, {s.vertical_key}"
            )
            x_dim = s.lat_key
        elif cross_section_mode.value == "Time vs Level" and s.has_time:
            df = s.source.execute(
                f"SELECT {s.time_key}, {s.vertical_key}, AVG({qv}) as {qv} "
                f"FROM {tbl}{where} GROUP BY {s.time_key}, {s.vertical_key} "
                f"ORDER BY {s.time_key}, {s.vertical_key}"
            )
            x_dim = s.time_key
        else:
            return pn.pane.Markdown("Selected mode requires lat or time dimension.")

        if df.empty:
            return pn.pane.Markdown("No data for selected range.")
        df = _apply_unit_conversion(df, var)
        label = _unit_label(var)

        plot = df.hvplot.heatmap(
            x=x_dim, y=s.vertical_key, C=var, cmap=cmap_select.value,
            title=f"Cross Section: {label} ({cross_section_mode.value})",
            width=_PLOT_W, height=_PLOT_H, colorbar=True,
        )
        return pn.Column(
            pn.pane.Markdown(f"**{s.vertical_key}** axis: higher values = lower altitude"),
            pn.pane.HoloViews(plot, sizing_mode="stretch_width"),
        )
    except Exception as e:
        return pn.pane.Markdown(f"**Error:** {e}")


# -- C4: Correlation Map --
corr_ref_lat = pn.widgets.FloatInput(name="Ref Lat", value=0.0, step=1.0, width=100)
corr_ref_lon = pn.widgets.FloatInput(name="Ref Lon", value=0.0, step=1.0, width=100)


@pn.depends(_update_counter, corr_ref_lat, corr_ref_lon, cmap_select)
def _correlation_map_view(trigger=None, ref_lat=None, ref_lon=None, cmap=None):
    s = state
    if not (s.has_time and s.has_spatial):
        return pn.pane.Markdown("Requires time + spatial dimensions.")
    var = _current_var()
    tbl = _table_name(var)
    where = _where_sql()
    try:
        # Find nearest grid cell to reference point
        grid = s.source.execute(
            f"SELECT DISTINCT {s.lat_key}, {s.lon_key} FROM {tbl} "
            f"ORDER BY ABS({s.lat_key} - {corr_ref_lat.value}) + "
            f"ABS({s.lon_key} - {corr_ref_lon.value}) LIMIT 1"
        )
        if grid.empty:
            return pn.pane.Markdown("No grid points found.")
        rlat = grid[s.lat_key].iloc[0]
        rlon = grid[s.lon_key].iloc[0]

        # Get reference time series
        qv = _q(var)
        ref_ts = s.source.execute(
            f"SELECT {s.time_key}, {qv} FROM {tbl} "
            f"WHERE {s.lat_key} = {rlat} AND {s.lon_key} = {rlon}"
            + (f" AND {' AND '.join(_filter_clauses())}" if _filter_clauses() else "")
            + f" ORDER BY {s.time_key}"
        )
        if ref_ts.empty or len(ref_ts) < 5:
            return pn.pane.Markdown("Not enough time steps at reference point.")

        # Get spatially binned data for correlation (avoids loading 68M rows)
        lat_sel, lon_sel, lat_grp, lon_grp = _binned_spatial_select()
        full = s.source.execute(
            f"SELECT {s.time_key}, {lat_sel}, {lon_sel}, AVG({qv}) as {qv} "
            f"FROM {tbl}{where} "
            f"GROUP BY {s.time_key}, {lat_grp}, {lon_grp}"
        )
        if full.empty:
            return pn.pane.Markdown("No data for selected range.")
        # Pivot and compute correlation
        pivot = full.pivot_table(
            index=s.time_key, columns=[s.lat_key, s.lon_key], values=var,
        )
        ref_series = ref_ts.set_index(s.time_key)[var]
        corrs = pivot.corrwith(ref_series)
        corr_df = corrs.reset_index()
        corr_df.columns = [s.lat_key, s.lon_key, "correlation"]

        return pn.Column(
            pn.pane.Markdown(
                f"**Reference:** ({rlat}, {rlon}) | "
                f"**Grid points:** {len(corr_df):,}"
            ),
            corr_df.hvplot.heatmap(
                x=s.lon_key, y=s.lat_key, C="correlation", cmap="RdBu_r",
                title=f"Correlation with ({rlat}, {rlon})",
                width=_PLOT_W, height=_PLOT_H, colorbar=True, clim=(-1, 1),
            ),
        )
    except Exception as e:
        return pn.pane.Markdown(f"**Error:** {e}")


# ================================================================
# COMPARE TABS
# ================================================================

compare_var_x = pn.widgets.Select(name="X Variable", options=[], value=None)
compare_var_y = pn.widgets.Select(name="Y Variable", options=[], value=None)


def _update_compare_options():
    tables = state.tables
    compare_var_x.options = tables
    compare_var_y.options = tables
    if len(tables) >= 2:
        compare_var_x.value = tables[0]
        compare_var_y.value = tables[1]
    elif len(tables) == 1:
        compare_var_x.value = tables[0]
        compare_var_y.value = tables[0]


@pn.depends(_update_counter, compare_var_x, compare_var_y)
def _cross_variable(trigger=None, vx=None, vy=None):
    s = state
    if not compare_var_x.value or not compare_var_y.value:
        return pn.pane.Markdown("Select two variables to compare.")

    var_x = compare_var_x.value
    var_y = compare_var_y.value

    try:
        sql_x = _build_sql_query(var_x, limit=50_000)
        sql_y = _build_sql_query(var_y, limit=50_000)
        df_x = s.source.execute(sql_x)
        df_y = s.source.execute(sql_y)
    except Exception as e:
        return pn.pane.Markdown(f"Error loading data: {e}")

    shared_coords = [c for c in df_x.columns if c in df_y.columns and c != var_x and c != var_y]
    if not shared_coords:
        return pn.pane.Markdown("No shared coordinates between variables.")

    merged = pd.merge(df_x, df_y, on=shared_coords, how="inner")
    if merged.empty:
        return pn.pane.Markdown("No overlapping data between variables.")

    if len(merged) > 10000:
        merged = merged.sample(10000, random_state=42)

    corr = merged[var_x].corr(merged[var_y])

    scatter = merged.hvplot.scatter(
        x=var_x, y=var_y, color="#00897B", alpha=0.3, size=3,
        title=f"{var_x} vs {var_y} (r = {corr:.3f})",
        width=_PLOT_W, height=_PLOT_H,
    )

    return pn.Column(
        pn.pane.Markdown(
            f"**Correlation:** {corr:.4f} | "
            f"**Points:** {len(merged):,} | "
            f"**Shared coords:** {', '.join(shared_coords)}",
            width=_PLOT_W,
        ),
        pn.pane.HoloViews(scatter, width=_PLOT_W, height=_PLOT_H + 20),
    )


@pn.depends(_update_counter)
def _statistics(trigger=None):
    s = state
    rows = []
    for var in s.tables:
        try:
            tbl = _table_name(var)
            qv = _q(var)
            where = _where_sql()
            null_clause = f" AND {qv} IS NOT NULL" if where else f" WHERE {qv} IS NOT NULL"
            stats = _cached_execute(
                f"SELECT COUNT({qv}) as cnt, AVG({qv}) as mean, "
                f"STDDEV({qv}) as std, MIN({qv}) as min_v, "
                f"MAX({qv}) as max_v FROM {tbl}{where}{null_clause}"
            )
            rows.append({
                "Variable": var,
                "Units": _get_units(var),
                "Count": f"{int(stats['cnt'].iloc[0]):,}",
                "Mean": f"{stats['mean'].iloc[0]:.4f}",
                "Std": f"{stats['std'].iloc[0]:.4f}",
                "Min": f"{stats['min_v'].iloc[0]:.4f}",
                "Max": f"{stats['max_v'].iloc[0]:.4f}",
            })
        except Exception:
            rows.append({"Variable": var, "Count": "Error"})

    stats_df = pd.DataFrame(rows)
    return pn.widgets.Tabulator(
        stats_df, width=_PLOT_W, show_index=False, disabled=True,
        layout="fit_columns",
    )


# ================================================================
# TOOLS TABS
# ================================================================

# -- SQL Explorer --
sql_input = pn.widgets.TextAreaInput(name="SQL Query", height=100, width=650)
sql_run = pn.widgets.Button(name="Run Query", button_type="primary")
sql_status = pn.pane.Markdown("")
sql_result_pane = pn.Column()


def _set_default_sql():
    s = state
    dims = list(s.dim_info.keys())[:2]
    tbl = _table_name(s.primary_var)
    rows = s.size_info.get("rows", 0)
    sql_input.value = (
        f"SELECT {', '.join(dims)}, AVG({_q(s.primary_var)}) as avg_val\n"
        f"FROM {tbl}\n"
        f"GROUP BY {', '.join(dims)}\n"
        f"ORDER BY {dims[0]}"
    )
    sql_status.object = f"*Tip: Dataset has {rows:,} rows. Use LIMIT for large queries.*"


def _run_sql(event):
    try:
        sql_status.object = "*Running...*"
        df = state.source.execute(sql_input.value)
        n = len(df)
        display_df = df.head(1000)
        sql_result_pane.clear()
        sql_result_pane.append(
            pn.widgets.Tabulator(
                display_df, width=_PLOT_W, show_index=False,
                disabled=True, layout="fit_columns",
                page_size=50, pagination="remote",
            )
        )
        sql_status.object = f"**{n:,} rows**" + (" (showing first 1,000)" if n > 1000 else "")
    except Exception as e:
        sql_result_pane.clear()
        sql_result_pane.append(pn.pane.Alert(str(e), alert_type="danger"))
        sql_status.object = "**Error**"


sql_run.on_click(_run_sql)
_set_default_sql()


# -- Data Export --
export_format = pn.widgets.Select(
    name="Format", options=["CSV", "Parquet", "JSON"], value="CSV", width=120,
)
export_btn = pn.widgets.Button(name="Export Filtered Data", button_type="warning", width=200)
export_status = pn.pane.Markdown("", width=400)
download_widget = pn.Column()


def _on_export(event):
    try:
        df = _get_filtered(limit=None)  # Export needs full data, no SQL LIMIT
        var = _current_var()
        fmt = export_format.value

        if fmt == "CSV":
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            content = buf.getvalue()
            filename = f"{var}_filtered.csv"
        elif fmt == "Parquet":
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            content = buf.getvalue()
            filename = f"{var}_filtered.parquet"
        elif fmt == "JSON":
            content = df.to_json(orient="records", date_format="iso")
            filename = f"{var}_filtered.json"

        if isinstance(content, str):
            content = content.encode("utf-8")

        fd = pn.widgets.FileDownload(
            callback=lambda c=content: io.BytesIO(c),
            filename=filename,
            button_type="success",
            label=f"Download {filename}",
        )
        download_widget.clear()
        download_widget.append(fd)
        export_status.object = f"**Ready:** {len(df):,} rows, {len(content):,} bytes"
    except Exception as e:
        export_status.object = f"**Error:** {e}"


export_btn.on_click(_on_export)


# -- Dataset Info --
info_pane = pn.pane.Markdown("", width=_PLOT_W)


def _build_info():
    s = state
    meta = s.source.get_metadata(s.primary_var)
    md = f"### Dataset: {s.name}\n{meta.get('description', 'N/A')}\n\n"
    md += f"**Backend:** {'xarray + DataFusion' if s.backend == 'xarray' else 'DuckDB'}  \n"
    md += f"**Variables:** {', '.join(s.tables)}  \n"
    md += f"**Data points:** {s.size_info['rows']:,} ({s.size_info['estimated_mb']} MB est.)\n\n"
    md += "| Dimension | Type | Role | Range | Size |\n|---|---|---|---|---|\n"
    for dim, d in s.dim_info.items():
        role = d.get("role", "-")
        md += f"| {dim} | {d['type']} | {role} | {d.get('min', 'N/A')} - {d.get('max', 'N/A')} | {d['size']} |\n"

    if len(s.tables) > 1:
        md += "\n### All Variables\n"
        for tbl_name in s.tables:
            tbl_meta = s.source.get_metadata(tbl_name)
            units = _get_units(tbl_name)
            unit_str = f" [{units}]" if units else ""
            md += f"- **{tbl_name}**{unit_str}: {tbl_meta.get('description', 'N/A')}\n"

    if s.ds is not None:
        ds_attrs = dict(s.ds.attrs)
        if ds_attrs:
            md += "\n### Attributes\n"
            for k, v in ds_attrs.items():
                md += f"- **{k}**: {v}\n"

    info_pane.object = md


_build_info()


# ================================================================
# NEW TOOLS FEATURES (Phase D)
# ================================================================

# -- D1: CF Metadata Inspector --
@pn.depends(_update_counter)
def _cf_inspector_view(trigger=None):
    s = state
    md = "### CF Metadata Inspector\n\n"

    if s.backend == "xarray" and s.ds is not None:
        # Global attributes
        md += "#### Global Attributes\n"
        for k, v in dict(s.ds.attrs).items():
            md += f"- **{k}**: {v}\n"
        md += "\n"

        # Per-variable details
        md += "#### Variable Metadata\n"
        for var in s.tables:
            if var in s.ds:
                attrs = dict(s.ds[var].attrs)
                enc = {k: v for k, v in s.ds[var].encoding.items()
                       if k in ("dtype", "scale_factor", "add_offset", "_FillValue",
                                "missing_value", "compression", "chunksizes")}
                md += f"\n**{var}**\n"
                md += f"- Shape: `{dict(s.ds[var].sizes)}`\n"
                md += f"- Dtype: `{s.ds[var].dtype}`\n"
                if attrs:
                    for ak, av in attrs.items():
                        md += f"- {ak}: `{av}`\n"
                if enc:
                    md += f"- Encoding: `{enc}`\n"

        # Per-coordinate details
        md += "\n#### Coordinate Metadata\n"
        for coord in s.ds.coords:
            attrs = dict(s.ds.coords[coord].attrs)
            md += f"\n**{coord}** (size: {s.ds.coords[coord].size})\n"
            for ak, av in attrs.items():
                md += f"- {ak}: `{av}`\n"
    else:
        # DuckDB / tabular backend
        md += "#### Column Information\n"
        try:
            info = s.source.conn.execute(
                f"SELECT column_name, data_type FROM information_schema.columns "
                f"WHERE table_name = '{s.source.table_name}'"
            ).fetchdf()
            for _, row in info.iterrows():
                col = row["column_name"]
                stats = s.source.execute(
                    f'SELECT COUNT(*) as total, COUNT("{col}") as non_null, '
                    f'COUNT(DISTINCT "{col}") as unique_vals FROM {s.source.table_name}'
                )
                md += (
                    f"- **{col}** ({row['data_type']}): "
                    f"{int(stats['non_null'].iloc[0]):,} non-null, "
                    f"{int(stats['unique_vals'].iloc[0]):,} unique\n"
                )
        except Exception as e:
            md += f"Error reading metadata: {e}\n"

    return pn.pane.Markdown(md, width=_PLOT_W)


# -- D2: Multi-Variable Overlay --
overlay_fill_var = pn.widgets.Select(name="Fill Variable", options=[], width=150)
overlay_contour_var = pn.widgets.Select(name="Contour Variable", options=[], width=150)
overlay_levels = pn.widgets.IntSlider(name="Contour Levels", start=3, end=20, value=8, width=200)


def _update_overlay_options():
    if len(state.tables) >= 2:
        overlay_fill_var.options = state.tables
        overlay_contour_var.options = state.tables
        overlay_fill_var.value = state.tables[0]
        overlay_contour_var.value = state.tables[1]


@pn.depends(_update_counter, overlay_fill_var, overlay_contour_var, overlay_levels, cmap_select)
def _overlay_view(trigger=None, fvar=None, cvar=None, levels=None, cmap=None):
    s = state
    if not s.has_spatial or len(s.tables) < 2:
        return pn.pane.Markdown("Requires 2+ variables with spatial dimensions.")
    fill_var = overlay_fill_var.value
    contour_var = overlay_contour_var.value
    if not fill_var or not contour_var:
        return pn.pane.Markdown("Select fill and contour variables.")
    try:
        fill_agg = _get_spatial_agg_sql(fill_var)
        contour_agg = _get_spatial_agg_sql(contour_var)
        if fill_agg.empty or contour_agg.empty:
            return pn.pane.Markdown("No data for selected range.")

        fill_agg = _apply_unit_conversion(fill_agg, fill_var)
        contour_agg = _apply_unit_conversion(contour_agg, contour_var)

        # Pivot both to grids
        fill_piv = fill_agg.pivot_table(
            index=s.lat_key, columns=s.lon_key, values=fill_var, aggfunc="mean",
        )
        contour_piv = contour_agg.pivot_table(
            index=s.lat_key, columns=s.lon_key, values=contour_var, aggfunc="mean",
        )

        fill_img = hv.Image(
            (fill_piv.columns.values, fill_piv.index.values, fill_piv.values),
            kdims=[s.lon_key, s.lat_key], vdims=[fill_var],
        ).opts(cmap=cmap_select.value, colorbar=True, width=_PLOT_W, height=_PLOT_H,
               title=f"{_unit_label(fill_var)} (fill) + {_unit_label(contour_var)} (contours)")

        contour_img = hv.Image(
            (contour_piv.columns.values, contour_piv.index.values, contour_piv.values),
            kdims=[s.lon_key, s.lat_key], vdims=[contour_var],
        )
        contours = hv.operation.contours(contour_img, levels=overlay_levels.value).opts(
            line_color="black", line_width=1.5,
        )

        return fill_img * contours
    except Exception as e:
        return pn.pane.Markdown(f"**Error:** {e}")


# -- D3: Quick Stats per Region --
_REGIONS = {
    "Global": (-90, 90, -180, 180),
    "Tropics (-23.5 to 23.5)": (-23.5, 23.5, -180, 180),
    "Arctic (>66.5N)": (66.5, 90, -180, 180),
    "Antarctic (<66.5S)": (-90, -66.5, -180, 180),
    "Northern Hemisphere": (0, 90, -180, 180),
    "Southern Hemisphere": (-90, 0, -180, 180),
    "Custom BBox": None,
}
region_select = pn.widgets.Select(
    name="Region", options=list(_REGIONS.keys()), value="Global", width=200,
)
region_lat_min = pn.widgets.FloatInput(name="Lat Min", value=-90, width=80)
region_lat_max = pn.widgets.FloatInput(name="Lat Max", value=90, width=80)
region_lon_min = pn.widgets.FloatInput(name="Lon Min", value=-180, width=80)
region_lon_max = pn.widgets.FloatInput(name="Lon Max", value=180, width=80)


@pn.depends(_update_counter, region_select, region_lat_min, region_lat_max,
            region_lon_min, region_lon_max)
def _region_stats_view(trigger=None, region=None, lat_lo=None, lat_hi=None,
                       lon_lo=None, lon_hi=None):
    s = state
    if not s.has_spatial:
        return pn.pane.Markdown("Requires spatial dimensions.")
    var = _current_var()
    tbl = _table_name(var)
    try:
        bounds = _REGIONS.get(region_select.value)
        if bounds is None:
            lat0, lat1 = region_lat_min.value, region_lat_max.value
            lon0, lon1 = region_lon_min.value, region_lon_max.value
        else:
            lat0, lat1, lon0, lon1 = bounds

        clauses = _filter_clauses()
        clauses.append(f"{s.lat_key} >= {lat0} AND {s.lat_key} <= {lat1}")
        clauses.append(f"{s.lon_key} >= {lon0} AND {s.lon_key} <= {lon1}")
        where = " WHERE " + " AND ".join(clauses)

        qv = _q(var)
        stats = s.source.execute(
            f"SELECT COUNT({qv}) as count, AVG({qv}) as mean, "
            f"STDDEV({qv}) as std, MIN({qv}) as min_val, MAX({qv}) as max_val "
            f"FROM {tbl}{where} AND {qv} IS NOT NULL"
        )
        if stats.empty:
            return pn.pane.Markdown("No data in selected region.")

        label = _unit_label(var)
        region_name = region_select.value
        if bounds is None:
            region_name = f"Custom ({lat0},{lon0}) to ({lat1},{lon1})"

        md = f"### Region Stats: {region_name}\n\n"
        md += f"**Variable:** {label}\n\n"
        md += "| Statistic | Value |\n|---|---|\n"
        md += f"| Count | {int(stats['count'].iloc[0]):,} |\n"
        md += f"| Mean | {stats['mean'].iloc[0]:.4f} |\n"
        md += f"| Std Dev | {stats['std'].iloc[0]:.4f} |\n"
        md += f"| Min | {stats['min_val'].iloc[0]:.4f} |\n"
        md += f"| Max | {stats['max_val'].iloc[0]:.4f} |\n"

        return pn.pane.Markdown(md, width=_PLOT_W)
    except Exception as e:
        return pn.pane.Markdown(f"**Error:** {e}")


# -- D4: Dataset Comparison --
compare_file_input = pn.widgets.FileInput(
    name="Compare Dataset", accept=",".join(_ALL_EXTENSIONS), multiple=False, height=60,
)
compare_path_input = pn.widgets.TextInput(
    name="Compare Path", placeholder="/data/other.nc", width=250,
)
compare_load_btn = pn.widgets.Button(name="Load Comparison", button_type="warning", width=200)
compare_status = pn.pane.Markdown("", width=400)
compare_state = None


def _on_compare_load(event):
    global compare_state
    try:
        if compare_path_input.value.strip():
            path = compare_path_input.value.strip()
            source, name, backend = _detect_and_load(path)
        elif compare_file_input.value:
            fname = compare_file_input.filename
            suffix = "." + fname.rsplit(".", 1)[-1] if "." in fname else ".nc"
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp.write(compare_file_input.value)
            tmp.flush()
            tmp.close()
            source, name, backend = _detect_and_load(tmp.name, fname)
        else:
            compare_status.object = "Upload a file or enter a path."
            return

        ds = source.dataset if backend == "xarray" else None
        compare_state = AppState(source, ds, name, backend=backend)
        compare_status.object = (
            f"**Loaded comparison:** {name} | "
            f"{len(compare_state.tables)} vars | "
            f"{compare_state.size_info['rows']:,} rows"
        )
    except Exception as e:
        compare_status.object = f"**Error:** {e}"


compare_load_btn.on_click(_on_compare_load)


@pn.depends(_update_counter)
def _dataset_comparison_view(trigger=None):
    if compare_state is None:
        return pn.pane.Markdown("Load a comparison dataset above.")
    s = state
    c = compare_state
    shared_vars = [v for v in s.tables if v in c.tables]
    if not shared_vars:
        return pn.pane.Markdown(
            f"No shared variables. Primary: {s.tables}, Comparison: {c.tables}"
        )

    md = f"### Comparison: {s.name} vs {c.name}\n\n"
    md += f"**Shared variables:** {', '.join(shared_vars)}\n\n"
    md += "| Variable | Primary Mean | Compare Mean | Difference |\n|---|---|---|---|\n"

    for var in shared_vars:
        try:
            p_tbl = _table_name(var)
            c_tbl = var if c.backend == "xarray" else c.source.table_name
            qv = _q(var)
            p_stats = s.source.execute(f"SELECT AVG({qv}) as m FROM {p_tbl}")
            c_stats = c.source.execute(f"SELECT AVG({qv}) as m FROM {c_tbl}")
            p_mean = p_stats["m"].iloc[0]
            c_mean = c_stats["m"].iloc[0]
            diff = c_mean - p_mean
            md += f"| {var} | {p_mean:.4f} | {c_mean:.4f} | {diff:+.4f} |\n"
        except Exception:
            md += f"| {var} | Error | Error | - |\n"

    # If both have spatial dims, show difference map for first shared var
    if s.has_spatial and c.has_spatial and shared_vars:
        var = shared_vars[0]
        try:
            p_tbl = _table_name(var)
            c_tbl = var if c.backend == "xarray" else c.source.table_name
            qv = _q(var)
            lat_sel, lon_sel, lat_grp, lon_grp = _binned_spatial_select()
            p_agg = s.source.execute(
                f"SELECT {lat_sel}, {lon_sel}, AVG({qv}) as {qv} "
                f"FROM {p_tbl} GROUP BY {lat_grp}, {lon_grp}"
            )
            # Use same binning for comparison dataset
            c_lat_sel, c_lon_sel = lat_sel, lon_sel
            c_lat_grp, c_lon_grp = lat_grp, lon_grp
            if c.lat_key != s.lat_key:
                c_lat_sel = c_lat_sel.replace(s.lat_key, c.lat_key)
                c_lat_grp = c_lat_grp.replace(s.lat_key, c.lat_key)
            if c.lon_key != s.lon_key:
                c_lon_sel = c_lon_sel.replace(s.lon_key, c.lon_key)
                c_lon_grp = c_lon_grp.replace(s.lon_key, c.lon_key)
            c_agg = c.source.execute(
                f"SELECT {c_lat_sel}, {c_lon_sel}, AVG({qv}) as {qv} "
                f"FROM {c_tbl} GROUP BY {c_lat_grp}, {c_lon_grp}"
            )
            merged = pd.merge(
                p_agg, c_agg, on=[s.lat_key, s.lon_key], suffixes=("_primary", "_compare"),
            )
            if not merged.empty:
                merged["difference"] = merged[f"{var}_compare"] - merged[f"{var}_primary"]
                vmax = max(abs(merged["difference"].min()), abs(merged["difference"].max()))
                plot = merged.hvplot.heatmap(
                    x=s.lon_key, y=s.lat_key, C="difference", cmap="RdBu_r",
                    title=f"Difference: {c.name} - {s.name} ({var})",
                    width=_PLOT_W, height=_PLOT_H, colorbar=True, clim=(-vmax, vmax),
                )
                return pn.Column(
                    pn.pane.Markdown(md, width=_PLOT_W),
                    pn.pane.HoloViews(plot, width=_PLOT_W, height=_PLOT_H + 20),
                )
        except Exception:
            pass  # Fall through to text-only view

    return pn.pane.Markdown(md, width=_PLOT_W)


# ================================================================
# TAB ASSEMBLY - Grouped sections
# ================================================================
main_tabs = pn.Tabs(tabs_location="above", dynamic=True)


def _rebuild_tabs():
    """Rebuild all tabs and info for the current dataset."""
    s = state

    _set_default_sql()
    _build_info()
    _update_compare_options()
    _populate_anim_steps()

    main_tabs.clear()

    # ---- EXPLORE GROUP ----
    explore_tabs = pn.Tabs(tabs_location="above")

    if s.has_spatial or len(s.dim_info) >= 2:
        map_controls = pn.Row(map_mode, show_coastlines, projection_select)
        explore_tabs.append(("Spatial Map", pn.Column(map_controls, pn.panel(_spatial_view))))

    # B1: Animation
    if s.has_time and s.has_spatial:
        explore_tabs.append(("Animation", pn.Column(
            anim_player, pn.panel(_animated_spatial_view),
        )))

    if s.lat_key:
        profile_tabs = pn.Tabs(("Latitude", pn.panel(_lat_profile)))
        if s.lon_key:
            profile_tabs.append(("Longitude", pn.panel(_lon_profile)))
        explore_tabs.append(("Profiles", profile_tabs))

    # B4: Transect
    if s.has_spatial:
        explore_tabs.append(("Transect", pn.Column(
            pn.Row(transect_start_lat, transect_start_lon, transect_end_lat, transect_end_lon),
            transect_npoints,
            pn.panel(_transect_view),
        )))

    # B2: Hovmoller
    if s.has_time and s.has_spatial:
        explore_tabs.append(("Hovmoller", pn.Column(
            hovmoller_axis, pn.panel(_hovmoller_view),
        )))

    # B3: Data Coverage
    if s.has_spatial:
        explore_tabs.append(("Data Coverage", pn.panel(_nan_coverage_view)))

    if s.has_vertical:
        explore_tabs.append(("Vertical Profile", pn.panel(_vertical_profile)))

    explore_tabs.append(("Distribution", pn.panel(_histogram)))

    if len(explore_tabs) > 0:
        main_tabs.append(("Explore", explore_tabs))

    # ---- TIME ANALYSIS GROUP ----
    if s.has_time:
        time_tabs = pn.Tabs(tabs_location="above")
        time_tabs.append(("Time Series", pn.panel(_time_series)))
        time_tabs.append(("Anomaly", pn.panel(_anomaly)))
        time_tabs.append(("Rolling Mean", pn.panel(_rolling)))
        time_tabs.append(("Trend", pn.panel(_trend)))
        time_tabs.append(("Climatology", pn.panel(_climatology_tab)))
        main_tabs.append(("Time Analysis", time_tabs))

    # ---- ANALYSIS GROUP (Phase C) ----
    analysis_tabs = pn.Tabs(tabs_location="above")

    # C1: Difference Maps
    if s.has_spatial and (s.has_time or s.has_vertical):
        _init_diff_widgets()
        time_controls = pn.Row(diff_period1_start, diff_period1_end,
                               diff_period2_start, diff_period2_end)
        level_controls = pn.Row(diff_level1, diff_level2)
        analysis_tabs.append(("Difference Map", pn.Column(
            diff_mode, time_controls, level_controls,
            pn.panel(_difference_map_view),
        )))

    # C4: Correlation Map
    if s.has_time and s.has_spatial:
        analysis_tabs.append(("Correlation Map", pn.Column(
            pn.Row(corr_ref_lat, corr_ref_lon),
            pn.panel(_correlation_map_view),
        )))

    # C3: Cross-Sections
    if s.has_vertical:
        analysis_tabs.append(("Cross-Sections", pn.Column(
            cross_section_mode, pn.panel(_cross_section_view),
        )))

    # C2: Spectral Analysis
    if s.has_time:
        analysis_tabs.append(("Spectral Analysis", pn.panel(_spectral_view)))

    if len(analysis_tabs) > 0:
        main_tabs.append(("Analysis", analysis_tabs))

    # ---- COMPARE GROUP ----
    compare_tabs = pn.Tabs(tabs_location="above")

    if len(s.tables) >= 2:
        compare_tabs.append(("Cross-Variable", pn.Column(
            pn.Row(compare_var_x, compare_var_y),
            pn.panel(_cross_variable),
        )))

    compare_tabs.append(("Statistics", pn.panel(_statistics)))

    if len(compare_tabs) > 0:
        main_tabs.append(("Compare", compare_tabs))

    # ---- TOOLS GROUP ----
    tools_tabs = pn.Tabs(tabs_location="above")
    tools_tabs.append(("SQL Explorer", pn.Column(
        sql_input, sql_run, sql_status, sql_result_pane,
    )))
    tools_tabs.append(("Export", pn.Column(
        "### Export Filtered Data",
        pn.Row(export_format, export_btn),
        export_status,
        download_widget,
    )))
    tools_tabs.append(("Dataset Info", info_pane))

    # D1: CF Inspector
    tools_tabs.append(("CF Inspector", pn.panel(_cf_inspector_view)))

    # D2: Variable Overlay
    if s.has_spatial and len(s.tables) >= 2:
        _update_overlay_options()
        tools_tabs.append(("Variable Overlay", pn.Column(
            pn.Row(overlay_fill_var, overlay_contour_var, overlay_levels),
            pn.panel(_overlay_view),
        )))

    # D3: Region Stats
    if s.has_spatial:
        tools_tabs.append(("Region Stats", pn.Column(
            region_select,
            pn.Row(region_lat_min, region_lat_max, region_lon_min, region_lon_max),
            pn.panel(_region_stats_view),
        )))

    # D4: Compare Datasets
    tools_tabs.append(("Compare Datasets", pn.Column(
        compare_file_input, compare_path_input, compare_load_btn, compare_status,
        pn.layout.Divider(),
        pn.panel(_dataset_comparison_view),
    )))

    main_tabs.append(("Tools", tools_tabs))


_rebuild_tabs()


# ----------------------------------------------------------------
# Template
# ----------------------------------------------------------------
sidebar = pn.Column(
    "## Load Data",
    file_input,
    path_input,
    load_btn,
    load_status,
    pn.layout.Divider(),
    "## Filters",
    controls_column,
    apply_btn,
    _update_counter,
    pn.layout.Divider(),
    "## Display",
    cmap_select,
    resolution_mode,
    kelvin_toggle,
    pn.layout.Divider(),
    "## Analysis",
    analysis_widgets_column,
    width=290,
)

template = pn.template.FastListTemplate(
    title="lumen-xarray Explorer",
    sidebar=[sidebar],
    main=[info_banner, main_tabs],
    accent_base_color="#1976D2",
    header_background="#1565C0",
)

# -- CLI arg support (panel serve ... --args file.nc) --
argv = getattr(pn.state, "argv", None) or sys.argv[1:]
_parser = argparse.ArgumentParser()
_parser.add_argument("dataset", nargs="?", default=None)
_cli, _ = _parser.parse_known_args(argv)
if _cli.dataset:
    _load_from_path(_cli.dataset)

template.servable()
