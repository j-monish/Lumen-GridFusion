"""
XArraySQLSource — SQL-backed xarray source for Lumen.

Uses xarray-sql (Apache DataFusion) to provide SQL query capabilities
over N-dimensional xarray datasets. This is the primary source for the
Lumen + Xarray integration, enabling Lumen AI agents to generate SQL
queries against scientific data (NetCDF, Zarr, HDF5, GRIB).

Architecture:
    NetCDF/Zarr file
        -> xarray.open_dataset() (lazy, dask-chunked)
        -> xarray-sql: XarrayContext registers dataset in DataFusion
        -> XArraySQLSource.execute() runs SQL via DataFusion
        -> Lumen Pipeline / AI agents consume DataFrames
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import param
import xarray as xr

from lumen.sources.base import BaseSQLSource, cached, cached_schema
from xarray_sql import XarrayContext

from ._base import XArrayMixin, XARRAY_ENGINES, XARRAY_EXTENSIONS, _detect_engine

logger = logging.getLogger(__name__)

# Safety limit: warn when a query would return more than this many rows
MAX_ROWS_WARNING = 10_000_000
# Default row limit for unbounded SELECT * queries
DEFAULT_ROW_LIMIT = 5_000_000


def _xr_dtype_to_pandas_type(dtype) -> str:
    """Map numpy/xarray dtype to a JSON schema type string."""
    kind = np.dtype(dtype).kind
    if kind == "f":
        return "number"
    elif kind in "iu":
        return "integer"
    elif kind == "b":
        return "boolean"
    elif kind in "US":
        return "string"
    elif kind == "M":
        return "string"  # datetime rendered as string in schema
    return "string"


class XArraySQLSource(BaseSQLSource, XArrayMixin):
    """
    SQL-queryable xarray data source for Lumen.

    Wraps xarray datasets with Apache DataFusion (via xarray-sql) to
    provide full SQL query support. Each data variable in the dataset
    is exposed as a separate SQL table with coordinate columns.

    Parameters
    ----------
    uri : str
        Path or URL to a NetCDF/Zarr/HDF5/GRIB file.
    engine : str or None
        xarray engine to use. Auto-detected from file extension if None.
    chunks : dict or str or None
        Chunking specification for dask-backed lazy loading.
        Use 'auto' for automatic chunking, or a dict like {'time': 100}.
    open_kwargs : dict
        Additional keyword arguments passed to xr.open_dataset().
    variables : list or None
        Subset of data variables to expose. None means all variables.
    tables : list or dict or None
        Inherited from BaseSQLSource. Populated automatically from dataset
        variables if not provided.

    Examples
    --------
    >>> source = XArraySQLSource(uri="air_temperature.nc")
    >>> source.get_tables()
    ['air']
    >>> df = source.execute("SELECT lat, lon, AVG(air) FROM air GROUP BY lat, lon")
    >>> schema = source.get_schema("air")
    """

    source_type: ClassVar[str] = "xarray"

    # DataFusion is PostgreSQL-compatible for sqlglot dialect translation
    dialect = "postgres"

    uri = param.String(default=None, allow_None=True, doc="""
        Path or URL to the xarray-compatible data file.
        Supports NetCDF (.nc), Zarr (.zarr), HDF5 (.h5), GRIB (.grib).""")

    engine = param.String(default=None, allow_None=True, doc="""
        xarray backend engine. Auto-detected from file extension if None.
        Options: netcdf4, h5netcdf, zarr, cfgrib.""")

    chunks = param.Parameter(default="auto", doc="""
        Dask chunk specification for lazy loading.
        Use 'auto', a dict like {'time': 100}, or None for eager loading.""")

    open_kwargs = param.Dict(default={}, doc="""
        Additional keyword arguments for xr.open_dataset().""")

    variables = param.List(default=None, allow_None=True, doc="""
        Subset of data variables to expose as tables.
        None means all variables are exposed.""")

    tables = param.ClassSelector(class_=(list, dict), default=None, allow_None=True, doc="""
        List or dict of tables. Populated automatically from dataset variables.""")

    _supports_sql: ClassVar[bool] = True

    def __init__(self, _dataset: xr.Dataset | None = None, **params):
        super().__init__(**params)
        self._dataset: xr.Dataset | None = _dataset
        self._ctx: XarrayContext | None = None
        self._registered_tables: set[str] = set()
        self._df_cache: dict[str, pd.DataFrame] = {}

        # Load dataset and register with DataFusion
        self._initialize()

    def _initialize(self):
        """Load the xarray dataset and register variables with DataFusion."""
        self._dataset = self._load_and_filter_dataset(
            dataset=self._dataset,
            uri=self.uri,
            engine=self.engine,
            chunks=self.chunks,
            open_kwargs=self.open_kwargs,
            variables=self.variables,
        )

        # Create DataFusion context and register each variable as a table
        self._ctx = XarrayContext()

        # xarray-sql requires chunks — resolve 'auto' and None to a dict
        if isinstance(self.chunks, dict):
            chunk_spec = self.chunks
        else:
            # Auto-chunk: use full dimension sizes as a single chunk per dim
            chunk_spec = dict(self._dataset.sizes)

        for var_name in self._dataset.data_vars:
            # xarray-sql requires a Dataset (not DataArray), so we create
            # a single-variable dataset for each variable
            var_ds = self._dataset[[var_name]]
            self._ctx.from_dataset(var_name, var_ds, chunks=chunk_spec)
            self._registered_tables.add(var_name)

        # Auto-populate tables param
        if self.tables is None:
            self.tables = list(self._registered_tables)

    @property
    def dataset(self) -> xr.Dataset:
        """Access the underlying xarray Dataset."""
        return self._dataset

    def get_tables(self) -> list[str]:
        """Return list of available table names (one per data variable)."""
        return sorted(self._registered_tables)

    def normalize_table(self, table: str) -> str:
        """
        Normalize table names for fuzzy matching.

        Strips quotes, lowercases, and matches against registered tables
        to handle minor variations from Lumen AI agents.
        """
        stripped = table.strip('"').strip("'").strip("`")
        # Exact match first
        if stripped in self._registered_tables:
            return stripped
        # Case-insensitive match
        lower_map = {t.lower(): t for t in self._registered_tables}
        if stripped.lower() in lower_map:
            return lower_map[stripped.lower()]
        return stripped

    def get_sql_expr(self, table: str | dict) -> str:
        """
        Return a SQL SELECT expression for the given table.

        Overrides BaseSQLSource to work with DataFusion table names
        directly, without requiring the `sql_expr` param.
        """
        if isinstance(table, dict):
            table = list(table.values())[0]
        if isinstance(self.tables, dict):
            table = self.tables.get(table, table)
        table = self.normalize_table(table)
        return f"SELECT * FROM {table}"

    def get_schema(
        self,
        table: str | None = None,
        limit: int | None = None,
        shuffle: bool = False,
    ) -> dict[str, dict[str, Any]] | dict[str, Any]:
        """
        Return JSON schema for the given table(s).

        Overrides BaseSQLSource.get_schema to query DataFusion directly
        instead of going through sqlglot SQL transforms.
        """
        from lumen.util import get_dataframe_schema

        if table is None:
            tables = self.get_tables()
        else:
            tables = [table]

        schemas = {}
        for entry in tables:
            # Get a sample row to determine column types
            sample = self.execute(f"SELECT * FROM {entry} LIMIT 1")
            schema = get_dataframe_schema(sample)["items"]["properties"]

            # Get total count
            count_df = self.execute(f"SELECT COUNT(*) as count FROM {entry}")
            count = int(count_df["count"].iloc[0])
            schema["__len__"] = count

            if not limit:
                # Patch min/max for numeric columns from full data
                enums, min_maxes = [], []
                for name, col_schema in schema.items():
                    if name == "__len__":
                        continue
                    if "enum" in col_schema:
                        enums.append(name)
                    elif "inclusiveMinimum" in col_schema:
                        min_maxes.append(name)

                for col in enums:
                    distinct = self.execute(
                        f"SELECT DISTINCT {col} FROM {entry} ORDER BY {col}"
                    )
                    schema[col]["enum"] = distinct[col].tolist()

                if min_maxes:
                    min_max_exprs = ", ".join(
                        f"MIN({c}) as {c}_min, MAX({c}) as {c}_max"
                        for c in min_maxes
                    )
                    mm = self.execute(f"SELECT {min_max_exprs} FROM {entry}")
                    for col in min_maxes:
                        kind = sample[col].dtype.kind
                        cast = int if kind in "iu" else float if kind == "f" else str if kind == "M" else lambda v: v
                        min_val = mm[f"{col}_min"].iloc[0]
                        max_val = mm[f"{col}_max"].iloc[0]
                        schema[col]["inclusiveMinimum"] = min_val if pd.isna(min_val) else cast(min_val)
                        schema[col]["inclusiveMaximum"] = max_val if pd.isna(max_val) else cast(max_val)

            schemas[entry] = schema

        return schemas if table is None else schemas[table]

    @cached
    def get(self, table: str, **query) -> pd.DataFrame:
        """
        Return a table as a DataFrame, optionally filtered by query params.

        Overrides BaseSQLSource.get to build SQL queries against DataFusion.
        Supports sql_transforms for Lumen pipeline integration.

        Parameters
        ----------
        table : str
            Name of the table (data variable).
        query : dict
            Column-value pairs for filtering. Special keys:
            - sql_transforms: list of SQLTransform objects to apply
        """
        sql_transforms = query.pop("sql_transforms", [])

        table = self.normalize_table(table)
        sql = f"SELECT * FROM {table}"
        conditions = []
        # NOTE: DataFusion does not support parameterized queries, so values
        # are inlined via repr(). This is safe here because DataFusion runs
        # in-process (no network boundary) and query values come from Lumen's
        # internal pipeline, not directly from untrusted user input.
        for col, val in query.items():
            if col.startswith("__"):
                continue
            if isinstance(val, list):
                vals_str = ", ".join(repr(v) for v in val)
                conditions.append(f"{col} IN ({vals_str})")
            elif isinstance(val, slice):
                if val.start is not None:
                    conditions.append(f"{col} >= {val.start!r}")
                if val.stop is not None:
                    conditions.append(f"{col} <= {val.stop!r}")
            else:
                conditions.append(f"{col} = {val!r}")

        if conditions:
            sql += " WHERE " + " AND ".join(conditions)

        # Apply Lumen SQL transforms (SQLFilter, SQLLimit, etc.)
        for transform in sql_transforms:
            try:
                sql = transform.apply(sql)
            except Exception:
                logger.warning(
                    f"SQL transform {type(transform).__name__} failed, skipping"
                )

        return self.execute(sql)

    def execute(
        self,
        sql_query: str,
        params: list | dict | None = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Execute a SQL query against the registered xarray data via DataFusion.

        Parameters
        ----------
        sql_query : str
            SQL query string. Tables correspond to dataset variable names.
            DataFusion supports PostgreSQL-compatible SQL syntax.
        params : list | dict | None
            Not supported by DataFusion — included for API compatibility.

        Returns
        -------
        pd.DataFrame
            Query results as a pandas DataFrame.

        Examples
        --------
        >>> source.execute("SELECT lat, lon, AVG(air) FROM air GROUP BY lat, lon")
        >>> source.execute("SELECT * FROM air WHERE lat > 60 LIMIT 10")
        """
        if params:
            # DataFusion doesn't support parameterized queries natively.
            # For basic cases, inline the parameters.
            if isinstance(params, list):
                for p in params:
                    sql_query = sql_query.replace("?", repr(p), 1)
            elif isinstance(params, dict):
                for k, v in params.items():
                    sql_query = sql_query.replace(f":{k}", repr(v))

        result = self._ctx.sql(sql_query)
        return result.to_pandas()

    async def execute_async(
        self,
        sql_query: str,
        params: list | dict | None = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Execute a SQL query asynchronously.

        Runs the synchronous execute() in a thread pool to avoid
        blocking the event loop. Used by Lumen AI agents.
        """
        return await asyncio.to_thread(self.execute, sql_query, params, *args, **kwargs)

    async def get_async(self, table: str, **query) -> pd.DataFrame:
        """
        Return a table asynchronously; optionally filtered.

        Used by Lumen AI pipeline for non-blocking data access.
        """
        sql = self.get_sql_expr(table)
        conditions = []
        for col, val in query.items():
            if col.startswith("__") or col == "sql_transforms":
                continue
            if isinstance(val, list):
                vals_str = ", ".join(repr(v) for v in val)
                conditions.append(f"{col} IN ({vals_str})")
            elif isinstance(val, slice):
                if val.start is not None:
                    conditions.append(f"{col} >= {val.start!r}")
                if val.stop is not None:
                    conditions.append(f"{col} <= {val.stop!r}")
            else:
                conditions.append(f"{col} = {val!r}")
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        return await self.execute_async(sql)

    def estimate_size(self, table: str | None = None) -> dict[str, Any]:
        """
        Estimate the memory size of a table for large dataset safety.

        Returns a dict with row count, estimated bytes, and whether
        it exceeds the safety threshold.
        """
        tables = [table] if table else self.get_tables()
        result = {}
        for tbl in tables:
            count_df = self.execute(f"SELECT COUNT(*) as count FROM {tbl}")
            n_rows = int(count_df["count"].iloc[0])
            # Estimate: each row is roughly n_cols * 8 bytes (float64)
            var = self._dataset[tbl]
            n_cols = len(var.dims) + 1  # coordinates + data variable
            est_bytes = n_rows * n_cols * 8
            result[tbl] = {
                "rows": n_rows,
                "columns": n_cols,
                "estimated_bytes": est_bytes,
                "estimated_mb": round(est_bytes / 1024**2, 1),
                "exceeds_warning": n_rows > MAX_ROWS_WARNING,
            }
        return result if table is None else result[tbl]

    def create_sql_expr_source(
        self,
        tables: dict[str, str],
        params: dict[str, list | dict] | None = None,
        **kwargs,
    ):
        """
        Create a new XArraySQLSource with custom SQL expressions as tables.

        Shares the same underlying dataset and DataFusion context to avoid
        expensive re-registration of variables for large datasets.
        """
        new_source = XArraySQLSource.__new__(XArraySQLSource)
        # Share dataset and context instead of re-initializing
        new_source._dataset = self._dataset
        new_source._ctx = self._ctx
        new_source._registered_tables = set(self._registered_tables)
        new_source._df_cache = {}
        new_source.uri = self.uri
        new_source.engine = self.engine
        new_source.chunks = self.chunks
        new_source.variables = self.variables
        new_source.open_kwargs = self.open_kwargs
        new_source.tables = tables
        new_source.dialect = self.dialect
        new_source.name = f"{self.name}_expr" if hasattr(self, 'name') else "xarray_expr"
        return new_source

    def _get_table_metadata(self, tables: list[str]) -> dict[str, Any]:
        """Delegate to shared mixin for metadata extraction."""
        return self._build_table_metadata(self._dataset, tables)

    def get_dimension_info(self, table: str | None = None) -> dict[str, Any]:
        """
        Return detailed dimension information for UI controls and AI context.

        Provides coordinate values, ranges, and types that the AI agent
        or UI needs to construct meaningful queries.
        """
        tables = [table] if table else list(self._dataset.data_vars)
        info = self._build_dimension_info(self._dataset, tables)
        return info if table is None else info.get(table, {})

    def clear_cache(self, *events):
        """Clear cached data and schema."""
        super().clear_cache(*events)
        self._df_cache.clear()

    def close(self):
        """Clean up resources."""
        if self._dataset is not None:
            self._dataset.close()
        self._ctx = None
        self._registered_tables.clear()
        self._df_cache.clear()

    def to_spec(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Serialize this source to a Lumen spec dict."""
        spec = {"type": self.source_type}
        if self.uri:
            spec["uri"] = str(self.uri)
        if self.engine:
            spec["engine"] = self.engine
        if self.chunks:
            spec["chunks"] = self.chunks
        if self.variables:
            spec["variables"] = self.variables
        if self.open_kwargs:
            spec["open_kwargs"] = self.open_kwargs
        return spec

    @classmethod
    def from_spec(cls, spec: dict[str, Any], **kwargs) -> XArraySQLSource:
        """Create an XArraySQLSource from a Lumen spec dict."""
        params = {k: v for k, v in spec.items() if k != "type"}
        params.update(kwargs)
        return cls(**params)

    def __repr__(self):
        tables = self.get_tables()
        dims = dict(self._dataset.sizes) if self._dataset else {}
        return (
            f"XArraySQLSource(uri={self.uri!r}, tables={tables}, "
            f"dims={dims}, dialect='{self.dialect}')"
        )
