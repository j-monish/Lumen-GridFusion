"""
Xarray-specific transforms for Lumen pipelines.

These transforms operate on DataFrames that were produced from xarray
datasets (via XArraySource or XArraySQLSource). They handle common
scientific data operations: dimension slicing, spatial filtering,
aggregation, time resampling, anomaly computation, and rolling windows.

All transforms follow Lumen's Transform API: they accept a DataFrame
and return a transformed DataFrame.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
import param

from lumen.transforms.base import Transform


class DimensionSlice(Transform):
    """
    Slice a DataFrame along a dimension (column) by range or explicit values.

    Works on coordinate columns produced by xarray-to-DataFrame conversion.
    Handles datetime and numeric type coercion automatically.

    Examples
    --------
    >>> DimensionSlice(dimension="time", start="2013-06-01", stop="2013-12-31").apply(df)
    >>> DimensionSlice(dimension="lat", values=[15.0, 30.0, 45.0]).apply(df)
    """

    transform_type: ClassVar[str] = "dimension_slice"

    dimension = param.String(doc="Column name to slice on.")
    start = param.Parameter(default=None, doc="Inclusive start value for range slice.")
    stop = param.Parameter(default=None, doc="Inclusive stop value for range slice.")
    values = param.List(default=None, allow_None=True, doc="Explicit values to select.")
    nearest = param.Boolean(default=False, doc="""
        If True, select the nearest values for numeric columns
        when exact matches aren't found.""")

    def apply(self, table: pd.DataFrame) -> pd.DataFrame:
        if self.dimension not in table.columns:
            return table

        col = table[self.dimension]

        if self.values is not None:
            if self.nearest and col.dtype.kind in "fiu":
                mask = pd.Series(False, index=table.index)
                for v in self.values:
                    idx = (col - float(v)).abs().idxmin()
                    mask.iloc[idx] = True
                return table[mask].reset_index(drop=True)
            return table[col.isin(self.values)].reset_index(drop=True)

        result = table
        if self.start is not None:
            start = self._coerce(self.start, col.dtype)
            result = result[result[self.dimension] >= start]
        if self.stop is not None:
            stop = self._coerce(self.stop, col.dtype)
            result = result[result[self.dimension] <= stop]

        return result.reset_index(drop=True)

    @staticmethod
    def _coerce(value, dtype):
        """Coerce a filter value to match the column's dtype."""
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return pd.Timestamp(value)
        elif pd.api.types.is_numeric_dtype(dtype):
            return float(value)
        return value


class SpatialBBox(Transform):
    """
    Filter rows to a lat/lon bounding box.

    Expects columns named 'lat' and 'lon' (or custom names via params).

    Examples
    --------
    >>> SpatialBBox(lat_min=30, lat_max=60, lon_min=200, lon_max=280).apply(df)
    """

    transform_type: ClassVar[str] = "spatial_bbox"

    lat_col = param.String(default="lat", doc="Latitude column name.")
    lon_col = param.String(default="lon", doc="Longitude column name.")
    lat_min = param.Number(default=None, allow_None=True)
    lat_max = param.Number(default=None, allow_None=True)
    lon_min = param.Number(default=None, allow_None=True)
    lon_max = param.Number(default=None, allow_None=True)

    def apply(self, table: pd.DataFrame) -> pd.DataFrame:
        if self.lat_col not in table.columns or self.lon_col not in table.columns:
            return table

        mask = pd.Series(True, index=table.index)
        if self.lat_min is not None:
            mask &= table[self.lat_col] >= self.lat_min
        if self.lat_max is not None:
            mask &= table[self.lat_col] <= self.lat_max
        if self.lon_min is not None:
            mask &= table[self.lon_col] >= self.lon_min
        if self.lon_max is not None:
            mask &= table[self.lon_col] <= self.lon_max

        return table[mask].reset_index(drop=True)


class DimensionAggregate(Transform):
    """
    Aggregate (reduce) over specified dimension columns.

    Removes the specified dimensions by grouping over the remaining
    columns and applying the aggregation function.

    Examples
    --------
    >>> DimensionAggregate(dimensions=["time"], method="mean").apply(df)
    >>> DimensionAggregate(dimensions=["lat", "lon"], method="mean",
    ...                    value_columns=["air"]).apply(df)
    """

    transform_type: ClassVar[str] = "dimension_aggregate"

    dimensions = param.List(doc="Dimension columns to aggregate over (remove).")
    method = param.Selector(
        default="mean",
        objects=["mean", "sum", "std", "min", "max", "median", "count"],
        doc="Aggregation function.",
    )
    value_columns = param.List(default=None, allow_None=True, doc="""
        Columns to aggregate. If None, all numeric non-group columns are used.""")

    def apply(self, table: pd.DataFrame) -> pd.DataFrame:
        dims = [d for d in self.dimensions if d in table.columns]
        if not dims:
            return table

        remaining = [c for c in table.columns if c not in dims]

        if self.value_columns:
            agg_cols = [c for c in self.value_columns if c in table.columns]
            group_cols = [c for c in remaining if c not in agg_cols]
        else:
            # Auto-detect: numeric non-dimension columns that aren't also
            # coordinate-like (i.e., columns that look like data values).
            # Heuristic: columns in `dimensions` list are coordinates;
            # remaining numeric columns are values to aggregate.
            numeric_remaining = [
                c for c in remaining
                if pd.api.types.is_numeric_dtype(table[c])
            ]
            non_numeric = [c for c in remaining if c not in numeric_remaining]

            # If all remaining are numeric, the ones matching common coordinate
            # names (lat, lon, x, y, level, etc.) become group cols
            coord_names = {"lat", "lon", "latitude", "longitude", "x", "y",
                          "level", "lev", "depth", "height", "z"}
            agg_cols = [c for c in numeric_remaining if c.lower() not in coord_names]
            group_cols = non_numeric + [c for c in numeric_remaining if c not in agg_cols]

            if not agg_cols:
                # Fallback: aggregate all numeric columns
                agg_cols = numeric_remaining
                group_cols = non_numeric

        if not agg_cols:
            return table

        if not group_cols:
            return table[agg_cols].agg(self.method).to_frame().T.reset_index(drop=True)

        return (
            table.groupby(group_cols, as_index=False)[agg_cols]
            .agg(self.method)
            .reset_index(drop=True)
        )


class TimeResample(Transform):
    """
    Resample time series data at a different frequency.

    Converts a time column to datetime, applies resampling, and aggregates.
    Optionally groups by spatial dimensions before resampling.

    Examples
    --------
    >>> TimeResample(time_col="time", freq="MS", method="mean").apply(df)
    >>> TimeResample(time_col="time", freq="YS", group_cols=["lat", "lon"]).apply(df)
    """

    transform_type: ClassVar[str] = "time_resample"

    time_col = param.String(default="time", doc="Name of the time column.")
    freq = param.String(default="MS", doc="""
        Resample frequency. Examples: D (daily), W (weekly), MS (month start),
        YS (year start), QS (quarter start), h (hourly).""")
    method = param.Selector(
        default="mean",
        objects=["mean", "sum", "std", "min", "max", "median", "count"],
    )
    group_cols = param.List(default=None, allow_None=True, doc="""
        Additional columns to group by before resampling (e.g., spatial dims).""")

    def apply(self, table: pd.DataFrame) -> pd.DataFrame:
        if self.time_col not in table.columns:
            return table

        df = table.copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col])

        numeric_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
            and c != self.time_col
            and (self.group_cols is None or c not in self.group_cols)
        ]

        if self.group_cols:
            valid_groups = [c for c in self.group_cols if c in df.columns]
            if valid_groups:
                result = (
                    df.groupby(valid_groups)
                    .resample(self.freq, on=self.time_col)[numeric_cols]
                    .agg(self.method)
                    .reset_index()
                )
                return result

        result = (
            df.resample(self.freq, on=self.time_col)[numeric_cols]
            .agg(self.method)
            .reset_index()
        )
        return result


class Anomaly(Transform):
    """
    Compute anomalies (deviations from a climatological mean).

    Subtracts the grouped mean from each value, producing a new column
    with the suffix '_anomaly'. Common groupings: month-of-year (seasonal
    cycle), day-of-year, or overall mean.

    Examples
    --------
    >>> Anomaly(time_col="time", value_col="air", groupby="month").apply(df)
    >>> Anomaly(value_col="air", groupby=None).apply(df)  # overall mean
    """

    transform_type: ClassVar[str] = "anomaly"

    time_col = param.String(default="time", doc="Time column name.")
    value_col = param.String(doc="Value column to compute anomalies for.")
    groupby = param.Selector(
        default="month",
        objects=["month", "dayofyear", "season", "hour", None],
        doc="""
        Grouping for the climatological mean:
        - month: group by month (1-12), standard for seasonal decomposition
        - dayofyear: group by day-of-year (1-366)
        - season: group by meteorological season (DJF, MAM, JJA, SON)
        - hour: group by hour-of-day (0-23)
        - None: subtract overall mean""",
    )

    def apply(self, table: pd.DataFrame) -> pd.DataFrame:
        if self.value_col not in table.columns:
            return table

        df = table.copy()
        out_col = f"{self.value_col}_anomaly"

        if self.groupby is None:
            df[out_col] = df[self.value_col] - df[self.value_col].mean()
            return df

        if self.time_col not in df.columns:
            df[out_col] = df[self.value_col] - df[self.value_col].mean()
            return df

        ts = pd.to_datetime(df[self.time_col])

        if self.groupby == "month":
            group_key = ts.dt.month
        elif self.groupby == "dayofyear":
            group_key = ts.dt.dayofyear
        elif self.groupby == "season":
            # Meteorological seasons: DJF=1, MAM=2, JJA=3, SON=4
            group_key = (ts.dt.month % 12 + 3) // 3
        elif self.groupby == "hour":
            group_key = ts.dt.hour
        else:
            df[out_col] = df[self.value_col] - df[self.value_col].mean()
            return df

        clim_mean = df.groupby(group_key)[self.value_col].transform("mean")
        df[out_col] = df[self.value_col] - clim_mean
        return df


class RollingWindow(Transform):
    """
    Apply a rolling window operation to smooth time series data.

    Creates a new column with the rolling statistic applied.

    Examples
    --------
    >>> RollingWindow(column="air", window=30, method="mean").apply(df)
    """

    transform_type: ClassVar[str] = "rolling_window"

    column = param.String(doc="Column to apply the rolling window to.")
    window = param.Integer(default=7, doc="Window size in number of rows.")
    method = param.Selector(
        default="mean",
        objects=["mean", "sum", "std", "min", "max", "median"],
    )
    center = param.Boolean(default=False, doc="Whether to center the window.")
    min_periods = param.Integer(default=1, doc="Minimum observations required.")
    output_column = param.String(default=None, allow_None=True, doc="""
        Name for the output column. Defaults to '{column}_rolling_{method}'.""")

    def apply(self, table: pd.DataFrame) -> pd.DataFrame:
        if self.column not in table.columns:
            return table

        df = table.copy()
        out_col = self.output_column or f"{self.column}_rolling_{self.method}"
        rolling = df[self.column].rolling(
            window=self.window, center=self.center, min_periods=self.min_periods
        )
        df[out_col] = getattr(rolling, self.method)()
        return df
