"""
XArraySource — DataFrame-based xarray source for Lumen.

A simpler, non-SQL source that converts xarray variables to DataFrames.
Use this when SQL support isn't needed, or when you want direct access
to xarray's native operations (select, aggregate, resample) rather than
going through SQL.

Complements XArraySQLSource: use XArraySQLSource for Lumen AI integration
(SQL agents), and XArraySource for direct programmatic access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import param
import xarray as xr

from lumen.sources.base import Source, cached, cached_schema, cached_metadata
from lumen.util import get_dataframe_schema

from ._base import XArrayMixin, _detect_engine


class XArraySource(Source, XArrayMixin):
    """
    DataFrame-based xarray source for Lumen.

    Loads xarray datasets and exposes each data variable as a flat DataFrame
    table. Provides native xarray operations (select, aggregate, resample)
    alongside Lumen's standard Source API.

    Parameters
    ----------
    uri : str
        Path or URL to the data file.
    engine : str or None
        xarray engine (auto-detected if None).
    chunks : dict or str or None
        Dask chunking specification.
    variables : list or None
        Subset of data variables to expose.
    include_coords : bool
        Whether to include all coordinates as columns in DataFrames.
    """

    source_type: ClassVar[str] = "xarray-basic"

    uri = param.String(default=None, allow_None=True, doc="""
        Path or URL to the xarray-compatible data file.""")

    engine = param.String(default=None, allow_None=True, doc="""
        xarray backend engine.""")

    chunks = param.Parameter(default=None, doc="""
        Dask chunk specification.""")

    open_kwargs = param.Dict(default={}, doc="""
        Additional kwargs for xr.open_dataset().""")

    variables = param.List(default=None, allow_None=True, doc="""
        Subset of data variables to expose as tables.""")

    include_coords = param.Boolean(default=True, doc="""
        Include all coordinates as columns in the DataFrame output.""")

    _supports_sql: ClassVar[bool] = False

    def __init__(self, _dataset: xr.Dataset | None = None, **params):
        super().__init__(**params)
        self._dataset: xr.Dataset | None = _dataset
        self._df_cache: dict[str, pd.DataFrame] = {}
        self._load_dataset()

    def _load_dataset(self):
        """Load the xarray dataset from URI or use provided dataset."""
        self._dataset = self._load_and_filter_dataset(
            dataset=self._dataset,
            uri=self.uri,
            engine=self.engine,
            chunks=self.chunks,
            open_kwargs=self.open_kwargs,
            variables=self.variables,
        )

    @property
    def dataset(self) -> xr.Dataset:
        """Access the underlying xarray Dataset."""
        return self._dataset

    def _to_dataframe(self, table: str) -> pd.DataFrame:
        """Convert an xarray variable to a pandas DataFrame with caching."""
        if table in self._df_cache:
            return self._df_cache[table]

        if table not in self._dataset.data_vars:
            raise ValueError(
                f"Table {table!r} not found. Available: {list(self._dataset.data_vars)}"
            )

        var = self._dataset[table]
        df = var.to_dataframe()

        if self.include_coords:
            df = df.reset_index()
        elif isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()

        df = df.dropna(subset=[table])
        self._df_cache[table] = df
        return df

    def get_tables(self) -> list[str]:
        """Return list of available tables (one per data variable)."""
        return sorted(self._dataset.data_vars)

    @cached
    def get(self, table: str, **query) -> pd.DataFrame:
        """
        Return a table as a DataFrame, optionally filtered.

        Parameters
        ----------
        table : str
            Name of the data variable.
        query : dict
            Column-value pairs for filtering. Values can be:
            - scalar: exact match
            - list: isin filter
            - slice: range filter
        """
        df = self._to_dataframe(table)

        for col, val in query.items():
            if col.startswith("__") or col not in df.columns:
                continue
            if isinstance(val, list):
                df = df[df[col].isin(val)]
            elif isinstance(val, slice):
                if val.start is not None:
                    df = df[df[col] >= val.start]
                if val.stop is not None:
                    df = df[df[col] <= val.stop]
            else:
                df = df[df[col] == val]

        return df.reset_index(drop=True)

    @cached_schema
    def get_schema(
        self,
        table: str | None = None,
        limit: int | None = None,
        shuffle: bool = False,
    ) -> dict:
        """Return JSON schema for the specified table(s)."""
        tables = [table] if table else self.get_tables()
        schemas = {}

        for tbl in tables:
            df = self._to_dataframe(tbl)
            if limit:
                df = df.sample(n=min(limit, len(df))) if shuffle else df.head(limit)

            schema = get_dataframe_schema(df)["items"]["properties"]
            schema["__len__"] = len(self._to_dataframe(tbl))

            # Enrich with xarray metadata
            var = self._dataset[tbl]
            schema["__xarray_meta__"] = {
                "dimensions": list(var.dims),
                "shape": list(var.shape),
                "coords": list(var.coords),
                "attrs": dict(var.attrs),
                "dtype": str(var.dtype),
            }
            schemas[tbl] = schema

        return schemas if table is None else schemas[table]

    @cached_metadata
    def get_metadata(self, table: str | list[str] | None = None) -> dict:
        """Return metadata for the specified table(s)."""
        if table is None:
            tables = self.get_tables()
        elif isinstance(table, str):
            tables = [table]
        else:
            tables = table
        return self._get_table_metadata(tables)

    def _get_table_metadata(self, tables: list[str]) -> dict[str, Any]:
        """Delegate to shared mixin for metadata extraction."""
        return self._build_table_metadata(self._dataset, tables)

    def get_dimension_info(self, table: str | None = None) -> dict[str, Any]:
        """Return detailed dimension information for UI controls."""
        tables = [table] if table else list(self._dataset.data_vars)
        info = self._build_dimension_info(self._dataset, tables)
        return info if table is None else info.get(table, {})

    # ---- Native xarray operations ----

    def select(self, table: str, **sel_kwargs) -> xr.DataArray:
        """Label-based selection on the xarray variable."""
        return self._dataset[table].sel(**sel_kwargs)

    def iselect(self, table: str, **isel_kwargs) -> xr.DataArray:
        """Integer-index selection on the xarray variable."""
        return self._dataset[table].isel(**isel_kwargs)

    def aggregate(
        self, table: str, dim: str | list[str], method: str = "mean"
    ) -> xr.DataArray:
        """Reduce a variable along dimension(s)."""
        var = self._dataset[table]
        func = getattr(var, method, None)
        if func is None:
            raise ValueError(f"Unknown aggregation method: {method}")
        return func(dim=dim)

    def resample(
        self, table: str, time_dim: str = "time", freq: str = "MS", method: str = "mean"
    ) -> xr.DataArray:
        """Resample a variable along a time dimension."""
        var = self._dataset[table]
        resampled = var.resample({time_dim: freq})
        func = getattr(resampled, method, None)
        if func is None:
            raise ValueError(f"Unknown resample method: {method}")
        return func()

    def clear_cache(self, *events):
        """Clear cached DataFrames."""
        super().clear_cache(*events)
        self._df_cache.clear()

    def close(self):
        """Close the underlying dataset."""
        if self._dataset is not None:
            self._dataset.close()

    def __repr__(self):
        tables = self.get_tables() if self._dataset else []
        return f"XArraySource(uri={self.uri!r}, tables={tables})"
