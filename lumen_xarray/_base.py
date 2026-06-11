"""
Shared mixin and utilities for xarray sources.

Provides common dataset loading, metadata extraction, dimension info,
and file format detection used by both XArraySQLSource and XArraySource.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


# Map file extensions to xarray engines
XARRAY_ENGINES = {
    ".nc": "netcdf4",
    ".nc4": "netcdf4",
    ".netcdf": "netcdf4",
    ".h5": "h5netcdf",
    ".hdf5": "h5netcdf",
    ".he5": "h5netcdf",
    ".zarr": "zarr",
    ".grib": "cfgrib",
    ".grib2": "cfgrib",
    ".grb": "cfgrib",
    ".grb2": "cfgrib",
}

XARRAY_EXTENSIONS = tuple(XARRAY_ENGINES.keys())


def _detect_engine(path: str) -> str | None:
    """Detect xarray engine from file extension."""
    suffix = Path(path).suffix.lower()
    return XARRAY_ENGINES.get(suffix)


class XArrayMixin:
    """
    Mixin providing shared xarray dataset operations.

    Subclasses must set self._dataset before calling mixin methods.
    """

    def _load_and_filter_dataset(
        self,
        dataset: xr.Dataset | None,
        uri: str | None,
        engine: str | None,
        chunks: dict | str | None,
        open_kwargs: dict,
        variables: list | None,
    ) -> xr.Dataset:
        """
        Load an xarray dataset from URI or use a provided one, then
        optionally filter to a subset of variables.

        Returns the loaded (and possibly filtered) dataset.
        """
        if dataset is not None:
            ds = dataset
        elif uri is not None:
            resolved_engine = engine or _detect_engine(uri)
            kw = dict(open_kwargs)
            if resolved_engine:
                kw["engine"] = resolved_engine
            if chunks is not None:
                kw["chunks"] = chunks
            ds = xr.open_dataset(uri, **kw)
        else:
            raise ValueError("Either 'uri' or '_dataset' must be provided.")

        if variables:
            available = list(ds.data_vars)
            missing = [v for v in variables if v not in available]
            if missing:
                raise ValueError(
                    f"Variables {missing} not found in dataset. "
                    f"Available: {available}"
                )
            ds = ds[variables]

        return ds

    def _build_table_metadata(self, ds: xr.Dataset, tables: list[str]) -> dict[str, Any]:
        """
        Build rich metadata for each table from xarray attributes.

        Extracts coordinate ranges, units, long_name, dimensions, and
        dataset-level attributes. Used by both sources and by Lumen AI
        agents to understand data structure.
        """
        metadata = {}

        for table in tables:
            if table not in ds.data_vars:
                continue

            var = ds[table]
            attrs = dict(var.attrs)
            columns = {}

            # Coordinate metadata
            for coord_name in var.dims:
                coord = ds.coords[coord_name]
                coord_attrs = dict(coord.attrs)
                col_meta = {
                    "data_type": str(coord.dtype),
                    "is_coordinate": True,
                    "is_dimension": True,
                }
                if "units" in coord_attrs:
                    col_meta["units"] = coord_attrs["units"]
                if "long_name" in coord_attrs:
                    col_meta["description"] = coord_attrs["long_name"]

                if coord.dtype.kind in "fiuM":
                    col_meta["min"] = str(coord.values.min())
                    col_meta["max"] = str(coord.values.max())
                    col_meta["size"] = int(coord.size)

                columns[coord_name] = col_meta

            # Data variable metadata
            col_meta = {"data_type": str(var.dtype), "is_coordinate": False}
            if "units" in attrs:
                col_meta["units"] = attrs["units"]
            if "long_name" in attrs:
                col_meta["description"] = attrs["long_name"]
            columns[table] = col_meta

            # Table-level description
            dims_str = " x ".join(f"{d}({ds.sizes[d]})" for d in var.dims)
            description = attrs.get("long_name", table)
            if "units" in attrs:
                description += f" [{attrs['units']}]"

            metadata[table] = {
                "description": f"{description} — dimensions: {dims_str}",
                "columns": columns,
                "dimensions": list(var.dims),
                "shape": list(var.shape),
                "dataset_attrs": dict(ds.attrs),
            }

        return metadata

    def _build_dimension_info(self, ds: xr.Dataset, tables: list[str]) -> dict[str, Any]:
        """
        Build detailed dimension information for UI controls and AI context.

        Returns coordinate values, ranges, types, and step sizes.
        """
        info = {}

        for tbl in tables:
            if tbl not in ds.data_vars:
                continue
            var = ds[tbl]
            dim_info = {}
            for dim in var.dims:
                coord = ds.coords[dim]
                vals = coord.values
                d = {
                    "size": int(coord.size),
                    "dtype": str(coord.dtype),
                }
                if coord.dtype.kind == "M":  # datetime
                    d["min"] = str(pd.Timestamp(vals.min()))
                    d["max"] = str(pd.Timestamp(vals.max()))
                    d["type"] = "datetime"
                elif coord.dtype.kind in "fiu":  # numeric
                    d["min"] = float(vals.min())
                    d["max"] = float(vals.max())
                    d["step"] = float(np.diff(vals[:2])[0]) if len(vals) > 1 else None
                    d["type"] = "numeric"
                else:
                    d["values"] = vals.tolist()
                    d["type"] = "categorical"
                dim_info[dim] = d
            info[tbl] = dim_info

        return info
