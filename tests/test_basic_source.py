"""
Tests for XArraySource — the DataFrame-based xarray source.

Tests cover:
- Construction and data loading
- get_tables, get_schema, get_metadata
- Filtering via get()
- Native xarray operations (select, aggregate, resample)
- File I/O (NetCDF, Zarr)
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from lumen_xarray.basic_source import XArraySource


class TestConstruction:

    def test_from_dataset(self, synthetic_dataset):
        source = XArraySource(_dataset=synthetic_dataset)
        assert "temperature" in source.get_tables()
        assert "pressure" in source.get_tables()

    def test_from_netcdf(self, nc_file):
        source = XArraySource(uri=nc_file)
        assert "temperature" in source.get_tables()

    def test_from_zarr(self, zarr_path):
        source = XArraySource(uri=zarr_path, engine="zarr")
        assert "temperature" in source.get_tables()

    def test_variable_filter(self, synthetic_dataset):
        source = XArraySource(_dataset=synthetic_dataset, variables=["pressure"])
        assert source.get_tables() == ["pressure"]

    def test_no_sql_support(self, synthetic_dataset):
        source = XArraySource(_dataset=synthetic_dataset)
        assert source._supports_sql is False

    def test_source_type(self, synthetic_dataset):
        source = XArraySource(_dataset=synthetic_dataset)
        assert source.source_type == "xarray-basic"


class TestGetData:

    def test_get_all(self, synthetic_dataset):
        source = XArraySource(_dataset=synthetic_dataset)
        df = source.get("temperature")
        assert isinstance(df, pd.DataFrame)
        assert "temperature" in df.columns
        assert "lat" in df.columns
        assert "lon" in df.columns
        assert "time" in df.columns
        assert len(df) == 10 * 5 * 4  # 200

    def test_get_filter_scalar(self, synthetic_dataset):
        source = XArraySource(_dataset=synthetic_dataset)
        df = source.get("temperature", lat=30.0)
        assert all(df["lat"] == 30.0)
        assert len(df) == 10 * 4  # time * lon

    def test_get_filter_list(self, synthetic_dataset):
        source = XArraySource(_dataset=synthetic_dataset)
        df = source.get("temperature", lat=[10.0, 50.0])
        assert set(df["lat"].unique()) == {10.0, 50.0}

    def test_get_filter_slice(self, synthetic_dataset):
        source = XArraySource(_dataset=synthetic_dataset)
        df = source.get("temperature", lat=slice(20.0, 40.0))
        assert all(df["lat"] >= 20.0)
        assert all(df["lat"] <= 40.0)

    def test_get_ignores_dunder_keys(self, synthetic_dataset):
        source = XArraySource(_dataset=synthetic_dataset)
        df = source.get("temperature", __limit=10)
        assert len(df) == 200  # ignores __limit


class TestSchema:

    def test_schema_single(self, synthetic_dataset):
        source = XArraySource(_dataset=synthetic_dataset)
        schema = source.get_schema("temperature")
        assert "temperature" in schema
        assert "lat" in schema
        assert "__len__" in schema
        assert schema["__len__"] == 200

    def test_schema_all(self, synthetic_dataset):
        source = XArraySource(_dataset=synthetic_dataset)
        schemas = source.get_schema()
        assert "temperature" in schemas
        assert "pressure" in schemas

    def test_schema_xarray_meta(self, synthetic_dataset):
        source = XArraySource(_dataset=synthetic_dataset)
        schema = source.get_schema("temperature")
        meta = schema["__xarray_meta__"]
        assert meta["dimensions"] == ["time", "lat", "lon"]
        assert meta["shape"] == [10, 5, 4]
        assert meta["dtype"] == "float64"

    def test_schema_with_limit(self, synthetic_dataset):
        source = XArraySource(_dataset=synthetic_dataset)
        schema = source.get_schema("temperature", limit=5)
        assert "__len__" in schema


class TestMetadata:

    def test_metadata_single(self, synthetic_dataset):
        source = XArraySource(_dataset=synthetic_dataset)
        meta = source.get_metadata("temperature")
        # Single table returns nested {table: {...}} or flat {...} depending on decorator
        if "temperature" in meta:
            assert "description" in meta["temperature"]
            assert "columns" in meta["temperature"]
        else:
            assert "description" in meta
            assert "columns" in meta

    def test_metadata_units(self, synthetic_dataset):
        source = XArraySource(_dataset=synthetic_dataset)
        meta = source.get_metadata("temperature")
        cols = meta.get("temperature", meta).get("columns", {})
        assert cols["temperature"]["units"] == "K"

    def test_metadata_all(self, synthetic_dataset):
        source = XArraySource(_dataset=synthetic_dataset)
        meta = source.get_metadata()
        assert "temperature" in meta
        assert "pressure" in meta


class TestNativeOps:

    def test_select(self, synthetic_dataset):
        source = XArraySource(_dataset=synthetic_dataset)
        result = source.select("temperature", lat=30.0)
        assert isinstance(result, xr.DataArray)
        assert "lat" not in result.dims

    def test_iselect(self, synthetic_dataset):
        source = XArraySource(_dataset=synthetic_dataset)
        result = source.iselect("temperature", time=0)
        assert isinstance(result, xr.DataArray)
        assert "time" not in result.dims

    def test_aggregate_mean(self, synthetic_dataset):
        source = XArraySource(_dataset=synthetic_dataset)
        result = source.aggregate("temperature", dim="time", method="mean")
        assert isinstance(result, xr.DataArray)
        assert "time" not in result.dims
        assert "lat" in result.dims

    def test_aggregate_sum(self, synthetic_dataset):
        source = XArraySource(_dataset=synthetic_dataset)
        result = source.aggregate("temperature", dim=["lat", "lon"], method="sum")
        assert "lat" not in result.dims
        assert "lon" not in result.dims

    def test_resample(self, air_dataset):
        source = XArraySource(_dataset=air_dataset)
        result = source.resample("air", freq="MS", method="mean")
        assert isinstance(result, xr.DataArray)
        # Monthly resampling of 2 years should give ~24 months
        assert result.sizes["time"] <= 24


class TestResourceManagement:

    def test_clear_cache(self, synthetic_dataset):
        source = XArraySource(_dataset=synthetic_dataset)
        _ = source.get("temperature")
        assert "temperature" in source._df_cache
        source.clear_cache()
        assert len(source._df_cache) == 0

    def test_close(self, synthetic_dataset):
        source = XArraySource(_dataset=synthetic_dataset)
        source.close()
        # Dataset should be closed

    def test_repr(self, synthetic_dataset):
        source = XArraySource(_dataset=synthetic_dataset)
        r = repr(source)
        assert "XArraySource" in r
