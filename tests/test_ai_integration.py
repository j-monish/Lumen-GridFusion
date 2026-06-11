"""
Tests for Lumen AI integration hooks.
"""

import tempfile
from pathlib import Path

import pytest
import xarray as xr

from lumen_xarray.ai import (
    TABLE_EXTENSIONS_XARRAY,
    build_xarray_source_code,
    get_extended_table_extensions,
    handle_xarray_upload,
    is_xarray_path,
    resolve_xarray_source,
)
from lumen_xarray.source import XArraySQLSource


class TestPathDetection:

    def test_netcdf_path(self):
        assert is_xarray_path("/data/file.nc")
        assert is_xarray_path("/data/file.nc4")
        assert is_xarray_path("/data/file.netcdf")

    def test_zarr_path(self):
        assert is_xarray_path("/data/file.zarr")

    def test_hdf5_path(self):
        assert is_xarray_path("/data/file.h5")
        assert is_xarray_path("/data/file.hdf5")

    def test_grib_path(self):
        assert is_xarray_path("/data/file.grib")
        assert is_xarray_path("/data/file.grib2")

    def test_non_xarray_path(self):
        assert not is_xarray_path("/data/file.csv")
        assert not is_xarray_path("/data/file.parquet")
        assert not is_xarray_path("/data/file.json")

    def test_remote_url(self):
        assert is_xarray_path("https://example.com/data.nc")
        assert is_xarray_path("https://example.com/data.zarr")
        assert not is_xarray_path("https://example.com/data.csv")

    def test_url_with_query(self):
        assert is_xarray_path("https://example.com/data.nc?token=abc")


class TestResolveSource:

    def test_resolve_netcdf(self, nc_file):
        source = resolve_xarray_source(nc_file)
        assert isinstance(source, XArraySQLSource)
        assert "temperature" in source.get_tables()

    def test_resolve_zarr(self, zarr_path):
        source = resolve_xarray_source(zarr_path, engine="zarr")
        assert isinstance(source, XArraySQLSource)

    def test_resolve_with_chunks(self, nc_file):
        source = resolve_xarray_source(nc_file, chunks={"time": 5})
        assert isinstance(source, XArraySQLSource)


class TestFileUpload:

    def test_upload_netcdf(self, synthetic_dataset, tmp_path):
        # Write to file, read back as bytes
        path = tmp_path / "upload_test.nc"
        synthetic_dataset.to_netcdf(path)
        file_bytes = path.read_bytes()

        result = handle_xarray_upload(file_bytes, "upload_test.nc")
        assert "source" in result
        assert "tables" in result
        assert "metadata" in result
        assert "dimension_info" in result
        assert "message" in result
        assert isinstance(result["source"], XArraySQLSource)
        assert "temperature" in result["tables"]

        # Clean up temp file
        temp_path = Path(result["temp_path"])
        if temp_path.exists():
            temp_path.unlink()


class TestCodeGeneration:

    def test_generate_netcdf(self):
        code = build_xarray_source_code("/data/climate.nc")
        assert "XArraySQLSource" in code
        assert "climate.nc" in code
        assert "netcdf4" in code

    def test_generate_zarr(self):
        code = build_xarray_source_code("/data/output.zarr")
        assert "zarr" in code

    def test_generate_custom_var(self):
        code = build_xarray_source_code("/data/file.nc", var_name="my_source")
        assert "my_source = XArraySQLSource" in code


class TestExtensions:

    def test_xarray_extensions(self):
        assert "nc" in TABLE_EXTENSIONS_XARRAY
        assert "zarr" in TABLE_EXTENSIONS_XARRAY
        assert "h5" in TABLE_EXTENSIONS_XARRAY
        assert "grib" in TABLE_EXTENSIONS_XARRAY

    def test_extended_extensions(self):
        exts = get_extended_table_extensions()
        # Should include both standard Lumen extensions and xarray ones
        assert "nc" in exts
