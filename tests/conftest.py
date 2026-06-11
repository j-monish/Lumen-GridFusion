"""Shared fixtures for lumen-xarray tests."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture
def air_dataset():
    """NOAA air temperature tutorial dataset (time x lat x lon)."""
    return xr.tutorial.open_dataset("air_temperature")


@pytest.fixture
def synthetic_dataset():
    """
    Synthetic 3D dataset with known values for deterministic testing.
    Dimensions: time(10) x lat(5) x lon(4)
    Variables: temperature, pressure
    """
    times = pd.date_range("2020-01-01", periods=10, freq="D")
    lats = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    lons = np.array([100.0, 110.0, 120.0, 130.0])

    np.random.seed(42)
    temp = np.random.uniform(250, 310, (10, 5, 4))
    pressure = np.random.uniform(950, 1050, (10, 5, 4))

    ds = xr.Dataset(
        {
            "temperature": (["time", "lat", "lon"], temp, {
                "units": "K",
                "long_name": "Air Temperature",
            }),
            "pressure": (["time", "lat", "lon"], pressure, {
                "units": "hPa",
                "long_name": "Surface Pressure",
            }),
        },
        coords={
            "time": times,
            "lat": ("lat", lats, {"units": "degrees_north"}),
            "lon": ("lon", lons, {"units": "degrees_east"}),
        },
        attrs={
            "title": "Synthetic Test Dataset",
            "source": "lumen-xarray tests",
        },
    )
    return ds


@pytest.fixture
def nc_file(synthetic_dataset, tmp_path):
    """Write synthetic dataset to a temporary NetCDF file."""
    path = tmp_path / "test_data.nc"
    synthetic_dataset.to_netcdf(path)
    return str(path)


@pytest.fixture
def zarr_path(synthetic_dataset, tmp_path):
    """Write synthetic dataset to a temporary Zarr store."""
    path = tmp_path / "test_data.zarr"
    synthetic_dataset.to_zarr(path)
    return str(path)


@pytest.fixture
def air_df(air_dataset):
    """Air temperature as a flat DataFrame."""
    return air_dataset["air"].to_dataframe().reset_index()
