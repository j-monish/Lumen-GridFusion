"""Create a custom multi-variable climate dataset for testing."""
import numpy as np
import pandas as pd
import xarray as xr

np.random.seed(42)

times = pd.date_range("2020-01-01", periods=730, freq="D")  # 2 years daily
lats = np.linspace(20, 70, 15)
lons = np.linspace(-130, -60, 20)

# Temperature with seasonal cycle + lat gradient
temp_base = 280 + 20 * np.cos(np.radians(lats))[:, None] * np.ones(len(lons))
seasonal = 15 * np.sin(2 * np.pi * np.arange(730) / 365)[:, None, None]
temp = temp_base[None, :, :] + seasonal + np.random.randn(730, 15, 20) * 3

# Precipitation with lat/seasonal pattern
precip = np.maximum(0, 5 + 3 * np.sin(2 * np.pi * np.arange(730) / 365)[:, None, None]
                     + np.random.randn(730, 15, 20) * 2)

# Wind speed
wind_base = 8 + 4 * np.cos(np.radians(lats))[:, None] * np.ones(len(lons))
wind = np.abs(wind_base[None, :, :] + np.random.randn(730, 15, 20) * 2)

ds = xr.Dataset({
    "temperature": (["time", "lat", "lon"], temp, {"units": "K", "long_name": "Surface Temperature"}),
    "precipitation": (["time", "lat", "lon"], precip, {"units": "mm/day", "long_name": "Daily Precipitation"}),
    "wind_speed": (["time", "lat", "lon"], wind, {"units": "m/s", "long_name": "Wind Speed at 10m"}),
}, coords={
    "time": times,
    "lat": lats,
    "lon": lons,
}, attrs={
    "title": "Custom Climate Dataset",
    "source": "Synthetic data for lumen-xarray demo",
    "history": "Generated with create_custom_dataset.py",
})

ds.to_netcdf("examples/custom_climate.nc")
print(f"Created custom_climate.nc: {ds}")
print(f"Variables: {list(ds.data_vars)}")
print(f"Shape: time={len(times)}, lat={len(lats)}, lon={len(lons)}")
print(f"Total points per var: {730*15*20:,}")
