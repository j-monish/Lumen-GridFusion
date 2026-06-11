"""
SQL Queries: Showcasing DataFusion SQL capabilities over xarray data.

Demonstrates the full range of SQL operations supported when querying
N-dimensional scientific data through XArraySQLSource.
"""

import xarray as xr

from lumen_xarray import XArraySQLSource

ds = xr.tutorial.open_dataset("air_temperature")
source = XArraySQLSource(_dataset=ds)

print("Dataset: 2-year NOAA air temperature (6-hourly, 25 lat x 53 lon)")
print(f"Total data points: {2920 * 25 * 53:,}")
print()

# --- 1. Descriptive Statistics ---
print("=" * 60)
print("1. DESCRIPTIVE STATISTICS")
print("=" * 60)

df = source.execute("""
    SELECT
        COUNT(*) as total_obs,
        AVG(air) as mean_temp,
        MIN(air) as min_temp,
        MAX(air) as max_temp
    FROM air
""")
print(df.to_string(index=False))
print()

# --- 2. Spatial Analysis ---
print("=" * 60)
print("2. SPATIAL ANALYSIS — Temperature by latitude band")
print("=" * 60)

df = source.execute("""
    SELECT
        CASE
            WHEN lat >= 60 THEN 'Arctic (60-75N)'
            WHEN lat >= 40 THEN 'Mid-latitude (40-60N)'
            WHEN lat >= 20 THEN 'Subtropical (20-40N)'
            ELSE 'Tropical (15-20N)'
        END as zone,
        AVG(air) as avg_temp,
        MIN(air) as min_temp,
        MAX(air) as max_temp,
        COUNT(*) as obs
    FROM air
    GROUP BY zone
    ORDER BY avg_temp
""")
print(df.to_string(index=False))
print()

# --- 3. Temporal Analysis ---
print("=" * 60)
print("3. TEMPORAL ANALYSIS — Monthly averages")
print("=" * 60)

df = source.execute("""
    SELECT
        EXTRACT(MONTH FROM time) as month,
        AVG(air) as avg_temp,
        MIN(air) as min_temp,
        MAX(air) as max_temp
    FROM air
    GROUP BY month
    ORDER BY month
""")
print(df.to_string(index=False))
print()

# --- 4. Extreme Events ---
print("=" * 60)
print("4. EXTREME EVENTS — Coldest observations")
print("=" * 60)

df = source.execute("""
    SELECT time, lat, lon, air
    FROM air
    WHERE air < 230
    ORDER BY air
    LIMIT 10
""")
print(df.to_string(index=False))
print()

# --- 5. Spatial Gradients ---
print("=" * 60)
print("5. SPATIAL GRADIENT — North-South temperature difference")
print("=" * 60)

df = source.execute("""
    SELECT
        lon,
        AVG(CASE WHEN lat >= 50 THEN air END) as north_avg,
        AVG(CASE WHEN lat <= 30 THEN air END) as south_avg,
        AVG(CASE WHEN lat <= 30 THEN air END) - AVG(CASE WHEN lat >= 50 THEN air END) as gradient
    FROM air
    GROUP BY lon
    ORDER BY lon
    LIMIT 10
""")
print(df.to_string(index=False))
print()

# --- 6. Percentiles ---
print("=" * 60)
print("6. PERCENTILE ANALYSIS")
print("=" * 60)

df = source.execute("""
    SELECT
        lat,
        APPROX_PERCENTILE_CONT(air, 0.1) as p10,
        APPROX_PERCENTILE_CONT(air, 0.5) as median,
        APPROX_PERCENTILE_CONT(air, 0.9) as p90,
        APPROX_PERCENTILE_CONT(air, 0.9) - APPROX_PERCENTILE_CONT(air, 0.1) as iqr_80
    FROM air
    GROUP BY lat
    ORDER BY lat
""")
print(df.to_string(index=False))
print()

# --- 7. Multi-variable example ---
print("=" * 60)
print("7. MULTI-VARIABLE — Synthetic dual-variable dataset")
print("=" * 60)

import numpy as np
import pandas as pd

# Create a dataset with temperature and pressure
times = pd.date_range("2020-01-01", periods=100, freq="D")
lats = np.array([20.0, 40.0, 60.0])
lons = np.array([100.0, 120.0])

np.random.seed(42)
multi_ds = xr.Dataset({
    "temperature": (["time", "lat", "lon"], np.random.uniform(250, 310, (100, 3, 2))),
    "pressure": (["time", "lat", "lon"], np.random.uniform(950, 1050, (100, 3, 2))),
}, coords={"time": times, "lat": lats, "lon": lons})

multi_source = XArraySQLSource(_dataset=multi_ds)
print(f"Tables: {multi_source.get_tables()}")

# Query each table independently
for table in multi_source.get_tables():
    df = multi_source.execute(f"""
        SELECT lat, AVG({table}) as avg_val, COUNT(*) as n
        FROM {table}
        GROUP BY lat
        ORDER BY lat
    """)
    print(f"\n{table}:")
    print(df.to_string(index=False))

print("\n✓ All SQL query patterns work with xarray data via DataFusion!")
