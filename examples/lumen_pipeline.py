"""
Lumen Pipeline Integration — XArraySQLSource in a Lumen Pipeline.

Shows how lumen-xarray integrates with Lumen's Pipeline system,
the same way DuckDB or other SQL sources do. This is what Lumen AI
agents use internally.

Run: PYTHONPATH=. python examples/lumen_pipeline.py
"""

import xarray as xr
from lumen.pipeline import Pipeline

from lumen_xarray import XArraySQLSource
from lumen_xarray.transforms import (
    DimensionAggregate, TimeResample, Anomaly, RollingWindow,
)


# ── 1. Create source from any xarray dataset ──
ds = xr.tutorial.open_dataset("air_temperature")
source = XArraySQLSource(_dataset=ds)

print("=== Source ===")
print(f"  Tables: {source.get_tables()}")
print(f"  Size:   {source.estimate_size('air')}")
print()


# ── 2. Lumen Pipeline with SQL source ──
pipeline = Pipeline(source=source, table="air")

print("=== Pipeline: raw data ===")
data = pipeline.data
print(f"  Shape: {data.shape}")
print(f"  Columns: {list(data.columns)}")
print(f"  Head:\n{data.head()}")
print()


# ── 3. Pipeline with transforms ──
pipeline_agg = Pipeline(
    source=source,
    table="air",
    transforms=[
        DimensionAggregate(dimensions=["lat", "lon"], method="mean", value_columns=["air"]),
        TimeResample(time_col="time", freq="MS"),
    ],
)

print("=== Pipeline: monthly spatial average ===")
data = pipeline_agg.data
print(f"  Shape: {data.shape}")
print(f"  Head:\n{data.head()}")
print()


# ── 4. Pipeline with anomaly ──
pipeline_anom = Pipeline(
    source=source,
    table="air",
    transforms=[
        DimensionAggregate(dimensions=["lat", "lon"], method="mean", value_columns=["air"]),
        TimeResample(time_col="time", freq="MS"),
        Anomaly(time_col="time", value_col="air", groupby="month"),
    ],
)

print("=== Pipeline: monthly anomaly ===")
data = pipeline_anom.data
print(f"  Shape: {data.shape}")
print(f"  Columns: {list(data.columns)}")
print(f"  Head:\n{data.head()}")
print()


# ── 5. Schema for AI agents ──
schema = source.get_schema("air")
print("=== Schema (used by Lumen AI) ===")
for col, info in schema.items():
    if col == "__len__":
        print(f"  Total rows: {info:,}")
    else:
        print(f"  {col}: {info}")
print()


# ── 6. Metadata for AI agents ──
meta = source.get_metadata("air")
print("=== Metadata (used by Lumen AI) ===")
print(f"  Description: {meta['description']}")
for col, info in meta["columns"].items():
    print(f"  {col}: {info}")
print()


# ── 7. Dimension info for UI widgets ──
dim_info = source.get_dimension_info("air")
print("=== Dimension Info (used for UI) ===")
for dim, info in dim_info.items():
    print(f"  {dim}: {info['type']} [{info.get('min', '?')} .. {info.get('max', '?')}] ({info['size']} values)")
print()


# ── 8. Direct SQL (what AI agents generate) ──
print("=== SQL Queries (what Lumen AI generates) ===")
queries = [
    "SELECT lat, AVG(air) as avg_temp FROM air GROUP BY lat ORDER BY lat",
    "SELECT EXTRACT(MONTH FROM time) as month, AVG(air) FROM air GROUP BY month ORDER BY month",
    "SELECT * FROM air WHERE lat > 60 AND air < 250 LIMIT 5",
]
for q in queries:
    print(f"\n  SQL: {q}")
    df = source.execute(q)
    print(f"  Result ({len(df)} rows):\n{df.to_string(index=False)}")

print("\nAll Lumen pipeline integrations work.")
