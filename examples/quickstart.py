"""
Quickstart: XArraySQLSource — Query scientific data with SQL

Demonstrates loading an xarray dataset and running SQL queries
against it via Apache DataFusion.
"""

import xarray as xr

from lumen_xarray import XArraySQLSource

# Load NOAA air temperature dataset (time x lat x lon)
ds = xr.tutorial.open_dataset("air_temperature")
print(f"Dataset: {ds}")
print()

# Create SQL-backed source
source = XArraySQLSource(_dataset=ds)
print(f"Source: {source}")
print(f"Tables: {source.get_tables()}")
print()

# --- SQL Queries ---

# 1. Basic select
print("=== Top 5 rows ===")
df = source.execute("SELECT * FROM air LIMIT 5")
print(df)
print()

# 2. Aggregation
print("=== Average temperature by latitude ===")
df = source.execute("""
    SELECT lat, AVG(air) as avg_temp
    FROM air
    GROUP BY lat
    ORDER BY lat
""")
print(df)
print()

# 3. Filtering
print("=== Arctic temperatures (lat > 60) ===")
df = source.execute("""
    SELECT time, lat, lon, air
    FROM air
    WHERE lat > 60
    LIMIT 10
""")
print(df)
print()

# 4. Complex query: hottest locations
print("=== Top 5 hottest grid cells ===")
df = source.execute("""
    SELECT lat, lon, AVG(air) as avg_temp
    FROM air
    GROUP BY lat, lon
    ORDER BY avg_temp DESC
    LIMIT 5
""")
print(df)
print()

# --- Lumen Source API ---

# 5. Schema (used by Lumen AI agents)
print("=== Schema ===")
schema = source.get_schema("air")
for col, info in schema.items():
    if col != "__len__":
        print(f"  {col}: {info}")
print(f"  Total rows: {schema['__len__']}")
print()

# 6. Metadata (used by Lumen AI for context)
print("=== Metadata ===")
meta = source.get_metadata("air")
print(f"  Description: {meta['description']}")
print(f"  Dimensions: {meta['dimensions']}")
for col, info in meta["columns"].items():
    print(f"  {col}: {info}")
print()

# 7. Dimension info (for UI sliders/controls)
print("=== Dimension Info ===")
dim_info = source.get_dimension_info("air")
for dim, info in dim_info.items():
    print(f"  {dim}: {info['type']} [{info.get('min', '?')} .. {info.get('max', '?')}] ({info['size']} values)")
print()

# 8. Standard get() with filter
print("=== get() with filter (lat=75.0) ===")
df = source.get("air", lat=75.0)
print(f"  Shape: {df.shape}")
print(df.head())

print("\nDone! All Lumen Source API methods work with xarray data via SQL.")
