"""
lumen-xarray: Native xarray support for Lumen.

Provides XArraySQLSource (SQL-backed via xarray-sql/DataFusion)
and XArraySource (DataFrame-based) for working with N-dimensional
scientific data in Lumen pipelines and Lumen AI.
"""

from .source import XArraySQLSource, XARRAY_ENGINES, XARRAY_EXTENSIONS
from .basic_source import XArraySource
from .transforms import (
    DimensionSlice,
    SpatialBBox,
    DimensionAggregate,
    TimeResample,
    Anomaly,
    RollingWindow,
)

__all__ = [
    "XArraySQLSource",
    "XArraySource",
    "DimensionSlice",
    "SpatialBBox",
    "DimensionAggregate",
    "TimeResample",
    "Anomaly",
    "RollingWindow",
]
