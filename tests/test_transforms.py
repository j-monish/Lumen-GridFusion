"""
Tests for xarray-specific Lumen transforms.

Tests each transform independently and in combination (integration).
"""

import numpy as np
import pandas as pd
import pytest

from lumen_xarray.transforms import (
    Anomaly,
    DimensionAggregate,
    DimensionSlice,
    RollingWindow,
    SpatialBBox,
    TimeResample,
)


@pytest.fixture
def sample_df():
    """A DataFrame mimicking xarray-to-DataFrame output."""
    times = pd.date_range("2020-01-01", periods=100, freq="D")
    lats = [10.0, 20.0, 30.0, 40.0, 50.0]
    rows = []
    np.random.seed(42)
    for t in times:
        for lat in lats:
            for lon in [100.0, 110.0, 120.0]:
                rows.append({
                    "time": t,
                    "lat": lat,
                    "lon": lon,
                    "air": 270 + lat / 5 + 10 * np.sin(2 * np.pi * t.dayofyear / 365) + np.random.normal(0, 2),
                })
    return pd.DataFrame(rows)


# ---- DimensionSlice ----

class TestDimensionSlice:

    def test_range_slice(self, sample_df):
        t = DimensionSlice(dimension="lat", start=20.0, stop=40.0)
        result = t.apply(sample_df)
        assert all(result["lat"] >= 20.0)
        assert all(result["lat"] <= 40.0)

    def test_values_slice(self, sample_df):
        t = DimensionSlice(dimension="lat", values=[10.0, 50.0])
        result = t.apply(sample_df)
        assert set(result["lat"].unique()) == {10.0, 50.0}

    def test_time_range(self, sample_df):
        t = DimensionSlice(dimension="time", start="2020-03-01", stop="2020-03-31")
        result = t.apply(sample_df)
        assert all(result["time"] >= pd.Timestamp("2020-03-01"))
        assert all(result["time"] <= pd.Timestamp("2020-03-31"))

    def test_missing_dimension_noop(self, sample_df):
        t = DimensionSlice(dimension="nonexistent", start=0, stop=1)
        result = t.apply(sample_df)
        assert len(result) == len(sample_df)

    def test_nearest_selection(self, sample_df):
        t = DimensionSlice(dimension="lat", values=[22.0], nearest=True)
        result = t.apply(sample_df)
        assert len(result) > 0  # Should find nearest to 22.0 (which is 20.0)

    def test_start_only(self, sample_df):
        t = DimensionSlice(dimension="lat", start=40.0)
        result = t.apply(sample_df)
        assert all(result["lat"] >= 40.0)

    def test_stop_only(self, sample_df):
        t = DimensionSlice(dimension="lat", stop=20.0)
        result = t.apply(sample_df)
        assert all(result["lat"] <= 20.0)


# ---- SpatialBBox ----

class TestSpatialBBox:

    def test_bbox_filter(self, sample_df):
        t = SpatialBBox(lat_min=20, lat_max=40, lon_min=100, lon_max=110)
        result = t.apply(sample_df)
        assert all(result["lat"] >= 20)
        assert all(result["lat"] <= 40)
        assert all(result["lon"] >= 100)
        assert all(result["lon"] <= 110)

    def test_partial_bbox(self, sample_df):
        t = SpatialBBox(lat_min=30)
        result = t.apply(sample_df)
        assert all(result["lat"] >= 30)
        assert len(result["lon"].unique()) == 3  # all lons included

    def test_missing_cols_noop(self, sample_df):
        df = sample_df.rename(columns={"lat": "latitude"})
        t = SpatialBBox(lat_min=30)
        result = t.apply(df)
        assert len(result) == len(df)

    def test_custom_col_names(self, sample_df):
        df = sample_df.rename(columns={"lat": "latitude", "lon": "longitude"})
        t = SpatialBBox(lat_col="latitude", lon_col="longitude", lat_min=30, lat_max=40)
        result = t.apply(df)
        assert all(result["latitude"] >= 30)
        assert all(result["latitude"] <= 40)


# ---- DimensionAggregate ----

class TestDimensionAggregate:

    def test_aggregate_time(self, sample_df):
        t = DimensionAggregate(dimensions=["time"], method="mean")
        result = t.apply(sample_df)
        assert "time" not in result.columns
        assert "lat" in result.columns
        assert "lon" in result.columns
        assert len(result) == 5 * 3  # lat * lon

    def test_aggregate_spatial(self, sample_df):
        t = DimensionAggregate(dimensions=["lat", "lon"], method="mean")
        result = t.apply(sample_df)
        assert "lat" not in result.columns
        assert "lon" not in result.columns
        assert "time" in result.columns
        assert len(result) == 100  # time only

    def test_aggregate_sum(self, sample_df):
        t = DimensionAggregate(
            dimensions=["time"], method="sum", value_columns=["air"]
        )
        result = t.apply(sample_df)
        assert len(result) == 15  # lat * lon

    def test_aggregate_all(self, sample_df):
        t = DimensionAggregate(dimensions=["time", "lat", "lon"], method="mean")
        result = t.apply(sample_df)
        assert len(result) == 1

    def test_missing_dimension(self, sample_df):
        t = DimensionAggregate(dimensions=["nonexistent"], method="mean")
        result = t.apply(sample_df)
        assert len(result) == len(sample_df)

    def test_methods(self, sample_df):
        for method in ["mean", "sum", "std", "min", "max", "count"]:
            t = DimensionAggregate(dimensions=["time"], method=method)
            result = t.apply(sample_df)
            assert len(result) > 0


# ---- TimeResample ----

class TestTimeResample:

    def test_monthly(self, sample_df):
        t = TimeResample(time_col="time", freq="MS", method="mean")
        result = t.apply(sample_df)
        assert len(result) < len(sample_df)
        assert "air" in result.columns

    def test_weekly(self, sample_df):
        t = TimeResample(time_col="time", freq="W")
        result = t.apply(sample_df)
        assert len(result) < len(sample_df)

    def test_with_groupby(self, sample_df):
        t = TimeResample(
            time_col="time", freq="MS", group_cols=["lat", "lon"]
        )
        result = t.apply(sample_df)
        assert "lat" in result.columns
        assert "lon" in result.columns
        # Should have more rows than without grouping
        simple = TimeResample(time_col="time", freq="MS").apply(sample_df)
        assert len(result) >= len(simple)

    def test_missing_time_col(self, sample_df):
        t = TimeResample(time_col="nonexistent", freq="MS")
        result = t.apply(sample_df)
        assert len(result) == len(sample_df)

    def test_sum_method(self, sample_df):
        t = TimeResample(time_col="time", freq="MS", method="sum")
        result = t.apply(sample_df)
        assert "air" in result.columns


# ---- Anomaly ----

class TestAnomaly:

    def test_monthly_anomaly(self, sample_df):
        t = Anomaly(time_col="time", value_col="air", groupby="month")
        result = t.apply(sample_df)
        assert "air_anomaly" in result.columns
        # Anomalies should be centered around zero
        assert abs(result["air_anomaly"].mean()) < 5

    def test_overall_anomaly(self, sample_df):
        t = Anomaly(value_col="air", groupby=None)
        result = t.apply(sample_df)
        assert "air_anomaly" in result.columns
        assert abs(result["air_anomaly"].mean()) < 1e-10

    def test_season_anomaly(self, sample_df):
        t = Anomaly(time_col="time", value_col="air", groupby="season")
        result = t.apply(sample_df)
        assert "air_anomaly" in result.columns

    def test_dayofyear_anomaly(self, sample_df):
        t = Anomaly(time_col="time", value_col="air", groupby="dayofyear")
        result = t.apply(sample_df)
        assert "air_anomaly" in result.columns

    def test_missing_value_col(self, sample_df):
        t = Anomaly(value_col="nonexistent", groupby="month")
        result = t.apply(sample_df)
        assert "nonexistent_anomaly" not in result.columns
        assert len(result) == len(sample_df)


# ---- RollingWindow ----

class TestRollingWindow:

    def test_rolling_mean(self, sample_df):
        t = RollingWindow(column="air", window=7, method="mean")
        result = t.apply(sample_df)
        assert "air_rolling_mean" in result.columns
        # Rolling mean should be smoother than raw
        raw_std = result["air"].std()
        rolling_std = result["air_rolling_mean"].dropna().std()
        assert rolling_std <= raw_std

    def test_custom_output_col(self, sample_df):
        t = RollingWindow(column="air", window=7, output_column="smoothed")
        result = t.apply(sample_df)
        assert "smoothed" in result.columns

    def test_rolling_sum(self, sample_df):
        t = RollingWindow(column="air", window=3, method="sum")
        result = t.apply(sample_df)
        assert "air_rolling_sum" in result.columns

    def test_missing_column(self, sample_df):
        t = RollingWindow(column="nonexistent", window=7)
        result = t.apply(sample_df)
        assert len(result) == len(sample_df)


# ---- Integration (chaining transforms) ----

class TestIntegration:

    def test_slice_then_aggregate(self, sample_df):
        """Slice to a region, then aggregate over time."""
        df = DimensionSlice(dimension="lat", start=20, stop=40).apply(sample_df)
        df = SpatialBBox(lon_min=100, lon_max=110).apply(df)
        result = DimensionAggregate(dimensions=["time"], method="mean").apply(df)
        assert all(result["lat"] >= 20)
        assert all(result["lat"] <= 40)
        assert all(result["lon"] <= 110)
        assert "time" not in result.columns

    def test_resample_then_anomaly(self, sample_df):
        """Resample to monthly, then compute anomalies."""
        df = TimeResample(time_col="time", freq="MS").apply(sample_df)
        result = Anomaly(time_col="time", value_col="air", groupby="month").apply(df)
        assert "air_anomaly" in result.columns

    def test_full_pipeline(self, sample_df):
        """Slice -> BBox -> Resample -> Rolling -> Anomaly."""
        df = DimensionSlice(dimension="time", start="2020-01-15", stop="2020-03-31").apply(sample_df)
        df = SpatialBBox(lat_min=20, lat_max=40).apply(df)
        df = DimensionAggregate(dimensions=["lat", "lon"], method="mean").apply(df)
        df = RollingWindow(column="air", window=3).apply(df)
        result = Anomaly(value_col="air", groupby=None).apply(df)
        assert "air_anomaly" in result.columns
        assert "air_rolling_mean" in result.columns
        assert len(result) > 0
