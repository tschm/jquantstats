"""Tests for the LazyData class — lazy Polars-backed financial returns container."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

from jquantstats import Data, LazyData

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_data() -> Data:
    """Build a minimal Data object with daily returns for two assets."""
    dates = [date(2023, 1, i) for i in range(1, 11)]
    returns = pl.DataFrame({"Date": dates, "A": [0.01 * i for i in range(10)], "B": [-0.005 * i for i in range(10)]})
    return Data.from_returns(returns=returns)


def _simple_data_with_benchmark() -> Data:
    """Build a minimal Data object with benchmark."""
    dates = [date(2023, 1, i) for i in range(1, 11)]
    returns = pl.DataFrame({"Date": dates, "A": [0.01 * i for i in range(10)]})
    benchmark = pl.DataFrame({"Date": dates, "BM": [0.005 * i for i in range(10)]})
    return Data.from_returns(returns=returns, benchmark=benchmark)


# ---------------------------------------------------------------------------
# Data.lazy() constructor
# ---------------------------------------------------------------------------


def test_data_lazy_returns_lazy_data():
    """Data.lazy() should return a LazyData instance."""
    data = _simple_data()
    lazy = data.lazy()
    assert isinstance(lazy, LazyData)


def test_data_lazy_fields_are_lazy_frames():
    """LazyData fields should be pl.LazyFrame instances."""
    data = _simple_data()
    lazy = data.lazy()
    assert isinstance(lazy.returns, pl.LazyFrame)
    assert isinstance(lazy.index, pl.LazyFrame)


def test_data_lazy_benchmark_is_none_when_no_benchmark():
    """LazyData.benchmark should be None when source Data has no benchmark."""
    data = _simple_data()
    lazy = data.lazy()
    assert lazy.benchmark is None


def test_data_lazy_benchmark_is_lazy_frame_when_present():
    """LazyData.benchmark should be a LazyFrame when source Data has a benchmark."""
    data = _simple_data_with_benchmark()
    lazy = data.lazy()
    assert isinstance(lazy.benchmark, pl.LazyFrame)


# ---------------------------------------------------------------------------
# LazyData.collect()
# ---------------------------------------------------------------------------


def test_collect_round_trips_to_data():
    """collect() should produce a Data whose returns/index match the source."""
    data = _simple_data()
    result = data.lazy().collect()
    assert isinstance(result, Data)
    assert_frame_equal(result.returns, data.returns)
    assert_frame_equal(result.index, data.index)


def test_collect_round_trips_benchmark():
    """collect() should preserve the benchmark frame."""
    data = _simple_data_with_benchmark()
    result = data.lazy().collect()
    assert result.benchmark is not None
    assert_frame_equal(result.benchmark, data.benchmark)


def test_collect_no_benchmark_remains_none():
    """collect() on a no-benchmark LazyData should return Data with None benchmark."""
    data = _simple_data()
    result = data.lazy().collect()
    assert result.benchmark is None


# ---------------------------------------------------------------------------
# LazyData.head() / tail()
# ---------------------------------------------------------------------------


def test_lazy_head_returns_first_n_rows():
    """head(n) should produce first n rows after collect."""
    data = _simple_data()
    result = data.lazy().head(3).collect()
    assert result.returns.shape[0] == 3
    assert_frame_equal(result.returns, data.returns.head(3))
    assert_frame_equal(result.index, data.index.head(3))


def test_lazy_tail_returns_last_n_rows():
    """tail(n) should produce last n rows after collect."""
    data = _simple_data()
    result = data.lazy().tail(4).collect()
    assert result.returns.shape[0] == 4
    assert_frame_equal(result.returns, data.returns.tail(4))
    assert_frame_equal(result.index, data.index.tail(4))


def test_lazy_head_with_benchmark():
    """head() should work when benchmark is present."""
    data = _simple_data_with_benchmark()
    result = data.lazy().head(2).collect()
    assert result.returns.shape[0] == 2
    assert result.benchmark is not None
    assert result.benchmark.shape[0] == 2


def test_lazy_tail_with_benchmark():
    """tail() should work when benchmark is present."""
    data = _simple_data_with_benchmark()
    result = data.lazy().tail(2).collect()
    assert result.returns.shape[0] == 2
    assert result.benchmark is not None
    assert result.benchmark.shape[0] == 2


# ---------------------------------------------------------------------------
# LazyData.truncate()
# ---------------------------------------------------------------------------


def test_lazy_truncate_by_start_and_end():
    """truncate(start, end) should filter rows by date lazily."""
    data = _simple_data()
    mid_start = data.index["Date"][2]
    mid_end = data.index["Date"][6]
    result = data.lazy().truncate(start=mid_start, end=mid_end).collect()
    assert result.index["Date"][0] == mid_start
    assert result.index["Date"][-1] == mid_end
    assert result.returns.shape[0] == 5


def test_lazy_truncate_start_only():
    """truncate(start=...) should keep rows from start onwards."""
    data = _simple_data()
    mid_start = data.index["Date"][3]
    result = data.lazy().truncate(start=mid_start).collect()
    assert result.index["Date"][0] == mid_start
    assert result.returns.shape[0] == data.returns.shape[0] - 3


def test_lazy_truncate_end_only():
    """truncate(end=...) should keep rows up to end inclusive."""
    data = _simple_data()
    mid_end = data.index["Date"][4]
    result = data.lazy().truncate(end=mid_end).collect()
    assert result.index["Date"][-1] == mid_end
    assert result.returns.shape[0] == 5


def test_lazy_truncate_no_bounds_returns_all():
    """truncate() with no bounds should return all rows unchanged."""
    data = _simple_data()
    result = data.lazy().truncate().collect()
    assert_frame_equal(result.returns, data.returns)
    assert_frame_equal(result.index, data.index)


def test_lazy_truncate_with_benchmark():
    """truncate() should filter benchmark rows in sync with returns."""
    data = _simple_data_with_benchmark()
    mid_start = data.index["Date"][2]
    mid_end = data.index["Date"][6]
    result = data.lazy().truncate(start=mid_start, end=mid_end).collect()
    assert result.benchmark is not None
    assert result.benchmark.shape[0] == result.returns.shape[0]


# ---------------------------------------------------------------------------
# LazyData.resample()
# ---------------------------------------------------------------------------


def test_lazy_resample_reduces_row_count():
    """resample("1y") on daily data should produce fewer rows."""
    dates = [date(2022, m, 15) for m in range(1, 13)] + [date(2023, m, 15) for m in range(1, 13)]
    returns = pl.DataFrame({"Date": dates, "A": [0.01] * 24})
    data = Data.from_returns(returns=returns)
    result = data.lazy().resample("1y").collect()
    # 2 years of monthly data → 2 yearly rows (or 3 with boundary)
    assert result.returns.shape[0] < data.returns.shape[0]


def test_lazy_resample_preserves_columns():
    """resample() should keep the same asset columns."""
    data = _simple_data()
    result = data.lazy().resample("1mo").collect()
    assert result.returns.columns == data.returns.columns


def test_lazy_resample_with_benchmark_preserves_benchmark_columns():
    """resample() with benchmark should keep benchmark columns."""
    data = _simple_data_with_benchmark()
    result = data.lazy().resample("1mo").collect()
    assert result.benchmark is not None
    assert result.benchmark.columns == data.benchmark.columns


# ---------------------------------------------------------------------------
# LazyData.scan_parquet() / scan_csv()
# ---------------------------------------------------------------------------


def test_scan_parquet_round_trip(tmp_path: Path):
    """scan_parquet should read data equivalent to reading eagerly."""
    data = _simple_data()
    parquet_path = str(tmp_path / "returns.parquet")
    data.all.write_parquet(parquet_path)

    lazy = LazyData.scan_parquet(parquet_path, date_col="Date")
    result = lazy.collect()

    assert_frame_equal(result.returns, data.returns)
    assert_frame_equal(result.index, data.index)


def test_scan_parquet_with_benchmark(tmp_path: Path):
    """scan_parquet should read benchmark data when path provided."""
    data = _simple_data_with_benchmark()
    returns_path = str(tmp_path / "returns.parquet")
    bench_path = str(tmp_path / "bench.parquet")
    pl.concat([data.index, data.returns], how="horizontal").write_parquet(returns_path)
    pl.concat([data.index, data.benchmark], how="horizontal").write_parquet(bench_path)

    lazy = LazyData.scan_parquet(returns_path, benchmark=bench_path, date_col="Date")
    result = lazy.collect()

    assert_frame_equal(result.returns, data.returns)
    assert result.benchmark is not None
    assert_frame_equal(result.benchmark, data.benchmark)


def test_scan_parquet_rf_adjustment(tmp_path: Path):
    """scan_parquet with rf!=0 should subtract rf from all return columns."""
    data = _simple_data()
    parquet_path = str(tmp_path / "returns.parquet")
    data.all.write_parquet(parquet_path)

    rf = 0.001
    lazy = LazyData.scan_parquet(parquet_path, rf=rf, date_col="Date")
    result = lazy.collect()

    expected = data.returns.with_columns([pl.col(c) - rf for c in data.returns.columns])
    assert_frame_equal(result.returns, expected)


def test_scan_parquet_truncate_pushdown(tmp_path: Path):
    """Truncation applied to a scan-backed LazyData should filter correctly."""
    data = _simple_data()
    parquet_path = str(tmp_path / "returns.parquet")
    data.all.write_parquet(parquet_path)

    mid_start = data.index["Date"][2]
    result = LazyData.scan_parquet(parquet_path, date_col="Date").truncate(start=mid_start).collect()
    assert result.index["Date"][0] == mid_start


def test_scan_csv_round_trip(tmp_path: Path):
    """scan_csv should produce the same result as reading eagerly."""
    data = _simple_data()
    csv_path = str(tmp_path / "returns.csv")
    data.all.write_csv(csv_path)

    lazy = LazyData.scan_csv(csv_path, date_col="Date")
    result = lazy.collect()

    # scan_csv infers the Date column type; use eager read as reference
    eager = Data.from_returns(pl.read_csv(csv_path, try_parse_dates=True))
    assert_frame_equal(result.returns, eager.returns)
    assert_frame_equal(result.index, eager.index)


def test_scan_csv_with_benchmark(tmp_path: Path):
    """scan_csv should read benchmark data when path provided."""
    data = _simple_data_with_benchmark()
    returns_csv = str(tmp_path / "returns.csv")
    bench_csv = str(tmp_path / "bench.csv")
    pl.concat([data.index, data.returns], how="horizontal").write_csv(returns_csv)
    pl.concat([data.index, data.benchmark], how="horizontal").write_csv(bench_csv)

    result = LazyData.scan_csv(returns_csv, benchmark=bench_csv, date_col="Date").collect()
    assert result.benchmark is not None
    assert result.benchmark.columns == data.benchmark.columns


def test_scan_csv_rf_adjustment(tmp_path: Path):
    """scan_csv with rf!=0 should subtract rf from all return columns."""
    data = _simple_data()
    csv_path = str(tmp_path / "returns.csv")
    data.all.write_csv(csv_path)

    rf = 0.002
    result = LazyData.scan_csv(csv_path, rf=rf, date_col="Date").collect()
    # Re-read eagerly and subtract RF for comparison
    eager = Data.from_returns(pl.read_csv(csv_path, try_parse_dates=True))
    expected = eager.returns.with_columns([pl.col(c) - rf for c in eager.returns.columns])
    assert_frame_equal(result.returns, expected)


# ---------------------------------------------------------------------------
# Chained lazy operations
# ---------------------------------------------------------------------------


def test_chain_truncate_resample():
    """Chaining truncate then resample lazily should produce a valid Data."""
    dates = [date(2022, m, 15) for m in range(1, 13)] + [date(2023, m, 15) for m in range(1, 13)]
    returns = pl.DataFrame({"Date": dates, "A": [0.01] * 24})
    data = Data.from_returns(returns=returns)

    result = data.lazy().truncate(start=date(2022, 3, 1), end=date(2023, 6, 30)).resample("1y").collect()
    assert isinstance(result, Data)
    assert result.returns.shape[0] > 0


def test_chain_head_collect_returns_lazy_data_type():
    """head() should return LazyData, not Data."""
    data = _simple_data()
    lazy = data.lazy().head(3)
    assert isinstance(lazy, LazyData)


def test_chain_tail_collect_returns_lazy_data_type():
    """tail() should return LazyData, not Data."""
    data = _simple_data()
    lazy = data.lazy().tail(3)
    assert isinstance(lazy, LazyData)


def test_chain_truncate_returns_lazy_data_type():
    """truncate() should return LazyData, not Data."""
    data = _simple_data()
    lazy = data.lazy().truncate(start=date(2023, 1, 3))
    assert isinstance(lazy, LazyData)


def test_chain_resample_returns_lazy_data_type():
    """resample() should return LazyData, not Data."""
    data = _simple_data()
    lazy = data.lazy().resample("1mo")
    assert isinstance(lazy, LazyData)
