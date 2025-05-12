"""Tests for the build_data function."""
import pandas as pd
import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

from jquantstats.api import build_data


def test_no_rf(returns):
    d = build_data(returns, date_col="Date")
    assert_frame_equal(d.returns, returns.drop("Date"))


def test_with_constant_rf(returns):
    """
    Tests that build_data works with a constant risk-free rate.

    Args:
        returns (pd.DataFrame): The returns fixture containing asset returns.

    Verifies:
        1. The function returns a Data object.
        2. The returns in the Data object match the input returns minus the risk-free rate.
        3. The benchmark is None when not provided.
    """
    # Test with a small constant and daily risk-free rate
    rf = 0.001
    result = build_data(returns=returns, rf=rf, date_col="Date")

    assert_series_equal(result.returns["Meta"], returns["Meta"] - rf)
    assert_series_equal(result.index.to_series(), returns["Date"])
    # Verify there's no benchmark
    assert result.benchmark is None


def test_with_series_rf(returns):
    """
    Tests that build_data works with a Series as the risk-free rate.

    Args:
        returns (pd.DataFrame): The returns fixture containing asset returns.

    Verifies:
        1. The function returns a Data object.
        2. The returns in the Data object match the input returns minus the risk-free rate.
        3. The risk-free rate Series is correctly aligned with the returns index.
    """
    # Create a Series with a constant risk-free rate
    date_col = "Date"
    rf_scalar = 0.001
    rf = returns.select([
        pl.col(date_col),
        pl.lit(rf_scalar).alias("rf")
    ])

    result = build_data(returns=returns, rf=rf)

    # Verify the returns are correctly adjusted by the risk-free rate
    assert_series_equal(result.returns["Meta"], returns["Meta"] - rf_scalar)


def test_with_benchmark(returns, benchmark):
    """
    Tests that build_data correctly handles a benchmark.

    Args:
        returns (pd.DataFrame): The returns fixture containing asset returns.
        benchmark (pd.Series): The benchmark fixture containing benchmark returns.

    Verifies:
        1. The function returns a Data object with both returns and benchmark.
        2. The returns in the Data object match the input returns.
        3. The benchmark in the Data object matches the input benchmark.
        4. The indices of the returns and benchmark are aligned.
    """
    result = build_data(returns=returns, benchmark=benchmark)
    assert result.benchmark.columns == ["SPY -- Benchmark"]

def test_with_pandas(returns, benchmark):
    x = returns.to_pandas().set_index("Date")
    y = benchmark.to_pandas().set_index("Date")

    data = build_data(returns=x, benchmark=y)
    pd.testing.assert_frame_equal(data.returns_pd, returns.to_pandas().set_index("Date"))
    pd.testing.assert_frame_equal(data.benchmark_pd,
                                  benchmark.to_pandas().set_index("Date").loc[data.benchmark_pd.index])
