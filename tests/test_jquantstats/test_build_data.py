"""Tests for the build_data function."""

import narwhals as nw
import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

from jquantstats.api import build_data


def test_no_rf(returns):
    """Tests that build_data works without a risk-free rate.

    Args:
        returns (pd.DataFrame): The returns fixture containing asset returns.

    Verifies:
        1. The function returns a Data object.
        2. The returns in the Data object match the input returns.
        3. The date column is correctly excluded from the returns.

    """
    d = build_data(returns, date_col="Date")
    assert_frame_equal(d.returns, returns.drop("Date"))


def test_with_constant_rf(returns):
    """Tests that build_data works with a constant risk-free rate.

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
    """Tests that build_data works with a Series as the risk-free rate.

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
    rf = returns.select([pl.col(date_col), pl.lit(rf_scalar).alias("rf")])

    result = build_data(returns=returns, rf=rf)

    # Verify the returns are correctly adjusted by the risk-free rate
    assert_series_equal(result.returns["Meta"], returns["Meta"] - rf_scalar)


def test_pandas_input(returns):
    """Tests that build_data accepts a pandas DataFrame as input."""
    pandas_returns = returns.to_pandas()
    d = build_data(pandas_returns, date_col="Date")
    assert_frame_equal(d.returns, returns.drop("Date"))


def test_with_benchmark(returns, benchmark_frame):
    """Tests that build_data correctly handles a benchmark.

    Args:
        returns (pd.DataFrame): The returns fixture containing asset returns.
        benchmark_frame (pd.DataFrame): The benchmark fixture containing benchmark returns.

    Verifies:
        1. The function returns a Data object with both returns and benchmark.
        2. The returns in the Data object match the input returns.
        3. The benchmark in the Data object matches the input benchmark.
        4. The indices of the returns and benchmark are aligned.

    """
    result = build_data(returns=returns, benchmark=benchmark_frame)
    b = result.benchmark
    assert b is not None
    assert b.columns == ["SPY -- Benchmark"]


def test_narwhals_polars_round_trip(returns):
    """Validates that a native Polars DataFrame round-trips correctly through build_data.

    Narwhals identifies polars DataFrames natively; this test ensures the
    fast-path (``isinstance(df, pl.DataFrame) → return df``) in ``_to_polars``
    is covered and produces identical output.

    Args:
        returns: The returns fixture (a polars DataFrame).

    Verifies:
        The Data.returns produced from a polars input equals the input minus
        the date column.
    """
    nw_frame = nw.from_native(returns, eager_only=True)
    # Round-trip through build_data — narwhals wraps polars transparently.
    d = build_data(nw_frame, date_col="Date")
    assert_frame_equal(d.returns, returns.drop("Date"))


def test_narwhals_pandas_round_trip(returns):
    """Validates that a narwhals-wrapped pandas DataFrame round-trips through build_data.

    ``build_data`` accepts any ``narwhals``-compatible eager frame via
    ``NativeFrame``.  This test exercises the conversion path
    ``nw.from_native(pandas_df, eager_only=True).to_polars()`` end-to-end.

    Args:
        returns: The returns fixture (a polars DataFrame used as source).

    Verifies:
        The Data.returns produced from a pandas-backed input matches the
        result produced from the native polars input.
    """
    pandas_returns = returns.to_pandas()
    nw_frame = nw.from_native(pandas_returns, eager_only=True)
    d = build_data(nw_frame, date_col="Date")
    assert_frame_equal(d.returns, returns.drop("Date"))
