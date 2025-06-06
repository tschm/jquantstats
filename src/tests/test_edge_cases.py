"""Tests for edge cases in the jquantstats package."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from jquantstats.api import build_data


@pytest.fixture
def data_no_benchmark(returns):
    """Fixture that returns a Data object with no benchmark.

    Args:
        returns: The returns fixture containing asset returns.

    Returns:
        _Data: A Data object with returns but no benchmark.

    """
    return build_data(returns=returns)


def test_copy_no_benchmark(data_no_benchmark):
    """Tests that the copy() method works correctly when benchmark is None.

    Args:
        data_no_benchmark: Fixture that returns a Data object with no benchmark.

    Verifies:
        1. The return value is a Data object.
        2. The copied object has the same returns as the original.
        3. The benchmark is None in both the original and the copy.
        4. Modifying the copied object does not affect the original.

    """
    # Create a copy of the data object
    data_copy = data_no_benchmark.copy()

    # Verify the copy has the same returns as the original
    assert_frame_equal(data_copy.returns, data_no_benchmark.returns)

    # Verify the benchmark is None in both the original and the copy
    assert data_no_benchmark.benchmark is None
    assert data_copy.benchmark is None

    # Verify that modifying the copy doesn't affect the original
    assert data_copy is not data_no_benchmark
    assert data_copy.returns is not data_no_benchmark.returns


def test_r_squared_no_benchmark(data_no_benchmark):
    """Tests that the r_squared() method raises an AttributeError when benchmark is None.

    Args:
        data_no_benchmark: Fixture that returns a Data object with no benchmark.

    Verifies:
        1. Calling r_squared() raises an AttributeError with the expected message.

    """
    # Verify that calling r_squared() raises an AttributeError
    with pytest.raises(AttributeError, match="No benchmark data available"):
        data_no_benchmark.stats.r_squared()


def test_non_overlapping_dates():
    """Tests that build_data raises a ValueError when returns and benchmark have non-overlapping dates.

    Verifies:
        1. Calling build_data with non-overlapping dates raises a ValueError with the expected message.
    """
    # Create returns data with dates in 2020
    returns_dates = [f"2020-01-{i:02d}" for i in range(1, 11)]
    returns = pl.DataFrame({
        "Date": returns_dates,
        "Asset": [0.01] * len(returns_dates)
    }).with_columns(pl.col("Date").str.to_date())

    # Create benchmark data with dates in 2010
    benchmark_dates = [f"2010-01-{i:02d}" for i in range(1, 11)]
    benchmark = pl.DataFrame({
        "Date": benchmark_dates,
        "Benchmark": [0.01] * len(benchmark_dates)
    }).with_columns(pl.col("Date").str.to_date())

    # Verify that calling build_data raises a ValueError
    with pytest.raises(ValueError, match="No overlapping dates between returns and benchmark."):
        build_data(returns=returns, benchmark=benchmark)
