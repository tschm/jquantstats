"""Tests for the build_data function."""

import numpy as np
import pandas as pd

from jquantstats.api import build_data


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
    result = build_data(returns=returns, rf=rf)

    # Verify the returns are correctly adjusted by the risk-free rate
    expected_returns = returns - rf
    pd.testing.assert_frame_equal(result.returns, expected_returns)

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
    rf = pd.Series(index=returns.index, data=0.001)
    result = build_data(returns=returns, rf=rf)

    # Verify the returns are correctly adjusted by the risk-free rate
    expected_returns = returns - rf
    pd.testing.assert_frame_equal(result.returns, expected_returns)


def test_with_series_rf_different_index(returns):
    """
    Tests that build_data correctly handles a Series risk-free rate with a different index.

    Args:
        returns (pd.DataFrame): The returns fixture containing asset returns.

    Verifies:
        1. The function returns a Data object.
        2. The risk-free rate Series is correctly filtered to match the returns index.
        3. The returns in the Data object match the input returns minus the filtered risk-free rate.
    """
    # Create a Series with a different index (extended by 5 days)
    extended_index = returns.index.union(
        pd.date_range(start=returns.index[-1] + pd.Timedelta(days=1), periods=5, freq="D")
    )
    rf = pd.Series(index=extended_index, data=0.001)

    result = build_data(returns=returns, rf=rf)

    # Verify the returns are correctly adjusted by the risk-free rate
    expected_returns = returns - rf[rf.index.isin(returns.index)]
    pd.testing.assert_frame_equal(result.returns, expected_returns)


def test_with_deannualized_rf(returns):
    """
    Tests that build_data correctly deannualizes the risk-free rate when nperiods is provided.

    Args:
        returns (pd.DataFrame): The returns fixture containing asset returns.

    Verifies:
        1. The function returns a Data object.
        2. The risk-free rate is correctly deannualized using the formula: (1 + rf)^(1/nperiods) - 1.
        3. The returns in the Data object match the input returns minus the deannualized risk-free rate.
    """
    # Use an annual risk-free rate of 5%
    annual_rf = 0.05
    nperiods = 252  # Assuming 252 trading days in a year

    result = build_data(returns=returns, rf=annual_rf, nperiods=nperiods)

    # Calculate the expected daily risk-free rate
    daily_rf = np.power(1 + annual_rf, 1.0 / nperiods) - 1.0

    # Verify the returns are correctly adjusted by the deannualized risk-free rate
    expected_returns = returns - daily_rf
    pd.testing.assert_frame_equal(result.returns, expected_returns)


def test_with_series_rf_and_nperiods(returns):
    """
    Tests that build_data correctly deannualizes a Series risk-free rate when nperiods is provided.

    Args:
        returns (pd.DataFrame): The returns fixture containing asset returns.

    Verifies:
        1. The function returns a Data object.
        2. The Series risk-free rate is correctly deannualized using the formula: (1 + rf)^(1/nperiods) - 1.
        3. The returns in the Data object match the input returns minus the deannualized risk-free rate.
    """
    # Create a Series with an annual risk-free rate of 5%
    annual_rf = pd.Series(index=returns.index, data=0.05)
    nperiods = 252  # Assuming 252 trading days in a year

    result = build_data(returns=returns, rf=annual_rf, nperiods=nperiods)

    # Calculate the expected daily risk-free rate
    daily_rf = np.power(1 + annual_rf, 1.0 / nperiods) - 1.0

    # Verify the returns are correctly adjusted by the deannualized risk-free rate
    expected_returns = returns - daily_rf
    pd.testing.assert_frame_equal(result.returns, expected_returns)


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

    # Find common dates between returns and benchmark
    common_dates = sorted(list(set(returns.index) & set(benchmark.index)))

    # Verify the returns match the input returns for common dates
    pd.testing.assert_frame_equal(result.returns, returns.loc[common_dates])

    # Verify the benchmark matches the input benchmark for common dates
    pd.testing.assert_series_equal(result.benchmark, benchmark.loc[common_dates])

    # Verify the indices are aligned
    pd.testing.assert_index_equal(result.returns.index, result.benchmark.index)


def test_with_benchmark_different_index(returns, benchmark):
    """
    Tests that build_data correctly aligns returns and benchmark with different indices.

    Args:
        returns (pd.DataFrame): The returns fixture containing asset returns.
        benchmark (pd.Series): The benchmark fixture containing benchmark returns.

    Verifies:
        1. The function returns a Data object with both returns and benchmark.
        2. The indices of the returns and benchmark in the Data object are aligned.
        3. Only the common dates between returns and benchmark are included.
    """
    # Create a benchmark with a different index (extended by 5 days)
    extended_index = benchmark.index.union(
        pd.date_range(start=benchmark.index[-1] + pd.Timedelta(days=1), periods=5, freq="D")
    )

    # Create a modified benchmark with the extended index
    # Make sure the index has the same name as the returns index
    extended_benchmark = pd.Series(
        index=pd.Index(extended_index, name=returns.index.name),
        data=np.random.randn(len(extended_index)) * 0.01
    )
    # Preserve the name of the original benchmark
    extended_benchmark.name = benchmark.name
    # Copy the values from the original benchmark
    extended_benchmark.loc[benchmark.index] = benchmark

    # Create a modified returns with the same index name
    modified_returns = returns.copy()

    result = build_data(returns=modified_returns, benchmark=extended_benchmark)

    # Verify the indices are aligned
    pd.testing.assert_index_equal(result.returns.index, result.benchmark.index)

    # Verify only common dates are included
    common_dates = sorted(list(set(modified_returns.index) & set(extended_benchmark.index)))
    assert len(result.returns) == len(common_dates)
    assert all(date in common_dates for date in result.returns.index)


def test_with_benchmark_and_rf(returns, benchmark):
    """
    Tests that build_data correctly handles both a benchmark and a risk-free rate.

    Args:
        returns (pd.DataFrame): The returns fixture containing asset returns.
        benchmark (pd.Series): The benchmark fixture containing benchmark returns.

    Verifies:
        1. The function returns a Data object with both returns and benchmark.
        2. The returns and benchmark in the Data object are correctly adjusted by the risk-free rate.
        3. The indices of the returns and benchmark are aligned.
    """
    rf = 0.001
    result = build_data(returns=returns, benchmark=benchmark, rf=rf)

    # Find common dates between returns and benchmark
    common_dates = sorted(list(set(returns.index) & set(benchmark.index)))

    # Verify the returns are correctly adjusted by the risk-free rate
    expected_returns = returns.loc[common_dates] - rf
    pd.testing.assert_frame_equal(result.returns, expected_returns)

    # Verify the benchmark is correctly adjusted by the risk-free rate
    expected_benchmark = benchmark.loc[common_dates] - rf
    pd.testing.assert_series_equal(result.benchmark, expected_benchmark)

    # Verify the indices are aligned
    pd.testing.assert_index_equal(result.returns.index, result.benchmark.index)


def test_with_series_input(returns):
    """
    Tests that build_data correctly handles a Series as input for returns.

    Args:
        returns (pd.DataFrame): The returns fixture containing asset returns.

    Verifies:
        1. The function returns a Data object.
        2. The returns in the Data object are converted to a DataFrame.
        3. The Series is converted to a DataFrame with a column named "returns".
    """
    # Extract a single column as a Series
    series = returns["Meta"]

    result = build_data(returns=series)

    # Verify the returns are a DataFrame
    assert isinstance(result.returns, pd.DataFrame)

    # Verify the DataFrame has a single column named "returns"
    assert result.returns.shape[1] == 1
    assert result.returns.columns[0] == "returns"

    # Verify the values match the input Series (ignoring the column name)
    pd.testing.assert_series_equal(
        result.returns["returns"],
        series,
        check_names=False
    )


def test_timezone_handling(returns):
    """
    Tests that build_data correctly handles timezone information.

    Args:
        returns (pd.DataFrame): The returns fixture containing asset returns.

    Verifies:
        1. The function returns a Data object with timezone-naive index.
        2. Timezone information is correctly removed from the returns.
    """
    # Add timezone information to the returns
    returns_with_tz = returns.copy()
    returns_with_tz.index = returns_with_tz.index.tz_localize("UTC")

    result = build_data(returns=returns_with_tz)

    # Verify the index is timezone-naive
    assert result.returns.index.tz is None

    # Verify the values match the input returns
    pd.testing.assert_frame_equal(result.returns, returns)
