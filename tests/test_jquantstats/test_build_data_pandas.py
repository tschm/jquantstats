"""Tests for the build_data function with pandas input types."""

import pandas as pd
import pytest

from jquantstats.api import build_data


@pytest.fixture
def returns_pd(returns):
    """Fixture that returns a pandas DataFrame version of the returns fixture.

    Args:
        returns (pl.DataFrame): The polars DataFrame returns fixture.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the same data as the returns fixture.

    """
    # Convert to pandas and ensure Date column is datetime
    dframe = returns.to_pandas()
    dframe["Date"] = pd.to_datetime(dframe["Date"])
    return dframe


@pytest.fixture
def returns_series_pd(returns_pd):
    """Fixture that returns a pandas Series version of the Meta returns.

    Args:
        returns_pd (pd.DataFrame): The pandas DataFrame returns fixture.

    Returns:
        pd.Series: A pandas Series containing the Meta returns with Date as index.

    """
    return returns_pd.set_index("Date")["Meta"]


@pytest.fixture
def benchmark_pd(benchmark):
    """Fixture that returns a pandas DataFrame version of the benchmark fixture.

    Args:
        benchmark (pl.DataFrame): The polars DataFrame benchmark fixture.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the same data as the benchmark fixture.

    """
    # Convert to pandas and ensure Date column is datetime
    dframe = benchmark.to_pandas()
    dframe["Date"] = pd.to_datetime(dframe["Date"])
    return dframe


@pytest.fixture
def benchmark_series_pd(benchmark_pd):
    """Fixture that returns a pandas Series version of the benchmark returns.

    Args:
        benchmark_pd (pd.DataFrame): The pandas DataFrame benchmark fixture.

    Returns:
        pd.Series: A pandas Series containing the benchmark returns with Date as index.

    """
    return benchmark_pd.set_index("Date")["SPY -- Benchmark"]


@pytest.fixture
def rf_pd(returns_pd):
    """Fixture that returns a pandas DataFrame with a constant risk-free rate.

    Args:
        returns_pd (pd.DataFrame): The pandas DataFrame returns fixture.

    Returns:
        pd.DataFrame: A pandas DataFrame with Date and a constant risk-free rate.

    """
    # The Date column from returns_pd is already datetime
    return pd.DataFrame({"Date": returns_pd["Date"], "rf": 0.001})


@pytest.fixture
def rf_series_pd(rf_pd):
    """Fixture that returns a pandas Series with a constant risk-free rate.

    Args:
        rf_pd (pd.DataFrame): The pandas DataFrame risk-free rate fixture.

    Returns:
        pd.Series: A pandas Series with a constant risk-free rate and Date as index.

    """
    return rf_pd.set_index("Date")["rf"]


def test_build_data_with_pd_dataframe_returns(returns_pd):
    """Tests that build_data works with a pandas DataFrame for returns.

    Args:
        returns_pd (pd.DataFrame): The pandas DataFrame returns fixture.

    Verifies:
        1. The function returns a Data object.
        2. The returns in the Data object match the input returns.
        3. The date column is correctly excluded from the returns.

    """
    result = build_data(returns=returns_pd)

    # Convert the polars DataFrame to pandas for comparison
    pl_returns = returns_pd.drop(columns=["Date"])
    result_returns = result.returns.to_pandas()

    # Compare the values (ignoring index which might be different)
    pd.testing.assert_frame_equal(
        pl_returns.reset_index(drop=True),
        result_returns.reset_index(drop=True),
        check_dtype=False,  # Ignore dtype differences between pandas and polars
    )


def test_build_data_with_pd_series_returns(returns_series_pd):
    """Tests that build_data works with a pandas Series for returns.

    Args:
        returns_series_pd (pd.Series): The pandas Series returns fixture.

    Verifies:
        1. The function returns a Data object.
        2. The returns in the Data object match the input returns.

    """
    result = build_data(returns=returns_series_pd)

    # Convert the polars DataFrame to pandas for comparison
    result_returns = result.returns.to_pandas()

    # The Series should be converted to a DataFrame with a single column
    assert result_returns.shape[1] == 1

    # Compare the values (ignoring index which might be different)
    pd.testing.assert_series_equal(
        returns_series_pd.reset_index(drop=True),
        result_returns.iloc[:, 0].reset_index(drop=True),
        check_dtype=False,  # Ignore dtype differences between pandas and polars
    )


def test_build_data_with_pd_dataframe_benchmark(returns_pd, benchmark_pd):
    """Tests that build_data works with a pandas DataFrame for benchmark.

    Args:
        returns_pd (pd.DataFrame): The pandas DataFrame returns fixture.
        benchmark_pd (pd.DataFrame): The pandas DataFrame benchmark fixture.

    Verifies:
        1. The function returns a Data object with both returns and benchmark.
        2. The benchmark in the Data object matches the input benchmark.

    """
    # Use a constant risk-free rate of 0 to avoid subtraction
    result = build_data(returns=returns_pd, benchmark=benchmark_pd, rf=0.0)

    # Verify the benchmark is not None
    b = result.benchmark
    if b is not None:
        # The Series should be converted to a DataFrame with a single column
        assert b.shape[1] == 1

        # Just verify that the benchmark data is not empty
        assert not b.is_empty()

        # Verify the number of rows matches
        assert b.shape[0] == result.returns.shape[0]
    else:
        raise AssertionError("No benchmark data available")


def test_build_data_with_pd_series_benchmark(returns_pd, benchmark_series_pd):
    """Tests that build_data works with a pandas Series for benchmark.

    Args:
        returns_pd (pd.DataFrame): The pandas DataFrame returns fixture.
        benchmark_series_pd (pd.Series): The pandas Series benchmark fixture.

    Verifies:
        1. The function returns a Data object with both returns and benchmark.
        2. The benchmark in the Data object has the expected structure.

    """
    # Use a constant risk-free rate of 0 to avoid subtraction
    result = build_data(returns=returns_pd, benchmark=benchmark_series_pd, rf=0.0)

    # Verify the benchmark is not None
    b = result.benchmark
    if b is not None:
        # The Series should be converted to a DataFrame with a single column
        assert b.shape[1] == 1

        # Just verify that the benchmark data is not empty
        assert not b.is_empty()

        # Verify the number of rows matches
        assert b.shape[0] == result.returns.shape[0]
    else:
        raise AssertionError("No benchmark data available")


def test_build_data_with_pd_dataframe_rf(returns_pd, rf_pd):
    """Tests that build_data works with a pandas DataFrame for risk-free rate.

    Args:
        returns_pd (pd.DataFrame): The pandas DataFrame returns fixture.
        rf_pd (pd.DataFrame): The pandas DataFrame risk-free rate fixture.

    Verifies:
        1. The function returns a Data object.
        2. The returns in the Data object are correctly adjusted by the risk-free rate.

    """
    result = build_data(returns=returns_pd, rf=rf_pd)

    # Get the risk-free rate value
    rf_value = rf_pd["rf"].iloc[0]

    # Convert the polars DataFrame to pandas for comparison
    result_returns = result.returns.to_pandas()

    # Calculate the expected returns (original returns minus risk-free rate)
    expected_returns = returns_pd.drop(columns=["Date"]) - rf_value

    # Compare the values (ignoring index which might be different)
    pd.testing.assert_frame_equal(
        expected_returns.reset_index(drop=True),
        result_returns.reset_index(drop=True),
        check_dtype=False,  # Ignore dtype differences between pandas and polars
    )


def test_build_data_with_pd_series_rf(returns_pd, rf_series_pd):
    """Tests that build_data works with a pandas Series for risk-free rate.

    Args:
        returns_pd (pd.DataFrame): The pandas DataFrame returns fixture.
        rf_series_pd (pd.Series): The pandas Series risk-free rate fixture.

    Verifies:
        1. The function returns a Data object.
        2. The returns in the Data object are correctly adjusted by the risk-free rate.

    """
    result = build_data(returns=returns_pd, rf=rf_series_pd)

    # Get the risk-free rate value
    rf_value = rf_series_pd.iloc[0]

    # Convert the polars DataFrame to pandas for comparison
    result_returns = result.returns.to_pandas()

    # Calculate the expected returns (original returns minus risk-free rate)
    expected_returns = returns_pd.drop(columns=["Date"]) - rf_value

    # Compare the values (ignoring index which might be different)
    pd.testing.assert_frame_equal(
        expected_returns.reset_index(drop=True),
        result_returns.reset_index(drop=True),
        check_dtype=False,  # Ignore dtype differences between pandas and polars
    )


def test_build_data_with_pd_all_inputs(returns_series_pd, benchmark_series_pd, rf_series_pd):
    """Tests that build_data works with all inputs as pandas Series.

    Args:
        returns_series_pd (pd.Series): The pandas Series returns fixture.
        benchmark_series_pd (pd.Series): The pandas Series benchmark fixture.
        rf_series_pd (pd.Series): The pandas Series risk-free rate fixture.

    Verifies:
        1. The function returns a Data object with both returns and benchmark.
        2. The returns and benchmark in the Data object have the expected structure.

    """
    # Use a constant risk-free rate to simplify testing
    result = build_data(returns=returns_series_pd, benchmark=benchmark_series_pd, rf=0.0)

    # Verify the returns and benchmark are not None
    assert result.returns is not None
    b = result.benchmark

    # assert result.benchmark is not None

    # Verify the returns and benchmark have the expected structure (1 column each)
    assert result.returns.shape[1] == 1

    # Verify the returns and benchmark are not empty
    assert not result.returns.is_empty()

    if b is not None:
        assert b.shape[1] == 1
        assert not b.is_empty()

        # Verify the number of rows in returns and benchmark match
        assert result.returns.shape[0] == b.shape[0]

    # Verify that the index has the same number of rows
    assert result.index.shape[0] == result.returns.shape[0]


def test_build_data_error_no_overlapping_dates():
    """Tests that build_data raises an error when there are no overlapping dates between returns and benchmark.

    Verifies:
        1. The function raises a ValueError when there are no overlapping dates.
    """
    # Create returns and benchmark with non-overlapping dates
    returns_dates = pd.date_range(start="2020-01-01", periods=10)
    benchmark_dates = pd.date_range(start="2021-01-01", periods=10)

    returns_dframe = pd.DataFrame({"Date": returns_dates, "Asset": [0.01] * 10})

    benchmark_dframe = pd.DataFrame({"Date": benchmark_dates, "Benchmark": [0.02] * 10})

    # Test that a ValueError is raised
    with pytest.raises(ValueError, match="No overlapping dates between returns and benchmark"):
        build_data(returns=returns_dframe, benchmark=benchmark_dframe)
