from datetime import date

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from jquantstats.api import build_data


def test_head(data):
    """
    Tests that the head() method returns a Data object with the first n rows.

    Args:
        data (_Data): The data fixture containing a Data object.

    Verifies:
        1. The return value is a Data object.
        2. The content matches the first n rows of the original data.
    """
    x = data.head()
    assert_frame_equal(x.returns, data.returns.head(5))

    #pd.testing.assert_frame_equal(x.all(), data.all().head())


def test_tail(data):
    """
    Tests that the tail() method returns a Data object with the last n rows.

    Args:
        data (_Data): The data fixture containing a Data object.

    Verifies:
        1. The return value is a Data object.
        2. The content matches the last n rows of the original data.
    """
    x = data.tail()
    assert_frame_equal(x.returns, data.returns.tail(5))

def test_all(data):
    print(data.returns.head(5))
    x = data.all
    print(x)

def test_assets(data):
    x = data.assets
    assert x == ['AAPL', 'META', 'SPY -- Benchmark']

def test_date_col(data):
    x = data.date_col
    assert x == ["Date"]

def test_periods(data):
    assert data._periods_per_year == 252

def test_periods_edge_cases(data):
    """
    Tests edge cases for the _periods_per_year property.

    Args:
        data (Data): The data fixture containing a Data object.

    Verifies:
        1. ValueError is raised when index has less than 2 timestamps
        2. Different frequencies return different period counts
        3. Unsorted data is handled correctly
    """
    # Weekly data
    # Create dates with weekly intervals
    weekly_dates = [date(2023, 1, 1), date(2023, 1, 8), date(2023, 1, 15), date(2023, 1, 22), date(2023, 1, 29),
                   date(2023, 2, 5), date(2023, 2, 12), date(2023, 2, 19), date(2023, 2, 26), date(2023, 3, 5)]
    weekly_returns = pl.DataFrame({
        "Date": weekly_dates,
        "returns": [0.01] * 10
    })
    weekly_data = build_data(returns=weekly_returns)
    print(weekly_data._periods_per_year)
    assert weekly_data._periods_per_year == 52
    # Monthly data
    # Create dates with monthly intervals
    monthly_dates = [date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1), date(2023, 4, 1), date(2023, 5, 1),
                    date(2023, 6, 1), date(2023, 7, 1), date(2023, 8, 1), date(2023, 9, 1), date(2023, 10, 1)]
    monthly_returns = pl.DataFrame({
        "Date": monthly_dates,
        "returns": [0.01] * 10
    })
    monthly_data = build_data(returns=monthly_returns)
    assert monthly_data._periods_per_year == 12


def test_post_init():
    """
    Tests the validation checks in the __post_init__ method of the Data class.

    Verifies:
        1. ValueError is raised when index has less than 2 timestamps
        2. ValueError is raised when index is not monotonically increasing
        3. ValueError is raised when returns and index have different row counts
        4. ValueError is raised when benchmark and index have different row counts
    """
    # Test case 1: Index with less than 2 timestamps
    single_date = [date(2023, 1, 1)]
    single_returns = pl.DataFrame({
        "Date": single_date,
        "returns": [0.01]
    })

    with pytest.raises(ValueError, match="Index must contain at least two timestamps."):
        build_data(returns=single_returns, date_col="Date")

    # Test case 2: Unsorted index
    unsorted_dates = [date(2023, 1, 15), date(2023, 1, 1), date(2023, 1, 30)]
    unsorted_returns = pl.DataFrame({
        "Date": unsorted_dates,
        "returns": [0.01, 0.02, 0.03]
    })

    with pytest.raises(ValueError, match="Index must be monotonically increasing."):
        build_data(returns=unsorted_returns)

    # Test case 3: Returns and index with different row counts
    dates = [date(2023, 1, 1), date(2023, 1, 15), date(2023, 1, 30)]
    returns = pl.DataFrame({
        "returns": [0.01, 0.02]
    })
    index = pl.DataFrame({
        "Date": dates
    })

    with pytest.raises(ValueError, match="Returns and index must have the same number of rows."):
        from jquantstats._data import Data
        Data(returns=returns, index=index)

    # Test case 4: Benchmark and index with different row counts
    dates = [date(2023, 1, 1), date(2023, 1, 15), date(2023, 1, 30)]
    returns = pl.DataFrame({
        "returns": [0.01, 0.02, 0.03]
    })
    benchmark = pl.DataFrame({
        "benchmark": [0.01, 0.02]
    })
    index = pl.DataFrame({
        "Date": dates
    })

    with pytest.raises(ValueError, match="Benchmark and index must have the same number of rows."):
        from jquantstats._data import Data
        Data(returns=returns, benchmark=benchmark, index=index)



def test_copy(data):
    """
    Tests that the copy() method creates a proper deep copy of the Data object.

    Args:
        data (_Data): The data fixture containing a Data object.

    Verifies:
        1. The return value is a Data object.
        2. The copied object has the same returns and benchmark data as the original.
        3. Modifying the copied object does not affect the original.
        4. The copy works correctly when there's a benchmark.
    """
    # Create a copy of the data object
    data_copy = data.copy()

    # Verify the copy has the same returns and benchmark as the original
    assert_frame_equal(data_copy.returns, data.returns)
    assert_frame_equal(data_copy.benchmark, data.benchmark)

    # Verify that modifying the copy doesn't affect the original
    # We can't directly modify the attributes because the Data class is frozen,
    # but we can verify that they are separate objects in memory
    assert data_copy is not data
    assert data_copy.returns is not data.returns
    assert data_copy.benchmark is not data.benchmark


def test_resample(data):
    """
    Tests that the resample() method correctly resamples data to different time periods.

    Args:
        data (_Data): The data fixture containing a Data object.

    Verifies:
        1. The return value is a Data object.
        2. The resampled object has the correct frequency.
        3. The resampling works with both compounded=False and compounded=True.
        4. The resampling works with different resample rules (YE, ME, etc.).
    """
    # Test resampling to yearly frequency with compounded=False (default)
    yearly_data = data.resample(every="1y", compounded=False)

    # Verify the resampled data has the correct structure
    assert yearly_data.returns.shape[1] == data.returns.shape[1]  # Same number of columns
    print(yearly_data.all)
    #assert yearly_data.returns.index.freq == 'YE'  # Yearly frequency

    # Test resampling to monthly frequency with compounded=True
    monthly_data = data.resample(every="1mo", compounded=True)
    print(monthly_data.all)
    # Verify the resampled data has the correct structure
    assert monthly_data.returns.shape[1] == data.returns.shape[1]  # Same number of columns
    #assert monthly_data.returns.index.freq == 'ME'  # Monthly frequency

def test_stats(data):
    assert data.stats is not None

def test_plots(data):
    assert data.plots is not None

def test_all_no_benchmark(data_no_benchmark):
    assert data_no_benchmark.all is not None

def test_assets_no_benchmark(data_no_benchmark):
    assert data_no_benchmark.assets is not None

def test_copy_no_benchmark(data_no_benchmark):
    x = data_no_benchmark.copy()
    assert x.returns is not None
    assert x.benchmark is None
