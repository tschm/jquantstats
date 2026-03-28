"""Tests for comparing jquantstats with quantstats library functionality."""

import pytest


@pytest.fixture
def stats(data):
    """Fixture that returns the stats property of the data fixture.

    Args:
        data: The data fixture containing a Data object.

    Returns:
        Stats: The stats property of the data fixture.

    """
    return data.stats


@pytest.fixture
def pandas_frame(data):
    """Fixture that returns the data as a pandas DataFrame with Date as index.

    Args:
        data: The data fixture containing a Data object.

    Returns:
        pd.DataFrame: A pandas DataFrame with Date as index and all data columns.

    """
    return data.all.to_pandas().set_index("Date")


@pytest.fixture
def aapl(pandas_frame):
    """Fixture that returns the AAPL returns from the data fixture.

    Args:
        pandas_frame: The data fixture containing a Data object.

    Returns:
        pd.Series: The AAPL returns as a pandas Series.

    """
    return pandas_frame["AAPL"]


@pytest.fixture
def benchmark_pd(pandas_frame):
    """Fixture that returns the benchmark returns as a pandas Series.

    Args:
        pandas_frame: The pandas_frame fixture containing all data as a pandas DataFrame.

    Returns:
        pd.Series: A pandas Series containing the SPY benchmark returns with Date as index.

    """
    return pandas_frame["SPY -- Benchmark"]
