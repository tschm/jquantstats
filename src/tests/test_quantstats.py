import numpy as np
import pytest
import quantstats as qs

qs.extend_pandas()

@pytest.fixture
def stats(data):
    """
    Fixture that returns the stats property of the data fixture.

    Args:
        data: The data fixture containing a Data object.

    Returns:
        Stats: The stats property of the data fixture.
    """
    return data.stats

@pytest.fixture
def benchmark_pd(data):
    return data.benchmark_pd

@pytest.fixture
def returns_pd(data):
    return data.returns_pd

def test_sharpe_ratio(stats, returns_pd):
    x = stats.sharpe()
    y = qs.stats.sharpe(returns_pd)
    print(x)
    print(y)
    #assert x["AAPL"] == pytest.approx(y["AAPL"], abs=1e-6)
    #assert x["META"] == pytest.approx(y["META"], abs=1e-6)

    y = qs.stats.sharpe(returns_pd["META"])
    print(y)

    y = qs.stats.sharpe(returns_pd["META"].dropna())
    print(y)

    divisor = returns_pd.std(ddof=1)
    print(divisor)

    res = returns_pd.mean() / divisor
    print(res)

    print((res/divisor) * np.sqrt(252))
