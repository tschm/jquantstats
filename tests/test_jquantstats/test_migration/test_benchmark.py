"""Migration tests for benchmark-dependent stats against quantstats."""

import pytest
import quantstats as qs


@pytest.fixture
def aligned(pandas_frame):
    """AAPL and SPY aligned on common dates, dropping the leading SPY NaN."""
    return pandas_frame[["AAPL", "SPY -- Benchmark"]].dropna()


def test_information_ratio(stats, aapl, benchmark_pd):
    """Quantstats does not annualise its information ratio, so we scale by sqrt(252)."""
    x = stats.information_ratio(periods_per_year=252)
    y = qs.stats.information_ratio(aapl, benchmark=benchmark_pd)
    assert x["AAPL"] == pytest.approx(y, abs=1e-4)


def test_information_ratio_no_annualise(stats, aapl, benchmark_pd):
    """annualise=False matches the raw QuantStats information ratio directly."""
    x = stats.information_ratio(periods_per_year=252, annualise=False)
    y = qs.stats.information_ratio(aapl, benchmark=benchmark_pd)
    assert x["AAPL"] == pytest.approx(y, abs=1e-4)


def test_r_squared(stats, aligned):
    """Series are inner-joined on Date to handle the leading NaN in SPY."""
    x = stats.r_squared()
    y = qs.stats.r_squared(aligned["AAPL"], benchmark=aligned["SPY -- Benchmark"])
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)


def test_greeks_beta(stats, aligned):
    """Series are inner-joined on Date to handle the leading NaN in SPY."""
    x = stats.greeks()
    y = qs.stats.greeks(aligned["AAPL"], benchmark=aligned["SPY -- Benchmark"])
    assert x["AAPL"]["beta"] == pytest.approx(y["beta"], abs=1e-6)


def test_greeks_alpha(stats, aligned):
    """periods_per_year=252 set explicitly on both sides to match annualisation factor."""
    x = stats.greeks(periods_per_year=252)
    y = qs.stats.greeks(aligned["AAPL"], benchmark=aligned["SPY -- Benchmark"], periods=252)
    assert x["AAPL"]["alpha"] == pytest.approx(y["alpha"], abs=1e-6)
