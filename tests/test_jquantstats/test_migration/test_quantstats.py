"""Tests for comparing jquantstats with quantstats library functionality."""

import pytest
import quantstats as qs


def test_omega(stats, aapl):
    """Compares omega ratio against quantstats for default, required_return, and rf variants."""
    x = stats.omega(periods=252)
    y = qs.stats.omega(aapl)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

    x_threshold = stats.omega(required_return=0.01, periods=252)
    y_threshold = qs.stats.omega(aapl, required_return=0.01)
    assert x_threshold["AAPL"] == pytest.approx(y_threshold, abs=1e-6)

    x_rf = stats.omega(rf=0.02, periods=252)
    y_rf = qs.stats.omega(aapl, rf=0.02)
    assert x_rf["AAPL"] == pytest.approx(y_rf, abs=1e-6)


def test_autocorrelation(stats, aapl):
    """Lag-1 autocorrelation matches pandas Series.autocorr(lag=1)."""
    x = stats.autocorr(lag=1)
    y = aapl.autocorr(lag=1)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)


def test_autocorrelation_lag5(stats, aapl):
    """Lag-5 autocorrelation matches pandas Series.autocorr(lag=5)."""
    x = stats.autocorr(lag=5)
    y = aapl.autocorr(lag=5)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)


def test_acf_matches_pandas(stats, aapl):
    """ACF values at each lag match pandas Series.autocorr()."""
    nlags = 10
    result = stats.acf(nlags=nlags)
    aapl_col = result["AAPL"].to_list()
    assert aapl_col[0] == pytest.approx(1.0)
    for k in range(1, nlags + 1):
        assert aapl_col[k] == pytest.approx(aapl.autocorr(lag=k), abs=1e-6)
