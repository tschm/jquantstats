import numpy as np
import pytest
import quantstats as qs


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
def aapl(data):
    return data.returns_pd["AAPL"].dropna()


def test_sharpe_ratio(stats, aapl):
    x = stats.sharpe()
    y = qs.stats.sharpe(aapl)

    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_skew(stats, aapl):
    x = stats.skew()
    y = qs.stats.skew(aapl)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_kurtosis(stats, aapl):
    x = stats.kurtosis()
    y = qs.stats.kurtosis(aapl)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_avg_return(stats, aapl):
    x = stats.avg_return()
    y = qs.stats.avg_return(aapl)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_avg_win(stats, aapl):
    x = stats.avg_win()
    y = qs.stats.avg_win(aapl)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_avg_loss(stats, aapl):
    x = stats.avg_loss()
    y = qs.stats.avg_loss(aapl)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_volatility(stats, aapl):
    x = stats.volatility()
    y = qs.stats.volatility(aapl)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_payoff_ratio(stats, aapl):
    x = stats.payoff_ratio()
    y = qs.stats.payoff_ratio(aapl)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_win_loss_ratio(stats, aapl):
    x = stats.win_loss_ratio()
    y = qs.stats.win_loss_ratio(aapl)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_profit_ratio(stats, aapl):
    x = stats.profit_ratio()
    y = qs.stats.profit_ratio(aapl)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_profit_factor(stats, aapl):
    x = stats.profit_factor()
    y = qs.stats.profit_factor(aapl)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_value_at_risk(stats, aapl):
    x = stats.value_at_risk(alpha=0.05)
    y = qs.stats.value_at_risk(aapl)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_conditional_value_at_risk(stats, aapl):
    x = stats.conditional_value_at_risk( alpha=0.05)
    y = qs.stats.conditional_value_at_risk(aapl, confidence=0.95)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_win_rate(stats, aapl):
    x = stats.win_rate()
    y = qs.stats.win_rate(aapl)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_kelly_criterion(stats, aapl):
    x = stats.kelly_criterion()
    y = qs.stats.kelly_criterion(aapl)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_gain_to_pain_ratio(stats, aapl):
    x = stats.gain_to_pain_ratio()
    y = qs.stats.gain_to_pain_ratio(aapl)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_risk_return_ratio(stats, aapl):
    x = stats.risk_return_ratio()
    y = qs.stats.risk_return_ratio(aapl)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_best(stats, aapl):
    x = stats.best()
    y = qs.stats.best(aapl)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_worst(stats, aapl):
    x = stats.worst()
    y = qs.stats.worst(aapl)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_exposure(stats, aapl):
    x = stats.exposure()
    y = qs.stats.exposure(aapl)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_sortino(stats, aapl):
    x = stats.sortino(periods=252)
    y = qs.stats.sortino(aapl)
    assert x["AAPL"] ==  pytest.approx(y, abs=1e-6)

def test_information_ratio(stats, aapl, benchmark_pd):
    x = stats.information_ratio()
    y = np.sqrt(252)*qs.stats.information_ratio(aapl, benchmark=benchmark_pd)
    assert x["AAPL"] ==  pytest.approx(y, abs=1e-6)
