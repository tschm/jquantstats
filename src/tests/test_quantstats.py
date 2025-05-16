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
    """
    Fixture that returns the benchmark_pd property of the data fixture.

    Args:
        data: The data fixture containing a Data object.

    Returns:
        pd.DataFrame: The benchmark data as a pandas DataFrame.
    """
    return data.benchmark_pd

@pytest.fixture
def aapl(data):
    """
    Fixture that returns the AAPL returns from the data fixture.

    Args:
        data: The data fixture containing a Data object.

    Returns:
        pd.Series: The AAPL returns as a pandas Series.
    """
    return data.returns_pd["AAPL"].dropna()


def test_sharpe_ratio(stats, aapl):
    """
    Tests that the sharpe method produces the same results as quantstats.

    Args:
        stats: The stats fixture containing a Stats object.
        aapl: The aapl fixture containing AAPL returns.

    Verifies:
        The sharpe ratio calculated by our library matches the one from quantstats.
    """
    x = stats.sharpe(periods=252)
    y = qs.stats.sharpe(aapl, periods=252)

    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_skew(stats, aapl):
    """
    Tests that the skew method produces the same results as quantstats.

    Args:
        stats: The stats fixture containing a Stats object.
        aapl: The aapl fixture containing AAPL returns.

    Verifies:
        The skewness calculated by our library matches the one from quantstats.
    """
    x = stats.skew()
    y = qs.stats.skew(aapl)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_kurtosis(stats, aapl):
    """
    Tests that the kurtosis method produces the same results as quantstats.

    Args:
        stats: The stats fixture containing a Stats object.
        aapl: The aapl fixture containing AAPL returns.

    Verifies:
        The kurtosis calculated by our library matches the one from quantstats.
    """
    x = stats.kurtosis()
    y = qs.stats.kurtosis(aapl)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_avg_return(stats, aapl):
    """
    Tests that the avg_return method produces the same results as quantstats.

    Args:
        stats: The stats fixture containing a Stats object.
        aapl: The aapl fixture containing AAPL returns.

    Verifies:
        The average return calculated by our library matches the one from quantstats.
    """
    x = stats.avg_return()
    y = qs.stats.avg_return(aapl)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_avg_win(stats, aapl):
    """
    Tests that the avg_win method produces the same results as quantstats.

    Args:
        stats: The stats fixture containing a Stats object.
        aapl: The aapl fixture containing AAPL returns.

    Verifies:
        The average win calculated by our library matches the one from quantstats.
    """
    x = stats.avg_win()
    y = qs.stats.avg_win(aapl)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_avg_loss(stats, aapl):
    """
    Tests that the avg_loss method produces the same results as quantstats.

    Args:
        stats: The stats fixture containing a Stats object.
        aapl: The aapl fixture containing AAPL returns.

    Verifies:
        The average loss calculated by our library matches the one from quantstats.
    """
    x = stats.avg_loss()
    y = qs.stats.avg_loss(aapl)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_volatility(stats, aapl):
    """
    Tests that the volatility method produces the same results as quantstats.

    Args:
        stats: The stats fixture containing a Stats object.
        aapl: The aapl fixture containing AAPL returns.

    Verifies:
        The volatility calculated by our library matches the one from quantstats.
    """
    x = stats.volatility(periods=252)
    y = qs.stats.volatility(aapl, periods=252)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_payoff_ratio(stats, aapl):
    """
    Tests that the payoff_ratio method produces the same results as quantstats.

    Args:
        stats: The stats fixture containing a Stats object.
        aapl: The aapl fixture containing AAPL returns.

    Verifies:
        The payoff ratio calculated by our library matches the one from quantstats.
    """
    x = stats.payoff_ratio()
    y = qs.stats.payoff_ratio(aapl)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_win_loss_ratio(stats, aapl):
    """
    Tests that the win_loss_ratio method produces the same results as quantstats.

    Args:
        stats: The stats fixture containing a Stats object.
        aapl: The aapl fixture containing AAPL returns.

    Verifies:
        The win/loss ratio calculated by our library matches the one from quantstats.
    """
    x = stats.win_loss_ratio()
    y = qs.stats.win_loss_ratio(aapl)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_profit_ratio(stats, aapl):
    """
    Tests that the profit_ratio method produces the same results as quantstats.

    Args:
        stats: The stats fixture containing a Stats object.
        aapl: The aapl fixture containing AAPL returns.

    Verifies:
        The profit ratio calculated by our library matches the one from quantstats.
    """
    x = stats.profit_ratio()
    y = qs.stats.profit_ratio(aapl)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_profit_factor(stats, aapl):
    """
    Tests that the profit_factor method produces the same results as quantstats.

    Args:
        stats: The stats fixture containing a Stats object.
        aapl: The aapl fixture containing AAPL returns.

    Verifies:
        The profit factor calculated by our library matches the one from quantstats.
    """
    x = stats.profit_factor()
    y = qs.stats.profit_factor(aapl)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_value_at_risk(stats, aapl):
    """
    Tests that the value_at_risk method produces the same results as quantstats.

    Args:
        stats: The stats fixture containing a Stats object.
        aapl: The aapl fixture containing AAPL returns.

    Verifies:
        The value at risk calculated by our library matches the one from quantstats.
    """
    x = stats.value_at_risk(alpha=0.05)
    y = qs.stats.value_at_risk(aapl)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_conditional_value_at_risk(stats, aapl):
    """
    Tests that the conditional_value_at_risk method produces the same results as quantstats.

    Args:
        stats: The stats fixture containing a Stats object.
        aapl: The aapl fixture containing AAPL returns.

    Verifies:
        The conditional value at risk calculated by our library matches the one from quantstats.
    """
    x = stats.conditional_value_at_risk( alpha=0.05)
    y = qs.stats.conditional_value_at_risk(aapl, confidence=0.95)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_win_rate(stats, aapl):
    """
    Tests that the win_rate method produces the same results as quantstats.

    Args:
        stats: The stats fixture containing a Stats object.
        aapl: The aapl fixture containing AAPL returns.

    Verifies:
        The win rate calculated by our library matches the one from quantstats.
    """
    x = stats.win_rate()
    y = qs.stats.win_rate(aapl)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_kelly_criterion(stats, aapl):
    """
    Tests that the kelly_criterion method produces the same results as quantstats.

    Args:
        stats: The stats fixture containing a Stats object.
        aapl: The aapl fixture containing AAPL returns.

    Verifies:
        The Kelly criterion calculated by our library matches the one from quantstats.
    """
    x = stats.kelly_criterion()
    y = qs.stats.kelly_criterion(aapl)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_gain_to_pain_ratio(stats, aapl):
    """
    Tests that the gain_to_pain_ratio method produces the same results as quantstats.

    Args:
        stats: The stats fixture containing a Stats object.
        aapl: The aapl fixture containing AAPL returns.

    Verifies:
        The gain-to-pain ratio calculated by our library matches the one from quantstats.
    """
    x = stats.gain_to_pain_ratio()
    y = qs.stats.gain_to_pain_ratio(aapl)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_risk_return_ratio(stats, aapl):
    """
    Tests that the risk_return_ratio method produces the same results as quantstats.

    Args:
        stats: The stats fixture containing a Stats object.
        aapl: The aapl fixture containing AAPL returns.

    Verifies:
        The risk-return ratio calculated by our library matches the one from quantstats.
    """
    x = stats.risk_return_ratio()
    y = qs.stats.risk_return_ratio(aapl)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_best(stats, aapl):
    """
    Tests that the best method produces the same results as quantstats.

    Args:
        stats: The stats fixture containing a Stats object.
        aapl: The aapl fixture containing AAPL returns.

    Verifies:
        The best return calculated by our library matches the one from quantstats.
    """
    x = stats.best()
    y = qs.stats.best(aapl)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_worst(stats, aapl):
    """
    Tests that the worst method produces the same results as quantstats.

    Args:
        stats: The stats fixture containing a Stats object.
        aapl: The aapl fixture containing AAPL returns.

    Verifies:
        The worst return calculated by our library matches the one from quantstats.
    """
    x = stats.worst()
    y = qs.stats.worst(aapl)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_exposure(stats, aapl):
    """
    Tests that the exposure method produces the same results as quantstats.

    Args:
        stats: The stats fixture containing a Stats object.
        aapl: The aapl fixture containing AAPL returns.

    Verifies:
        The exposure calculated by our library matches the one from quantstats.
    """
    x = stats.exposure()
    y = qs.stats.exposure(aapl)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_sortino(stats, aapl):
    """
    Tests that the sortino method produces the same results as quantstats.

    Args:
        stats: The stats fixture containing a Stats object.
        aapl: The aapl fixture containing AAPL returns.

    Verifies:
        The Sortino ratio calculated by our library matches the one from quantstats.
    """
    x = stats.sortino(periods=252)
    y = qs.stats.sortino(aapl)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)

def test_information_ratio(stats, aapl, benchmark_pd):
    """
    Tests that the information_ratio method produces the same results as quantstats.

    Args:
        stats: The stats fixture containing a Stats object.
        aapl: The aapl fixture containing AAPL returns.
        benchmark_pd: The benchmark_pd fixture containing benchmark returns.

    Verifies:
        The information ratio calculated by our library matches the one from quantstats.
    """
    x = stats.information_ratio()
    y = np.sqrt(252)*qs.stats.information_ratio(aapl, benchmark=benchmark_pd)
    assert x["AAPL"] ==  pytest.approx(y, abs=1e-6)
