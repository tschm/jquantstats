import numpy as np
import pandas as pd
import pytest


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

def test_skew(stats):
    result = stats.skew()
    assert result["META"] == pytest.approx(0.4220690178941095)

def test_kurtosis(stats):
    result = stats.kurtosis()
    assert result["META"] == pytest.approx(20.477943361921824)

def test_avg_return(stats):
    result = stats.avg_return()
    assert result["META"] == pytest.approx(0.001142572799759818)

def test_avg_win(stats):
    result = stats.avg_win()
    assert result["META"] == pytest.approx(0.016643066469975223)

def test_avg_loss(stats):
    result = stats.avg_loss()
    assert result["META"] == pytest.approx(-0.01614062422517277)

def test_volatility(stats):
    result = stats.volatility(periods=252, annualize=True)
    assert result["META"] == pytest.approx(0.40072031010504233)

def test_rolling_volatility(stats):
    result = stats.rolling_volatility(rolling_period=20, periods_per_year=252)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == stats.all.shape
    # First 19 values should be NaN (rolling window of 20)
    assert result.iloc[:19].isna().all().all()


def test_tail_ratio(stats):
    result = stats.tail_ratio(cutoff=0.95)
    assert result["META"] == pytest.approx(0.9789355685152505)

def test_payoff_ratio(stats):
    result = stats.payoff_ratio()
    assert result["META"] == pytest.approx(1.0383990517002815)

def test_win_loss_ratio(stats):
    result = stats.win_loss_ratio()
    assert result["META"] == pytest.approx(1.0383990517002815)

def test_profit_ratio(stats):
    result = stats.profit_ratio()
    assert result["META"] == pytest.approx(0.9383417987749899)

def test_profit_factor(stats):
    result = stats.profit_factor()
    assert result["META"] == pytest.approx(1.149125608578595)

def test_common_sense_ratio(stats):
    result = stats.common_sense_ratio()
    assert result["META"] == pytest.approx(1.12491993092932)

def test_value_at_risk(stats):
    result = stats.value_at_risk(alpha=0.05)
    assert result["META"] == pytest.approx(-0.036720613002961255)

def test_var(stats):
    result = stats.var(alpha=0.05)
    assert result["META"] == pytest.approx(-0.036720613002961255)

def test_conditional_value_at_risk(stats):
    result = stats.conditional_value_at_risk( alpha=0.05)
    assert result["META"] == pytest.approx(-0.05541208494628702)

def test_cvar(stats):
    result = stats.cvar(alpha=0.05)
    assert result["META"] == pytest.approx(-0.05541208494628702)

def test_expected_shortfall(stats):
    result = stats.expected_shortfall(alpha=0.05)
    assert result["META"] == pytest.approx(-0.05541208494628702)

def test_win_rate(stats):
    result = stats.win_rate()
    assert result["META"] == pytest.approx(0.525309)

def test_kelly_criterion(stats):
    result = stats.kelly_criterion()
    assert result["META"] == pytest.approx(0.06817093826936971)

def test_gain_to_pain_ratio(stats):
    result = stats.gain_to_pain_ratio()
    assert result["META"] == pytest.approx(0.14912560857859494)

def test_risk_return_ratio(stats):
    result = stats.risk_return_ratio()
    assert result["META"] == pytest.approx(0.045095921921619944)

def test_outlier_win_ratio(stats):
    result = stats.outlier_win_ratio(quantile=0.99)
    assert result["META"] == pytest.approx(3.870328455823392)

def test_outlier_loss_ratio(stats):
    result = stats.outlier_loss_ratio(quantile=0.01)
    assert result["META"] == pytest.approx(3.909307423414421)

def test_best(stats):
    result = stats.best()
    assert result["META"] == pytest.approx(0.2961146917048886)

def test_worst(stats):
    result = stats.worst()
    assert result["META"] == pytest.approx(-0.2639010078964036)

def test_exposure(stats):
    result = stats.exposure()
    assert result["META"] == pytest.approx(0.40)

def test_sharpe(stats):
    result = stats.sharpe(periods=252)
    assert result["META"] == pytest.approx(0.7158755672867543)

def test_rolling_sharpe(stats):
    result = stats.rolling_sharpe(rolling_period=20, periods_per_year=252)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == stats.all.shape
    # First 19 values should be NaN (rolling window of 20)
    assert result.iloc[:19].isna().all().all()

def test_sortino(stats):
    result = stats.sortino(periods=252)
    assert result["META"] == pytest.approx(0.7311766729290573)

def test_rolling_sortino(stats):
    result = stats.rolling_sortino(rolling_period=20, periods_per_year=252)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == stats.all.shape

    # First 19 values should be NaN (rolling window of 20)
    assert result.iloc[:19].isna().all().all()

def test_adjusted_sortino(stats):
    result = stats.adjusted_sortino(periods=252)
    assert result["META"] == pytest.approx(0.5170199836735547)

def test_edge_cases(edge):
    assert np.isnan(edge.stats.profit_ratio()["returns"])# == {"returns": np.nan, "Benchmark": np.nan}
    assert np.isnan(edge.stats.gain_to_pain_ratio()["returns"]) # == {"returns": np.nan, "Benchmark": np.nan}



def test_information_ratio(stats):
    result = stats.information_ratio()
    print(result)
    #assert False

    #assert result["AAPL"] == pytest.approx(0.025195605347769847)

def test_greeks(stats):
    result = stats.greeks(periods=252.0)
    assert isinstance(result, pd.DataFrame)
    assert "alpha" in result.index
    assert "beta" in result.index

    assert result["Benchmark"]["beta"] == 1.0
    assert result["Benchmark"]["alpha"] == 0.0

    assert result["AAPL"]["beta"] == pytest.approx(1.1090322781954098)
    assert result["AAPL"]["alpha"] == pytest.approx(0.1576003006124853)

def test_r_squared(stats):
    result = stats.r_squared()
    print(result)
    assert isinstance(result, pd.Series)
    #assert 0 <= result <= 1  # R-squared should be between 0 and 1

def test_r2(stats):
    result = stats.r2()
    expected = stats.r_squared()
    pd.testing.assert_series_equal(result, expected)
