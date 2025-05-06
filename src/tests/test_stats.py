import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm


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
    assert isinstance(result, pd.Series)
    expected = stats.all.skew()
    assert result.equals(expected)

def test_kurtosis(stats):
    result = stats.kurtosis()
    assert isinstance(result, pd.Series)
    expected = stats.all.kurtosis()
    assert result.equals(expected)

def test_avg_return(stats):
    result = stats.avg_return()
    assert isinstance(result, pd.Series)
    # Manual calculation for comparison
    non_zero_returns = stats.all[stats.all != 0].dropna()
    expected = non_zero_returns.mean()
    assert result.equals(expected)

def test_avg_win(stats):
    result = stats.avg_win()
    assert isinstance(result, pd.Series)
    # Manual calculation for comparison
    positive_returns = stats.all[stats.all > 0].dropna()
    expected = positive_returns.mean()
    assert result.equals(expected)

def test_avg_loss(stats):
    result = stats.avg_loss()
    assert isinstance(result, pd.Series)
    # Manual calculation for comparison
    negative_returns = stats.all[stats.all < 0].dropna()
    expected = negative_returns.mean()
    assert result.equals(expected)

def test_volatility(stats):
    result = stats.volatility(periods=252, annualize=True)
    assert isinstance(result, pd.Series)
    # Manual calculation for comparison
    expected = stats.all.std() * np.sqrt(252)
    assert result.equals(expected)

def test_rolling_volatility(stats):
    result = stats.rolling_volatility(rolling_period=20, periods_per_year=252)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == stats.all.shape
    # First 19 values should be NaN (rolling window of 20)
    assert result.iloc[:19].isna().all().all()

def test_implied_volatility(stats):
    result = stats.implied_volatility(periods=252)
    assert isinstance(result, pd.Series)
    # Should be similar to regular volatility but using log returns
    log_returns = np.log(1 + stats.all)
    expected = log_returns.std() * np.sqrt(252)
    pd.testing.assert_series_equal(result, expected)

def test_autocorr_penalty(stats):
    result = stats.autocorr_penalty()
    assert isinstance(result, pd.Series)
    assert (result >= 1).all()  # Penalty should be at least 1

def test_tail_ratio(stats):
    result = stats.tail_ratio(cutoff=0.95)
    assert isinstance(result, pd.Series)
    assert (result > 0).all()  # Tail ratio should be positive
    # Manual calculation for comparison
    expected = abs(stats.all.quantile(0.95) / stats.all.quantile(0.05))
    pd.testing.assert_series_equal(result, expected)

def test_payoff_ratio(stats):
    result = stats.payoff_ratio()
    assert isinstance(result, pd.Series)
    assert (result > 0).all()  # Payoff ratio should be positive
    # Manual calculation for comparison
    expected = stats.avg_win() / abs(stats.avg_loss())
    pd.testing.assert_series_equal(result, expected)

def test_win_loss_ratio(stats):
    result = stats.win_loss_ratio()
    assert isinstance(result, pd.Series)
    expected = stats.payoff_ratio()
    pd.testing.assert_series_equal(result, expected)

def test_profit_ratio(stats):
    result = stats.profit_ratio()
    assert isinstance(result, pd.Series)
    # Manual calculation for comparison
    wins = stats.all[stats.all >= 0]
    loss = stats.all[stats.all < 0]
    win_ratio = abs(wins.mean() / wins.count())
    loss_ratio = abs(loss.mean() / loss.count())
    expected = win_ratio / loss_ratio
    pd.testing.assert_series_equal(result, expected)

def test_profit_factor(stats):
    result = stats.profit_factor()
    assert isinstance(result, pd.Series)
    # Manual calculation for comparison
    expected = abs(stats.all[stats.all >= 0].sum() / stats.all[stats.all < 0].sum())
    pd.testing.assert_series_equal(result, expected)

def test_cpc_index(stats):
    result = stats.cpc_index()
    assert isinstance(result, pd.Series)
    # Manual calculation for comparison
    expected = stats.profit_factor() * stats.win_rate() * stats.win_loss_ratio()
    pd.testing.assert_series_equal(result, expected)

def test_common_sense_ratio(stats):
    result = stats.common_sense_ratio()
    assert isinstance(result, pd.Series)
    # Manual calculation for comparison
    expected = stats.profit_factor() * stats.tail_ratio()
    pd.testing.assert_series_equal(result, expected)

def test_value_at_risk(stats):
    result = stats.value_at_risk(confidence=0.95)
    assert isinstance(result, pd.Series)
    # Correct column-wise mean and std
    mu = stats.all.mean()
    sigma = stats.all.std()
    expected = norm.ppf(1 - 0.95, loc=mu, scale=sigma)
    expected = pd.Series(expected, index=stats.all.columns)
    pd.testing.assert_series_equal(result, expected)


def test_var(stats):
    result = stats.var( confidence=0.95)
    expected = stats.value_at_risk( confidence=0.95)
    pd.testing.assert_series_equal(result, expected)

def test_conditional_value_at_risk(stats):
    result = stats.conditional_value_at_risk( confidence=0.95)
    assert isinstance(result, pd.Series)
    # Should be less than or equal to VaR
    var = stats.value_at_risk(confidence=0.95)
    assert (result <= var).all()

def test_cvar(stats):
    result = stats.cvar(confidence=0.95)
    print(result)
    expected = stats.conditional_value_at_risk(confidence=0.95)
    pd.testing.assert_series_equal(result, expected)

def test_expected_shortfall(stats):
    result = stats.expected_shortfall(confidence=0.95)
    expected = stats.conditional_value_at_risk(confidence=0.95)
    pd.testing.assert_series_equal(result, expected)

def test_win_rate(stats):
    result = stats.win_rate()
    assert isinstance(result, pd.Series)
    assert result["META"] == pytest.approx(0.525309)

def test_kelly_criterion(stats):
    result = stats.kelly_criterion()
    assert isinstance(result, pd.Series)
    # Manual calculation for comparison
    win_loss_ratio = stats.payoff_ratio()
    win_prob = stats.win_rate()
    lose_prob = 1 - win_prob
    expected = ((win_loss_ratio * win_prob) - lose_prob) / win_loss_ratio
    pd.testing.assert_series_equal(result, expected)

def test_gain_to_pain_ratio(stats):
    result = stats.gain_to_pain_ratio()
    assert isinstance(result, pd.Series)
    # Manual calculation for comparison
    downside = abs(stats.all[stats.all < 0].sum())
    expected = stats.all.sum() / downside
    pd.testing.assert_series_equal(result, expected)

def test_risk_return_ratio(stats):
    result = stats.risk_return_ratio()
    assert isinstance(result, pd.Series)
    # Manual calculation for comparison
    expected = stats.all.mean() / stats.all.std()
    pd.testing.assert_series_equal(result, expected)

def test_outlier_win_ratio(stats):
    result = stats.outlier_win_ratio(quantile=0.99)
    assert isinstance(result, pd.Series)
    # Manual calculation for comparison
    expected = stats.all.quantile(0.99).mean() / stats.all[stats.all >= 0].mean()
    pd.testing.assert_series_equal(result, expected)

def test_outlier_loss_ratio(stats):
    result = stats.outlier_loss_ratio(quantile=0.01)
    assert isinstance(result, pd.Series)
    # Manual calculation for comparison
    expected = stats.all.quantile(0.01).mean() / stats.all[stats.all < 0].mean()
    pd.testing.assert_series_equal(result, expected)

def test_best(stats):
    result = stats.best()
    assert isinstance(result, pd.Series)
    expected = stats.all.max()
    pd.testing.assert_series_equal(result, expected)

def test_worst(stats):
    result = stats.worst()
    assert isinstance(result, pd.Series)
    expected = stats.all.min()
    pd.testing.assert_series_equal(result, expected)

def test_exposure(stats):
    result = stats.exposure()
    assert isinstance(result, pd.Series)
    assert result["META"] == 0.40

def test_expected_return(stats):
    result = stats.expected_return()
    assert isinstance(result, pd.Series)
    assert result["META"] == pytest.approx(0.0003285438070024238)


def test_geometric_mean(stats):
    result = stats.geometric_mean()
    assert isinstance(result, pd.Series)
    assert result["META"] == pytest.approx(0.0003285438070024238)

def test_ghpr(stats):
    result = stats.ghpr()
    assert isinstance(result, pd.Series)
    assert result["META"] == pytest.approx(0.0003285438070024238)

def test_outliers(stats):
    result = stats.outliers(quantile=0.95)
    assert isinstance(result, pd.DataFrame)
    # assert len(result) <= len(returns)
    # Check that all values are above the 95th percentile
    threshold = stats.all.quantile(0.95)
    assert all(result > threshold)

def test_sharpe(stats):
    result = stats.sharpe(periods=252, smart=False)
    assert isinstance(result, pd.Series)
    # Manual calculation for comparison
    expected = stats.all.mean() / stats.all.std() * np.sqrt(252)
    pd.testing.assert_series_equal(result, expected)

def test_sharpe_smart(stats):
    result = stats.sharpe(periods=252, smart=True)
    assert isinstance(result, pd.Series)
    # Manual calculation for comparison
    expected = stats.all.mean() / (stats.all.std() * stats.autocorr_penalty()) * np.sqrt(252)
    pd.testing.assert_series_equal(result, expected)

def test_rolling_sharpe(stats):
    result = stats.rolling_sharpe(rolling_period=20, periods_per_year=252)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == stats.all.shape
    # First 19 values should be NaN (rolling window of 20)
    assert result.iloc[:19].isna().all().all()

def test_sortino(stats):
    result = stats.sortino(periods=252, smart=False)
    assert result["META"] == pytest.approx(0.67301334753882)

def test_smart_sortino(stats):
    result = stats.sortino(periods=252, smart=True)
    assert result["META"] == pytest.approx(0.6572328550179357)

def test_rolling_sortino(stats):
    result = stats.rolling_sortino(rolling_period=20, periods_per_year=252)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == stats.all.shape

    # First 19 values should be NaN (rolling window of 20)
    assert result.iloc[:19].isna().all().all()

def test_adjusted_sortino(stats):
    result = stats.adjusted_sortino(periods=252, smart=False)
    assert result["META"] == pytest.approx(0.47589230187375825)

def test_edge_cases(edge):
    assert edge.stats.profit_ratio().isna().all()
    assert edge.stats.cpc_index().isna().all()
    assert edge.stats.gain_to_pain_ratio().isna().all()

def test_information_ratio(stats):
    result = stats.information_ratio()
    assert result["AAPL"] == pytest.approx(0.025195605347769847)

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
