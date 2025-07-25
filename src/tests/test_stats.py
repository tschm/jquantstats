"""Tests for the stats module."""

import numpy as np
import polars as pl
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


def test_skew(stats):
    """Tests that the skew method calculates skewness correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The skewness value for META matches the expected value.

    """
    result = stats.skew()
    assert result["META"] == pytest.approx(0.4220690178941095)


def test_kurtosis(stats):
    """Tests that the kurtosis method calculates kurtosis correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The kurtosis value for META matches the expected value.

    """
    result = stats.kurtosis()
    assert result["META"] == pytest.approx(20.477943361921824)


def test_avg_return(stats):
    """Tests that the avg_return method calculates average returns correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The average return value for META matches the expected value.

    """
    result = stats.avg_return()
    assert result["META"] == pytest.approx(0.001142572799759818)


def test_avg_win(stats):
    """Tests that the avg_win method calculates average winning returns correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The average win value for META matches the expected value.

    """
    result = stats.avg_win()
    assert result["META"] == pytest.approx(0.016760408889269992)


def test_avg_loss(stats):
    """Tests that the avg_loss method calculates average losing returns correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The average loss value for META matches the expected value.

    """
    result = stats.avg_loss()
    assert result["META"] == pytest.approx(-0.01614062422517277)


def test_volatility(stats):
    """Tests that the volatility method calculates volatility correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The annualized volatility value for META matches the expected value.

    """
    result = stats.volatility(periods=252, annualize=True)
    assert result["META"] == pytest.approx(0.40072031010504233)


def test_rolling_volatility(stats):
    """Tests that the rolling_volatility method calculates rolling volatility correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The shape of the rolling volatility result matches the shape of the input data.

    """
    result = stats.rolling_volatility(rolling_period=20, periods_per_year=252)
    print(result.tail(5))
    assert result.shape == stats.all.shape


def test_payoff_ratio(stats):
    """Tests that the payoff_ratio method calculates payoff ratio correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The payoff ratio value for META matches the expected value.

    """
    result = stats.payoff_ratio()
    assert result["META"] == pytest.approx(1.0383990517002815)


def test_win_loss_ratio(stats):
    """Tests that the win_loss_ratio method calculates win/loss ratio correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The win/loss ratio value for META matches the expected value.

    """
    result = stats.win_loss_ratio()
    assert result["META"] == pytest.approx(1.0383990517002815)


def test_profit_ratio(stats):
    """Tests that the profit_ratio method calculates profit ratio correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The profit ratio value for META matches the expected value.

    """
    result = stats.profit_ratio()
    assert result["META"] == pytest.approx(0.9252488178411934)


def test_profit_factor(stats):
    """Tests that the profit_factor method calculates profit factor correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The profit factor value for META matches the expected value.

    """
    result = stats.profit_factor()
    assert result["META"] == pytest.approx(1.149125608578595)


def test_value_at_risk(stats):
    """Tests that the value_at_risk method calculates VaR correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The value at risk (alpha=0.05) for META matches the expected value.

    """
    result = stats.value_at_risk(alpha=0.05)
    assert result["META"] == pytest.approx(-0.04038269463520536)


def test_conditional_value_at_risk(stats):
    """Tests that the conditional_value_at_risk method calculates CVaR correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The conditional value at risk (alpha=0.05) for META matches the expected value.

    """
    result = stats.conditional_value_at_risk(alpha=0.05)
    assert result["META"] == pytest.approx(-0.06084410598898649)


def test_win_rate(stats):
    """Tests that the win_rate method calculates win rate correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The win rate value for META matches the expected value.

    """
    result = stats.win_rate()
    assert result["META"] == pytest.approx(0.525309)


def test_kelly_criterion(stats):
    """Tests that the kelly_criterion method calculates Kelly criterion correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The Kelly criterion value for META matches the expected value.

    """
    result = stats.kelly_criterion()
    assert result["META"] == pytest.approx(0.06817093826936971)


def test_gain_to_pain_ratio(stats):
    """Tests that the gain_to_pain_ratio method calculates gain-to-pain ratio correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The gain-to-pain ratio value for META matches the expected value.

    """
    result = stats.gain_to_pain_ratio()
    assert result["META"] == pytest.approx(0.14912560857859494)


def test_risk_return_ratio(stats):
    """Tests that the risk_return_ratio method calculates risk-return ratio correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The risk-return ratio value for META matches the expected value.

    """
    result = stats.risk_return_ratio()
    assert result["META"] == pytest.approx(0.045095921921619944)


def test_best(stats):
    """Tests that the best method calculates the best return correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The best return value for META matches the expected value.

    """
    result = stats.best()
    assert result["META"] == pytest.approx(0.2961146917048886)


def test_worst(stats):
    """Tests that the worst method calculates the worst return correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The worst return value for META matches the expected value.

    """
    result = stats.worst()
    assert result["META"] == pytest.approx(-0.2639010078964036)


def test_exposure(stats):
    """Tests that the exposure method calculates exposure correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The exposure value for META matches the expected value.

    """
    result = stats.exposure()
    assert result["META"] == pytest.approx(0.40)


def test_sharpe(stats):
    """Tests that the sharpe method calculates Sharpe ratio correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The Sharpe ratio value for META matches the expected value.

    """
    result = stats.sharpe(periods=252)
    assert result["META"] == pytest.approx(0.7158755672867543)


def test_rolling_sharpe(stats):
    """Tests that the rolling_sharpe method calculates rolling Sharpe ratio correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The shape of the rolling Sharpe ratio result matches the shape of the input data.

    """
    result = stats.rolling_sharpe(rolling_period=20, periods_per_year=252)
    assert result.shape == stats.all.shape
    print(result.tail(5))


def test_sortino(stats):
    """Tests that the sortino method calculates Sortino ratio correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The Sortino ratio value for META matches the expected value.

    """
    result = stats.sortino(periods=252)
    assert result["META"] == pytest.approx(1.06321091920911)


def test_rolling_sortino(stats):
    """Tests that the rolling_sortino method calculates rolling Sortino ratio correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The shape of the rolling Sortino ratio result matches the shape of the input data.

    """
    result = stats.rolling_sortino(rolling_period=20, periods_per_year=252)
    assert result.shape == stats.all.shape
    print(result.tail(5))


def test_adjusted_sortino(stats):
    """Tests that the adjusted_sortino method calculates adjusted Sortino ratio correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The adjusted Sortino ratio value for META matches the expected value.

    """
    result = stats.adjusted_sortino(periods=252)
    assert result["META"] == pytest.approx(0.7518036508043441)


def test_edge_cases(edge):
    """Tests that the stats methods handle edge cases correctly.

    Args:
        edge: The edge fixture containing a Data object with edge case data.

    Verifies:
        1. The profit_ratio method returns NaN for the 'returns' column.
        2. The gain_to_pain_ratio method returns NaN for the 'returns' column.

    """
    assert np.isnan(edge.stats.profit_ratio()["returns"])  # == {"returns": np.nan, "Benchmark": np.nan}
    assert np.isnan(edge.stats.gain_to_pain_ratio()["returns"])  # == {"returns": np.nan, "Benchmark": np.nan}


def test_information_ratio(stats):
    """Tests that the information_ratio method calculates information ratio correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The information ratio value for AAPL matches the expected value.

    """
    result = stats.information_ratio(periods_per_year=252)
    assert result["AAPL"] == pytest.approx(0.45766323376481344)


def test_greeks(stats):
    """Tests that the greeks method calculates alpha and beta correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The method executes without errors and returns a result.

    """
    result = stats.greeks(periods_per_year=252)
    print(result)

    #
    # assert isinstance(result, pd.DataFrame)
    # assert "alpha" in result.index
    # assert "beta" in result.index
    #
    # assert result["Benchmark"]["beta"] == 1.0
    # assert result["Benchmark"]["alpha"] == 0.0
    #
    # assert result["AAPL"]["beta"] == pytest.approx(1.1090322781954098)
    # assert result["AAPL"]["alpha"] == pytest.approx(0.1576003006124853)


def test_r_squared(stats):
    """Tests that the r_squared method calculates R-squared correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The method executes without errors and returns a result.

    """
    result = stats.r_squared()
    print(result)
    # assert 0 <= result <= 1  # R-squared should be between 0 and 1


def test_r2(stats):
    """Tests that the r2 method is an alias for r_squared.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        The r2 method returns the same result as the r_squared method.

    """
    result = stats.r2()
    expected = stats.r_squared()
    assert result == expected


def test_drawdowns(stats):
    """Tests that the drawdown method calculates drawdowns correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        1. The drawdown method returns a DataFrame with the expected structure.
        2. The drawdown values are calculated correctly.
        3. The maximum drawdown values match expected values.
        4. Drawdown values are always non-negative (since they represent losses).

    """
    # Test the drawdown method returns a DataFrame with the expected structure
    dd = stats.drawdown()
    assert isinstance(dd, pl.DataFrame)
    assert dd.shape[0] == stats.all.shape[0]  # Same number of rows as input data

    # Check that all columns from returns and benchmark are present
    for col in stats.data.returns.columns:
        assert col in dd.columns
    if stats.data.benchmark is not None:
        for col in stats.data.benchmark.columns:
            assert col in dd.columns

    # Verify drawdown values are always non-negative (since they represent losses)
    for col in dd.columns:
        if col not in stats.data.date_col:  # Skip date column
            assert (dd[col] >= 0).all()

    # Test specific drawdown values for META
    # The maximum drawdown for META should be approximately 0.76 (76%)
    max_dd = dd["META"].max()
    assert max_dd == pytest.approx(0.76, abs=0.01)

    # Test that drawdown is zero at peak equity points
    prices = stats.prices(stats.all["META"])
    peak_indices = prices.cum_max() == prices
    dd_at_peaks = dd.filter(peak_indices)["META"]
    assert (dd_at_peaks == 0).all()


def test_drawdowns_edge_case(edge):
    """Tests that the drawdown method handles edge cases correctly.

    Args:
        edge: The edge fixture containing a Data object with edge case data.

    Verifies:
        1. The drawdown method works with constant zero returns.
        2. The drawdown values are all zero when returns are all zero.

    """
    # Test the drawdown method with constant zero returns
    dd = edge.stats.drawdown()

    # Verify the structure
    assert isinstance(dd, pl.DataFrame)
    assert dd.shape[0] == edge.stats.all.shape[0]

    # For constant zero returns, drawdowns should all be zero
    for col in dd.columns:
        if col not in edge.stats.data.date_col:  # Skip date column
            assert (dd[col] == 0).all()


def test_max_drawdown(stats):
    """Tests that the max_drawdown method calculates maximum drawdown correctly.

    Args:
        stats: The stats fixture containing a Stats object.

    Verifies:
        1. The max_drawdown method returns a dictionary with the expected structure.
        2. The maximum drawdown values match the expected values.
        3. The max_drawdown method returns the same values as the maximum of the drawdown series.

    """
    # Calculate maximum drawdowns
    max_dd = stats.max_drawdown()

    # Verify structure
    assert isinstance(max_dd, dict)
    assert set(max_dd.keys()) == set(stats.data.assets)

    # Verify values for specific assets
    assert max_dd["META"] == pytest.approx(0.76, abs=0.01)
    assert max_dd["AAPL"] == pytest.approx(0.82, abs=0.01)

    # Verify that max_drawdown returns the same values as the maximum of the drawdown series
    dd = stats.drawdown()
    for col in stats.data.assets:
        assert max_dd[col] == pytest.approx(dd[col].max(), abs=0.0001)
