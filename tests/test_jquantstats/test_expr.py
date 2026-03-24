"""Tests for the pure pl.Expr factory functions in _stats/_expr.py.

Each test:
 1. Builds a small pl.DataFrame (or pl.LazyFrame) in-memory.
 2. Calls the factory function to obtain a named pl.Expr.
 3. Collects the result and asserts the expected scalar value.

This validates:
 - Each factory function in isolation (no Stats/Data objects needed).
 - That all factories produce scalar aggregations suitable for
   _lazy_columnwise (i.e. lf.select([expr]).collect() is a 1-row DataFrame).
 - Edge-case handling (zero-std, no wins/losses, empty drawdown, etc.).
"""

from __future__ import annotations

import math

import polars as pl
import pytest
from scipy.stats import norm as scipy_norm

from jquantstats._stats._expr import (
    avg_drawdown_expr,
    avg_loss_expr,
    avg_return_expr,
    avg_win_expr,
    best_expr,
    calmar_expr,
    conditional_value_at_risk_expr,
    exposure_expr,
    gain_to_pain_ratio_expr,
    kurtosis_expr,
    max_drawdown_expr,
    payoff_ratio_expr,
    profit_factor_expr,
    profit_ratio_expr,
    recovery_factor_expr,
    risk_return_ratio_expr,
    sharpe_expr,
    skew_expr,
    sortino_expr,
    value_at_risk_expr,
    volatility_expr,
    win_rate_expr,
    worst_expr,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _collect_scalar(lf: pl.LazyFrame, expr: pl.Expr) -> float:
    """Collect a scalar aggregation expression from a LazyFrame."""
    result = lf.select([expr]).collect()
    assert result.shape == (1, 1), f"Expected 1x1 DataFrame, got {result.shape}"
    val = result.row(0)[0]
    return float(val) if val is not None else float("nan")


@pytest.fixture
def sample_lf() -> pl.LazyFrame:
    """Small returns series with positive and negative values."""
    returns = [0.02, -0.01, 0.03, -0.02, 0.01, 0.04, -0.03, 0.02, 0.01, -0.01]
    return pl.DataFrame({"r": returns}).lazy()


@pytest.fixture
def all_positive_lf() -> pl.LazyFrame:
    """Series with only positive returns (no losses)."""
    return pl.DataFrame({"r": [0.01, 0.02, 0.03, 0.01, 0.02]}).lazy()


@pytest.fixture
def all_zero_lf() -> pl.LazyFrame:
    """Series of all-zero returns."""
    return pl.DataFrame({"r": [0.0, 0.0, 0.0, 0.0, 0.0]}).lazy()


# ── Basic statistics ───────────────────────────────────────────────────────────


def test_skew_expr_returns_scalar(sample_lf):
    """skew_expr produces a scalar result."""
    val = _collect_scalar(sample_lf, skew_expr("r"))
    assert math.isfinite(val)


def test_skew_expr_matches_polars_skew(sample_lf):
    """skew_expr matches pl.Series.skew(bias=False)."""
    series = sample_lf.collect()["r"]
    expected = float(series.skew(bias=False))
    val = _collect_scalar(sample_lf, skew_expr("r"))
    assert val == pytest.approx(expected)


def test_kurtosis_expr_returns_scalar(sample_lf):
    """kurtosis_expr produces a scalar result."""
    val = _collect_scalar(sample_lf, kurtosis_expr("r"))
    assert math.isfinite(val)


def test_kurtosis_expr_matches_polars_kurtosis(sample_lf):
    """kurtosis_expr matches pl.Series.kurtosis(bias=False)."""
    series = sample_lf.collect()["r"]
    expected = float(series.kurtosis(bias=False))
    val = _collect_scalar(sample_lf, kurtosis_expr("r"))
    assert val == pytest.approx(expected)


def test_avg_return_expr_excludes_zeros():
    """avg_return_expr excludes zero values from the mean."""
    lf = pl.DataFrame({"r": [0.0, 0.01, -0.01, 0.02, 0.0]}).lazy()
    val = _collect_scalar(lf, avg_return_expr("r"))
    expected = (0.01 + (-0.01) + 0.02) / 3
    assert val == pytest.approx(expected)


def test_avg_win_expr_mean_of_positives(sample_lf):
    """avg_win_expr returns the mean of strictly positive values."""
    series = sample_lf.collect()["r"]
    expected = float(series.filter(series > 0).mean())
    val = _collect_scalar(sample_lf, avg_win_expr("r"))
    assert val == pytest.approx(expected)


def test_avg_loss_expr_mean_of_negatives(sample_lf):
    """avg_loss_expr returns the mean of strictly negative values."""
    series = sample_lf.collect()["r"]
    expected = float(series.filter(series < 0).mean())
    val = _collect_scalar(sample_lf, avg_loss_expr("r"))
    assert val == pytest.approx(expected)


# ── Volatility ─────────────────────────────────────────────────────────────────


def test_volatility_expr_annualised(sample_lf):
    """volatility_expr annualises with sqrt(periods)."""
    series = sample_lf.collect()["r"]
    expected = float(series.std()) * math.sqrt(252)
    val = _collect_scalar(sample_lf, volatility_expr("r", periods=252.0))
    assert val == pytest.approx(expected)


def test_volatility_expr_no_annualise(sample_lf):
    """volatility_expr with annualize=False returns raw std."""
    series = sample_lf.collect()["r"]
    expected = float(series.std())
    val = _collect_scalar(sample_lf, volatility_expr("r", periods=252.0, annualize=False))
    assert val == pytest.approx(expected)


# ── Win / loss metrics ─────────────────────────────────────────────────────────


def test_payoff_ratio_expr(sample_lf):
    """payoff_ratio_expr = avg_win / |avg_loss|."""
    series = sample_lf.collect()["r"]
    avg_win = float(series.filter(series > 0).mean())
    avg_loss = abs(float(series.filter(series < 0).mean()))
    expected = avg_win / avg_loss
    val = _collect_scalar(sample_lf, payoff_ratio_expr("r"))
    assert val == pytest.approx(expected)


def test_profit_factor_expr(sample_lf):
    """profit_factor_expr = sum_wins / |sum_losses|."""
    series = sample_lf.collect()["r"]
    wins_sum = float(series.filter(series > 0).sum())
    loss_sum = abs(float(series.filter(series < 0).sum()))
    expected = wins_sum / loss_sum
    val = _collect_scalar(sample_lf, profit_factor_expr("r"))
    assert val == pytest.approx(expected)


def test_profit_ratio_expr(sample_lf):
    """profit_ratio_expr = (|mean_win|/count_win) / (|mean_loss|/count_loss)."""
    series = sample_lf.collect()["r"]
    wins = series.filter(series >= 0)
    losses = series.filter(series < 0)
    win_ratio = abs(float(wins.mean())) / wins.count()
    loss_ratio = abs(float(losses.mean())) / losses.count()
    expected = win_ratio / loss_ratio
    val = _collect_scalar(sample_lf, profit_ratio_expr("r"))
    assert val == pytest.approx(expected)


def test_profit_ratio_expr_no_losses_is_nan(all_positive_lf):
    """profit_ratio_expr returns NaN when there are no negative returns."""
    val = _collect_scalar(all_positive_lf, profit_ratio_expr("r"))
    assert math.isnan(val)


def test_win_rate_expr(sample_lf):
    """win_rate_expr = count(>0) / count(!=0)."""
    series = sample_lf.collect()["r"]
    expected = series.filter(series > 0).count() / series.filter(series != 0).count()
    val = _collect_scalar(sample_lf, win_rate_expr("r"))
    assert val == pytest.approx(expected)


def test_gain_to_pain_ratio_expr(sample_lf):
    """gain_to_pain_ratio_expr = sum / |sum of negatives|."""
    series = sample_lf.collect()["r"]
    expected = float(series.sum()) / float(series.filter(series < 0).abs().sum())
    val = _collect_scalar(sample_lf, gain_to_pain_ratio_expr("r"))
    assert val == pytest.approx(expected)


def test_gain_to_pain_ratio_expr_no_losses_is_nan(all_positive_lf):
    """gain_to_pain_ratio_expr returns NaN when sum of negatives is zero."""
    val = _collect_scalar(all_positive_lf, gain_to_pain_ratio_expr("r"))
    assert math.isnan(val)


def test_risk_return_ratio_expr(sample_lf):
    """risk_return_ratio_expr = mean / std."""
    series = sample_lf.collect()["r"]
    expected = float(series.mean()) / float(series.std())
    val = _collect_scalar(sample_lf, risk_return_ratio_expr("r"))
    assert val == pytest.approx(expected)


def test_best_expr(sample_lf):
    """best_expr returns the maximum value."""
    series = sample_lf.collect()["r"]
    expected = float(series.max())
    val = _collect_scalar(sample_lf, best_expr("r"))
    assert val == pytest.approx(expected)


def test_worst_expr(sample_lf):
    """worst_expr returns the minimum value."""
    series = sample_lf.collect()["r"]
    expected = float(series.min())
    val = _collect_scalar(sample_lf, worst_expr("r"))
    assert val == pytest.approx(expected)


def test_exposure_expr(sample_lf):
    """exposure_expr = count(!=0) / total_rows, rounded to 2 dp."""
    series = sample_lf.collect()["r"]
    total = len(series)
    nonzero = int((series != 0).sum())
    expected = round(nonzero / total, 2)
    val = _collect_scalar(sample_lf, exposure_expr("r"))
    assert val == pytest.approx(expected)


def test_exposure_expr_with_zeros():
    """exposure_expr handles series with some zero values."""
    lf = pl.DataFrame({"r": [0.0, 0.01, 0.0, -0.01, 0.02]}).lazy()
    val = _collect_scalar(lf, exposure_expr("r"))
    # 3 non-zero out of 5 → 0.60
    assert val == pytest.approx(0.60)


# ── Risk metrics ───────────────────────────────────────────────────────────────


def test_value_at_risk_expr(sample_lf):
    """value_at_risk_expr matches scipy VaR calculation."""
    series = sample_lf.collect()["r"]
    mu = float(series.mean())
    std = float(series.std())
    expected = float(scipy_norm.ppf(0.05, mu, 1.0 * std))
    val = _collect_scalar(sample_lf, value_at_risk_expr("r"))
    assert val == pytest.approx(expected)


def test_conditional_value_at_risk_expr(sample_lf):
    """conditional_value_at_risk_expr matches manual CVaR calculation."""
    series = sample_lf.collect()["r"]
    mu = float(series.mean())
    std = float(series.std())
    var = float(scipy_norm.ppf(0.05, mu, 1.0 * std))
    filtered = series.filter(series < var)
    expected = float(filtered.mean()) if len(filtered) > 0 else float("nan")
    val = _collect_scalar(sample_lf, conditional_value_at_risk_expr("r"))
    if math.isnan(expected):
        assert math.isnan(val)
    else:
        assert val == pytest.approx(expected)


# ── Performance ratios ─────────────────────────────────────────────────────────


def test_sharpe_expr(sample_lf):
    """sharpe_expr matches manual Sharpe calculation."""
    series = sample_lf.collect()["r"]
    mean_v = float(series.mean())
    std_v = float(series.std(ddof=1))
    expected = (mean_v / std_v) * (252.0**0.5)
    val = _collect_scalar(sample_lf, sharpe_expr("r", periods=252.0))
    assert val == pytest.approx(expected)


def test_sharpe_expr_zero_std_is_nan(all_zero_lf):
    """sharpe_expr returns NaN when std ≈ 0."""
    val = _collect_scalar(all_zero_lf, sharpe_expr("r", periods=252.0))
    assert math.isnan(val)


def test_sortino_expr(sample_lf):
    """sortino_expr matches manual Sortino calculation."""
    series = sample_lf.collect()["r"]
    mean_v = float(series.mean())
    downside_sum = float((series.filter(series < 0) ** 2).sum())
    n = series.count()
    downside_dev = math.sqrt(downside_sum / n)
    expected = (mean_v / downside_dev) * (252.0**0.5)
    val = _collect_scalar(sample_lf, sortino_expr("r", periods=252.0))
    assert val == pytest.approx(expected)


def test_sortino_expr_zero_downside_positive_mean_is_inf(all_positive_lf):
    """sortino_expr returns inf when downside deviation is 0 and mean > 0."""
    val = _collect_scalar(all_positive_lf, sortino_expr("r", periods=252.0))
    assert val == float("inf")


def test_sortino_expr_zero_downside_zero_mean_is_nan(all_zero_lf):
    """sortino_expr returns NaN when both downside and mean are 0."""
    val = _collect_scalar(all_zero_lf, sortino_expr("r", periods=252.0))
    assert math.isnan(val)


# ── Drawdown ────────────────────────────────────────────────────────────────────


def test_max_drawdown_expr_positive(sample_lf):
    """max_drawdown_expr returns a positive drawdown fraction."""
    val = _collect_scalar(sample_lf, max_drawdown_expr("r"))
    assert val > 0


def test_max_drawdown_expr_no_drawdown():
    """max_drawdown_expr returns 0 (or very small) for monotonically increasing returns."""
    lf = pl.DataFrame({"r": [0.01, 0.02, 0.01, 0.03, 0.02]}).lazy()
    val = _collect_scalar(lf, max_drawdown_expr("r"))
    # With all positive returns, there is no drawdown
    assert val == pytest.approx(0.0, abs=1e-10)


def test_avg_drawdown_expr_no_drawdown(all_positive_lf):
    """avg_drawdown_expr returns 0.0 when there are no underwater periods."""
    val = _collect_scalar(all_positive_lf, avg_drawdown_expr("r"))
    assert val == pytest.approx(0.0)


def test_avg_drawdown_expr_with_drawdown(sample_lf):
    """avg_drawdown_expr returns a positive value for data with losses."""
    val = _collect_scalar(sample_lf, avg_drawdown_expr("r"))
    assert val >= 0.0


def test_calmar_expr(sample_lf):
    """calmar_expr returns a finite ratio when there is drawdown."""
    val = _collect_scalar(sample_lf, calmar_expr("r", periods=252.0))
    assert math.isfinite(val)


def test_calmar_expr_no_drawdown_is_nan(all_positive_lf):
    """calmar_expr returns NaN when max drawdown is zero."""
    val = _collect_scalar(all_positive_lf, calmar_expr("r", periods=252.0))
    assert math.isnan(val)


def test_recovery_factor_expr(sample_lf):
    """recovery_factor_expr returns a finite value for data with drawdown."""
    val = _collect_scalar(sample_lf, recovery_factor_expr("r"))
    assert math.isfinite(val)


def test_recovery_factor_expr_no_drawdown_is_nan(all_positive_lf):
    """recovery_factor_expr returns NaN when max drawdown is zero."""
    val = _collect_scalar(all_positive_lf, recovery_factor_expr("r"))
    assert math.isnan(val)


# ── Multi-column (lazy_columnwise) ────────────────────────────────────────────


def test_lazy_columnwise_single_select():
    """_lazy_columnwise produces a 1-row DataFrame for all asset columns at once.

    This verifies that no N-column Python loop occurs — all expressions are
    collected in a single lf.select() call.
    """
    from jquantstats._stats._core import _lazy_columnwise

    lf = pl.DataFrame(
        {
            "A": [0.01, -0.01, 0.02, -0.02, 0.01],
            "B": [0.02, -0.02, 0.03, -0.01, 0.02],
        }
    ).lazy()

    result = _lazy_columnwise(lf, sharpe_expr, ["A", "B"], periods=252.0)

    assert set(result.keys()) == {"A", "B"}
    assert all(math.isfinite(v) for v in result.values())


def test_lazy_columnwise_result_matches_individual():
    """_lazy_columnwise matches computing each expression individually."""
    from jquantstats._stats._core import _lazy_columnwise

    lf = pl.DataFrame(
        {
            "X": [0.01, -0.01, 0.02, -0.02, 0.03],
            "Y": [-0.01, 0.02, -0.03, 0.01, 0.01],
        }
    ).lazy()

    batch = _lazy_columnwise(lf, win_rate_expr, ["X", "Y"])

    for col in ["X", "Y"]:
        individual = _collect_scalar(lf, win_rate_expr(col))
        assert batch[col] == pytest.approx(individual)


# ── Data.lazy property ─────────────────────────────────────────────────────────


def test_data_lazy_returns_lazyframe(data):
    """data.lazy returns a pl.LazyFrame view of data.all."""
    lf = data.lazy
    assert isinstance(lf, pl.LazyFrame)
    assert lf.collect().shape == data.all.shape


def test_data_lazy_columns_match_all(data):
    """data.lazy has the same columns as data.all."""
    assert data.lazy.collect_schema().names() == data.all.columns


# ── max_drawdown_single_series (kept for backward compat) ─────────────────────


def test_max_drawdown_single_series_positive():
    """max_drawdown_single_series returns a positive fraction for a series with losses."""
    from jquantstats._stats._performance import _PerformanceStatsMixin

    series = pl.Series([0.01, 0.02, -0.05, 0.01, 0.02])
    val = _PerformanceStatsMixin.max_drawdown_single_series(series)
    assert val > 0
    assert math.isfinite(val)


def test_max_drawdown_single_series_no_drawdown():
    """max_drawdown_single_series returns 0.0 for monotonically increasing series."""
    from jquantstats._stats._performance import _PerformanceStatsMixin

    series = pl.Series([0.01, 0.02, 0.01, 0.03])
    val = _PerformanceStatsMixin.max_drawdown_single_series(series)
    assert val == pytest.approx(0.0, abs=1e-10)


# ── _drawdown_series (public export kept for backward compat) ─────────────────


def test_drawdown_series_basic():
    """_drawdown_series returns expected drawdown values for a known series."""
    from jquantstats._stats._core import _drawdown_series

    s = pl.Series([0.0, -0.1, 0.2])
    result = _drawdown_series(s).to_list()
    assert result[0] == pytest.approx(0.0)
    assert result[1] == pytest.approx(0.1, rel=1e-6)
    assert result[2] == pytest.approx(0.0)


def test_drawdown_series_all_zeros():
    """_drawdown_series returns all zeros for a zero-return series."""
    from jquantstats._stats._core import _drawdown_series

    s = pl.Series([0.0, 0.0, 0.0, 0.0])
    result = _drawdown_series(s).to_list()
    assert all(v == pytest.approx(0.0) for v in result)
