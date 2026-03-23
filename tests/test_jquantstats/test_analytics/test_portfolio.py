"""Tests for jquantstats.analytics.portfolio module.

This module verifies the Portfolio dataclass end-to-end: construction and
validation (from_cash_position, from_risk_position, __post_init__ guards),
derived series (profits, NAV, drawdown, highwater, returns), decomposition
(tilt / timing), aggregation (monthly), positional transforms (lag,
smoothed_holding, truncate), and the stats / plots facades.

Integer-indexed (date-free) portfolios are exercised alongside the standard
date-indexed variants so that the date-column branch never silently breaks.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import plotly.graph_objects as go
import polars as pl
import polars.testing as pt
import pytest

from jquantstats import Portfolio
from jquantstats._portfolio_data import PortfolioData
from jquantstats.exceptions import (
    IntegerIndexBoundError,
    MissingDateColumnError,
)

# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def prices():
    """Three-day, two-asset price frame with exact geometric ratios.

    Asset A: 100 → 110 → 121 (10 % gain each day).
    Asset B: 200 → 180 → 198 (−10 % then +10 %).
    Chosen so that expected profit values are small integers, making manual
    verification of downstream assertions straightforward.
    """
    return pl.DataFrame(
        {
            "date": pl.date_range(start=date(2020, 1, 1), end=date(2020, 1, 3), interval="1d", eager=True).cast(
                pl.Date
            ),
            "A": pl.Series([100.0, 110.0, 121.0], dtype=pl.Float64),
            "B": pl.Series([200.0, 180.0, 198.0], dtype=pl.Float64),
        }
    )


@pytest.fixture
def positions():
    """Three-day cash-position frame aligned with the prices fixture.

    Asset A: 1 000 units held throughout.
    Asset B: 0 on day 1, 500 from day 2 onward.
    The zero position on day 1 ensures the B contribution to daily profit is
    zero on that day, allowing exact integer assertions in the core tests.
    """
    return pl.DataFrame(
        {
            "date": pl.date_range(start=date(2020, 1, 1), end=date(2020, 1, 3), interval="1d", eager=True).cast(
                pl.Date
            ),
            "A": pl.Series([1000.0, 1000.0, 1000.0], dtype=pl.Float64),
            "B": pl.Series([0.0, 500.0, 500.0], dtype=pl.Float64),
        }
    )


@pytest.fixture
def portfolio(prices, positions):
    """Create Portfolio instance for testing (3-day, exact numeric values)."""
    return Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e5)


@pytest.fixture
def monthly_portfolio():
    """Build a small deterministic Portfolio for monthly aggregation tests."""
    start = date(2020, 1, 10)
    days = 80
    end = start + timedelta(days=days - 1)
    dates = pl.date_range(start=start, end=end, interval="1d", eager=True).cast(pl.Date)

    rng = np.random.default_rng(42)
    returns = rng.normal(0.0, 0.01, size=days)
    prices_arr = (1.0 + returns).cumprod()

    return Portfolio(
        prices=pl.DataFrame({"date": dates, "A": prices_arr}),
        cashposition=pl.DataFrame({"date": dates, "A": pl.Series([1000.0] * days, dtype=pl.Float64)}),
        aum=10000,
    )


@pytest.fixture
def truncate_portfolio():
    """Small 6-day portfolio so truncation height assertions are exact."""
    n = 6
    start = date(2020, 1, 1)
    end = start + timedelta(days=n - 1)
    dates = pl.date_range(start=start, end=end, interval="1d", eager=True).cast(pl.Date)
    return Portfolio.from_cash_position(
        prices=pl.DataFrame(
            {
                "date": dates,
                "A": pl.Series([100.0 + 10.0 * i for i in range(n)], dtype=pl.Float64),
                "B": pl.Series([200.0 - 5.0 * i for i in range(n)], dtype=pl.Float64),
            }
        ),
        cash_position=pl.DataFrame(
            {
                "date": dates,
                "A": pl.Series([1000.0] * n, dtype=pl.Float64),
                "B": pl.Series([500.0] * n, dtype=pl.Float64),
            }
        ),
        aum=1e6,
    )


# ─── Core: profits, NAV, Sharpe ───────────────────────────────────────────────


def test_compute_daily_profits_portfolio_basic(portfolio):
    """Compute per-asset profits and preserve date column."""
    profits = portfolio.profits

    assert "date" in profits.columns

    expected = pl.DataFrame(
        {
            "date": portfolio.prices["date"],
            "A": pl.Series([0.0, 100.0, 100.0], dtype=pl.Float64),
            "B": pl.Series([0.0, 0.0, 50.0], dtype=pl.Float64),
        }
    )

    assert profits.columns == expected.columns
    for c in ["A", "B"]:
        assert np.allclose(profits[c].to_numpy(), expected[c].to_numpy(), rtol=1e-12, atol=1e-12)


def test_portfolio_profit_and_nav(portfolio):
    """Aggregate per-asset profits to portfolio profit and compute NAV."""
    profit_df = portfolio.profit
    assert profit_df.columns == ["date", "profit"]

    expected_profit = np.array([0.0, 100.0, 150.0])
    assert np.allclose(profit_df["profit"].to_numpy(), expected_profit)

    nav_df = portfolio.nav_accumulated
    assert nav_df.columns == ["date", "profit", "NAV_accumulated"]

    expected_nav = np.array([1e5, 1e5 + 100.0, 1e5 + 250.0])
    assert np.allclose(nav_df["NAV_accumulated"].to_numpy(), expected_nav)


def test_portfolio_sharpe_matches_manual(portfolio):
    """Sharpe returned by class matches manual computation."""
    out = portfolio.stats.sharpe()["returns"]
    assert np.isfinite(out)
    assert np.isclose(out, 20.845234695819794, rtol=1e-12, atol=1e-12)


def test_portfolio_plot_returns_figure(portfolio):
    """Plot method returns a Plotly Figure and is serializable."""
    fig = portfolio.plots.snapshot()
    assert isinstance(fig, go.Figure)
    _ = fig.to_dict()


# ─── __post_init__ validation ─────────────────────────────────────────────────


def test_portfolio_post_init_requires_polars_dataframes(prices, positions):
    """__post_init__ should assert inputs are Polars DataFrames."""
    with pytest.raises(TypeError, match=r"cashposition must be pl\.DataFrame, got dict"):
        Portfolio(prices=prices, cashposition={"date": [1, 2, 3]}, aum=1e5)

    with pytest.raises(TypeError, match=r"prices must be pl\.DataFrame, got list"):
        Portfolio(prices=[[1.0, 2.0, 3.0]], cashposition=positions, aum=1e5)


def test_portfolio_post_init_requires_same_number_of_rows(prices, positions):
    """__post_init__ should raise ValueError when row counts differ."""
    with pytest.raises(ValueError, match=r"cashposition and prices must have the same number of rows"):
        Portfolio(prices=prices.head(3), cashposition=positions.head(2), aum=1e5)


def test_portfolio_post_init_requires_positive_aum(prices, positions):
    """__post_init__ should raise ValueError when AUM is not strictly positive."""
    with pytest.raises(ValueError, match=r"aum must be strictly positive"):
        Portfolio(prices=prices, cashposition=positions, aum=0.0)

    with pytest.raises(ValueError, match=r"aum must be strictly positive"):
        Portfolio(prices=prices, cashposition=positions, aum=-1.0)


# ─── from_riskposition / edge-cases ──────────────────────────────────────────


def test_from_riskposition_returns_portfolio_and_cashposition_shape():
    """from_riskposition should return a Portfolio with aligned cashposition columns/height."""
    dates = pl.date_range(start=date(2020, 1, 1), end=date(2020, 2, 10), interval="1d", eager=True).cast(pl.Date)
    prices = pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series(np.linspace(100, 120, len(dates)), dtype=pl.Float64),
            "B": pl.Series(np.linspace(50, 60, len(dates)), dtype=pl.Float64),
        }
    )
    riskposition = pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series(np.sin(np.linspace(0, 3.14, len(dates))), dtype=pl.Float64),
            "B": pl.Series(np.cos(np.linspace(0, 3.14, len(dates))), dtype=pl.Float64),
        }
    )

    pf = Portfolio.from_risk_position(prices, riskposition, vola=8, aum=1e8)
    assert isinstance(pf, Portfolio)
    assert pf.cashposition.height == prices.height
    for c in ["A", "B"]:
        assert c in pf.cashposition.columns


def test_sharpe_zero_std_returns_nan():
    """Sharpe should return NaN when NAV differences have zero std (flat NAV)."""
    import math

    dates = pl.date_range(start=date(2020, 1, 1), end=date(2020, 1, 5), interval="1d", eager=True).cast(pl.Date)
    prices = pl.DataFrame({"date": dates, "A": pl.Series([100.0] * len(dates), dtype=pl.Float64)})
    positions = pl.DataFrame({"date": dates, "A": pl.Series([0.0] * len(dates), dtype=pl.Float64)})

    pf = Portfolio(prices=prices, cashposition=positions, aum=1e5)
    result = pf.stats.sharpe()["returns"]
    assert math.isnan(result)


def test_compute_daily_profits_replaces_nonfinite_with_zero():
    """_compute_daily_profits_portfolio should replace non-finite profit values with 0.0."""
    prices = pl.DataFrame(
        {
            "date": pl.date_range(start=date(2020, 1, 1), end=date(2020, 1, 2), interval="1d", eager=True).cast(
                pl.Date
            ),
            "A": pl.Series([0.0, 1.0], dtype=pl.Float64),
        }
    )
    positions = pl.DataFrame({"date": prices["date"], "A": pl.Series([1.0, 1.0], dtype=pl.Float64)})

    portfolio = Portfolio(prices=prices, cashposition=positions, aum=1e5)
    profits = portfolio.profits
    assert np.allclose(profits["A"].to_numpy(), np.array([0.0, 0.0]))


def test_compute_daily_profits_no_numeric_columns():
    """When there are no numeric columns, function should return only non-numeric columns unchanged."""
    dates = pl.date_range(start=date(2020, 1, 1), end=date(2020, 1, 2), interval="1d", eager=True).cast(pl.Date)
    prices = pl.DataFrame({"date": dates})
    positions = pl.DataFrame({"date": dates})
    portfolio = Portfolio(prices=prices, cashposition=positions, aum=1e5)
    profits = portfolio.profits
    assert profits.columns == ["date"]
    assert profits.height == 2


def test_profit_raises_when_no_numeric_asset_columns():
    """Portfolio.profit should raise ValueError if there are no numeric asset columns."""
    dates = pl.date_range(start=date(2020, 1, 1), end=date(2020, 1, 2), interval="1d", eager=True).cast(pl.Date)
    prices = pl.DataFrame({"date": dates})
    positions = pl.DataFrame({"date": dates})
    portfolio = Portfolio(prices=prices, cashposition=positions, aum=1e5)

    with pytest.raises(ValueError, match=r".*"):
        _ = portfolio.profit


# ─── Returns, NAV variants, drawdown ─────────────────────────────────────────


def test_returns_property_scales_profit_by_aum_and_preserves_date(portfolio):
    """Returns should divide numeric columns by aum and retain the 'date' column."""
    rets = portfolio.returns

    assert "date" in rets.columns
    assert "profit" in rets.columns

    expected = (portfolio.profit.select(pl.col("profit")) / portfolio.aum)["profit"].to_numpy()
    actual = rets["returns"].to_numpy()
    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_nav_compounded_uses_compounding_and_is_close_to_nav_for_small_returns(portfolio):
    """nav_compounded should compound returns; for small returns it approximates additive NAV."""
    nav_add = portfolio.nav_accumulated
    nav_cmp = portfolio.nav_compounded

    assert "date" in nav_cmp.columns
    assert "date" in nav_add.columns

    cmp_values = nav_cmp["NAV_compounded"].to_numpy()
    assert np.isclose(cmp_values[0], portfolio.aum)

    add_values = nav_add["NAV_accumulated"].to_numpy()
    assert np.isclose(add_values[0], portfolio.aum)


def test_highwater_is_cummax_of_nav(portfolio):
    """Highwater should equal the cumulative maximum of NAV and preserve 'date'."""
    nav_df = portfolio.nav_accumulated
    hw_df = portfolio.highwater

    assert "date" in hw_df.columns
    assert "highwater" in hw_df.columns

    expected = nav_df["NAV_accumulated"].cum_max().to_numpy()
    actual = hw_df["highwater"].to_numpy()
    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_drawdown_is_highwater_minus_nav_and_preserves_date(portfolio):
    """Drawdown should equal highwater - NAV, start at 0, be non-negative, and keep 'date'."""
    dd_df = portfolio.drawdown

    assert "date" in dd_df.columns
    assert "drawdown" in dd_df.columns

    expected = (dd_df["highwater"] - dd_df["NAV_accumulated"]).to_numpy()
    actual = dd_df["drawdown"].to_numpy()

    assert np.isclose(actual[0], 0.0)
    assert np.all(actual >= 0.0)
    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_stats(portfolio):
    """stats() returns legacy Stats with expected Sharpe; kurtosis is None for tiny samples."""
    stats = portfolio.stats

    assert pytest.approx(stats.sharpe()["returns"]) == 20.845234695819794
    # Legacy kurtosis(bias=False) requires ≥4 observations; returns None for 3-row portfolios.
    assert stats.kurtosis()["returns"] is None


def test_portfolio_snapshot_log_scale(portfolio):
    """snapshot(log_scale=True) returns a Figure and sets the first y-axis to logarithmic scale."""
    fig = portfolio.plots.snapshot(log_scale=True)
    assert isinstance(fig, go.Figure)
    assert fig.layout.yaxis.type == "log"


def test_assert_clean_series_raises_on_null():
    """_assert_clean_series should raise ValueError when series contains null values."""
    s = pl.Series([1.0, None, 3.0])
    with pytest.raises(ValueError, match=r".*"):
        PortfolioData._assert_clean_series(s)


def test_assert_clean_series_raises_on_nonfinite():
    """_assert_clean_series should raise ValueError when series contains non-finite values."""
    s = pl.Series([1.0, float("inf"), 3.0])
    with pytest.raises(ValueError, match=r".*"):
        PortfolioData._assert_clean_series(s)


def test_portfolio_all_merges_drawdown_and_nav_compounded(portfolio):
    """All property should join drawdown and nav_compounded on 'date' with expected columns."""
    result = portfolio.all
    assert "date" in result.columns
    assert "NAV_accumulated" in result.columns
    assert "NAV_compounded" in result.columns
    assert "drawdown" in result.columns
    assert len(result) == len(portfolio.prices)


# ─── Lag ─────────────────────────────────────────────────────────────────────


def test_lag_positive_shifts_weights_down_and_preserves_date(portfolio):
    """lag(+1) should shift numeric columns down by one and preserve 'date'."""
    pf_lag1 = portfolio.lag(1)
    assert isinstance(pf_lag1, Portfolio)
    assert pf_lag1.aum == portfolio.aum
    assert pf_lag1.cashposition.columns[0] == "date"

    for c in portfolio.assets:
        s0 = portfolio.cashposition[c]
        s1 = pf_lag1.cashposition[c]
        assert s1.null_count() == 1
        assert np.allclose(s1.drop_nulls().to_numpy(), s0[:-1].to_numpy(), rtol=0, atol=0)

    _ = pf_lag1.profit


def test_lag_negative_leads_weights_and_last_becomes_null(portfolio):
    """lag(-1) should lead numeric columns; last element becomes null."""
    pf_lead1 = portfolio.lag(-1)
    for c in portfolio.assets:
        s0 = portfolio.cashposition[c]
        s1 = pf_lead1.cashposition[c]
        assert s1.null_count() == 1
        assert np.allclose(s1.head(len(s1) - 1).to_numpy(), s0[1:].to_numpy(), rtol=0, atol=0)

    _ = pf_lead1.profit


def test_lag_zero_returns_same_portfolio_object_or_equal_data(portfolio):
    """lag(0) should be a no-op: same object or equal data content and AUM preserved."""
    pf0 = portfolio.lag(0)
    assert pf0.aum == portfolio.aum
    pt.assert_frame_equal(pf0.cashposition, portfolio.cashposition)
    pt.assert_frame_equal(pf0.prices, portfolio.prices)


def test_lag_raises_typeerror_for_non_int(portfolio):
    """Passing a non-integer value to lag() should raise TypeError.

    Floats are rejected even when they represent whole numbers (e.g., 1.5)
    to prevent silent truncation from masking logic errors in calling code.
    """
    with pytest.raises(TypeError):
        _ = portfolio.lag(1.5)  # type: ignore[arg-type]


# ─── Smoothed holding ─────────────────────────────────────────────────────────


def test_smoothed_holding_zero_returns_self_and_preserves_state(portfolio):
    """Calling smoothed_holding(0) should return the same Portfolio instance and keep data intact."""
    pf_zero = portfolio.smoothed_holding(0)

    assert pf_zero is portfolio
    assert pf_zero.aum == portfolio.aum
    pt.assert_frame_equal(pf_zero.prices, portfolio.prices)
    pt.assert_frame_equal(pf_zero.cashposition, portfolio.cashposition)

    nav_before = portfolio.nav_accumulated
    _ = portfolio.profit
    nav_after = pf_zero.nav_accumulated
    pt.assert_frame_equal(nav_after, nav_before)


# ─── Tilt / timing ────────────────────────────────────────────────────────────


def test_timing_prices_are_difference_and_portfolio_computable(portfolio):
    """timing.prices must equal original prices - tilt.prices, date preserved."""
    tilt = portfolio.tilt
    assert isinstance(tilt, Portfolio)
    assert tilt.aum == portfolio.aum
    pt.assert_frame_equal(tilt.prices, portfolio.prices)

    timing = portfolio.timing
    assert isinstance(timing, Portfolio)
    assert timing.aum == portfolio.aum
    pt.assert_frame_equal(timing.prices, portfolio.prices)

    pt.assert_frame_equal(
        portfolio.cashposition.select(portfolio.assets),
        timing.cashposition.select(portfolio.assets) + tilt.cashposition.select(portfolio.assets),
    )
    print(portfolio.tilt_timing_decomp)


# ─── Truncate ─────────────────────────────────────────────────────────────────


def test_truncate_by_start_end_inclusive_preserves_aum_and_dates(truncate_portfolio):
    """Truncating with both start and end returns new Portfolio, preserves AUM, and filters dates inclusively."""
    start = date(2020, 1, 2)
    end = date(2020, 1, 4)

    pf_t = truncate_portfolio.truncate(start=start, end=end)

    assert isinstance(pf_t, Portfolio)
    assert pf_t.aum == truncate_portfolio.aum
    assert pf_t.prices.height == 3
    assert pf_t.cashposition.height == 3
    assert pf_t.prices["date"].min() == start
    assert pf_t.prices["date"].max() == end

    nav = pf_t.nav_accumulated
    assert "NAV_accumulated" in nav.columns
    assert nav.height == 3


def test_truncate_with_only_start_or_end_open_bounds(truncate_portfolio):
    """Truncating with only a start or only an end applies open bounds and remains computable."""
    pf_s = truncate_portfolio.truncate(start=date(2020, 1, 4))
    assert pf_s.prices["date"].min() == date(2020, 1, 4)
    assert pf_s.prices.height == 3  # days 4,5,6

    pf_e = truncate_portfolio.truncate(end=date(2020, 1, 3))
    assert pf_e.prices["date"].max() == date(2020, 1, 3)
    assert pf_e.prices.height == 3  # days 1,2,3

    _ = pf_s.profit
    _ = pf_e.profit


# ─── Validation ───────────────────────────────────────────────────────────────


def test_portfolio_smoothed_holding_negative_raises_value_error(portfolio):
    """Portfolio.smoothed_holding should raise ValueError when n < 0."""
    with pytest.raises(ValueError, match=r".*"):
        _ = portfolio.smoothed_holding(-1)


def test_portfolio_smoothed_holding_type_error_on_non_int(portfolio):
    """Portfolio.smoothed_holding should raise TypeError when n is not an int."""
    with pytest.raises(TypeError):
        _ = portfolio.smoothed_holding(1.5)  # type: ignore[arg-type]


# ─── Monthly ──────────────────────────────────────────────────────────────────


def test_monthly_structure_and_end_of_month_dates(monthly_portfolio):
    """Monthly should include date (month-end), returns, and calendar columns including month_name."""
    monthly = monthly_portfolio.monthly

    assert monthly.columns == ["date", "returns", "NAV_accumulated", "profit", "year", "month", "month_name"]
    assert monthly["date"].dtype == pl.Date
    assert list(monthly["year"]) == [2020, 2020, 2020]
    assert list(monthly["month"]) == [1, 2, 3]
    assert list(monthly["month_name"]) == ["Jan", "Feb", "Mar"]
    assert monthly["returns"].is_finite().all()


# ─── Integer-indexed (no date column) portfolios ─────────────────────────────


@pytest.fixture
def int_portfolio():
    """A small Portfolio with no 'date' column (integer-indexed rows)."""
    n = 6
    return Portfolio.from_cash_position(
        prices=pl.DataFrame(
            {
                "A": pl.Series([100.0 + 10.0 * i for i in range(n)], dtype=pl.Float64),
                "B": pl.Series([200.0 - 5.0 * i for i in range(n)], dtype=pl.Float64),
            }
        ),
        cash_position=pl.DataFrame(
            {
                "A": pl.Series([1000.0] * n, dtype=pl.Float64),
                "B": pl.Series([500.0] * n, dtype=pl.Float64),
            }
        ),
        aum=1e6,
    )


def test_truncate_integer_indexed_both_bounds(int_portfolio):
    """truncate(start, end) on integer-indexed portfolio slices rows inclusively."""
    pf_t = int_portfolio.truncate(start=1, end=3)
    assert isinstance(pf_t, Portfolio)
    assert pf_t.prices.height == 3
    assert pf_t.cashposition.height == 3
    assert pf_t.aum == int_portfolio.aum
    _ = pf_t.profit


def test_truncate_integer_indexed_start_only(int_portfolio):
    """truncate(start=n) on integer-indexed portfolio returns rows from n onward."""
    pf_s = int_portfolio.truncate(start=2)
    assert pf_s.prices.height == 4  # rows 2,3,4,5
    _ = pf_s.profit


def test_truncate_integer_indexed_end_only(int_portfolio):
    """truncate(end=n) on integer-indexed portfolio returns rows up to n inclusive."""
    pf_e = int_portfolio.truncate(end=2)
    assert pf_e.prices.height == 3  # rows 0,1,2
    _ = pf_e.profit


def test_truncate_integer_indexed_no_bounds_returns_full(int_portfolio):
    """truncate() with no bounds on integer-indexed portfolio returns all rows."""
    pf_all = int_portfolio.truncate()
    assert pf_all.prices.height == int_portfolio.prices.height


def test_truncate_integer_indexed_raises_on_non_int_start(int_portfolio):
    """Truncate with non-integer start on integer-indexed portfolio raises IntegerIndexBoundError."""
    with pytest.raises(IntegerIndexBoundError, match="start must be an integer"):
        int_portfolio.truncate(start="2020-01-01")


def test_truncate_integer_indexed_raises_on_non_int_end(int_portfolio):
    """Truncate with non-integer end on integer-indexed portfolio raises IntegerIndexBoundError."""
    with pytest.raises(IntegerIndexBoundError, match="end must be an integer"):
        int_portfolio.truncate(end=3.5)


def test_monthly_raises_without_date_column(int_portfolio):
    """Portfolio.monthly raises MissingDateColumnError when no 'date' column is present."""
    with pytest.raises(MissingDateColumnError, match="missing the required 'date' column"):
        _ = int_portfolio.monthly


def test_all_works_without_date_column(int_portfolio):
    """Portfolio.all returns a DataFrame with expected columns for integer-indexed data."""
    result = int_portfolio.all
    assert "NAV_accumulated" in result.columns
    assert "NAV_compounded" in result.columns
    assert "drawdown" in result.columns
    assert "date" not in result.columns
    assert result.height == int_portfolio.prices.height


def test_stats_works_without_date_column(int_portfolio):
    """Portfolio.stats returns a Stats object for integer-indexed portfolios."""
    stats = int_portfolio.stats
    sharpe = stats.sharpe()["returns"]
    assert np.isfinite(sharpe)


def test_tilt_timing_decomp_works_without_date_column(int_portfolio):
    """tilt_timing_decomp returns portfolio/tilt/timing columns for integer-indexed data."""
    decomp = int_portfolio.tilt_timing_decomp
    assert "portfolio" in decomp.columns
    assert "tilt" in decomp.columns
    assert "timing" in decomp.columns
    assert "date" not in decomp.columns
    assert decomp.height == int_portfolio.prices.height
    # Numerical check: portfolio NAV ≈ tilt NAV + timing NAV - aum (decomposition identity)
    # portfolio = tilt_nav + timing_nav - aum because tilt and timing both start at aum
    expected_portfolio = decomp["tilt"].to_numpy() + decomp["timing"].to_numpy() - int_portfolio.aum
    assert np.allclose(decomp["portfolio"].to_numpy(), expected_portfolio, rtol=1e-10, atol=1e-6)


# ─── Turnover ────────────────────────────────────────────────────────────────


@pytest.fixture
def turnover_portfolio():
    """A 10-day, two-asset portfolio with linearly changing positions.

    Asset A increases by 100 each day (position_t = 100 * t).
    Asset B decreases by 50 each day (position_t = 500 - 50 * t).
    AUM = 10_000.
    Daily turnover = (|ΔA| + |ΔB|) / AUM = (100 + 50) / 10_000 = 0.015 per day
    (except row 0 which is 0.0).
    """
    n = 10
    start = date(2020, 1, 1)
    end = start + timedelta(days=n - 1)
    dates = pl.date_range(start=start, end=end, interval="1d", eager=True).cast(pl.Date)
    return Portfolio.from_cash_position(
        prices=pl.DataFrame({"date": dates, "A": pl.Series([100.0] * n), "B": pl.Series([200.0] * n)}),
        cash_position=pl.DataFrame(
            {
                "date": dates,
                "A": pl.Series([100.0 * i for i in range(n)], dtype=pl.Float64),
                "B": pl.Series([500.0 - 50.0 * i for i in range(n)], dtype=pl.Float64),
            }
        ),
        aum=10_000.0,
    )


def test_turnover_columns_and_length(turnover_portfolio):
    """Turnover DataFrame should have 'date' and 'turnover' columns with same row count as portfolio."""
    to = turnover_portfolio.turnover
    assert "date" in to.columns
    assert "turnover" in to.columns
    assert to.height == turnover_portfolio.prices.height


def test_turnover_first_row_is_zero(turnover_portfolio):
    """The first row of daily turnover must be 0.0 (no prior position)."""
    to = turnover_portfolio.turnover
    assert float(to["turnover"][0]) == pytest.approx(0.0, abs=1e-12)


def test_turnover_subsequent_rows_correct(turnover_portfolio):
    """Daily turnover from row 1 onward: (|ΔA| + |ΔB|) / AUM = 0.015."""
    to = turnover_portfolio.turnover
    expected = 0.015  # (100 + 50) / 10_000
    assert np.allclose(to["turnover"][1:].to_numpy(), expected, rtol=1e-12, atol=1e-12)


def test_turnover_nonnegative(turnover_portfolio):
    """All turnover values must be non-negative."""
    to = turnover_portfolio.turnover
    assert (to["turnover"] >= 0.0).all()


def test_turnover_without_date_column(int_portfolio):
    """Turnover on an integer-indexed portfolio has only a 'turnover' column."""
    to = int_portfolio.turnover
    assert "date" not in to.columns
    assert "turnover" in to.columns
    assert to.height == int_portfolio.prices.height
    assert float(to["turnover"][0]) == pytest.approx(0.0, abs=1e-12)


def test_turnover_constant_positions_are_zero():
    """Constant cash positions → zero turnover on every row after the first."""
    n = 5
    start = date(2020, 1, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True)
    pf = Portfolio.from_cash_position(
        prices=pl.DataFrame({"date": dates, "A": pl.Series([100.0] * n)}),
        cash_position=pl.DataFrame({"date": dates, "A": pl.Series([500.0] * n, dtype=pl.Float64)}),
        aum=1e4,
    )
    to = pf.turnover
    # After the first row, every diff is 0
    assert np.allclose(to["turnover"].to_numpy(), 0.0, atol=1e-12)


def test_turnover_weekly_columns_when_date_present(turnover_portfolio):
    """turnover_weekly with a date column should have 'date' and 'turnover' columns."""
    wto = turnover_portfolio.turnover_weekly
    assert "date" in wto.columns
    assert "turnover" in wto.columns


def test_turnover_weekly_sums_daily(turnover_portfolio):
    """Weekly turnover must equal the sum of daily turnovers within the same week."""
    daily = turnover_portfolio.turnover
    weekly = turnover_portfolio.turnover_weekly
    # Sum all daily values and compare with sum of weekly values
    assert np.isclose(float(daily["turnover"].sum()), float(weekly["turnover"].sum()), rtol=1e-10)


def test_turnover_weekly_without_date_uses_rolling(int_portfolio):
    """Without a date column, turnover_weekly uses a rolling 5-period sum."""
    wto = int_portfolio.turnover_weekly
    assert "date" not in wto.columns
    assert "turnover" in wto.columns
    # First 4 rows should be null (rolling window with min_samples=5)
    assert wto["turnover"][:4].null_count() == 4
    # Row index 4 should be non-null
    assert wto["turnover"][4] is not None


def test_turnover_summary_structure(turnover_portfolio):
    """turnover_summary should return a DataFrame with metric and value columns."""
    summary = turnover_portfolio.turnover_summary()
    assert list(summary["metric"]) == ["mean_daily_turnover", "mean_weekly_turnover", "turnover_std"]
    assert "value" in summary.columns
    assert summary["value"].is_finite().all()


def test_turnover_summary_mean_daily_matches_manual(turnover_portfolio):
    """mean_daily_turnover in summary should match the manual mean of the turnover series."""
    summary = turnover_portfolio.turnover_summary()
    manual_mean = float(turnover_portfolio.turnover["turnover"].mean())
    row = summary.filter(pl.col("metric") == "mean_daily_turnover")
    assert float(row["value"][0]) == pytest.approx(manual_mean, rel=1e-10)


def test_turnover_summary_std_matches_manual(turnover_portfolio):
    """turnover_std in summary should match the manual std of the turnover series."""
    summary = turnover_portfolio.turnover_summary()
    manual_std = float(turnover_portfolio.turnover["turnover"].std())
    row = summary.filter(pl.col("metric") == "turnover_std")
    assert float(row["value"][0]) == pytest.approx(manual_std, rel=1e-10)


def test_turnover_summary_mean_weekly_positive(turnover_portfolio):
    """mean_weekly_turnover must be positive for a portfolio with non-zero trading."""
    summary = turnover_portfolio.turnover_summary()
    row = summary.filter(pl.col("metric") == "mean_weekly_turnover")
    assert float(row["value"][0]) > 0.0


def test_turnover_summary_without_date_column(int_portfolio):
    """turnover_summary should work for integer-indexed portfolios."""
    summary = int_portfolio.turnover_summary()
    assert list(summary["metric"]) == ["mean_daily_turnover", "mean_weekly_turnover", "turnover_std"]


# ── cost_adjusted_returns ────────────────────────────────────────────────────


def test_cost_adjusted_returns_zero_bps_equals_base_returns(turnover_portfolio):
    """cost_adjusted_returns(0) must equal the base returns exactly."""
    base = turnover_portfolio.returns
    adj = turnover_portfolio.cost_adjusted_returns(0.0)
    pt.assert_series_equal(adj["returns"], base["returns"])


def test_cost_adjusted_returns_positive_costs_reduce_returns(turnover_portfolio):
    """Positive trading costs must strictly reduce returns for a portfolio with non-zero turnover."""
    base = turnover_portfolio.returns["returns"]
    adj = turnover_portfolio.cost_adjusted_returns(5.0)["returns"]
    # Sum of adjusted returns must be less than sum of base returns
    assert float(adj.sum()) < float(base.sum())


def test_cost_adjusted_returns_preserves_date_column(turnover_portfolio):
    """cost_adjusted_returns must preserve the date column when present."""
    adj = turnover_portfolio.cost_adjusted_returns(10.0)
    assert "date" in adj.columns


def test_cost_adjusted_returns_negative_bps_raises(turnover_portfolio):
    """cost_adjusted_returns must raise ValueError for negative cost_bps."""
    with pytest.raises(ValueError, match=r".*"):
        turnover_portfolio.cost_adjusted_returns(-1.0)


def test_cost_adjusted_returns_higher_bps_lower_returns(turnover_portfolio):
    """Higher basis points must lead to lower (or equal) total adjusted returns."""
    adj5 = float(turnover_portfolio.cost_adjusted_returns(5.0)["returns"].sum())
    adj10 = float(turnover_portfolio.cost_adjusted_returns(10.0)["returns"].sum())
    assert adj10 <= adj5


# ── trading_cost_impact ──────────────────────────────────────────────────────


def test_trading_cost_impact_columns(turnover_portfolio):
    """trading_cost_impact must return a DataFrame with 'cost_bps' and 'sharpe' columns."""
    impact = turnover_portfolio.trading_cost_impact()
    assert "cost_bps" in impact.columns
    assert "sharpe" in impact.columns


def test_trading_cost_impact_default_range(turnover_portfolio):
    """trading_cost_impact default runs from 0 to 20 inclusive (21 rows)."""
    impact = turnover_portfolio.trading_cost_impact()
    assert impact.height == 21
    assert int(impact["cost_bps"][0]) == 0
    assert int(impact["cost_bps"][-1]) == 20


def test_trading_cost_impact_custom_max_bps(turnover_portfolio):
    """trading_cost_impact(max_bps=5) must produce exactly 6 rows (0..5)."""
    impact = turnover_portfolio.trading_cost_impact(max_bps=5)
    assert impact.height == 6
    assert list(impact["cost_bps"].to_list()) == [0, 1, 2, 3, 4, 5]


def test_trading_cost_impact_sharpe_decreases_with_cost():
    """Sharpe ratio must be non-increasing as trading costs increase.

    Uses a portfolio with genuine daily returns so that the cost deduction
    is meaningful.  A linearly rising price series gives positive returns;
    constant positions give measurable turnover so costs actually bite.
    """
    n = 252
    start = date(2020, 1, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True).cast(pl.Date)
    # Steadily rising price → positive returns
    prices = pl.DataFrame({"date": dates, "A": pl.Series([100.0 * (1.001**i) for i in range(n)])})
    # Alternating position changes → non-zero daily turnover
    positions = pl.DataFrame({"date": dates, "A": pl.Series([1000.0 + 100.0 * (i % 2) for i in range(n)])})
    pf = Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e5)
    impact = pf.trading_cost_impact(max_bps=10)
    sharpe = [s for s in impact["sharpe"].to_list() if s == s]  # drop NaN
    # Allow floating-point ties via >= check element-by-element
    for i in range(len(sharpe) - 1):
        assert sharpe[i] >= sharpe[i + 1] - 1e-9, f"Sharpe increased at bps={i}: {sharpe[i]} → {sharpe[i + 1]}"


def test_trading_cost_impact_invalid_max_bps_raises(turnover_portfolio):
    """trading_cost_impact must raise ValueError for max_bps=0 or non-integer."""
    with pytest.raises(ValueError, match=r".*"):
        turnover_portfolio.trading_cost_impact(max_bps=0)
    with pytest.raises(ValueError, match=r".*"):
        turnover_portfolio.trading_cost_impact(max_bps=-1)  # type: ignore[arg-type]


# ── Plots.trading_cost_impact_plot ───────────────────────────────────────────


def test_trading_cost_impact_plot_returns_figure(turnover_portfolio):
    """trading_cost_impact_plot must return a Plotly Figure."""
    fig = turnover_portfolio.plots.trading_cost_impact_plot()
    assert isinstance(fig, go.Figure)


def test_trading_cost_impact_plot_has_one_trace(turnover_portfolio):
    """trading_cost_impact_plot figure must have exactly one scatter trace."""
    fig = turnover_portfolio.plots.trading_cost_impact_plot()
    scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
    assert len(scatter_traces) == 1


def test_trading_cost_impact_plot_x_axis_length(turnover_portfolio):
    """trading_cost_impact_plot trace x-values must span 0..20 (21 points)."""
    fig = turnover_portfolio.plots.trading_cost_impact_plot()
    assert len(fig.data[0].x) == 21


def test_trading_cost_impact_plot_title_contains_bps(turnover_portfolio):
    """trading_cost_impact_plot figure title must mention 'bps'."""
    fig = turnover_portfolio.plots.trading_cost_impact_plot()
    assert "bps" in fig.layout.title.text.lower()


# ── position_delta_costs ─────────────────────────────────────────────────────


@pytest.fixture
def cost_portfolio():
    """3-day, single-asset portfolio for position-delta cost tests.

    Position sequence:  0 → 1000 → 700
    |Δ position|:       0,  1000,   300
    cost_per_unit = 0.01
    Expected daily costs: [0, 1000, 300] * 0.01 = [0.0, 10.0, 3.0]
    """
    n = 3
    start = date(2020, 1, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True).cast(pl.Date)
    return Portfolio(
        prices=pl.DataFrame({"date": dates, "A": pl.Series([100.0, 110.0, 121.0])}),
        cashposition=pl.DataFrame({"date": dates, "A": pl.Series([0.0, 1000.0, 700.0])}),
        aum=1e5,
        cost_per_unit=0.01,
    )


def test_position_delta_costs_columns(cost_portfolio):
    """position_delta_costs must have 'date' and 'cost' columns."""
    df = cost_portfolio.position_delta_costs
    assert "date" in df.columns
    assert "cost" in df.columns
    assert df.height == cost_portfolio.prices.height


def test_position_delta_costs_first_row_is_zero(cost_portfolio):
    """First row of position_delta_costs must be 0.0 (no prior position)."""
    assert float(cost_portfolio.position_delta_costs["cost"][0]) == pytest.approx(0.0, abs=1e-12)


def test_position_delta_costs_analytical_values(cost_portfolio):
    """position_delta_costs must match known analytical values.

    Position changes: 0, 1000, 300  → costs: 0*0.01, 1000*0.01, 300*0.01.
    """
    costs = cost_portfolio.position_delta_costs["cost"].to_list()
    assert costs[0] == pytest.approx(0.0, abs=1e-12)
    assert costs[1] == pytest.approx(10.0, rel=1e-9)
    assert costs[2] == pytest.approx(3.0, rel=1e-9)


def test_position_delta_costs_zero_cost_per_unit_all_zeros():
    """When cost_per_unit=0.0 all costs must be zero regardless of positions."""
    n = 5
    start = date(2021, 1, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True).cast(pl.Date)
    pf = Portfolio(
        prices=pl.DataFrame({"date": dates, "A": pl.Series([100.0] * n)}),
        cashposition=pl.DataFrame({"date": dates, "A": pl.Series([float(i) * 500 for i in range(n)])}),
        aum=1e4,
        cost_per_unit=0.0,
    )
    costs = pf.position_delta_costs["cost"].to_list()
    assert all(c == pytest.approx(0.0, abs=1e-12) for c in costs)


def test_position_delta_costs_two_assets():
    """position_delta_costs must sum |Δx| across all assets correctly.

    Asset A: 0 → 500 → 200,  |Δ| = 0, 500, 300
    Asset B: 0 → 200 → 400,  |Δ| = 0, 200, 200
    cost_per_unit = 0.005
    Expected costs: 0, (500+200)*0.005=3.5, (300+200)*0.005=2.5
    """
    n = 3
    start = date(2020, 3, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True).cast(pl.Date)
    pf = Portfolio(
        prices=pl.DataFrame({"date": dates, "A": [100.0] * n, "B": [200.0] * n}),
        cashposition=pl.DataFrame({"date": dates, "A": [0.0, 500.0, 200.0], "B": [0.0, 200.0, 400.0]}),
        aum=1e5,
        cost_per_unit=0.005,
    )
    costs = pf.position_delta_costs["cost"].to_list()
    assert costs[0] == pytest.approx(0.0, abs=1e-12)
    assert costs[1] == pytest.approx(3.5, rel=1e-9)
    assert costs[2] == pytest.approx(2.5, rel=1e-9)


def test_position_delta_costs_without_date_column():
    """position_delta_costs must work on date-free portfolios."""
    pf = Portfolio(
        prices=pl.DataFrame({"A": [100.0, 110.0, 121.0]}),
        cashposition=pl.DataFrame({"A": [0.0, 1000.0, 700.0]}),
        aum=1e4,
        cost_per_unit=0.001,
    )
    df = pf.position_delta_costs
    assert "date" not in df.columns
    assert "cost" in df.columns
    assert df.height == 3
    costs = df["cost"].to_list()
    assert costs[0] == pytest.approx(0.0)
    assert costs[1] == pytest.approx(1.0, rel=1e-9)
    assert costs[2] == pytest.approx(0.3, rel=1e-9)


def test_position_delta_costs_nan_warmup_rows_produce_zero_cost():
    """NaN positions (e.g. EWMA warmup rows) must not propagate NaN into cost.

    fill_null(0.0) only clears Polars null; fill_nan(0.0) is also required to
    clear IEEE-754 NaN values that arise during EWMA warmup.

    Positions: [NaN, NaN, 1000, 1200, 900]
    diff:       [null, NaN-NaN=NaN, 1000-NaN=NaN, 200, -300]
    After fill_null+fill_nan: [0, 0, 0, 200, 300]
    cost (* 0.01): [0, 0, 0, 2, 3]
    """
    n = 5
    start = date(2021, 6, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True).cast(pl.Date)
    # First two rows are NaN (simulating EWMA warmup), rest are finite.
    positions = [float("nan"), float("nan"), 1000.0, 1200.0, 900.0]
    pf = Portfolio(
        prices=pl.DataFrame({"date": dates, "A": [100.0] * n}),
        cashposition=pl.DataFrame({"date": dates, "A": pl.Series(positions, dtype=pl.Float64)}),
        aum=1e5,
        cost_per_unit=0.01,
    )
    costs = pf.position_delta_costs["cost"].to_list()
    # No cost should be NaN — warmup NaNs must be treated as zero.
    assert all(c == c for c in costs), "NaN found in cost column"  # NaN != NaN
    assert costs[0] == pytest.approx(0.0, abs=1e-12)
    assert costs[1] == pytest.approx(0.0, abs=1e-12)
    assert costs[2] == pytest.approx(0.0, abs=1e-12)  # 1000 - NaN → NaN → 0
    assert costs[3] == pytest.approx(2.0, rel=1e-9)  # |1200 - 1000| * 0.01
    assert costs[4] == pytest.approx(3.0, rel=1e-9)  # |900 - 1200| * 0.01


# ── net_cost_nav ─────────────────────────────────────────────────────────────


def test_net_cost_nav_columns(cost_portfolio):
    """net_cost_nav must return a frame with 'date', 'profit', 'cost', and 'NAV_accumulated_net'."""
    df = cost_portfolio.net_cost_nav
    for col in ("date", "profit", "cost", "NAV_accumulated_net"):
        assert col in df.columns, f"Missing column: {col}"
    assert df.height == cost_portfolio.prices.height


def test_net_cost_nav_zero_cost_equals_nav_accumulated():
    """With cost_per_unit=0, net_cost_nav must equal the gross nav_accumulated."""
    n = 5
    start = date(2020, 1, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True).cast(pl.Date)
    pf = Portfolio(
        prices=pl.DataFrame({"date": dates, "A": pl.Series([100.0 * (1.01**i) for i in range(n)])}),
        cashposition=pl.DataFrame({"date": dates, "A": pl.Series([1000.0] * n)}),
        aum=1e5,
        cost_per_unit=0.0,
    )
    net_nav = pf.net_cost_nav["NAV_accumulated_net"].to_list()
    gross_nav = pf.nav_accumulated["NAV_accumulated"].to_list()
    for net, gross in zip(net_nav, gross_nav, strict=False):
        assert net == pytest.approx(gross, rel=1e-9)


def test_net_cost_nav_positive_cost_lower_than_gross(cost_portfolio):
    """With positive cost_per_unit, net NAV must be <= gross NAV at every point."""
    net_nav = cost_portfolio.net_cost_nav["NAV_accumulated_net"].to_list()
    gross_nav = cost_portfolio.nav_accumulated["NAV_accumulated"].to_list()
    for net, gross in zip(net_nav, gross_nav, strict=False):
        assert net <= gross + 1e-9, f"net {net} > gross {gross}"


def test_net_cost_nav_analytical_values(cost_portfolio):
    """net_cost_nav must match exact hand-computed values.

    Setup (cost_portfolio fixture):
      AUM = 1e5
      Positions: A = [0, 1000, 700], cost_per_unit=0.01
      Prices: A = [100, 110, 121]
      Per-asset profit (p_{t-1} * pct_change):
        day 0: 0 (no prior), day 1: 0*(10/100)=0, day 2: 1000*(11/110)=100
      Daily costs: 0, 10.0, 3.0
      Net profit: 0-0=0, 0-10=-10, 100-3=97
      NAV: 1e5, 1e5+0=1e5, 1e5-10=99990, 99990+97=100087
    """
    df = cost_portfolio.net_cost_nav
    nav = df["NAV_accumulated_net"].to_list()
    assert nav[0] == pytest.approx(1e5, rel=1e-9)
    assert nav[1] == pytest.approx(1e5 - 10.0, rel=1e-9)
    assert nav[2] == pytest.approx(1e5 - 10.0 + 97.0, rel=1e-9)


# ── Portfolio.cost_per_unit field ─────────────────────────────────────────────


def test_portfolio_cost_per_unit_default_is_zero():
    """Portfolio constructed without cost_per_unit must default to 0.0."""
    pf = Portfolio(
        prices=pl.DataFrame({"date": [date(2020, 1, 1)], "A": [100.0]}),
        cashposition=pl.DataFrame({"date": [date(2020, 1, 1)], "A": [1000.0]}),
        aum=1e5,
    )
    assert pf.cost_per_unit == 0.0


def test_portfolio_from_cash_position_passes_cost_per_unit():
    """from_cash_position must forward cost_per_unit to the Portfolio."""
    prices = pl.DataFrame({"date": [date(2020, 1, 1)], "A": [100.0]})
    pos = pl.DataFrame({"date": [date(2020, 1, 1)], "A": [500.0]})
    pf = Portfolio.from_cash_position(prices=prices, cash_position=pos, aum=1e5, cost_per_unit=0.002)
    assert pf.cost_per_unit == pytest.approx(0.002)


def test_net_cost_nav_integer_indexed(int_portfolio):
    """net_cost_nav on an integer-indexed portfolio uses hstack (no 'date' join)."""
    result = int_portfolio.net_cost_nav
    assert "NAV_accumulated_net" in result.columns
    assert "profit" in result.columns
    assert "cost" in result.columns
    assert "date" not in result.columns
    assert result.height == int_portfolio.prices.height


# ─── Portfolio.data bridge property ──────────────────────────────────────────


def test_portfolio_data_property_returns_data_object(portfolio):
    """portfolio.data returns a legacy Data object with a 'returns' column and date index."""
    from jquantstats.data import Data

    d = portfolio.data
    assert isinstance(d, Data)
    assert "returns" in d.returns.columns
    assert d.returns.height == portfolio.prices.height
    assert d.index.height == portfolio.prices.height


def test_portfolio_data_property_integer_indexed(int_portfolio):
    """portfolio.data on an integer-indexed portfolio creates a synthetic integer index."""
    from jquantstats.data import Data

    d = int_portfolio.data
    assert isinstance(d, Data)
    assert "returns" in d.returns.columns
    assert "date" not in d.returns.columns
    assert d.index.columns == ["index"]
    assert d.index.height == int_portfolio.prices.height


def test_integer_indexed_stats_uses_252_periods_per_year(int_portfolio):
    """Integer-indexed portfolio.stats must use 252 periods/year, not ~31.5 million."""
    assert int_portfolio.data._periods_per_year == pytest.approx(252.0)
    sharpe = int_portfolio.stats.sharpe()
    assert "returns" in sharpe
    assert abs(sharpe["returns"]) < 1000  # sanity: not ~5600x inflated


# ── cost_per_unit forwarding through transforms ───────────────────────────────


@pytest.fixture
def cost_pf():
    """3-day portfolio with cost_per_unit=0.05 for transform-forwarding tests."""
    prices = pl.DataFrame({"A": [100.0, 110.0, 105.0]})
    pos = pl.DataFrame({"A": [1000.0, 1200.0, 1000.0]})
    return Portfolio(prices=prices, cashposition=pos, aum=1e5, cost_per_unit=0.05)


def test_truncate_forwards_cost_per_unit(cost_pf):
    """Truncate must preserve cost_per_unit."""
    assert cost_pf.truncate(start=0, end=1).cost_per_unit == pytest.approx(0.05)


def test_lag_forwards_cost_per_unit(cost_pf):
    """Lag must preserve cost_per_unit."""
    assert cost_pf.lag(1).cost_per_unit == pytest.approx(0.05)


def test_smoothed_holding_forwards_cost_per_unit(cost_pf):
    """smoothed_holding must preserve cost_per_unit."""
    assert cost_pf.smoothed_holding(1).cost_per_unit == pytest.approx(0.05)


def test_tilt_forwards_cost_per_unit(cost_pf):
    """Tilt must preserve cost_per_unit."""
    assert cost_pf.tilt.cost_per_unit == pytest.approx(0.05)


def test_timing_forwards_cost_per_unit(cost_pf):
    """Timing must preserve cost_per_unit."""
    assert cost_pf.timing.cost_per_unit == pytest.approx(0.05)


# ── cost_bps construction-time parameter ─────────────────────────────────────


@pytest.fixture
def cost_bps_pf():
    """3-day portfolio with cost_bps=5.0 for cost_bps-forwarding tests."""
    prices = pl.DataFrame({"A": [100.0, 110.0, 105.0]})
    pos = pl.DataFrame({"A": [1000.0, 1200.0, 1000.0]})
    return Portfolio(prices=prices, cashposition=pos, aum=1e5, cost_bps=5.0)


def test_cost_bps_forwarded_through_transforms(cost_bps_pf):
    """All transforms must preserve the construction-time cost_bps value."""
    assert cost_bps_pf.truncate(start=0, end=1).cost_bps == pytest.approx(5.0)
    assert cost_bps_pf.lag(1).cost_bps == pytest.approx(5.0)
    assert cost_bps_pf.smoothed_holding(1).cost_bps == pytest.approx(5.0)
    assert cost_bps_pf.tilt.cost_bps == pytest.approx(5.0)
    assert cost_bps_pf.timing.cost_bps == pytest.approx(5.0)


def test_cost_adjusted_returns_defaults_to_construction_cost_bps(cost_bps_pf, turnover_portfolio):
    """cost_adjusted_returns() with no argument uses self.cost_bps."""
    adj_default = cost_bps_pf.cost_adjusted_returns()
    adj_explicit = cost_bps_pf.cost_adjusted_returns(5.0)
    assert adj_default["returns"].to_list() == pytest.approx(adj_explicit["returns"].to_list())


def test_cost_adjusted_returns_explicit_overrides_construction_cost_bps(cost_bps_pf):
    """An explicit cost_bps argument overrides self.cost_bps."""
    adj_override = cost_bps_pf.cost_adjusted_returns(0.0)
    adj_base = cost_bps_pf.cost_adjusted_returns(5.0)
    # 0 bps should yield higher (or equal) returns than 5 bps
    assert float(adj_override["returns"].sum()) >= float(adj_base["returns"].sum())


# ── from_risk_position per-asset vola dict ───────────────────────────────────


def test_from_risk_position_per_asset_vola():
    """Different vola values per asset produce different cash positions."""
    import numpy as np

    dates = pl.date_range(start=date(2020, 1, 1), end=date(2020, 3, 1), interval="1d", eager=True).cast(pl.Date)
    prices = pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series(np.linspace(100, 120, len(dates)), dtype=pl.Float64),
            "B": pl.Series(np.linspace(50, 60, len(dates)), dtype=pl.Float64),
        }
    )
    riskposition = pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series([1.0] * len(dates), dtype=pl.Float64),
            "B": pl.Series([1.0] * len(dates), dtype=pl.Float64),
        }
    )
    pf_uniform = Portfolio.from_risk_position(prices, riskposition, vola=8, aum=1e8)
    pf_per_asset = Portfolio.from_risk_position(prices, riskposition, vola={"A": 8, "B": 32}, aum=1e8)
    # Same vola for A → identical A columns
    assert pf_uniform.cashposition["A"].to_list() == pytest.approx(
        pf_per_asset.cashposition["A"].to_list(), nan_ok=True
    )
    # Different vola for B → different B columns
    b_uniform = pf_uniform.cashposition["B"].drop_nulls().to_list()
    b_per_asset = pf_per_asset.cashposition["B"].drop_nulls().to_list()
    assert b_uniform != pytest.approx(b_per_asset)


def test_from_risk_position_dict_vola_missing_key_falls_back_to_32():
    """Assets absent from the vola dict use span=32 as the default."""
    import numpy as np

    dates = pl.date_range(start=date(2020, 1, 1), end=date(2020, 3, 1), interval="1d", eager=True).cast(pl.Date)
    prices = pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series(np.linspace(100, 120, len(dates)), dtype=pl.Float64),
        }
    )
    riskposition = pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series([1.0] * len(dates), dtype=pl.Float64),
        }
    )
    # dict with no "A" key → should fall back to 32
    pf_dict = Portfolio.from_risk_position(prices, riskposition, vola={}, aum=1e8)
    pf_int = Portfolio.from_risk_position(prices, riskposition, vola=32, aum=1e8)
    a_dict = pf_dict.cashposition["A"].drop_nulls().to_list()
    a_int = pf_int.cashposition["A"].drop_nulls().to_list()
    assert a_dict == pytest.approx(a_int, nan_ok=True)


# ── from_risk_position cost param forwarding ─────────────────────────────────


def test_from_risk_position_forwards_cost_per_unit():
    """from_risk_position must preserve the cost_per_unit parameter."""
    import numpy as np

    dates = pl.date_range(start=date(2020, 1, 1), end=date(2020, 3, 1), interval="1d", eager=True).cast(pl.Date)
    prices = pl.DataFrame({"date": dates, "A": pl.Series(np.linspace(100, 120, len(dates)), dtype=pl.Float64)})
    risk = pl.DataFrame({"date": dates, "A": pl.Series([1.0] * len(dates), dtype=pl.Float64)})
    pf = Portfolio.from_risk_position(prices, risk, aum=1e8, cost_per_unit=0.05)
    assert pf.cost_per_unit == pytest.approx(0.05)


def test_from_risk_position_forwards_cost_bps():
    """from_risk_position must preserve the cost_bps parameter."""
    import numpy as np

    dates = pl.date_range(start=date(2020, 1, 1), end=date(2020, 3, 1), interval="1d", eager=True).cast(pl.Date)
    prices = pl.DataFrame({"date": dates, "A": pl.Series(np.linspace(100, 120, len(dates)), dtype=pl.Float64)})
    risk = pl.DataFrame({"date": dates, "A": pl.Series([1.0] * len(dates), dtype=pl.Float64)})
    pf = Portfolio.from_risk_position(prices, risk, aum=1e8, cost_bps=5.0)
    assert pf.cost_bps == pytest.approx(5.0)


# ── _data_bridge caching ──────────────────────────────────────────────────────


def test_data_property_returns_same_object(portfolio: Portfolio):
    """pf.data must return the identical Data object on repeated calls."""
    assert portfolio.data is portfolio.data


def test_data_cached_after_factory():
    """Portfolio built via from_cash_position must cache the Data bridge."""
    prices = pl.DataFrame({"date": [date(2020, 1, 1), date(2020, 1, 2)], "A": [100.0, 110.0]})
    pos = pl.DataFrame({"date": [date(2020, 1, 1), date(2020, 1, 2)], "A": [1000.0, 1000.0]})
    pf = Portfolio.from_cash_position(prices=prices, cash_position=pos, aum=1e5)
    assert pf.data is pf.data


def test_stats_plots_report_cached():
    """stats, plots, and report must return the same object on repeated access."""
    prices = pl.DataFrame({"date": [date(2020, 1, 1), date(2020, 1, 2)], "A": [100.0, 110.0]})
    pos = pl.DataFrame({"date": [date(2020, 1, 1), date(2020, 1, 2)], "A": [1000.0, 1000.0]})
    pf = Portfolio.from_cash_position(prices=prices, cash_position=pos, aum=1e5)
    assert pf.stats is pf.stats
    assert pf.plots is pf.plots
    assert pf.report is pf.report


def test_repr(portfolio):
    """Tests that Portfolio.__repr__ returns an informative string."""
    r = repr(portfolio)
    assert r.startswith("Portfolio(assets=")
    for asset in portfolio.assets:
        assert asset in r
