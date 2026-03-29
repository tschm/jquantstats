"""Tests for Portfolio data bridge, caching, repr/describe, and computation properties."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from jquantstats import Portfolio
from jquantstats.data import Data

# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def prices_single():
    """Three-day single-asset price frame (A: 100 → 110 → 121)."""
    return pl.DataFrame(
        {
            "date": pl.date_range(start=date(2020, 1, 1), end=date(2020, 1, 3), interval="1d", eager=True).cast(
                pl.Date
            ),
            "A": pl.Series([100.0, 110.0, 121.0], dtype=pl.Float64),
        }
    )


@pytest.fixture
def positions_single(prices_single):
    """Three-day cash-position frame aligned with the prices_single fixture."""
    return pl.DataFrame(
        {
            "date": prices_single["date"],
            "A": pl.Series([1000.0, 1000.0, 1000.0], dtype=pl.Float64),
        }
    )


@pytest.fixture
def portfolio_single(prices_single, positions_single):
    """Portfolio instance built from the prices_single and positions_single fixtures."""
    return Portfolio(prices=prices_single, cashposition=positions_single, aum=1e5)


# ─── Portfolio.data bridge property ──────────────────────────────────────────


def test_portfolio_data_property_returns_data_object(portfolio):
    """portfolio.data returns a legacy Data object with a 'returns' column and date index."""
    d = portfolio.data
    assert isinstance(d, Data)
    assert "returns" in d.returns.columns
    assert d.returns.height == portfolio.prices.height
    assert d.index.height == portfolio.prices.height


def test_portfolio_data_property_integer_indexed(int_portfolio):
    """portfolio.data on an integer-indexed portfolio creates a synthetic integer index."""
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


# ─── _data_bridge caching ─────────────────────────────────────────────────────


def test_data_property_returns_same_object(portfolio):
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


# ─── repr and describe ────────────────────────────────────────────────────────


def test_repr(portfolio):
    """Tests that Portfolio.__repr__ returns an informative string."""
    r = repr(portfolio)
    assert r.startswith("Portfolio(assets=")
    assert "rows=" in r
    assert "start=" in r
    assert "end=" in r
    for asset in portfolio.assets:
        assert asset in r


def test_describe(portfolio):
    """Tests that Portfolio.describe() returns a tidy summary DataFrame."""
    df = portfolio.describe()
    assert "asset" in df.columns
    assert "start" in df.columns
    assert "end" in df.columns
    assert "rows" in df.columns
    assert len(df) == len(portfolio.assets)
    for asset in portfolio.assets:
        assert asset in df["asset"].to_list()


def test_repr_integer_indexed(int_portfolio):
    """Portfolio.__repr__ omits start/end for integer-indexed (no date) portfolios."""
    r = repr(int_portfolio)
    assert r.startswith("Portfolio(assets=")
    assert "rows=" in r
    assert "start=" not in r
    assert "end=" not in r


# ─── Construction factories ───────────────────────────────────────────────────


def test_from_cash_position_returns_portfolio(prices_single, positions_single):
    """Portfolio.from_cash_position returns a Portfolio instance."""
    pf = Portfolio.from_cash_position(prices=prices_single, cash_position=positions_single, aum=2e5)
    assert isinstance(pf, Portfolio)
    assert pf.aum == 2e5


def test_from_risk_position_returns_portfolio(prices_single, positions_single):
    """Portfolio.from_risk_position returns a Portfolio instance."""
    pf = Portfolio.from_risk_position(prices=prices_single, risk_position=positions_single, vola=2, aum=1e5)
    assert isinstance(pf, Portfolio)
    assert pf.assets == ["A"]


# ─── Computation properties ───────────────────────────────────────────────────


def test_portfolio_assets(portfolio_single):
    """Portfolio.assets lists numeric column names from prices."""
    assert portfolio_single.assets == ["A"]


def test_portfolio_profits_columns(portfolio_single):
    """Portfolio.profits contains the asset column."""
    assert "A" in portfolio_single.profits.columns


def test_portfolio_profit_columns(portfolio_single):
    """Portfolio.profit contains a 'profit' column."""
    assert "profit" in portfolio_single.profit.columns


def test_portfolio_nav_accumulated(portfolio_single):
    """Portfolio.nav_accumulated contains a 'NAV_accumulated' column."""
    assert "NAV_accumulated" in portfolio_single.nav_accumulated.columns


def test_portfolio_returns(portfolio_single):
    """Portfolio.returns contains a 'returns' column."""
    assert "returns" in portfolio_single.returns.columns


def test_portfolio_nav_compounded(portfolio_single):
    """Portfolio.nav_compounded contains a 'NAV_compounded' column."""
    assert "NAV_compounded" in portfolio_single.nav_compounded.columns


def test_portfolio_highwater(portfolio_single):
    """Portfolio.highwater contains a 'highwater' column."""
    assert "highwater" in portfolio_single.highwater.columns


def test_portfolio_drawdown(portfolio_single):
    """Portfolio.drawdown contains both 'drawdown' and 'drawdown_pct' columns."""
    assert "drawdown" in portfolio_single.drawdown.columns
    assert "drawdown_pct" in portfolio_single.drawdown.columns


def test_portfolio_all_columns(portfolio_single):
    """Portfolio.all merges NAV, drawdown, and compounded NAV columns."""
    df = portfolio_single.all
    assert "NAV_accumulated" in df.columns
    assert "NAV_compounded" in df.columns
    assert "drawdown" in df.columns
