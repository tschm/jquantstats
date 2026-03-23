"""Tests for jquantstats.analytics._portfolio_data.PortfolioData.

Verifies that PortfolioData can be instantiated directly, that its factory
classmethods work, that all pure data properties are accessible, and that
Portfolio correctly delegates every PortfolioData property via composition.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from jquantstats import Portfolio
from jquantstats._portfolio_data import PortfolioData

# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def prices():
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
def positions(prices):
    """Three-day cash-position frame aligned with the prices fixture."""
    return pl.DataFrame(
        {
            "date": prices["date"],
            "A": pl.Series([1000.0, 1000.0, 1000.0], dtype=pl.Float64),
        }
    )


@pytest.fixture
def portfolio_data(prices, positions):
    """PortfolioData instance built from the prices and positions fixtures."""
    return PortfolioData(prices=prices, cashposition=positions, aum=1e5)


# ─── Composition relationship ─────────────────────────────────────────────────


def test_portfolio_is_not_subclass_of_portfolio_data():
    """Portfolio must NOT subclass PortfolioData (composition, not inheritance)."""
    assert not issubclass(Portfolio, PortfolioData)


def test_portfolio_instance_is_not_portfolio_data(prices, positions):
    """A Portfolio instance must not pass isinstance checks for PortfolioData."""
    pf = Portfolio(prices=prices, cashposition=positions, aum=1e5)
    assert not isinstance(pf, PortfolioData)


# ─── PortfolioData direct instantiation ──────────────────────────────────────


def test_portfolio_data_instantiation(portfolio_data):
    """PortfolioData can be instantiated directly with the expected AUM."""
    assert portfolio_data.aum == 1e5


def test_portfolio_data_assets(portfolio_data):
    """PortfolioData.assets lists numeric column names from prices."""
    assert portfolio_data.assets == ["A"]


# ─── PortfolioData factory methods ───────────────────────────────────────────


def test_from_cash_position_returns_portfolio_data(prices, positions):
    """PortfolioData.from_cash_position returns a PortfolioData instance."""
    pd_obj = PortfolioData.from_cash_position(prices=prices, cash_position=positions, aum=2e5)
    assert isinstance(pd_obj, PortfolioData)
    assert pd_obj.aum == 2e5


def test_from_cash_position_on_portfolio_subclass_returns_portfolio(prices, positions):
    """from_cash_position called on Portfolio returns a Portfolio, not bare PortfolioData."""
    pf = Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e5)
    assert isinstance(pf, Portfolio)


def test_from_risk_position_returns_portfolio_data(prices, positions):
    """PortfolioData.from_risk_position returns a PortfolioData with expected assets."""
    pd_obj = PortfolioData.from_risk_position(prices=prices, risk_position=positions, vola=2, aum=1e5)
    assert isinstance(pd_obj, PortfolioData)
    assert pd_obj.assets == ["A"]


def test_from_risk_position_on_portfolio_subclass_returns_portfolio(prices, positions):
    """from_risk_position called on Portfolio returns a Portfolio instance."""
    pf = Portfolio.from_risk_position(prices=prices, risk_position=positions, vola=2, aum=1e5)
    assert isinstance(pf, Portfolio)


# ─── PortfolioData data properties ───────────────────────────────────────────


def test_portfolio_data_profits_columns(portfolio_data):
    """PortfolioData.profits contains the asset column."""
    assert "A" in portfolio_data.profits.columns


def test_portfolio_data_profit_columns(portfolio_data):
    """PortfolioData.profit contains a 'profit' column."""
    assert "profit" in portfolio_data.profit.columns


def test_portfolio_data_nav_accumulated(portfolio_data):
    """PortfolioData.nav_accumulated contains a 'NAV_accumulated' column."""
    assert "NAV_accumulated" in portfolio_data.nav_accumulated.columns


def test_portfolio_data_returns(portfolio_data):
    """PortfolioData.returns contains a 'returns' column."""
    assert "returns" in portfolio_data.returns.columns


def test_portfolio_data_nav_compounded(portfolio_data):
    """PortfolioData.nav_compounded contains a 'NAV_compounded' column."""
    assert "NAV_compounded" in portfolio_data.nav_compounded.columns


def test_portfolio_data_highwater(portfolio_data):
    """PortfolioData.highwater contains a 'highwater' column."""
    assert "highwater" in portfolio_data.highwater.columns


def test_portfolio_data_drawdown(portfolio_data):
    """PortfolioData.drawdown contains both 'drawdown' and 'drawdown_pct' columns."""
    assert "drawdown" in portfolio_data.drawdown.columns
    assert "drawdown_pct" in portfolio_data.drawdown.columns


def test_portfolio_data_all_columns(portfolio_data):
    """PortfolioData.all merges NAV, drawdown, and compounded NAV columns."""
    df = portfolio_data.all
    assert "NAV_accumulated" in df.columns
    assert "NAV_compounded" in df.columns
    assert "drawdown" in df.columns


# ─── PortfolioData lacks analytics methods ───────────────────────────────────


def test_portfolio_data_has_no_stats():
    """PortfolioData must not define a 'stats' attribute (belongs to Portfolio)."""
    assert not hasattr(PortfolioData, "stats")


def test_portfolio_data_has_no_plots():
    """PortfolioData must not define a 'plots' attribute (belongs to Portfolio)."""
    assert not hasattr(PortfolioData, "plots")


def test_portfolio_data_has_no_report():
    """PortfolioData must not define a 'report' attribute (belongs to Portfolio)."""
    assert not hasattr(PortfolioData, "report")


def test_portfolio_data_has_no_tilt():
    """PortfolioData must not define a 'tilt' attribute (belongs to Portfolio)."""
    assert not hasattr(PortfolioData, "tilt")


def test_portfolio_data_has_no_turnover():
    """PortfolioData must not define a 'turnover' attribute (belongs to Portfolio)."""
    assert not hasattr(PortfolioData, "turnover")


# ─── Portfolio delegates all PortfolioData properties ─────────────────────────


def test_portfolio_delegates_profits(prices, positions):
    """Portfolio exposes profits via delegation to the internal PortfolioData."""
    pf = Portfolio(prices=prices, cashposition=positions, aum=1e5)
    assert "A" in pf.profits.columns


def test_portfolio_delegates_profit(prices, positions):
    """Portfolio exposes profit via delegation to the internal PortfolioData."""
    pf = Portfolio(prices=prices, cashposition=positions, aum=1e5)
    assert "profit" in pf.profit.columns


def test_portfolio_delegates_nav_accumulated(prices, positions):
    """Portfolio exposes nav_accumulated via delegation to the internal PortfolioData."""
    pf = Portfolio(prices=prices, cashposition=positions, aum=1e5)
    assert "NAV_accumulated" in pf.nav_accumulated.columns


def test_portfolio_delegates_returns(prices, positions):
    """Portfolio exposes returns via delegation to the internal PortfolioData."""
    pf = Portfolio(prices=prices, cashposition=positions, aum=1e5)
    assert "returns" in pf.returns.columns


def test_portfolio_delegates_drawdown(prices, positions):
    """Portfolio exposes drawdown via delegation to the internal PortfolioData."""
    pf = Portfolio(prices=prices, cashposition=positions, aum=1e5)
    assert "drawdown" in pf.drawdown.columns


def test_portfolio_data_values_match_portfolio(prices, positions):
    """PortfolioData and Portfolio produce identical derived series values."""
    pd_obj = PortfolioData(prices=prices, cashposition=positions, aum=1e5)
    pf = Portfolio(prices=prices, cashposition=positions, aum=1e5)

    assert pd_obj.profit["profit"].to_list() == pf.profit["profit"].to_list()
    assert pd_obj.returns["returns"].to_list() == pf.returns["returns"].to_list()
    assert pd_obj.drawdown["drawdown"].to_list() == pf.drawdown["drawdown"].to_list()
