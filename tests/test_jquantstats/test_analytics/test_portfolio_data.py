"""Tests for Portfolio data computation properties.

Verifies that all derived data series (profits, NAV, returns, drawdown, etc.)
are accessible and correct directly on the Portfolio class.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from jquantstats import Portfolio

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
def portfolio(prices, positions):
    """Portfolio instance built from the prices and positions fixtures."""
    return Portfolio(prices=prices, cashposition=positions, aum=1e5)


# ─── Factory methods ──────────────────────────────────────────────────────────


def test_from_cash_position_returns_portfolio(prices, positions):
    """Portfolio.from_cash_position returns a Portfolio instance."""
    pf = Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=2e5)
    assert isinstance(pf, Portfolio)
    assert pf.aum == 2e5


def test_from_risk_position_returns_portfolio(prices, positions):
    """Portfolio.from_risk_position returns a Portfolio instance."""
    pf = Portfolio.from_risk_position(prices=prices, risk_position=positions, vola=2, aum=1e5)
    assert isinstance(pf, Portfolio)
    assert pf.assets == ["A"]


# ─── Data properties ──────────────────────────────────────────────────────────


def test_portfolio_assets(portfolio):
    """Portfolio.assets lists numeric column names from prices."""
    assert portfolio.assets == ["A"]


def test_portfolio_profits_columns(portfolio):
    """Portfolio.profits contains the asset column."""
    assert "A" in portfolio.profits.columns


def test_portfolio_profit_columns(portfolio):
    """Portfolio.profit contains a 'profit' column."""
    assert "profit" in portfolio.profit.columns


def test_portfolio_nav_accumulated(portfolio):
    """Portfolio.nav_accumulated contains a 'NAV_accumulated' column."""
    assert "NAV_accumulated" in portfolio.nav_accumulated.columns


def test_portfolio_returns(portfolio):
    """Portfolio.returns contains a 'returns' column."""
    assert "returns" in portfolio.returns.columns


def test_portfolio_nav_compounded(portfolio):
    """Portfolio.nav_compounded contains a 'NAV_compounded' column."""
    assert "NAV_compounded" in portfolio.nav_compounded.columns


def test_portfolio_highwater(portfolio):
    """Portfolio.highwater contains a 'highwater' column."""
    assert "highwater" in portfolio.highwater.columns


def test_portfolio_drawdown(portfolio):
    """Portfolio.drawdown contains both 'drawdown' and 'drawdown_pct' columns."""
    assert "drawdown" in portfolio.drawdown.columns
    assert "drawdown_pct" in portfolio.drawdown.columns


def test_portfolio_all_columns(portfolio):
    """Portfolio.all merges NAV, drawdown, and compounded NAV columns."""
    df = portfolio.all
    assert "NAV_accumulated" in df.columns
    assert "NAV_compounded" in df.columns
    assert "drawdown" in df.columns
