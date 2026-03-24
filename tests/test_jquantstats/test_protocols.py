"""Tests for internal Protocol/interface definitions.

Verifies that concrete implementations satisfy each @runtime_checkable protocol
and that the protocol modules are importable.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from jquantstats import Portfolio
from jquantstats._plots._protocol import DataLike as PlotsDataLike
from jquantstats._plots._protocol import PortfolioLike as PlotsPortfolioLike
from jquantstats._reports._protocol import DataLike as ReportsDataLike
from jquantstats._reports._protocol import PortfolioLike as ReportsPortfolioLike
from jquantstats._stats._protocol import DataLike as StatsDataLike


@pytest.fixture
def simple_portfolio():
    """Minimal two-asset Portfolio for protocol checks."""
    dates = pl.date_range(start=date(2020, 1, 1), end=date(2020, 1, 5), interval="1d", eager=True).cast(pl.Date)
    prices = pl.DataFrame({"date": dates, "A": pl.Series([100.0, 101.0, 102.0, 103.0, 104.0], dtype=pl.Float64)})
    positions = pl.DataFrame({"date": dates, "A": pl.Series([1000.0] * 5, dtype=pl.Float64)})
    return Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e5)


def test_data_satisfies_stats_data_like(data):
    """Data satisfies the StatsDataLike protocol."""
    assert isinstance(data, StatsDataLike)


def test_data_satisfies_plots_data_like(data):
    """Data satisfies the PlotsDataLike protocol."""
    assert isinstance(data, PlotsDataLike)


def test_data_satisfies_reports_data_like(data):
    """Data satisfies the ReportsDataLike protocol."""
    assert isinstance(data, ReportsDataLike)


def test_portfolio_satisfies_plots_portfolio_like(simple_portfolio):
    """Portfolio satisfies the PlotsPortfolioLike protocol."""
    assert isinstance(simple_portfolio, PlotsPortfolioLike)


def test_portfolio_satisfies_reports_portfolio_like(simple_portfolio):
    """Portfolio satisfies the ReportsPortfolioLike protocol."""
    assert isinstance(simple_portfolio, ReportsPortfolioLike)


def test_non_data_does_not_satisfy_stats_protocol():
    """A plain object does not satisfy StatsDataLike."""
    assert not isinstance("not-a-data", StatsDataLike)


def test_non_data_does_not_satisfy_plots_protocol():
    """A plain object does not satisfy PlotsDataLike."""
    assert not isinstance(42, PlotsDataLike)
