"""Snapshot tests for jquantstats.analytics._plots — assert figure structure stays stable.

These tests capture the structural fingerprint (trace types, trace names, and
key layout properties) of every plot produced by the analytics ``Plots``
facade.  They detect accidental structural regressions — e.g. a refactor that
removes a trace, renames a chart title, or changes subplot layout — without
being coupled to specific data values.

Run with ``--snapshot-update`` to regenerate stored snapshots after an
intentional structural change:

    uv run pytest tests/test_jquantstats/test_analytics/test_plot_snapshots.py --snapshot-update
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

from jquantstats import Portfolio

from ..plot_test_utils import figure_structure

# ─── Extra fixture needed for rolling / multi-year plots ─────────────────────


@pytest.fixture
def long_portfolio() -> Portfolio:
    """Portfolio spanning ~2 calendar years used to test rolling and annual-breakdown plots."""
    n = 756  # ~756 calendar days (~2 years); includes weekends/holidays
    start = date(2020, 1, 1)
    end = start + timedelta(days=n - 1)
    dates = pl.date_range(start=start, end=end, interval="1d", eager=True).cast(pl.Date)

    a = pl.Series([100.0 * (1.001**i) for i in range(n)], dtype=pl.Float64)
    b = pl.Series([200.0 + 5.0 * np.sin(0.1 * i) for i in range(n)], dtype=pl.Float64)
    prices = pl.DataFrame({"date": dates, "A": a, "B": b})

    pos_a = pl.Series([1000.0 + float(i % 10) for i in range(n)], dtype=pl.Float64)
    pos_b = pl.Series([500.0 + float(i % 5) for i in range(n)], dtype=pl.Float64)
    positions = pl.DataFrame({"date": dates, "A": pos_a, "B": pos_b})

    return Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e6)


# ─── Snapshot ────────────────────────────────────────────────────────────────


def test_snapshot_structure(portfolio: Portfolio, snapshot: SnapshotAssertion):
    """snapshot() figure structure should not change unexpectedly.

    Asserts the number of traces, their types and names, and key layout
    properties of the default performance-dashboard snapshot chart.
    """
    fig = portfolio.plots.snapshot()
    assert figure_structure(fig) == snapshot


def test_snapshot_log_scale_structure(portfolio: Portfolio, snapshot: SnapshotAssertion):
    """snapshot(log_scale=True) figure structure should not change unexpectedly.

    The log-scale variant must produce the same structural fingerprint as
    the default chart.
    """
    fig = portfolio.plots.snapshot(log_scale=True)
    assert figure_structure(fig) == snapshot


# ─── Lagged performance ───────────────────────────────────────────────────────


def test_lagged_performance_plot_structure(portfolio: Portfolio, snapshot: SnapshotAssertion):
    """lagged_performance_plot() figure structure should not change unexpectedly.

    Asserts the default 5-trace layout (lag 0 … lag 4) is stable.
    """
    fig = portfolio.plots.lagged_performance_plot()
    assert figure_structure(fig) == snapshot


# ─── Smoothed holdings ────────────────────────────────────────────────────────


def test_smoothed_holdings_performance_plot_structure(portfolio: Portfolio, snapshot: SnapshotAssertion):
    """smoothed_holdings_performance_plot() figure structure should not change unexpectedly.

    Asserts the default 5-trace layout (smooth 0 … smooth 4) is stable.
    """
    fig = portfolio.plots.smoothed_holdings_performance_plot()
    assert figure_structure(fig) == snapshot


# ─── Lead/lag IR ─────────────────────────────────────────────────────────────


def test_lead_lag_ir_plot_structure(portfolio: Portfolio, snapshot: SnapshotAssertion):
    """lead_lag_ir_plot() figure structure should not change unexpectedly.

    Asserts the single-bar-trace layout with the expected title is stable.
    """
    fig = portfolio.plots.lead_lag_ir_plot()
    assert figure_structure(fig) == snapshot


# ─── Correlation heatmap ──────────────────────────────────────────────────────


def test_correlation_heatmap_structure(portfolio: Portfolio, snapshot: SnapshotAssertion):
    """correlation_heatmap() figure structure should not change unexpectedly.

    Asserts the single Heatmap trace and title are stable.
    """
    fig = portfolio.plots.correlation_heatmap()
    assert figure_structure(fig) == snapshot


# ─── Monthly returns heatmap ─────────────────────────────────────────────────


def test_monthly_returns_heatmap_structure(portfolio: Portfolio, snapshot: SnapshotAssertion):
    """monthly_returns_heatmap() figure structure should not change unexpectedly.

    Asserts the single Heatmap trace layout and title are stable.
    """
    fig = portfolio.plots.monthly_returns_heatmap()
    assert figure_structure(fig) == snapshot


# ─── Rolling Sharpe ──────────────────────────────────────────────────────────


def test_rolling_sharpe_plot_structure(long_portfolio: Portfolio, snapshot: SnapshotAssertion):
    """rolling_sharpe_plot() figure structure should not change unexpectedly.

    Uses the 3-year portfolio so the rolling window produces meaningful data.
    Asserts one Scatter trace per asset column and the expected title.
    """
    fig = long_portfolio.plots.rolling_sharpe_plot(window=63)
    assert figure_structure(fig) == snapshot


# ─── Rolling Volatility ───────────────────────────────────────────────────────


def test_rolling_volatility_plot_structure(long_portfolio: Portfolio, snapshot: SnapshotAssertion):
    """rolling_volatility_plot() figure structure should not change unexpectedly.

    Uses the 3-year portfolio so the rolling window produces meaningful data.
    Asserts one Scatter trace per asset column and the expected title.
    """
    fig = long_portfolio.plots.rolling_volatility_plot(window=63)
    assert figure_structure(fig) == snapshot


# ─── Annual Sharpe ────────────────────────────────────────────────────────────


def test_annual_sharpe_plot_structure(long_portfolio: Portfolio, snapshot: SnapshotAssertion):
    """annual_sharpe_plot() figure structure should not change unexpectedly.

    Uses the 3-year portfolio to ensure multiple calendar years are present.
    Asserts one Bar trace per asset column and the expected title.
    """
    fig = long_portfolio.plots.annual_sharpe_plot()
    assert figure_structure(fig) == snapshot
