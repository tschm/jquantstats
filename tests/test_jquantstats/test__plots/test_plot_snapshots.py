"""Snapshot tests for jquantstats._plots — assert figure structure stays stable.

These tests capture the structural fingerprint of each Plotly figure (trace
types, trace names, key layout properties) without storing the raw data
arrays.  Any unintended change to the number of traces, their types, the
chart title, subplot layout, or hover behaviour will cause the corresponding
assertion to fail, helping catch regressions early.

Run with ``--snapshot-update`` to regenerate stored snapshots after an
intentional structural change:

    uv run pytest tests/test_jquantstats/test__plots/test_plot_snapshots.py --snapshot-update
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

from jquantstats import Data, Portfolio

from ..plot_test_utils import figure_structure

# ─── Data.plots snapshot tests ───────────────────────────────────────────────


@pytest.fixture
def plots(data):
    """Return the Plots facade attached to the shared data fixture."""
    return data.plots


def test_plot_snapshot_structure(plots, snapshot: SnapshotAssertion):
    """plot_snapshot() structure should not change unexpectedly.

    Captures the number of traces, their types and names, and key layout
    properties for the default (multi-ticker + benchmark) snapshot chart.
    """
    fig = plots.plot_snapshot()
    assert figure_structure(fig) == snapshot


def test_plot_snapshot_log_scale_structure(plots, snapshot: SnapshotAssertion):
    """plot_snapshot(log_scale=True) layout structure should not change unexpectedly.

    The log-scale variant should produce the same structural fingerprint as
    the default chart; only the y-axis scale type differs.
    """
    fig = plots.plot_snapshot(log_scale=True)
    assert figure_structure(fig) == snapshot


def test_plot_snapshot_single_symbol_structure(returns, snapshot: SnapshotAssertion):
    """Single-symbol plot_snapshot() structure should not change unexpectedly.

    When only one ticker is provided (no benchmark), the function uses
    green/red bar colouring instead of per-ticker colours.  This test
    verifies the trace count and types remain stable.
    """
    fig = Data.from_returns(returns=returns).plots.plot_snapshot()
    assert figure_structure(fig) == snapshot


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


# ─── Portfolio.plots snapshot tests ──────────────────────────────────────────


def test_snapshot_structure(pf: Portfolio, snapshot: SnapshotAssertion):
    """snapshot() figure structure should not change unexpectedly.

    Asserts the number of traces, their types and names, and key layout
    properties of the default performance-dashboard snapshot chart.
    """
    fig = pf.plots.snapshot()
    assert figure_structure(fig) == snapshot


def test_snapshot_log_scale_structure(pf: Portfolio, snapshot: SnapshotAssertion):
    """snapshot(log_scale=True) figure structure should not change unexpectedly.

    The log-scale variant must produce the same structural fingerprint as
    the default chart.
    """
    fig = pf.plots.snapshot(log_scale=True)
    assert figure_structure(fig) == snapshot


# ─── Lagged performance ───────────────────────────────────────────────────────


def test_lagged_performance_plot_structure(pf: Portfolio, snapshot: SnapshotAssertion):
    """lagged_performance_plot() figure structure should not change unexpectedly.

    Asserts the default 5-trace layout (lag 0 … lag 4) is stable.
    """
    fig = pf.plots.lagged_performance_plot()
    assert figure_structure(fig) == snapshot


# ─── Smoothed holdings ────────────────────────────────────────────────────────


def test_smoothed_holdings_performance_plot_structure(pf: Portfolio, snapshot: SnapshotAssertion):
    """smoothed_holdings_performance_plot() figure structure should not change unexpectedly.

    Asserts the default 5-trace layout (smooth 0 … smooth 4) is stable.
    """
    fig = pf.plots.smoothed_holdings_performance_plot()
    assert figure_structure(fig) == snapshot


# ─── Lead/lag IR ─────────────────────────────────────────────────────────────


def test_lead_lag_ir_plot_structure(pf: Portfolio, snapshot: SnapshotAssertion):
    """lead_lag_ir_plot() figure structure should not change unexpectedly.

    Asserts the single-bar-trace layout with the expected title is stable.
    """
    fig = pf.plots.lead_lag_ir_plot()
    assert figure_structure(fig) == snapshot


# ─── Correlation heatmap ──────────────────────────────────────────────────────


def test_correlation_heatmap_structure(pf: Portfolio, snapshot: SnapshotAssertion):
    """correlation_heatmap() figure structure should not change unexpectedly.

    Asserts the single Heatmap trace and title are stable.
    """
    fig = pf.plots.correlation_heatmap()
    assert figure_structure(fig) == snapshot


# ─── Monthly returns heatmap ─────────────────────────────────────────────────


def test_monthly_returns_heatmap_structure(pf: Portfolio, snapshot: SnapshotAssertion):
    """monthly_returns_heatmap() figure structure should not change unexpectedly.

    Asserts the single Heatmap trace layout and title are stable.
    """
    fig = pf.plots.monthly_returns_heatmap()
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
