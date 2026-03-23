"""Snapshot tests for jquantstats._plots — assert figure structure stays stable.

These tests capture the structural fingerprint of each Plotly figure (trace
types, trace names, key layout properties) without storing the raw data
arrays.  Any unintended change to the number of traces, their types, the
chart title, subplot layout, or hover behaviour will cause the corresponding
assertion to fail, helping catch regressions early.

Run with ``--snapshot-update`` to regenerate stored snapshots after an
intentional structural change:

    uv run pytest tests/test_jquantstats/test_plot_snapshots.py --snapshot-update
"""

from __future__ import annotations

import pytest
from syrupy.assertion import SnapshotAssertion

from jquantstats import build_data

from .plot_test_utils import figure_structure


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
    fig = build_data(returns=returns).plots.plot_snapshot()
    assert figure_structure(fig) == snapshot
