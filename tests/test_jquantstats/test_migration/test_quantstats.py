"""Tests for comparing jquantstats with quantstats library functionality."""

import pytest
import quantstats as qs


def test_calmar(stats, aapl):
    """Compares calmar ratio against quantstats."""
    x = stats.calmar(periods=252)
    y = qs.stats.calmar(aapl, periods=252)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)


def test_recovery_factor(stats, aapl):
    """Compares recovery_factor against quantstats."""
    x = stats.recovery_factor()
    y = qs.stats.recovery_factor(aapl)
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)
