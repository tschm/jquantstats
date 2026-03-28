"""Parametrized smoke tests for DataPlots methods."""

import plotly.graph_objects as go
import pytest


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("plot_snapshot", {}),
        ("plot_snapshot", {"log_scale": True}),
        ("plot_returns", {}),
        ("plot_returns", {"log_scale": True}),
        ("plot_log_returns", {}),
        ("plot_daily_returns", {}),
        ("plot_yearly_returns", {}),
        ("plot_yearly_returns", {"compounded": False}),
        ("plot_monthly_returns", {}),
        ("plot_monthly_returns", {"compounded": False}),
        ("plot_monthly_heatmap", {}),
        ("plot_monthly_heatmap", {"asset": "AAPL"}),
        ("plot_histogram", {}),
        ("plot_histogram", {"bins": 20}),
        ("plot_distribution", {}),
        ("plot_distribution", {"compounded": False}),
        ("plot_drawdown", {}),
        ("plot_drawdowns_periods", {}),
        ("plot_drawdowns_periods", {"n": 3}),
        ("plot_earnings", {}),
        ("plot_earnings", {"start_balance": 1e6, "compounded": False}),
        ("plot_rolling_sharpe", {}),
        ("plot_rolling_sharpe", {"rolling_period": 63, "periods_per_year": 252}),
        ("plot_rolling_sortino", {}),
        ("plot_rolling_sortino", {"rolling_period": 63}),
        ("plot_rolling_volatility", {}),
        ("plot_rolling_volatility", {"rolling_period": 63}),
        ("plot_rolling_beta", {}),
        ("plot_rolling_beta", {"rolling_period2": None}),
    ],
)
def test_plot_returns_figure(data, method, kwargs):
    """Each plot method returns a non-empty go.Figure."""
    fig = getattr(data.plots, method)(**kwargs)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("plot_returns", {}),
        ("plot_log_returns", {}),
        ("plot_daily_returns", {}),
        ("plot_drawdown", {}),
    ],
)
def test_plot_includes_benchmark_trace(data, method, kwargs):
    """Plots that render all columns include a trace for the benchmark."""
    fig = getattr(data.plots, method)(**kwargs)
    trace_names = [t.name for t in fig.data]
    assert any("Benchmark" in name or "SPY" in name for name in trace_names)


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("plot_snapshot", {}),
        ("plot_returns", {}),
        ("plot_log_returns", {}),
        ("plot_daily_returns", {}),
        ("plot_yearly_returns", {}),
        ("plot_monthly_returns", {}),
        ("plot_drawdown", {}),
        ("plot_drawdowns_periods", {}),
    ],
)
def test_plot_has_title(data, method, kwargs):
    """Each plot has a non-empty title."""
    fig = getattr(data.plots, method)(**kwargs)
    assert fig.layout.title.text


def test_plot_monthly_heatmap_axes(data):
    """Monthly heatmap has 12 x-axis labels (Jan-Dec)."""
    fig = data.plots.plot_monthly_heatmap()
    heatmap = fig.data[0]
    assert len(heatmap.x) == 12


def test_plot_drawdowns_periods_shading(data):
    """plot_drawdowns_periods adds vrect shapes for the drawdown periods."""
    fig = data.plots.plot_drawdowns_periods(n=3)
    assert len(fig.layout.shapes) > 0


def test_plot_distribution_one_subplot_per_asset(data):
    """plot_distribution produces 5 box traces per asset."""
    fig = data.plots.plot_distribution()
    n_assets = len(data.assets)
    assert len(fig.data) == 5 * n_assets


def test_plot_earnings_scale(data):
    """plot_earnings y-values start near start_balance."""
    start = 50_000.0
    fig = data.plots.plot_earnings(start_balance=start)
    first_y = fig.data[0].y[0]
    assert abs(first_y - start) / start < 0.05


def test_plot_rolling_beta_two_windows(data):
    """plot_rolling_beta with two windows produces 2 traces per asset."""
    fig = data.plots.plot_rolling_beta(rolling_period=63, rolling_period2=126)
    n_return_assets = len(data.returns.columns)
    assert len(fig.data) == 2 * n_return_assets


def test_plot_rolling_beta_no_benchmark_raises(data_no_benchmark):
    """plot_rolling_beta raises AttributeError when no benchmark is attached."""
    with pytest.raises(AttributeError):
        data_no_benchmark.plots.plot_rolling_beta()
