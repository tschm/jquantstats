# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.20.4",
#     "jquantstats",
#     "numpy>=2.0.0",
#     "polars>=1.0.0",
#     "plotly>=6.0.0",
#     "jinja2>=3.1.0",
# ]
# [tool.uv.sources]
# jquantstats = { path = "../../..", editable = true }
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")

with app.setup:
    from datetime import date, timedelta

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl

    from jquantstats import Portfolio


@app.cell
def cell_intro() -> None:
    """Render the portfolio construction introduction."""
    mo.md(
        r"""
        # 🏗️ jquantstats — Portfolio Construction & Attribution

        This notebook compares the **three Portfolio constructors** and then digs into
        **tilt/timing attribution**, **turnover analysis**, and **cost sensitivity**.

        | Constructor | Inputs | Use case |
        |-------------|--------|----------|
        | `from_cash_position` | Cash $ value per asset | Direct exposure control |
        | `from_position` | Unit quantity per asset | Shares / contracts |
        | `from_risk_position` | Risk units + EWMA vol | Vol-targeting strategies |

        **Data:** 500-day synthetic 3-asset portfolio (assets A, B, C).
        """
    )
    return


# ── Synthetic data ────────────────────────────────────────────────────────────


@app.cell
def cell_data():
    """Generate synthetic price and position data for three assets."""
    n = 500
    start = date(2021, 1, 4)
    dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True).cast(pl.Date)

    rng = np.random.default_rng(7)

    # Asset A: uptrend + moderate noise
    price_a = pl.Series(
        [100.0 * np.exp(0.0004 * i + 0.012 * rng.standard_normal()) for i in range(n)],
        dtype=pl.Float64,
    )
    # Asset B: mean-reverting
    price_b = pl.Series(
        [50.0 + 8.0 * np.sin(0.04 * i) + 1.5 * rng.standard_normal() for i in range(n)],
        dtype=pl.Float64,
    )
    # Asset C: high-volatility uptrend
    price_c = pl.Series(
        [200.0 * np.exp(0.0002 * i + 0.025 * rng.standard_normal()) for i in range(n)],
        dtype=pl.Float64,
    )

    prices = pl.DataFrame({"date": dates, "A": price_a, "B": price_b, "C": price_c})

    # Cash positions: time-varying exposure
    cash_a = pl.Series([10_000.0 + 20.0 * i for i in range(n)], dtype=pl.Float64)
    cash_b = pl.Series([8_000.0 + float(i % 10) * 500.0 for i in range(n)], dtype=pl.Float64)
    cash_c = pl.Series([5_000.0 - float(i % 25) * 100.0 for i in range(n)], dtype=pl.Float64)
    cash_position = pl.DataFrame({"date": dates, "A": cash_a, "B": cash_b, "C": cash_c})

    # Unit positions (shares): derived from cash / price
    unit_a = cash_a / price_a
    unit_b = cash_b / price_b
    unit_c = cash_c / price_c
    unit_position = pl.DataFrame({"date": dates, "A": unit_a, "B": unit_b, "C": unit_c})

    # Risk positions: same $ units but constructor will de-vol them
    risk_position = cash_position.clone()

    mo.md(
        f"""
        **Synthetic data generated** ✅

        - Assets: A, B, C
        - Trading days: {n}
        - Date range: `{start}` → `{(start + timedelta(days=n - 1))}`
        """
    )
    return (cash_position, dates, n, price_a, price_b, price_c, prices, risk_position, start, unit_position)


# ── Three constructors ────────────────────────────────────────────────────────


@app.cell
def cell_constructors_header() -> None:
    """Constructors section header."""
    mo.md(
        """
        ---
        ## 1 · Three constructors — side-by-side comparison

        We build the same strategy three ways and compare the resulting NAV curves.
        """
    )
    return


@app.cell
def cell_build_portfolios(prices, cash_position, unit_position, risk_position):
    """Build three portfolios using different constructors."""
    pf_cash = Portfolio.from_cash_position(
        prices=prices,
        cash_position=cash_position,
        aum=1_000_000.0,
        cost_bps=2.0,
    )
    pf_unit = Portfolio.from_position(
        prices=prices,
        position=unit_position,
        aum=1_000_000.0,
        cost_bps=2.0,
    )
    pf_risk = Portfolio.from_risk_position(
        prices=prices,
        risk_position=risk_position,
        aum=1_000_000.0,
        vola=32,
        cost_bps=2.0,
    )

    mo.md(
        """
        | Constructor | Description |
        |-------------|-------------|
        | `from_cash_position` | Raw $ exposure — NAV driven directly by price P&L |
        | `from_position` | Unit positions × price — equivalent when initialised from cash/price |
        | `from_risk_position` | Risk units ÷ EWMA vol — position sizes shrink in volatile regimes |
        """
    )
    return (pf_cash, pf_risk, pf_unit)


@app.cell
def cell_nav_comparison(pf_cash, pf_unit, pf_risk):
    """Plot NAV curves for all three portfolios."""
    fig = go.Figure()

    for _label, _pf in [
        ("from_cash_position", pf_cash),
        ("from_position", pf_unit),
        ("from_risk_position", pf_risk),
    ]:
        _nav = _pf.nav_accumulated
        _date_col = _nav.columns[0]
        _nav_col = "NAV_accumulated"
        fig.add_trace(
            go.Scatter(
                x=_nav[_date_col].to_list(),
                y=_nav[_nav_col].to_list(),
                mode="lines",
                name=_label,
            )
        )

    fig.update_layout(
        title="NAV Comparison — Three Portfolio Constructors",
        xaxis_title="Date",
        yaxis_title="Cumulative NAV ($)",
        height=440,
    )
    return (fig,)


@app.cell
def cell_nav_comparison_show(fig) -> None:
    """Display NAV comparison chart."""
    fig
    return


@app.cell
def cell_constructor_stats(pf_cash, pf_unit, pf_risk) -> None:
    """Compare Sharpe and max drawdown across the three portfolios."""
    _rows = []
    for _label, _pf in [
        ("from_cash_position", pf_cash),
        ("from_position", pf_unit),
        ("from_risk_position", pf_risk),
    ]:
        _sharpe = _pf.stats.sharpe()
        _max_dd = _pf.stats.max_drawdown()
        _cagr = _pf.stats.cagr()
        _rows.append(
            {
                "Constructor": _label,
                "Sharpe": round(next(iter(_sharpe.values())), 3),
                "Max Drawdown": round(next(iter(_max_dd.values())), 4),
                "CAGR": round(next(iter(_cagr.values())), 4),
            }
        )
    mo.plain_text(str(pl.DataFrame(_rows)))
    return


# ── Tilt / timing attribution ─────────────────────────────────────────────────


@app.cell
def cell_attribution_header() -> None:
    """Attribution section header."""
    mo.md(
        """
        ---
        ## 2 · Tilt / timing attribution

        Brinson-style attribution decomposes NAV into two components:

        - **Tilt** — NAV from the *time-average* position (what you would earn
          holding constant weights throughout the period)
        - **Timing** — NAV from position *deviations* around the average
          (the return from varying weights over time)

        A timing contribution near zero means the variation in weights added
        little value beyond a simple buy-and-hold.
        """
    )
    return


@app.cell
def cell_attribution(pf_cash):
    """Compute tilt/timing decomposition."""
    decomp = pf_cash.tilt_timing_decomp
    return (decomp,)


@app.cell
def cell_attribution_chart(decomp) -> None:
    """Plot tilt/timing NAV decomposition."""
    _fig = go.Figure()
    _date_col = decomp.columns[0]

    for _col in ["portfolio", "tilt", "timing"]:
        if _col in decomp.columns:
            _fig.add_trace(
                go.Scatter(
                    x=decomp[_date_col].to_list(),
                    y=decomp[_col].to_list(),
                    mode="lines",
                    name=_col.capitalize(),
                )
            )

    _fig.update_layout(
        title="Tilt / Timing Attribution — Cumulative NAV",
        xaxis_title="Date",
        yaxis_title="Cumulative NAV ($)",
        height=440,
    )
    _fig
    return


# ── Turnover analysis ─────────────────────────────────────────────────────────


@app.cell
def cell_turnover_header() -> None:
    """Turnover section header."""
    mo.md(
        """
        ---
        ## 3 · Turnover analysis

        Turnover measures how actively positions change.  High turnover magnifies
        the impact of trading costs on realised returns.
        """
    )
    return


@app.cell
def cell_turnover(pf_cash):
    """Compute turnover statistics."""
    to_summary = pf_cash.turnover_summary()
    to_daily = pf_cash.turnover
    return (to_daily, to_summary)


@app.cell
def cell_turnover_summary(to_summary) -> None:
    """Display turnover summary."""
    mo.md("### Turnover summary statistics")
    mo.plain_text(str(to_summary))
    return


@app.cell
def cell_turnover_chart(to_daily) -> None:
    """Plot daily turnover."""
    _date_col = to_daily.columns[0]
    _fig = go.Figure()
    for _col in to_daily.columns[1:]:
        _fig.add_trace(
            go.Scatter(
                x=to_daily[_date_col].to_list(),
                y=to_daily[_col].to_list(),
                mode="lines",
                name=_col,
                opacity=0.7,
            )
        )
    _fig.update_layout(
        title="Daily Turnover per Asset",
        xaxis_title="Date",
        yaxis_title="Turnover ($ change in position)",
        height=400,
    )
    _fig
    return


# ── Cost sensitivity ──────────────────────────────────────────────────────────


@app.cell
def cell_cost_header() -> None:
    """Cost sensitivity section header."""
    mo.md(
        """
        ---
        ## 4 · Trading cost sensitivity

        The `trading_cost_impact_plot` sweeps one-way trading costs from 0 to
        `max_bps` and shows how the Sharpe ratio degrades.  This helps identify
        the **break-even cost** — the maximum fee the strategy can absorb.
        """
    )
    return


@app.cell
def cell_cost_chart(pf_cash):
    """Plot trading cost impact."""
    fig_cost = pf_cash.plots.trading_cost_impact_plot(max_bps=30)
    return (fig_cost,)


@app.cell
def cell_cost_chart_show(fig_cost) -> None:
    """Display cost impact chart."""
    fig_cost
    return


# ── Smoothed holdings ─────────────────────────────────────────────────────────


@app.cell
def cell_smooth_header() -> None:
    """Smoothed holdings section header."""
    mo.md(
        """
        ---
        ## 5 · Smoothed holdings performance

        Position smoothing (rolling-mean entry) reduces turnover at the cost of
        slower signal response.  The plot compares NAV curves for different
        smoothing windows (0 = no smoothing, 5 = 5-day average).
        """
    )
    return


@app.cell
def cell_smooth_chart(pf_cash):
    """Plot smoothed holdings performance."""
    fig_smooth = pf_cash.plots.smoothed_holdings_performance_plot(windows=[0, 1, 3, 5, 10])
    return (fig_smooth,)


@app.cell
def cell_smooth_chart_show(fig_smooth) -> None:
    """Display smoothed holdings figure."""
    fig_smooth
    return


# ── Execution lag analysis ────────────────────────────────────────────────────


@app.cell
def cell_lag_header() -> None:
    """Execution lag section header."""
    mo.md(
        """
        ---
        ## 6 · Execution lag analysis

        Signals often have a delay between generation and execution.  The
        lead/lag IR plot shows how much the Sharpe ratio degrades as we shift
        positions by 1–10 days.  A steep drop indicates the strategy is
        highly sensitive to execution timing.
        """
    )
    return


@app.cell
def cell_lag_chart(pf_cash):
    """Plot lead/lag information ratio."""
    fig_lag = pf_cash.plots.lead_lag_ir_plot(start=0, end=10)
    return (fig_lag,)


@app.cell
def cell_lag_chart_show(fig_lag) -> None:
    """Display lead/lag IR chart."""
    fig_lag
    return


@app.cell
def cell_lagged_perf(pf_cash):
    """Plot NAV curves at multiple execution lags."""
    fig_lp = pf_cash.plots.lagged_performance_plot(lags=[0, 1, 2, 5])
    return (fig_lp,)


@app.cell
def cell_lagged_perf_show(fig_lp) -> None:
    """Display lagged performance chart."""
    fig_lp
    return


if __name__ == "__main__":
    app.run()
