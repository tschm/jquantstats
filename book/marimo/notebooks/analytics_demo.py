# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.13.15",
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

__generated_with = "0.13.15"
app = marimo.App(width="medium")

with app.setup:
    from datetime import date, timedelta

    import marimo as mo
    import numpy as np
    import polars as pl

    from jquantstats.analytics import Portfolio


@app.cell
def cell_intro() -> None:
    """Render the analytics demo introduction."""
    mo.md(
        r"""
        # 📊 jquantstats — Portfolio Analytics Demo

        This notebook demonstrates the **Portfolio analytics** subpackage from `jquantstats`.

        It covers:
        1. 📈 **Synthetic data generation** — create a 2-asset price and position series
        2. 🏗️ **Portfolio construction** — use `Portfolio.from_cash_position(...)`
        3. 📉 **Portfolio analytics** — NAV, drawdown, statistics, and visualisations
        4. 📋 **HTML report generation** — self-contained report via `portfolio.report.to_html()`
        """
    )
    return


@app.cell
def cell_data():
    """Generate synthetic 2-asset price and position data."""
    n = 120
    start = date(2020, 1, 1)
    end = start + timedelta(days=n - 1)
    dates = pl.date_range(start=start, end=end, interval="1d", eager=True).cast(pl.Date)

    rng = np.random.default_rng(42)

    # Asset A: trending upward with small noise
    price_a = pl.Series(
        [100.0 * (1.001**i) * (1 + 0.005 * rng.standard_normal()) for i in range(n)],
        dtype=pl.Float64,
    )
    # Asset B: mean-reverting sine pattern
    price_b = pl.Series(
        [200.0 + 10.0 * np.sin(0.1 * i) + 2.0 * rng.standard_normal() for i in range(n)],
        dtype=pl.Float64,
    )

    prices = pl.DataFrame({"date": dates, "A": price_a, "B": price_b})

    pos_a = pl.Series([1000.0 + 5.0 * i for i in range(n)], dtype=pl.Float64)
    pos_b = pl.Series([500.0 + float(i % 5) * 50.0 for i in range(n)], dtype=pl.Float64)
    positions = pl.DataFrame({"date": dates, "A": pos_a, "B": pos_b})

    mo.md(f"Generated **{n}-day** synthetic portfolio with **2 assets** (A, B).")
    return dates, n, pos_a, pos_b, positions, price_a, price_b, prices, rng, start, end


@app.cell
def cell_portfolio(prices, positions):
    """Build the Portfolio from cash positions."""
    portfolio = Portfolio.from_cash_position(
        prices=prices,
        cash_position=positions,
        aum=1_000_000.0,
    )
    mo.md(
        f"""
        **Portfolio created** ✅

        - Assets: `{portfolio.assets}`
        - AUM: `{portfolio.aum:,.0f}`
        - Periods: `{prices.height}`
        """
    )
    return (portfolio,)


@app.cell
def cell_stats(portfolio):
    """Display portfolio statistics summary."""
    summary = portfolio.stats.summary()
    mo.md("## 📊 Statistics Summary")
    return (summary,)


@app.cell
def cell_stats_table(summary) -> None:
    """Show the stats table."""
    mo.plain_text(str(summary))
    return


@app.cell
def cell_snapshot(portfolio):
    """Render the portfolio snapshot chart."""
    mo.md("## 📈 Portfolio Snapshot")
    fig = portfolio.plots.snapshot()
    return (fig,)


@app.cell
def cell_snapshot_plot(fig) -> None:
    """Display snapshot figure."""
    fig
    return


@app.cell
def cell_lead_lag(portfolio):
    """Render the lead/lag IR chart."""
    mo.md("## 🔀 Lead/Lag Information Ratio")
    fig_ll = portfolio.plots.lead_lag_ir_plot()
    return (fig_ll,)


@app.cell
def cell_lead_lag_plot(fig_ll) -> None:
    """Display lead/lag figure."""
    fig_ll
    return


@app.cell
def cell_report(portfolio):
    """Generate and display the HTML report."""
    mo.md("## 📋 HTML Report")
    html = portfolio.report.to_html(title="Analytics Demo Report")
    mo.md(f"Report generated: **{len(html):,}** characters of HTML.")
    return (html,)


@app.cell
def cell_report_preview(html) -> None:
    """Show a snippet of the report HTML."""
    snippet = html[:500] + "..."
    mo.plain_text(snippet)
    return


if __name__ == "__main__":
    app.run()
