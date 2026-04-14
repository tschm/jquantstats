# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.20.4",
#     "jquantstats",
#     "yfinance>=0.2.0",
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
    import os
    from datetime import date, timedelta
    from pathlib import Path

    import marimo as mo
    import polars as pl
    import yfinance as yf

    from jquantstats import Portfolio

    SHARES = 500
    TICKERS = ["META", "AAPL"]
    END_DATE = date.today()
    START_DATE = END_DATE - timedelta(days=5 * 365)
    NOTEBOOK_DIR = Path(__file__).parent


@app.cell
def cell_intro() -> None:
    """Render the introduction."""
    mo.md(
        rf"""
        # 📈 yfinance Portfolio Demo

        This notebook loads **{TICKERS[0]}** and **{TICKERS[1]}** closing prices from Yahoo Finance
        for the last 5 years, allocates **{SHARES:,} shares** in each, and produces a
        full `jquantstats` portfolio report.

        | Parameter | Value |
        |-----------|-------|
        | Tickers | `{", ".join(TICKERS)}` |
        | Shares per asset | `{SHARES:,}` |
        | Start date | `{START_DATE}` |
        | End date | `{END_DATE}` |
        """
    )
    return


@app.cell
def cell_load_prices():
    """Download adjusted closing prices from Yahoo Finance."""
    raw = yf.download(
        TICKERS,
        start=START_DATE.isoformat(),
        end=END_DATE.isoformat(),
        auto_adjust=True,
        progress=False,
    )
    # yfinance returns a MultiIndex DataFrame; pull the Close level
    close = raw["Close"][TICKERS].dropna()

    # Build Polars DataFrame without pyarrow by passing a plain dict
    prices = pl.DataFrame(
        {
            "date": [d.date() for d in close.index.to_pydatetime()],
            **{ticker: close[ticker].tolist() for ticker in TICKERS},
        }
    )

    mo.md(
        f"""
        **Prices loaded** ✅

        - Tickers: `{TICKERS}`
        - Rows: `{prices.height:,}`
        - Date range: `{prices["date"].min()}` → `{prices["date"].max()}`
        """
    )
    return (prices,)


@app.cell
def cell_build_positions(prices):
    """Build a constant share position of SHARES units per asset."""
    n = prices.height
    positions = prices.select("date").with_columns([pl.lit(float(SHARES)).alias(ticker) for ticker in TICKERS])
    mo.md(
        f"""
        **Positions constructed** ✅

        - `{SHARES:,}` shares held in each of `{TICKERS}` for every trading day
        - Total rows: `{n:,}`
        """
    )
    return (positions,)


@app.cell
def cell_build_portfolio(prices, positions):
    """Create the Portfolio from share positions."""
    # AUM = initial market value of the positions
    first_row = prices.filter(pl.col("date") == prices["date"].min())
    aum = sum(first_row[ticker][0] * SHARES for ticker in TICKERS)

    portfolio = Portfolio.from_position(
        prices=prices,
        position=positions,
        aum=float(aum),
    )
    mo.md(
        f"""
        **Portfolio created** ✅

        - Assets: `{portfolio.assets}`
        - Initial AUM: `${aum:,.0f}`
        """
    )
    return (portfolio,)


@app.cell
def cell_snapshot(portfolio):
    """Render the portfolio snapshot chart."""
    mo.md("## 📊 Portfolio Snapshot")
    fig = portfolio.plots.snapshot()
    return (fig,)


@app.cell
def cell_snapshot_show(fig) -> None:
    """Display snapshot figure."""
    fig
    return


@app.cell
def cell_report(portfolio):
    """Generate the full HTML portfolio report."""
    mo.md("## 📋 HTML Report")
    html = portfolio.report.to_html(title="META + AAPL — 500 Shares Each")
    mo.md(f"Report generated: **{len(html):,}** characters.")
    return (html,)


@app.cell
def cell_report_export(html) -> None:
    """Save the report to NOTEBOOK_OUTPUT_FOLDER if set, otherwise skip."""
    _output_folder = os.environ.get("NOTEBOOK_OUTPUT_FOLDER")
    if _output_folder:
        _path = Path(_output_folder) / "yfinance_report.html"
        _path.write_text(html)
        mo.md(f"✅ Report saved to `{_path}`")
    else:
        mo.md("ℹ️ `NOTEBOOK_OUTPUT_FOLDER` is not set — artifact saving skipped (set automatically by `rhiza_marimo`).")
    return


if __name__ == "__main__":
    app.run()
