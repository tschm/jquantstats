# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.20.4",
#     "jquantstats",
#     "kaleido==1.2.0",
#     "polars>=1.0.0",
#     "plotly>=6.0.0",
# ]
# [tool.uv.sources]
# jquantstats = { path = "../../..", editable = true }
# ///

from pathlib import Path

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")

_HERE = Path(__file__).parent
_DATA = _HERE / "data"
_ASSETS = _HERE.parent.parent.parent / "assets"

with app.setup:
    import marimo as mo
    import polars as pl

    from jquantstats import Data


@app.cell
def cell_intro() -> None:
    """Render the generate_svgs introduction."""
    mo.md(
        r"""
        # 🖼️ jquantstats — SVG Chart Generator

        This notebook builds a **1/n equal-weight portfolio** from AAPL + META daily
        returns and exports publication-quality SVG files to the repository `assets/`
        directory.

        It covers:
        1. 📂 **Data loading** — read AAPL + META returns and benchmark from CSV
        2. ⚖️ **Portfolio construction** — equal-weight (½ AAPL + ½ META) daily return
        3. 📊 **Chart generation** — snapshot plots via `Data.plots.plot_snapshot(...)`
        4. 💾 **SVG export** — write charts to `assets/` using Kaleido
        """
    )
    return


@app.cell
def cell_load_data():
    """Load raw returns and benchmark from CSV files."""
    portfolio_raw = pl.read_csv(_DATA / "portfolio.csv", try_parse_dates=True).with_columns(
        [
            pl.col("AAPL").cast(pl.Float64, strict=False),
            pl.col("META").cast(pl.Float64, strict=False),
            pl.col("Date").cast(pl.Date, strict=False),
        ]
    )

    benchmark_raw = pl.read_csv(_DATA / "benchmark.csv", try_parse_dates=True)

    mo.md(
        f"""
        **Data loaded** ✅

        - Portfolio rows: `{portfolio_raw.height}`
        - Portfolio columns: `{portfolio_raw.columns}`
        - Benchmark rows: `{benchmark_raw.height}`
        """
    )
    return benchmark_raw, portfolio_raw


@app.cell
def cell_portfolio(benchmark_raw, portfolio_raw):
    """Construct the 1/n equal-weight portfolio and individual-asset series."""
    assets = [c for c in portfolio_raw.columns if c != "Date"]
    n = len(assets)

    equal_weight_returns = portfolio_raw.with_columns((sum(pl.col(a) for a in assets) / n).alias("1/n Portfolio"))

    portfolio_1n = equal_weight_returns.select(["Date", "1/n Portfolio"])
    all_assets_cols = equal_weight_returns.select(["Date", *assets, "1/n Portfolio"])

    mo.md(
        f"""
        **1/n portfolio constructed** ✅

        - Assets: `{assets}`
        - Equal weight: `{1 / n:.1%}` each
        - Series built: `portfolio_1n`, `all_assets_cols`
        """
    )
    return all_assets_cols, assets, benchmark_raw, equal_weight_returns, n, portfolio_1n


@app.cell
def cell_data_objects(all_assets_cols, benchmark_raw, portfolio_1n):
    """Wrap returns in Data objects."""
    data_1n = Data.from_returns(returns=portfolio_1n, benchmark=benchmark_raw)
    data_all = Data.from_returns(returns=all_assets_cols, benchmark=benchmark_raw)

    mo.md("**Data objects created** ✅  (`data_1n`, `data_all`)")
    return data_1n, data_all


@app.cell
def cell_charts(data_1n, data_all):
    """Generate snapshot figures."""
    fig_1n = data_1n.plots.plot_snapshot(title="1/n Equal-Weight Portfolio (AAPL + META)")
    fig_all = data_all.plots.plot_snapshot(title="AAPL vs META vs 1/n Equal-Weight Portfolio")

    mo.md("## 📊 1/n Portfolio Snapshot")
    return fig_1n, fig_all


@app.cell
def cell_display_1n(fig_1n) -> None:
    """Display 1/n portfolio snapshot."""
    fig_1n


@app.cell
def cell_display_all(fig_all) -> None:
    """Display all-assets snapshot."""
    mo.md("## 📊 All Assets Snapshot")
    fig_all


@app.cell
def cell_save(fig_1n, fig_all) -> None:
    """Save SVG files to assets/."""
    _ASSETS.mkdir(exist_ok=True)

    charts = {
        "snapshot_1n": fig_1n,
        "snapshot_all": fig_all,
    }

    saved = []
    for name, fig in charts.items():
        path = _ASSETS / f"{name}.svg"
        fig.write_image(str(path), format="svg")
        saved.append(str(path))

    mo.md("**SVGs saved** ✅\n\n" + "\n".join(f"- `{p}`" for p in saved))


if __name__ == "__main__":
    app.run()
