"""Demo for jquantstats."""

# /// script
# dependencies = [
#     "marimo==0.18.4",
#     "jquantstats",
# ]
#
# [tool.uv.sources]
# jquantstats = { path = "../..", editable=true }
#
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")

with app.setup:
    import plotly.io as pio
    import polars as pl

    import jquantstats as jqs

    # Ensure Plotly works with Marimo
    pio.renderers.default = "plotly_mimetype"


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _download(mo):
    returns = pl.read_csv(str(mo.notebook_location() / "public" / "portfolio.csv"), try_parse_dates=True).with_columns(
        [
            pl.col("AAPL").cast(pl.Float64, strict=False),
            pl.col("META").cast(pl.Float64, strict=False),
            pl.col("Date").cast(pl.Date, strict=False),
        ]
    )

    benchmark = pl.read_csv(str(mo.notebook_location() / "public" / "benchmark.csv"), try_parse_dates=True)
    return returns, benchmark


@app.cell
def _(benchmark, returns):
    data = jqs.build_data(returns=returns, benchmark=benchmark, date_col="Date")
    return data


@app.cell
def _(data):
    fig = data.plots.plot_snapshot(log_scale=True)
    fig
    return


if __name__ == "__main__":
    app.run()
