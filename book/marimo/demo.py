"""Demo for jquantstats."""

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from jquantstats.api import build_data

    return build_data, mo, pl


@app.cell
def _(mo, pl):
    returns = pl.read_csv(str(mo.notebook_location() / "public" / "portfolio.csv"), try_parse_dates=True).with_columns(
        [
            pl.col("AAPL").cast(pl.Float64, strict=False),
            pl.col("META").cast(pl.Float64, strict=False),
            pl.col("Date").cast(pl.Date, strict=False),
        ]
    )

    benchmark = pl.read_csv(str(mo.notebook_location() / "public" / "benchmark.csv"), try_parse_dates=True)

    return benchmark, returns


@app.cell
def _(benchmark, build_data, returns):
    data = build_data(returns=returns, benchmark=benchmark, date_col="Date")
    return (data,)


@app.cell
def _(data):
    data.all
    return


@app.cell
def _(data):
    data.assets
    return


@app.cell
def _(data):
    data.date_col
    return


@app.cell
def _(data):
    data.returns
    return


@app.cell
def _(data):
    data.stats.sharpe()
    return


@app.cell
def _(data):
    data
    fig = data.plots.plot_snapshot(log_scale=True)
    fig
    return


if __name__ == "__main__":
    app.run()
