import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    from pathlib import Path

    import polars as pl

    from jquantstats.api import build_data

    path = Path(__file__).parent
    print(path)
    return build_data, path, pl


@app.cell
def _(path, pl):
    returns = pl.read_csv(path / "data" / "portfolio.csv", try_parse_dates=True).with_columns(
        [
            pl.col("AAPL").cast(pl.Float64, strict=False),
            pl.col("META").cast(pl.Float64, strict=False),
            pl.col("Date").cast(pl.Date, strict=False),
        ]
    )

    benchmark = pl.read_csv(path / "data" / "benchmark.csv", try_parse_dates=True)
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
    data.all_pd

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
    fig = data.plots.plot_snapshot(log_scale=True)
    fig.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
