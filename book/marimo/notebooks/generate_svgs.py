# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "jquantstats",
#     "kaleido==1.2.0",
#     "polars>=1.0.0",
#     "plotly>=6.0.0",
# ]
# [tool.uv.sources]
# jquantstats = { path = "../../..", editable = true }
# ///

"""Generate SVG charts for the 1/n (equal-weight) portfolio.

Loads AAPL + META daily returns from the bundled data directory, constructs
a simple 1/n equal-weight portfolio (½ AAPL + ½ META), and exports a set of
publication-quality SVG files to ``assets/`` at the repository root.

Usage::

    uv run book/marimo/notebooks/generate_svgs.py
"""

from pathlib import Path

import polars as pl

from jquantstats import Data

# ── Paths ─────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent
_DATA = _HERE / "data"
_ASSETS = _HERE.parent.parent.parent / "assets"


def main() -> None:
    """Load data, build the 1/n portfolio, and write SVG charts to ``assets/``."""
    _ASSETS.mkdir(exist_ok=True)

    # ── Load raw returns ──────────────────────────────────────────────────────

    portfolio_raw = pl.read_csv(_DATA / "portfolio.csv", try_parse_dates=True).with_columns(
        [
            pl.col("AAPL").cast(pl.Float64, strict=False),
            pl.col("META").cast(pl.Float64, strict=False),
            pl.col("Date").cast(pl.Date, strict=False),
        ]
    )

    benchmark_raw = pl.read_csv(_DATA / "benchmark.csv", try_parse_dates=True)

    # ── 1/n equal-weight portfolio return ────────────────────────────────────

    assets = [c for c in portfolio_raw.columns if c != "Date"]
    n = len(assets)

    # Compute equal-weight combined daily return and keep all individual assets
    equal_weight_returns = portfolio_raw.with_columns((sum(pl.col(a) for a in assets) / n).alias("1/n Portfolio"))

    # Series showing only the 1/n portfolio vs benchmark
    portfolio_1n = equal_weight_returns.select(["Date", "1/n Portfolio"])

    # Series showing all individual assets + 1/n portfolio vs benchmark
    all_assets_cols = equal_weight_returns.select(["Date", *assets, "1/n Portfolio"])

    # ── Build Data objects ────────────────────────────────────────────────────

    data_1n = Data.from_returns(returns=portfolio_1n, benchmark=benchmark_raw)
    data_all = Data.from_returns(returns=all_assets_cols, benchmark=benchmark_raw)

    # ── Generate and save SVGs ────────────────────────────────────────────────

    charts = {
        "snapshot_1n": data_1n.plots.plot_snapshot(title="1/n Equal-Weight Portfolio (AAPL + META)"),
        "snapshot_all": data_all.plots.plot_snapshot(title="AAPL vs META vs 1/n Equal-Weight Portfolio"),
    }

    for name, fig in charts.items():
        path = _ASSETS / f"{name}.svg"
        fig.write_image(str(path), format="svg")
        print(f"Saved {path}")

    print("Done — SVG files written to", _ASSETS)


if __name__ == "__main__":
    main()
