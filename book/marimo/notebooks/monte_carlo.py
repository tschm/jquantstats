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
    from pathlib import Path

    import marimo as mo
    import plotly.graph_objects as go
    import polars as pl

    from jquantstats import Data

    NOTEBOOK_DIR = Path(__file__).parent


@app.cell
def cell_intro() -> None:
    """Render the Monte Carlo notebook introduction."""
    mo.md(
        r"""
        # 🎲 jquantstats — Monte Carlo Simulation

        This notebook demonstrates the **block-bootstrap Monte Carlo** engine built into `jquantstats`.

        Block bootstrap preserves the autocorrelation structure of returns by resampling
        contiguous blocks rather than individual observations — giving more realistic
        simulated paths than plain i.i.d. resampling.

        | Section | What you get |
        |---------|--------------|
        | **Fan chart** | `data.plots.montecarlo()` — simulated paths vs observed |
        | **Metric distributions** | `data.plots.montecarlo_distribution()` for Sharpe, drawdown, CAGR |
        | **Raw simulation data** | `data.stats.montecarlo_*()` — percentile tables |

        **Data:** real AAPL + META daily returns (≈ 11 k rows) with SPY as benchmark.
        """
    )
    return


@app.cell
def cell_load_data():
    """Load portfolio returns and SPY benchmark from CSV."""
    data_dir = NOTEBOOK_DIR / "data"

    returns_df = pl.read_csv(data_dir / "portfolio.csv", try_parse_dates=True).with_columns(
        [
            pl.col("AAPL").cast(pl.Float64, strict=False),
            pl.col("META").cast(pl.Float64, strict=False),
            pl.col("Date").cast(pl.Date, strict=False),
        ]
    )
    benchmark_df = pl.read_csv(data_dir / "benchmark.csv", try_parse_dates=True).select(["Date", "SPY -- Benchmark"])
    data = Data.from_returns(returns=returns_df, benchmark=benchmark_df)

    mo.md(
        f"""
        **Data loaded** ✅

        - Assets: `{data.assets}`
        - Rows: `{returns_df.height:,}`
        - Date range: `{returns_df["Date"].min()}` → `{returns_df["Date"].max()}`

        Each Monte Carlo run samples **block-bootstrap paths** of 252 trading-day
        horizon from the full return history.
        """
    )
    return (data,)


# ── Fan chart ─────────────────────────────────────────────────────────────────


@app.cell
def cell_fan_header() -> None:
    """Fan chart section header."""
    mo.md(
        """
        ---
        ## 1 · Simulated paths vs observed — `data.plots.montecarlo()`

        Each faint line is one block-bootstrap simulation; the bold line is the
        **observed trailing-year** cumulative return.  The spread of the fan
        represents the range of outcomes consistent with historical return dynamics.
        """
    )
    return


@app.cell
def cell_fan_chart(data):
    """Generate Monte Carlo fan chart (100 paths, 252-day horizon)."""
    fig_fan = data.plots.montecarlo(
        n=100,
        period=252,
        title="Monte Carlo Fan Chart — AAPL & META (100 paths, 1-year horizon)",
    )
    return (fig_fan,)


@app.cell
def cell_fan_chart_show(fig_fan) -> None:
    """Display fan chart."""
    fig_fan
    return


# ── Metric distributions ──────────────────────────────────────────────────────


@app.cell
def cell_dist_header() -> None:
    """Metric distributions section header."""
    mo.md(
        """
        ---
        ## 2 · Metric distributions — `data.plots.montecarlo_distribution()`

        Running 1 000 simulations and plotting the distribution of three key metrics
        reveals the **uncertainty band** around each statistic.  The vertical line
        marks the historically observed value.
        """
    )
    return


@app.cell
def cell_sharpe_dist(data):
    """Monte Carlo Sharpe ratio distribution."""
    mo.md("### 2a · Sharpe ratio distribution")
    fig_sharpe_dist = data.plots.montecarlo_distribution(
        n=1000,
        period=252,
        metric="sharpe",
        title="Simulated Sharpe Ratio Distribution (1 000 paths, 1-year)",
    )
    return (fig_sharpe_dist,)


@app.cell
def cell_sharpe_dist_show(fig_sharpe_dist) -> None:
    """Display Sharpe distribution."""
    fig_sharpe_dist
    return


@app.cell
def cell_dd_dist(data):
    """Monte Carlo drawdown distribution."""
    mo.md("### 2b · Max drawdown distribution")
    fig_dd_dist = data.plots.montecarlo_distribution(
        n=1000,
        period=252,
        metric="drawdown",
        title="Simulated Max Drawdown Distribution (1 000 paths, 1-year)",
    )
    return (fig_dd_dist,)


@app.cell
def cell_dd_dist_show(fig_dd_dist) -> None:
    """Display drawdown distribution."""
    fig_dd_dist
    return


@app.cell
def cell_cagr_dist(data):
    """Monte Carlo CAGR distribution."""
    mo.md("### 2c · CAGR distribution")
    fig_cagr_dist = data.plots.montecarlo_distribution(
        n=1000,
        period=252,
        metric="cagr",
        title="Simulated CAGR Distribution (1 000 paths, 1-year)",
    )
    return (fig_cagr_dist,)


@app.cell
def cell_cagr_dist_show(fig_cagr_dist) -> None:
    """Display CAGR distribution."""
    fig_cagr_dist
    return


# ── Raw simulation data & percentile tables ───────────────────────────────────


@app.cell
def cell_raw_header() -> None:
    """Raw simulation data section header."""
    mo.md(
        """
        ---
        ## 3 · Percentile tables from raw simulation data

        `data.stats.montecarlo_*()` returns a `(n_simulations × n_assets)` DataFrame.
        We summarise it into percentile tables to answer questions like:
        *"What Sharpe can we expect in the worst 5% of years?"*
        """
    )
    return


@app.cell
def cell_mc_sharpe(data):
    """Run raw Monte Carlo Sharpe simulations."""
    mc_sharpe = data.stats.montecarlo_sharpe(n=2000, period=252)
    return (mc_sharpe,)


@app.cell
def cell_mc_dd(data):
    """Run raw Monte Carlo drawdown simulations."""
    mc_dd = data.stats.montecarlo_drawdown(n=2000, period=252)
    return (mc_dd,)


@app.cell
def cell_mc_cagr(data):
    """Run raw Monte Carlo CAGR simulations."""
    mc_cagr = data.stats.montecarlo_cagr(n=2000, period=252)
    return (mc_cagr,)


@app.cell
def cell_percentile_table(mc_sharpe, mc_dd, mc_cagr, data) -> None:
    """Build and display percentile summary table."""
    quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    rows = []
    for _asset in data.assets:
        sharpe_q = mc_sharpe[_asset].quantile(quantiles, interpolation="linear")
        dd_q = mc_dd[_asset].quantile(quantiles, interpolation="linear")
        cagr_q = mc_cagr[_asset].quantile(quantiles, interpolation="linear")
        for i, q in enumerate(quantiles):
            rows.append(
                {
                    "Asset": _asset,
                    "Percentile": f"P{int(q * 100):02d}",
                    "Sharpe": round(float(sharpe_q[i]), 3),
                    "Max Drawdown (%)": round(float(dd_q[i]) * 100, 2),
                    "CAGR (%)": round(float(cagr_q[i]) * 100, 2),
                }
            )

    summary_df = pl.DataFrame(rows)
    mo.md("### Percentile summary across 2 000 simulations (1-year horizon)")
    mo.plain_text(str(summary_df))
    return


# ── Horizon sensitivity ───────────────────────────────────────────────────────


@app.cell
def cell_horizon_header() -> None:
    """Horizon sensitivity section header."""
    mo.md(
        """
        ---
        ## 4 · Horizon sensitivity — how uncertainty changes with time

        Shorter horizons mean more variance in outcomes.  We run the same 500-path
        simulation across three horizons (63, 126, 252 days) and compare the
        interquartile range of the Sharpe distribution for each asset.
        """
    )
    return


@app.cell
def cell_horizon_chart(data):
    """Build horizon-sensitivity chart for Sharpe IQR."""
    _horizons = {"63-day (Qtr)": 63, "126-day (6mo)": 126, "252-day (1yr)": 252}
    fig = go.Figure()

    for _label, _period in _horizons.items():
        _mc = data.stats.montecarlo_sharpe(n=500, period=_period)
        for _asset in data.assets:
            _s = _mc[_asset]
            fig.add_trace(
                go.Box(
                    y=_s.to_list(),
                    name=f"{_asset} / {_label}",
                    boxmean=True,
                )
            )

    fig.update_layout(
        title="Sharpe Ratio Distribution by Horizon (500 paths each)",
        yaxis_title="Simulated Annualised Sharpe",
        xaxis_title="Asset / Horizon",
        height=480,
    )
    return (fig,)


@app.cell
def cell_horizon_chart_show(fig) -> None:
    """Display horizon sensitivity chart."""
    fig
    return


if __name__ == "__main__":
    app.run()
