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
    """Render the risk metrics introduction."""
    mo.md(
        r"""
        # ⚠️ jquantstats — Risk Metrics Deep Dive

        This notebook explores the full breadth of **risk and downside metrics**
        available in `jquantstats`.  All metrics are computed on real AAPL + META
        daily returns (≈ 11 k rows) with SPY as the benchmark.

        | Group | Metrics covered |
        |-------|-----------------|
        | **Tail risk** | VaR, CVaR, tail ratio, serenity index |
        | **Drawdown** | Max DD, avg DD, max DD duration, worst periods |
        | **Win/loss** | Win rate, payoff ratio, profit factor, gain-to-pain |
        | **Position sizing** | Kelly criterion, risk-of-ruin |
        | **Distribution shape** | Skew, kurtosis, outlier ratios |
        | **Autocorrelation** | ACF, autocorr penalty, smart ratios |
        | **Benchmark** | Up/down capture, alpha, beta, information ratio |
        """
    )
    return


@app.cell
def cell_load_data():
    """Load portfolio returns and benchmark from CSV."""
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
        - Benchmark: SPY
        - Rows: `{returns_df.height:,}`
        - Date range: `{returns_df["Date"].min()}` → `{returns_df["Date"].max()}`
        """
    )
    return (data,)


# ── Tail risk ─────────────────────────────────────────────────────────────────


@app.cell
def cell_tail_header() -> None:
    """Tail risk section header."""
    mo.md(
        """
        ---
        ## 1 · Tail risk — VaR, CVaR, tail ratio, serenity index

        - **VaR (5%)** — the worst daily return we expect to beat 95 % of the time
        - **CVaR (5%)** — expected loss *given* we are in the worst 5 % of days
        - **Tail ratio** — abs(95th-pct gain) / abs(5th-pct loss); > 1 means fat right tail
        - **Serenity index** — CAGR / (max drawdown × downside volatility)
        """
    )
    return


@app.cell
def cell_tail_metrics(data):
    """Compute tail risk metrics."""
    var_5 = data.stats.value_at_risk(alpha=0.05)
    cvar_5 = data.stats.conditional_value_at_risk(confidence=0.95)
    tail = data.stats.tail_ratio()
    serenity = data.stats.serenity_index()
    return (cvar_5, serenity, tail, var_5)


@app.cell
def cell_tail_table(var_5, cvar_5, tail, serenity, data) -> None:
    """Display tail risk table."""
    _rows = [
        {"Metric": "VaR (5%)", **{a: f"{var_5[a]:.4f}" for a in data.assets}},
        {"Metric": "CVaR (5%)", **{a: f"{cvar_5[a]:.4f}" for a in data.assets}},
        {"Metric": "Tail Ratio", **{a: f"{tail[a]:.3f}" for a in data.assets}},
        {"Metric": "Serenity Index", **{a: f"{serenity[a]:.3f}" for a in data.assets}},
    ]
    mo.plain_text(str(pl.DataFrame(_rows)))
    return


@app.cell
def cell_var_chart(data):
    """Plot daily return distribution with VaR/CVaR annotations."""
    _var_5 = data.stats.value_at_risk(alpha=0.05)
    _cvar_5 = data.stats.conditional_value_at_risk(confidence=0.95)

    fig_var = go.Figure()
    _all_df = data.all

    for _asset in data.assets:
        _returns = _all_df[_asset].drop_nulls().cast(pl.Float64).to_list()
        fig_var.add_trace(
            go.Histogram(
                x=_returns,
                name=_asset,
                opacity=0.6,
                nbinsx=120,
                histnorm="probability",
            )
        )
        fig_var.add_vline(
            x=_var_5[_asset],
            line_dash="dash",
            annotation_text=f"{_asset} VaR",
            annotation_position="top right",
        )
        fig_var.add_vline(
            x=_cvar_5[_asset],
            line_dash="dot",
            annotation_text=f"{_asset} CVaR",
            annotation_position="top left",
        )

    fig_var.update_layout(
        title="Daily Return Distribution with VaR (dashed) and CVaR (dotted) at 5%",
        xaxis_title="Daily Return",
        yaxis_title="Probability",
        barmode="overlay",
        height=440,
    )
    return (fig_var,)


@app.cell
def cell_var_chart_show(fig_var) -> None:
    """Display VaR chart."""
    fig_var
    return


# ── Drawdown analysis ─────────────────────────────────────────────────────────


@app.cell
def cell_dd_header() -> None:
    """Drawdown section header."""
    mo.md(
        """
        ---
        ## 2 · Drawdown analysis

        - **Max drawdown** — peak-to-trough decline
        - **Avg drawdown** — average of all underwater excursions
        - **Max DD duration** — longest time-to-recovery in periods
        - **Ulcer index** — RMS of drawdown series (penalises long, deep drawdowns)
        - **Ulcer Performance Index** — return / ulcer index
        - **Recovery factor** — total return / max drawdown magnitude
        - **Calmar ratio** — CAGR / max drawdown magnitude
        """
    )
    return


@app.cell
def cell_dd_metrics(data):
    """Compute drawdown family metrics."""
    max_dd = data.stats.max_drawdown()
    avg_dd = data.stats.avg_drawdown()
    max_dur = data.stats.max_drawdown_duration()
    ulcer = data.stats.ulcer_index()
    upi = data.stats.ulcer_performance_index()
    recovery = data.stats.recovery_factor()
    calmar = data.stats.calmar()
    return (avg_dd, calmar, max_dd, max_dur, recovery, ulcer, upi)


@app.cell
def cell_dd_table(max_dd, avg_dd, max_dur, ulcer, upi, recovery, calmar, data) -> None:
    """Display drawdown metrics table."""
    _rows = [
        {"Metric": "Max Drawdown", **{a: f"{max_dd[a]:.4f}" for a in data.assets}},
        {"Metric": "Avg Drawdown", **{a: f"{avg_dd[a]:.4f}" for a in data.assets}},
        {"Metric": "Max DD Duration (days)", **{a: f"{max_dur[a]:.0f}" for a in data.assets}},
        {"Metric": "Ulcer Index", **{a: f"{ulcer[a]:.4f}" for a in data.assets}},
        {"Metric": "Ulcer Performance Index", **{a: f"{upi[a]:.3f}" for a in data.assets}},
        {"Metric": "Recovery Factor", **{a: f"{recovery[a]:.3f}" for a in data.assets}},
        {"Metric": "Calmar Ratio", **{a: f"{calmar[a]:.3f}" for a in data.assets}},
    ]
    mo.plain_text(str(pl.DataFrame(_rows)))
    return


@app.cell
def cell_worst_periods(data):
    """Show worst-5 periods per asset."""
    mo.md("### Worst 5 periods per asset")
    worst = data.stats.worst_n_periods(n=5)
    # worst is {asset: [val1, val2, ...]} — reformat as a DataFrame
    max_len = max(len(v) for v in worst.values())
    _rows = []
    for i in range(max_len):
        _row = {"Rank": i + 1}
        for asset, vals in worst.items():
            _row[asset] = f"{vals[i]:.4f}" if i < len(vals) else ""
        _rows.append(_row)
    worst_df = pl.DataFrame(_rows)
    return (worst_df,)


@app.cell
def cell_worst_periods_show(worst_df) -> None:
    """Display worst periods."""
    mo.plain_text(str(worst_df))
    return


# ── Win/loss metrics ──────────────────────────────────────────────────────────


@app.cell
def cell_winloss_header() -> None:
    """Win/loss section header."""
    mo.md(
        """
        ---
        ## 3 · Win/loss metrics

        - **Win rate** — fraction of positive periods
        - **Monthly win rate** — fraction of positive months
        - **Payoff ratio** — avg win / |avg loss|
        - **Profit factor** — sum of wins / |sum of losses|
        - **Gain-to-pain ratio** — total gains / total |losses|
        - **CPC index** — win rate × payoff ratio (common sense ratio variant)
        """
    )
    return


@app.cell
def cell_winloss_metrics(data):
    """Compute win/loss metrics."""
    win_rate = data.stats.win_rate()
    monthly_win = data.stats.monthly_win_rate()
    payoff = data.stats.payoff_ratio()
    profit_f = data.stats.profit_factor()
    g2p = data.stats.gain_to_pain_ratio()
    cpc = data.stats.cpc_index()
    return (cpc, g2p, monthly_win, payoff, profit_f, win_rate)


@app.cell
def cell_winloss_table(win_rate, monthly_win, payoff, profit_f, g2p, cpc, data) -> None:
    """Display win/loss metrics table."""
    _rows = [
        {"Metric": "Win Rate (daily)", **{a: f"{win_rate[a]:.3f}" for a in data.assets}},
        {"Metric": "Monthly Win Rate", **{a: f"{monthly_win[a]:.3f}" for a in data.assets}},
        {"Metric": "Payoff Ratio", **{a: f"{payoff[a]:.3f}" for a in data.assets}},
        {"Metric": "Profit Factor", **{a: f"{profit_f[a]:.3f}" for a in data.assets}},
        {"Metric": "Gain-to-Pain", **{a: f"{g2p[a]:.3f}" for a in data.assets}},
        {"Metric": "CPC Index", **{a: f"{cpc[a]:.3f}" for a in data.assets}},
    ]
    mo.plain_text(str(pl.DataFrame(_rows)))
    return


# ── Kelly criterion & position sizing ─────────────────────────────────────────


@app.cell
def cell_kelly_header() -> None:
    """Kelly criterion section header."""
    mo.md(
        """
        ---
        ## 4 · Kelly criterion & risk-of-ruin

        - **Kelly criterion** — theoretically optimal leverage fraction: `win_rate - (1 - win_rate) / payoff_ratio`
        - **Risk-of-ruin** — probability of losing a fixed fraction of capital under the current return distribution

        Kelly > 1 means even full Kelly is under-leveraged relative to the edge;
        in practice use half-Kelly or quarter-Kelly to account for estimation error.
        """
    )
    return


@app.cell
def cell_kelly_metrics(data):
    """Compute Kelly criterion and risk-of-ruin."""
    kelly = data.stats.kelly_criterion()
    ror = data.stats.risk_of_ruin()
    return (kelly, ror)


@app.cell
def cell_kelly_table(kelly, ror, data) -> None:
    """Display Kelly / risk-of-ruin table."""
    _rows = [
        {"Metric": "Kelly Criterion", **{a: f"{kelly[a]:.4f}" for a in data.assets}},
        {"Metric": "Risk of Ruin", **{a: f"{ror[a]:.6f}" for a in data.assets}},
    ]
    mo.plain_text(str(pl.DataFrame(_rows)))
    return


# ── Distribution shape ────────────────────────────────────────────────────────


@app.cell
def cell_shape_header() -> None:
    """Distribution shape section header."""
    mo.md(
        """
        ---
        ## 5 · Distribution shape — skew, kurtosis, outliers

        A normal distribution has skew = 0 and excess kurtosis = 0.
        Financial returns typically exhibit **negative skew** (larger left tail) and
        **positive excess kurtosis** (fat tails).
        """
    )
    return


@app.cell
def cell_shape_metrics(data):
    """Compute distribution shape metrics."""
    skew = data.stats.skew()
    kurt = data.stats.kurtosis()
    # outliers() returns a Series of extreme returns; len gives the count
    outlier_series = data.stats.outliers()
    outlier_count = {a: len(outlier_series[a]) for a in data.assets}
    out_win = data.stats.outlier_win_ratio()
    out_loss = data.stats.outlier_loss_ratio()
    return (kurt, out_loss, out_win, outlier_count, skew)


@app.cell
def cell_shape_table(skew, kurt, outlier_count, out_win, out_loss, data) -> None:
    """Display distribution shape table."""
    _rows = [
        {"Metric": "Skewness", **{a: f"{skew[a]:.4f}" for a in data.assets}},
        {"Metric": "Excess Kurtosis", **{a: f"{kurt[a]:.4f}" for a in data.assets}},
        {"Metric": "Outlier Count (>P95)", **{a: str(outlier_count[a]) for a in data.assets}},
        {"Metric": "Outlier Win Ratio", **{a: f"{out_win[a]:.4f}" for a in data.assets}},
        {"Metric": "Outlier Loss Ratio", **{a: f"{out_loss[a]:.4f}" for a in data.assets}},
    ]
    mo.plain_text(str(pl.DataFrame(_rows)))
    return


# ── Autocorrelation ───────────────────────────────────────────────────────────


@app.cell
def cell_acf_header() -> None:
    """Autocorrelation section header."""
    mo.md(
        """
        ---
        ## 6 · Autocorrelation & smart ratios

        Return autocorrelation matters for risk: a strategy with positively
        autocorrelated returns (trending) has **higher true volatility** than
        the i.i.d. estimate suggests.  The `autocorr_penalty` adjusts for this.

        - **Lag-1 autocorrelation** — correlation between consecutive returns
        - **Autocorr penalty** — multiplicative adjustment to the Sharpe denominator
        - **Smart Sharpe** — Sharpe × autocorr penalty (lower when returns are trending)
        - **Smart Sortino** — Sortino × autocorr penalty
        """
    )
    return


@app.cell
def cell_acf_metrics(data):
    """Compute autocorrelation metrics."""
    ac1 = data.stats.autocorr(lag=1)
    penalty = data.stats.autocorr_penalty()
    smart_sharpe = data.stats.smart_sharpe()
    sharpe = data.stats.sharpe()
    smart_sortino = data.stats.smart_sortino()
    sortino = data.stats.sortino()
    return (ac1, penalty, sharpe, smart_sharpe, smart_sortino, sortino)


@app.cell
def cell_acf_table(ac1, penalty, sharpe, smart_sharpe, sortino, smart_sortino, data) -> None:
    """Display autocorrelation metrics table."""
    _rows = [
        {"Metric": "Lag-1 Autocorrelation", **{a: f"{ac1[a]:.4f}" for a in data.assets}},
        {"Metric": "Autocorr Penalty", **{a: f"{penalty[a]:.4f}" for a in data.assets}},
        {"Metric": "Sharpe (standard)", **{a: f"{sharpe[a]:.4f}" for a in data.assets}},
        {"Metric": "Smart Sharpe", **{a: f"{smart_sharpe[a]:.4f}" for a in data.assets}},
        {"Metric": "Sortino (standard)", **{a: f"{sortino[a]:.4f}" for a in data.assets}},
        {"Metric": "Smart Sortino", **{a: f"{smart_sortino[a]:.4f}" for a in data.assets}},
    ]
    mo.plain_text(str(pl.DataFrame(_rows)))
    return


# ── Benchmark analytics ───────────────────────────────────────────────────────


@app.cell
def cell_bench_header() -> None:
    """Benchmark analytics section header."""
    mo.md(
        """
        ---
        ## 7 · Benchmark analytics — alpha, beta, capture ratios

        Requires a benchmark series in the `Data` object (here: SPY).

        - **Alpha** — annualised return unexplained by the benchmark
        - **Beta** — sensitivity of returns to benchmark movements
        - **R²** — fraction of variance explained by the benchmark
        - **Up capture** — return in up-market months relative to benchmark
        - **Down capture** — return in down-market months relative to benchmark
        - **Information ratio** — (return − benchmark) / tracking error
        """
    )
    return


@app.cell
def cell_bench_metrics(data):
    """Compute benchmark-relative metrics."""
    bench_series = data.benchmark[data.benchmark.columns[0]]
    greeks = data.stats.greeks()
    up_cap = data.stats.up_capture(benchmark=bench_series)
    down_cap = data.stats.down_capture(benchmark=bench_series)
    ir = data.stats.information_ratio()
    r2 = data.stats.r_squared()
    return (bench_series, down_cap, greeks, ir, r2, up_cap)


@app.cell
def cell_bench_table(greeks, up_cap, down_cap, ir, r2, data) -> None:
    """Display benchmark metrics table."""
    _rows = [
        {"Metric": "Alpha (ann.)", **{a: f"{greeks[a].get('alpha', float('nan')):.4f}" for a in data.assets}},
        {"Metric": "Beta", **{a: f"{greeks[a].get('beta', float('nan')):.4f}" for a in data.assets}},
        {"Metric": "R²", **{a: f"{r2[a]:.4f}" for a in data.assets}},
        {"Metric": "Information Ratio", **{a: f"{ir[a]:.4f}" for a in data.assets}},
        {"Metric": "Up Capture", **{a: f"{up_cap[a]:.4f}" for a in data.assets}},
        {"Metric": "Down Capture", **{a: f"{down_cap[a]:.4f}" for a in data.assets}},
    ]
    mo.plain_text(str(pl.DataFrame(_rows)))
    return


@app.cell
def cell_capture_chart(data, bench_series):
    """Plot up vs down capture scatter."""
    _up_cap = data.stats.up_capture(benchmark=bench_series)
    _down_cap = data.stats.down_capture(benchmark=bench_series)

    fig_cap = go.Figure()
    for _asset in data.assets:
        fig_cap.add_trace(
            go.Scatter(
                x=[_down_cap[_asset]],
                y=[_up_cap[_asset]],
                mode="markers+text",
                name=_asset,
                text=[_asset],
                textposition="top center",
                marker={"size": 14},
            )
        )

    # Diagonal: up = down capture (neutral)
    lo = min(list(_down_cap.values()) + list(_up_cap.values())) * 0.9
    hi = max(list(_down_cap.values()) + list(_up_cap.values())) * 1.1
    fig_cap.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            name="Benchmark (1:1)",
            line={"dash": "dash", "color": "grey"},
            showlegend=True,
        )
    )

    fig_cap.update_layout(
        title="Up vs Down Capture vs SPY<br><sup>Above the diagonal = better relative performance</sup>",
        xaxis_title="Down Capture Ratio",
        yaxis_title="Up Capture Ratio",
        height=440,
    )
    return (fig_cap,)


@app.cell
def cell_capture_chart_show(fig_cap) -> None:
    """Display capture ratio chart."""
    fig_cap
    return


# ── Rolling risk overview ─────────────────────────────────────────────────────


@app.cell
def cell_rolling_risk_header() -> None:
    """Rolling risk section header."""
    mo.md(
        """
        ---
        ## 8 · Rolling risk overview

        Rolling windows reveal how risk characteristics evolve through time.
        A 126-day (≈ 6-month) window is used here.
        """
    )
    return


@app.cell
def cell_rolling_vol(data):
    """Rolling volatility chart."""
    fig_rv = data.plots.rolling_volatility(rolling_period=126, title="Rolling Annualised Volatility (126-day)")
    return (fig_rv,)


@app.cell
def cell_rolling_vol_show(fig_rv) -> None:
    """Display rolling volatility chart."""
    fig_rv
    return


@app.cell
def cell_rolling_beta(data):
    """Rolling beta chart."""
    fig_rb = data.plots.rolling_beta(rolling_period=126, title="Rolling Beta vs SPY (126-day)")
    return (fig_rb,)


@app.cell
def cell_rolling_beta_show(fig_rb) -> None:
    """Display rolling beta chart."""
    fig_rb
    return


if __name__ == "__main__":
    app.run()
