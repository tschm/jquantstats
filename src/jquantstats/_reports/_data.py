"""Financial report generation from returns data."""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from ._protocol import DataLike

# ── Formatting helpers ────────────────────────────────────────────────────────


def _is_finite(v: object) -> bool:
    """Return True when *v* is a real, finite number."""
    if not isinstance(v, (int, float)):
        return False
    return math.isfinite(float(v))


def _fmt(value: object, fmt: str = ".4f", suffix: str = "") -> str:
    """Format *value* for display; return ``"N/A"`` for non-finite values."""
    if not _is_finite(value):
        return "N/A"
    return f"{float(value):{fmt}}{suffix}"


def _safe(fn: Any, *args: Any, **kwargs: Any) -> dict[str, float]:
    """Call ``fn(*args, **kwargs)`` and return ``{}`` on any exception."""
    try:
        return fn(*args, **kwargs)  # type: ignore[no-any-return]
    except Exception:
        return {}


def _pct(d: dict[str, float]) -> dict[str, float]:
    """Multiply every finite value in *d* by 100."""
    return {k: v * 100.0 if _is_finite(v) else float("nan") for k, v in d.items()}


# ── Period-return helpers ─────────────────────────────────────────────────────


def _comp_since(all_df: pl.DataFrame, date_col: str, asset_cols: list[str], cutoff: Any) -> dict[str, float]:
    """Compounded return for each asset from *cutoff* to the last date."""
    filtered = all_df.filter(pl.col(date_col) >= cutoff)
    result: dict[str, float] = {}
    for col in asset_cols:
        s = filtered[col].drop_nulls().cast(pl.Float64)
        result[col] = float((1.0 + s).product()) - 1.0 if len(s) > 0 else float("nan")
    return result


def _cagr_since(
    all_df: pl.DataFrame,
    date_col: str,
    asset_cols: list[str],
    cutoff: Any,
    periods_per_year: float,
) -> dict[str, float]:
    """Annualised CAGR for each asset from *cutoff* to the last date."""
    filtered = all_df.filter(pl.col(date_col) >= cutoff)
    result: dict[str, float] = {}
    for col in asset_cols:
        s = filtered[col].drop_nulls().cast(pl.Float64)
        n = len(s)
        if n < 2:
            result[col] = float("nan")
            continue
        total = float((1.0 + s).product()) - 1.0
        years = n / periods_per_year
        result[col] = float(abs(1.0 + total) ** (1.0 / years) - 1.0) * (1 if total >= 0 else -1)
    return result


# ── Metrics-table HTML renderer ───────────────────────────────────────────────

_SECTION_SPANS: list[tuple[str, list[str]]] = [
    (
        "Overview",
        [
            "Start Period",
            "End Period",
            "Time in Market",
            "Cumulative Return",
            "CAGR",
        ],
    ),
    (
        "Risk-Adjusted Ratios",
        [
            "Sharpe",
            "Prob. Sharpe Ratio",
            "Sortino",
            "Sortino / √2",
            "Omega",
        ],
    ),
    (
        "Drawdown",
        [
            "Max Drawdown",
            "Max DD Duration",
            "Avg Drawdown",
            "Recovery Factor",
            "Ulcer Index",
            "Serenity Index",
        ],
    ),
    (
        "Trading",
        [
            "Gain/Pain Ratio",
            "Gain/Pain (1M)",
            "Payoff Ratio",
            "Profit Factor",
            "Common Sense Ratio",
            "CPC Index",
            "Tail Ratio",
            "Outlier Win Ratio",
            "Outlier Loss Ratio",
        ],
    ),
    (
        "Recent Returns",
        [
            "MTD",
            "3M",
            "6M",
            "YTD",
            "1Y",
            "3Y (ann.)",
            "5Y (ann.)",
            "All-time (ann.)",
        ],
    ),
    (
        "Smart Ratios",
        ["Smart Sharpe", "Smart Sortino", "Smart Sortino / √2"],
    ),
    (
        "Risk",
        [
            "Volatility (ann.)",
            "Calmar",
            "Risk-Adjusted Return",
            "Risk-Return Ratio",
            "Ulcer Performance Index",
            "Skew",
            "Kurtosis",
        ],
    ),
    (
        "Averages",
        [
            "Avg. Return",
            "Avg. Win",
            "Avg. Loss",
            "Win/Loss Ratio",
            "Profit Ratio",
            "Win Rate",
            "Monthly Win Rate",
        ],
    ),
    (
        "Expected Returns",
        ["Expected Daily", "Expected Monthly", "Expected Yearly"],
    ),
    (
        "Tail Risk",
        [
            "Kelly Criterion",
            "Risk of Ruin",
            "Daily VaR",
            "Expected Shortfall (cVaR)",
        ],
    ),
    (
        "Streaks",
        ["Max Consecutive Wins", "Max Consecutive Losses"],
    ),
    (
        "Best / Worst",
        ["Best Day", "Worst Day"],
    ),
    (
        "Benchmark",
        ["Beta", "Alpha", "Correlation", "R²", "Treynor Ratio"],
    ),
]

_PCT_METRICS: frozenset[str] = frozenset(
    {
        "Time in Market",
        "Cumulative Return",
        "CAGR",
        "Prob. Sharpe Ratio",
        "Max Drawdown",
        "Avg Drawdown",
        "MTD",
        "3M",
        "6M",
        "YTD",
        "1Y",
        "3Y (ann.)",
        "5Y (ann.)",
        "All-time (ann.)",
        "Volatility (ann.)",
        "Risk-Adjusted Return",
        "Avg. Return",
        "Avg. Win",
        "Avg. Loss",
        "Win Rate",
        "Monthly Win Rate",
        "Expected Daily",
        "Expected Monthly",
        "Expected Yearly",
        "Kelly Criterion",
        "Risk of Ruin",
        "Daily VaR",
        "Expected Shortfall (cVaR)",
        "Best Day",
        "Worst Day",
        "Alpha",
        "Correlation",
    }
)


def _metrics_table_html(df: pl.DataFrame) -> str:
    """Render a metrics DataFrame as a styled HTML table with section headers.

    Args:
        df: DataFrame with a ``"Metric"`` column and one column per asset.

    Returns:
        An HTML ``<table>`` string.

    """
    assets = [c for c in df.columns if c != "Metric"]
    rows_by_label: dict[str, dict[str, object]] = {
        str(row["Metric"]): {a: row.get(a) for a in assets} for row in df.iter_rows(named=True)
    }

    n_cols = len(assets) + 1
    header_cells = "".join(f'<th class="asset-header">{a}</th>' for a in assets)
    parts: list[str] = []

    rendered: set[str] = set()
    for section_label, section_metrics in _SECTION_SPANS:
        section_rows: list[str] = []
        for label in section_metrics:
            if label not in rows_by_label:
                continue
            vals = rows_by_label[label]
            rendered.add(label)
            fmt = ".2%" if label in _PCT_METRICS else ".2f"
            cells = "".join(f'<td class="metric-value">{_fmt(vals.get(a), fmt)}</td>' for a in assets)
            section_rows.append(f'<tr><td class="metric-name">{label}</td>{cells}</tr>\n')

        if section_rows:
            parts.append(
                f'<tr class="table-section-header"><td colspan="{n_cols}"><strong>{section_label}</strong></td></tr>\n'
            )
            parts.extend(section_rows)

    # Anything not matched by a section (e.g. string-valued rows like dates)
    for label, vals in rows_by_label.items():
        if label in rendered:
            continue
        raw = next(iter(vals.values()), None)
        if isinstance(raw, str):
            cells = "".join(f'<td class="metric-value">{vals.get(a, "")}</td>' for a in assets)
        else:
            cells = "".join(f'<td class="metric-value">{_fmt(vals.get(a), ".4f")}</td>' for a in assets)
        parts.append(f'<tr><td class="metric-name">{label}</td>{cells}</tr>\n')

    return (
        '<table class="stats-table">'
        "<thead><tr>"
        f'<th class="metric-header">Metric</th>{header_cells}'
        "</tr></thead>"
        f"<tbody>{''.join(parts)}</tbody>"
        "</table>"
    )


def _drawdowns_section_html(data: Any, assets: list[str]) -> str:
    """Render worst-5 drawdown periods per asset as HTML tables.

    Args:
        data: The DataLike object (accessed via ``getattr`` for stats).
        assets: List of asset column names to render.

    Returns:
        HTML string containing one table per asset.

    """
    stats = getattr(data, "stats", None)
    if stats is None:
        return "<p>No drawdown data available.</p>"

    parts: list[str] = []
    try:
        dd_dict: dict[str, pl.DataFrame] = stats.drawdown_details()
    except Exception:
        return "<p>Drawdown details unavailable.</p>"

    for asset in assets:
        df = dd_dict.get(asset)
        if df is None or len(df) == 0:
            parts.append(f"<h3>{asset}</h3><p>No drawdown periods found.</p>")
            continue

        worst5 = df.sort("max_drawdown").head(5)
        rows = "".join(
            f"<tr>"
            f"<td>{row.get('start', '')}</td>"
            f"<td>{row.get('valley', '')}</td>"
            f"<td>{row.get('end', '') or '—'}</td>"
            f"<td>{_fmt(row.get('max_drawdown'), '.2%')}</td>"
            f"<td>{row.get('duration', '') or '—'}</td>"
            f"</tr>"
            for row in worst5.iter_rows(named=True)
        )
        parts.append(
            f"<h3>{asset}</h3>"
            '<table class="stats-table">'
            "<thead><tr>"
            "<th>Start</th><th>Valley</th><th>End</th><th>Max DD</th><th>Duration</th>"
            "</tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )

    return "\n".join(parts)


def _try_plotly_div(fig: Any, include_cdn: bool = False) -> str:
    """Convert a Plotly figure to an HTML div string.

    Args:
        fig: A Plotly Figure object (or anything with ``to_html``).
        include_cdn: Include the Plotly JS CDN ``<script>`` tag. Defaults to False.

    Returns:
        An HTML string, or an empty string if conversion fails.

    """
    try:
        import plotly.io as pio

        return pio.to_html(  # type: ignore[no-any-return]
            fig,
            full_html=False,
            include_plotlyjs="cdn" if include_cdn else False,
        )
    except Exception:
        return ""


_REPORT_CSS = """
body{margin:0;font-family:system-ui,sans-serif;background:#0f1117;color:#e2e8f0}
h1{color:#90cdf4;margin:0 0 4px}
h2{color:#63b3ed;border-bottom:1px solid #2d3748;padding-bottom:6px}
h3{color:#a0aec0;margin:16px 0 6px}
header{padding:24px 32px;background:linear-gradient(135deg,#1a202c,#2d3748);border-bottom:1px solid #4a5568}
.period-info{color:#a0aec0;font-size:.85rem;margin-top:4px}
main{padding:24px 32px}
section{margin-bottom:40px}
.stats-table{border-collapse:collapse;width:100%;font-size:.85rem}
.stats-table th,.stats-table td{padding:6px 12px;text-align:right;border-bottom:1px solid #2d3748}
.stats-table th:first-child,.stats-table td:first-child{text-align:left}
.metric-header,.asset-header{background:#1a202c;color:#90cdf4;font-weight:600}
.metric-name{color:#cbd5e0}
.metric-value{font-family:monospace;color:#e2e8f0}
.table-section-header td{background:#1a202c;color:#68d391;font-size:.75rem;text-transform:uppercase;
letter-spacing:.08em;padding:8px 12px}
footer{padding:16px 32px;color:#718096;font-size:.75rem;border-top:1px solid #2d3748}
"""


def _build_full_html(
    title: str,
    period_info: str,
    assets_str: str,
    metrics_html: str,
    drawdowns_html: str,
    charts_html: str,
) -> str:
    """Assemble the full HTML report from its component parts.

    Args:
        title: Page and ``<h1>`` title.
        period_info: Period metadata string for the header.
        assets_str: Comma-separated asset names for the header.
        metrics_html: Pre-rendered metrics ``<table>`` HTML.
        drawdowns_html: Pre-rendered worst-drawdowns HTML.
        charts_html: Pre-rendered Plotly chart divs.

    Returns:
        A complete, self-contained HTML document string.

    """
    from datetime import date

    footer_date = str(date.today())
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>{_REPORT_CSS}</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <div class="period-info">{period_info}</div>
  <div class="period-info">Assets: {assets_str}</div>
</header>
<main>
  <section id="metrics">
    <h2>Performance Metrics</h2>
    {metrics_html}
  </section>
  <section id="drawdowns">
    <h2>Worst 5 Drawdown Periods</h2>
    {drawdowns_html}
  </section>
  <section id="charts">
    <h2>Charts</h2>
    {charts_html}
  </section>
</main>
<footer>Generated by jquantstats · {footer_date}</footer>
</body>
</html>"""


# ── Reports dataclass ─────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class Reports:
    """A class for generating financial reports from Data objects.

    This class provides methods for calculating and formatting various financial metrics
    into report-ready formats such as DataFrames.

    Attributes:
        data (DataLike): The financial data object to generate reports from.

    """

    data: DataLike

    def metrics(
        self,
        mode: str = "basic",
        periods_per_year: int | float = 252,
        rf: float = 0.0,
    ) -> pl.DataFrame:
        """Comprehensive performance metrics table matching ``qs.reports.metrics``.

        Computes an ordered set of performance, risk, and trading metrics for
        every asset in the dataset and returns them as a tidy DataFrame.

        Args:
            mode: ``"basic"`` (default) for core metrics, ``"full"`` for the
                extended set including smart ratios, expected returns, streaks,
                best/worst periods, win rates, and benchmark greeks.
            periods_per_year: Annualisation factor. Defaults to 252.
            rf: Annualised risk-free rate used in ratio calculations.
                Defaults to 0.0.

        Returns:
            pl.DataFrame: One row per metric, one column per asset, plus a
            leading ``"Metric"`` column with the metric label.

        """
        s = self.data.stats
        ppy = float(periods_per_year)
        is_full = mode.lower() == "full"

        rows: list[tuple[str, dict[str, Any]]] = []

        def _row(label: str, vals: dict[str, Any]) -> None:
            """Append a labelled metric row to *rows*."""
            rows.append((label, vals))

        # ── Overview ──────────────────────────────────────────────────────────
        all_df: pl.DataFrame | None = getattr(self.data, "all", None)
        asset_cols: list[str] = []
        date_col: str | None = None
        has_dates = False

        if all_df is not None:
            date_col = all_df.columns[0]
            asset_cols = [c for c in all_df.columns if c != date_col]
            has_dates = all_df[date_col].dtype.is_temporal()

        _row("Time in Market", _pct(_safe(s.exposure)))
        _row("Cumulative Return", _pct(_safe(s.comp)))
        _row("CAGR", _pct(_safe(s.cagr, periods=ppy)))

        # ── Risk-Adjusted Ratios ──────────────────────────────────────────────
        _row("Sharpe", _safe(s.sharpe, periods=ppy))
        _row("Prob. Sharpe Ratio", _pct(_safe(s.probabilistic_sharpe_ratio)))
        _row("Sortino", _safe(s.sortino, periods=ppy))
        _row("Sortino / √2", _safe(s.adjusted_sortino, periods=ppy))
        _row("Omega", _safe(s.omega, periods=ppy))

        # ── Drawdown ──────────────────────────────────────────────────────────
        _row("Max Drawdown", _pct(_safe(s.max_drawdown)))
        _row("Max DD Duration", _safe(s.max_drawdown_duration))
        _row("Avg Drawdown", _pct(_safe(s.avg_drawdown)))
        _row("Recovery Factor", _safe(s.recovery_factor))
        _row("Ulcer Index", _safe(s.ulcer_index))
        _row("Serenity Index", _safe(s.serenity_index))

        # ── Trading ───────────────────────────────────────────────────────────
        _row("Gain/Pain Ratio", _safe(s.gain_to_pain_ratio))
        _row("Gain/Pain (1M)", _safe(s.gain_to_pain_ratio, aggregate="ME"))
        _row("Payoff Ratio", _safe(s.payoff_ratio))
        _row("Profit Factor", _safe(s.profit_factor))
        _row("Common Sense Ratio", _safe(s.common_sense_ratio))
        _row("CPC Index", _safe(s.cpc_index))
        _row("Tail Ratio", _safe(s.tail_ratio))
        _row("Outlier Win Ratio", _safe(s.outlier_win_ratio))
        _row("Outlier Loss Ratio", _safe(s.outlier_loss_ratio))

        # ── Recent Returns (date-filtered) ───────────────────────────────────
        if has_dates and date_col is not None and all_df is not None:
            from datetime import timedelta

            end_dt = all_df[date_col].max()
            # Use Python date arithmetic; Polars dates are datetime.date
            today = end_dt  # type: ignore[assignment]

            def _cutoff_days(n: int) -> Any:
                """Return the date *n* days before *today*."""
                return today - timedelta(days=n)  # type: ignore[operator]

            def _cutoff_months(n: int) -> Any:
                """Return the date *n* calendar months before *today*."""
                import calendar

                y = today.year  # type: ignore[union-attr]
                m = today.month  # type: ignore[union-attr]
                for _ in range(n):
                    m -= 1
                    if m == 0:
                        m = 12
                        y -= 1
                d = min(today.day, calendar.monthrange(y, m)[1])  # type: ignore[union-attr]
                from datetime import date as _date

                return _date(y, m, d)

            mtd_start = today.replace(day=1)  # type: ignore[union-attr]
            ytd_start = today.replace(month=1, day=1)  # type: ignore[union-attr]

            _row("MTD", _pct(_comp_since(all_df, date_col, asset_cols, mtd_start)))
            _row("3M", _pct(_comp_since(all_df, date_col, asset_cols, _cutoff_months(3))))
            _row("6M", _pct(_comp_since(all_df, date_col, asset_cols, _cutoff_months(6))))
            _row("YTD", _pct(_comp_since(all_df, date_col, asset_cols, ytd_start)))
            _row("1Y", _pct(_comp_since(all_df, date_col, asset_cols, _cutoff_months(12))))
            _row("3Y (ann.)", _pct(_cagr_since(all_df, date_col, asset_cols, _cutoff_months(36), ppy)))
            _row("5Y (ann.)", _pct(_cagr_since(all_df, date_col, asset_cols, _cutoff_months(60), ppy)))
            _row("All-time (ann.)", _pct(_safe(s.cagr, periods=ppy)))

        # ── Full-mode extensions ──────────────────────────────────────────────
        if is_full:
            # Smart ratios
            _row("Smart Sharpe", _safe(s.smart_sharpe, periods=ppy))
            ss = _safe(s.smart_sortino, periods=ppy)
            _row("Smart Sortino", ss)
            _row("Smart Sortino / √2", {k: v / math.sqrt(2) for k, v in ss.items() if _is_finite(v)})

            # Risk
            _row("Volatility (ann.)", _pct(_safe(s.volatility, periods=ppy)))
            _row("Calmar", _safe(s.calmar, periods=ppy))
            _row("Risk-Adjusted Return", _pct(_safe(s.rar, periods=ppy)))
            _row("Risk-Return Ratio", _safe(s.risk_return_ratio))
            _row("Ulcer Performance Index", _safe(s.ulcer_performance_index))
            _row("Skew", _safe(s.skew))
            _row("Kurtosis", _safe(s.kurtosis))

            # Averages
            _row("Avg. Return", _pct(_safe(s.avg_return)))
            _row("Avg. Win", _pct(_safe(s.avg_win)))
            _row("Avg. Loss", _pct(_safe(s.avg_loss)))
            _row("Win/Loss Ratio", _safe(s.win_loss_ratio))
            _row("Profit Ratio", _safe(s.profit_ratio))
            _row("Win Rate", _pct(_safe(s.win_rate)))
            _row("Monthly Win Rate", _pct(_safe(s.monthly_win_rate)))

            # Expected returns
            _row("Expected Daily", _pct(_safe(s.expected_return)))
            _row("Expected Monthly", _pct(_safe(s.expected_return, aggregate="monthly")))
            _row("Expected Yearly", _pct(_safe(s.expected_return, aggregate="yearly")))

            # Tail risk
            _row("Kelly Criterion", _pct(_safe(s.kelly_criterion)))
            _row("Risk of Ruin", _pct(_safe(s.risk_of_ruin)))
            _row("Daily VaR", _pct(_safe(s.value_at_risk)))
            _row("Expected Shortfall (cVaR)", _pct(_safe(s.conditional_value_at_risk)))

            # Streaks
            _row("Max Consecutive Wins", _safe(s.consecutive_wins))
            _row("Max Consecutive Losses", _safe(s.consecutive_losses))

            # Best / Worst
            _row("Best Day", _pct(_safe(s.best)))
            _row("Worst Day", _pct(_safe(s.worst)))

            # Benchmark greeks (only if benchmark present)
            try:
                greeks = s.greeks()
                if greeks:
                    beta = {k: v["beta"] for k, v in greeks.items()}
                    alpha = {k: v["alpha"] * 100.0 for k, v in greeks.items()}
                    _row("Beta", beta)
                    _row("Alpha", alpha)
            except Exception:  # noqa: S110
                pass  # nosec B110

            try:
                bench_obj = getattr(self.data, "benchmark", None)
                if bench_obj is not None and all_df is not None and date_col is not None:
                    bench_col = bench_obj.columns[0]
                    corr_dict: dict[str, float] = {}
                    for ac in asset_cols:
                        corr_val = float(
                            all_df[ac]
                            .drop_nulls()
                            .cast(pl.Float64)
                            .pearson_corr(all_df[bench_col].drop_nulls().cast(pl.Float64))
                        )
                        corr_dict[ac] = corr_val * 100.0
                    _row("Correlation", corr_dict)
            except Exception:  # noqa: S110
                pass  # nosec B110

            _row("R²", _safe(s.r2))
            _row("Treynor Ratio", _safe(s.treynor_ratio, periods=ppy))

        # ── Build DataFrame ───────────────────────────────────────────────────
        # Collect all asset names seen across all rows
        all_assets: list[str] = []
        seen: set[str] = set()
        for _, vals in rows:
            for k in vals:
                if k not in seen:
                    all_assets.append(k)
                    seen.add(k)

        return pl.DataFrame([{"Metric": label, **{a: vals.get(a) for a in all_assets}} for label, vals in rows])

    def full(
        self,
        title: str = "Performance Report",
        periods_per_year: int | float = 252,
        rf: float = 0.0,
    ) -> str:
        """Generate a self-contained HTML performance report.

        Combines a comprehensive metrics table (full mode), worst-5 drawdown
        periods per asset, and interactive Plotly charts into a single
        dark-themed HTML document.

        Args:
            title: Page ``<h1>`` title. Defaults to ``"Performance Report"``.
            periods_per_year: Annualisation factor passed to
                :py:meth:`metrics`. Defaults to 252.
            rf: Annualised risk-free rate. Defaults to 0.0.

        Returns:
            str: A complete, self-contained HTML document.

        """
        # ── Metrics ───────────────────────────────────────────────────────────
        metrics_df = self.metrics(mode="full", periods_per_year=periods_per_year, rf=rf)
        assets = [c for c in metrics_df.columns if c != "Metric"]
        metrics_html = _metrics_table_html(metrics_df)

        # ── Period info for header ────────────────────────────────────────────
        all_df: pl.DataFrame | None = getattr(self.data, "all", None)
        period_info = ""
        if all_df is not None:
            date_col = all_df.columns[0]
            if all_df[date_col].dtype.is_temporal():
                start_dt = all_df[date_col].min()
                end_dt = all_df[date_col].max()
                n = len(all_df)
                period_info = f"{start_dt} → {end_dt} | {n:,} observations"

        # ── Drawdowns ─────────────────────────────────────────────────────────
        drawdowns_html = _drawdowns_section_html(self.data, assets)

        # ── Charts ────────────────────────────────────────────────────────────
        plots = getattr(self.data, "plots", None)
        chart_parts: list[str] = []
        if plots is not None:
            _chart_methods = [
                ("plot_snapshot", {}),
                ("plot_returns", {}),
                ("plot_drawdown", {}),
                ("plot_rolling_sharpe", {}),
                ("plot_rolling_volatility", {}),
                ("plot_monthly_heatmap", {}),
                ("plot_yearly_returns", {}),
                ("plot_histogram", {}),
            ]
            for i, (method, kwargs) in enumerate(_chart_methods):
                fn = getattr(plots, method, None)
                if fn is None:
                    continue
                div = _try_plotly_div(fn(**kwargs), include_cdn=(i == 0))
                if div:
                    chart_parts.append(f'<div style="margin-bottom:24px">{div}</div>')

        charts_html = "\n".join(chart_parts) if chart_parts else "<p>No charts available.</p>"

        return _build_full_html(
            title=title,
            period_info=period_info,
            assets_str=", ".join(assets),
            metrics_html=metrics_html,
            drawdowns_html=drawdowns_html,
            charts_html=charts_html,
        )
