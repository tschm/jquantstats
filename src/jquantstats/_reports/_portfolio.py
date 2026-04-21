"""HTML report generation for portfolio analytics.

This module defines the Report facade which produces a self-contained HTML
document containing all relevant performance numbers and interactive Plotly
visualisations for a Portfolio.

Examples:
    >>> import dataclasses
    >>> from jquantstats._reports import Report
    >>> dataclasses.is_dataclass(Report)
    True
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeGuard

import plotly.graph_objects as go
import plotly.io as pio
import polars as pl
from jinja2 import Environment, FileSystemLoader, select_autoescape

if TYPE_CHECKING:
    from ._protocol import PortfolioLike

# templates/ lives one level above this subpackage (at src/jquantstats/templates/)
_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
_env = Environment(
    loader=FileSystemLoader(_TEMPLATES_DIR),
    autoescape=select_autoescape(["html"]),
)


# ── Formatting helpers ────────────────────────────────────────────────────────


def _is_finite(v: Any) -> TypeGuard[int | float]:
    """Return True when *v* is a real, finite number."""
    if not isinstance(v, (int, float)):
        return False
    return math.isfinite(float(v))


def _fmt(value: Any, fmt: str = ".4f", suffix: str = "") -> str:
    """Format *value* for display in an HTML table cell.

    Returns ``"N/A"`` for ``None``, ``NaN``, or non-finite values.
    """
    if not _is_finite(value):
        return "N/A"
    return f"{float(value):{fmt}}{suffix}"


# ── Stats table ───────────────────────────────────────────────────────────────

_METRIC_FORMATS: dict[str, tuple[str, str]] = {
    "avg_return": (".2%", ""),
    "avg_win": (".2%", ""),
    "avg_loss": (".2%", ""),
    "best": (".2%", ""),
    "worst": (".2%", ""),
    "sharpe": (".2f", ""),
    "calmar": (".2f", ""),
    "recovery_factor": (".2f", ""),
    "max_drawdown": (".2%", ""),
    "avg_drawdown": (".2%", ""),
    "max_drawdown_duration": (".0f", " days"),
    "win_rate": (".1%", ""),
    "monthly_win_rate": (".1%", ""),
    "profit_factor": (".2f", ""),
    "payoff_ratio": (".2f", ""),
    "volatility": (".2%", ""),
    "skew": (".2f", ""),
    "kurtosis": (".2f", ""),
    "value_at_risk": (".2%", ""),
    "conditional_value_at_risk": (".2%", ""),
}

_METRIC_LABELS: dict[str, str] = {
    "avg_return": "Avg Return",
    "avg_win": "Avg Win",
    "avg_loss": "Avg Loss",
    "best": "Best Period",
    "worst": "Worst Period",
    "sharpe": "Sharpe Ratio",
    "calmar": "Calmar Ratio",
    "recovery_factor": "Recovery Factor",
    "max_drawdown": "Max Drawdown",
    "avg_drawdown": "Avg Drawdown",
    "max_drawdown_duration": "Max DD Duration",
    "win_rate": "Win Rate",
    "monthly_win_rate": "Monthly Win Rate",
    "profit_factor": "Profit Factor",
    "payoff_ratio": "Payoff Ratio",
    "volatility": "Volatility (ann.)",
    "skew": "Skewness",
    "kurtosis": "Kurtosis",
    "value_at_risk": "VaR (95 %)",
    "conditional_value_at_risk": "CVaR (95 %)",
}

# Metrics where the *highest* value across assets is highlighted.
_HIGHER_IS_BETTER: frozenset[str] = frozenset(
    {"sharpe", "calmar", "recovery_factor", "win_rate", "monthly_win_rate", "profit_factor", "payoff_ratio"}
)

_CATEGORIES: list[tuple[str, list[str]]] = [
    ("Returns", ["avg_return", "avg_win", "avg_loss", "best", "worst"]),
    ("Risk-Adjusted Performance", ["sharpe", "calmar", "recovery_factor"]),
    ("Drawdown", ["max_drawdown", "avg_drawdown", "max_drawdown_duration"]),
    ("Win / Loss", ["win_rate", "monthly_win_rate", "profit_factor", "payoff_ratio"]),
    ("Distribution & Risk", ["volatility", "skew", "kurtosis", "value_at_risk", "conditional_value_at_risk"]),
]


def _stats_table_html(summary: pl.DataFrame) -> str:
    """Render a stats summary DataFrame as a styled HTML table.

    Args:
        summary: Output of `Stats.summary` — one row per metric,
            one column per asset plus a ``metric`` column.

    Returns:
        An HTML ``<table>`` string ready to embed in a page.
    """
    assets = [c for c in summary.columns if c != "metric"]

    # Build a fast lookup: metric_name → {asset: value}
    metric_data: dict[str, dict[str, Any]] = {}
    for row in summary.iter_rows(named=True):
        name = str(row["metric"])
        metric_data[name] = {a: row.get(a) for a in assets}

    header_cells = "".join(f'<th class="asset-header">{a}</th>' for a in assets)
    rows_html_parts: list[str] = []

    for category_label, metrics in _CATEGORIES:
        rows_html_parts.append(
            f'<tr class="table-section-header">'
            f'<td colspan="{len(assets) + 1}"><strong>{category_label}</strong></td>'
            f"</tr>\n"
        )
        for metric in metrics:
            if metric not in metric_data:
                continue
            fmt, suffix = _METRIC_FORMATS.get(metric, (".4f", ""))
            label = _METRIC_LABELS.get(metric, metric.replace("_", " ").title())
            values = metric_data[metric]

            # Find the best asset to highlight (only for higher-is-better metrics)
            best_asset: str | None = None
            if metric in _HIGHER_IS_BETTER:
                finite_pairs = [(a, float(v)) for a, v in values.items() if _is_finite(v)]
                if finite_pairs:
                    best_asset = max(finite_pairs, key=lambda x: x[1])[0]

            cells = "".join(
                f'<td class="metric-value{"  best-value" if a == best_asset else ""}">'
                f"{_fmt(values.get(a), fmt, suffix)}</td>"
                for a in assets
            )
            rows_html_parts.append(f'<tr><td class="metric-name">{label}</td>{cells}</tr>\n')

    rows_html = "".join(rows_html_parts)
    return (
        '<table class="stats-table">'
        "<thead><tr>"
        f'<th class="metric-header">Metric</th>{header_cells}'
        "</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table>"
    )


# ── Report dataclass ──────────────────────────────────────────────────────────


def _figure_div(fig: go.Figure, include_plotlyjs: bool | str) -> str:
    """Return an HTML div string for *fig*.

    Args:
        fig: Plotly figure to serialise.
        include_plotlyjs: Passed directly to `plotly.io.to_html`.
            Pass ``"cdn"`` for the first figure so the CDN script tag is
            injected; pass ``False`` for all subsequent figures.

    Returns:
        HTML string (not a full page).
    """
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=include_plotlyjs,
    )


@dataclasses.dataclass(frozen=True)
class Report:
    """Facade for generating HTML reports from a Portfolio.

    Provides a `to_html` method that assembles a self-contained,
    dark-themed HTML document with a performance-statistics table and
    multiple interactive Plotly charts.

    Usage::

        report = portfolio.report
        html_str = report.to_html()
        report.save("output/report.html")
    """

    portfolio: PortfolioLike

    def to_html(self, title: str = "JQuantStats Portfolio Report") -> str:
        """Render a full HTML report as a string.

        The document is self-contained: Plotly.js is loaded once from the
        CDN and all charts are embedded as ``<div>`` elements.  No external
        CSS framework is required.

        Args:
            title: HTML ``<title>`` text and visible page heading.

        Returns:
            A complete HTML document as a `str`.
        """
        pf = self.portfolio

        # ── Metadata ──────────────────────────────────────────────────────────
        has_date = "date" in pf.prices.columns
        if has_date:
            dates = pf.prices["date"]
            start_date = str(dates.min())
            end_date = str(dates.max())
            n_periods = pf.prices.height
            period_info = f"{start_date} → {end_date} &nbsp;|&nbsp; {n_periods:,} periods"
        else:
            start_date = ""
            end_date = ""
            period_info = f"{pf.prices.height:,} periods"

        assets_list = ", ".join(pf.assets)

        # ── Figures ───────────────────────────────────────────────────────────
        # The first chart includes Plotly.js from CDN; subsequent ones reuse it.
        _first = True

        def _div(fig: go.Figure) -> str:
            """Serialise *fig* to an HTML div, embedding Plotly.js only on the first call."""
            nonlocal _first
            include = "cdn" if _first else False
            _first = False
            return _figure_div(fig, include)

        def _try_div(build_fig: Callable[[], go.Figure]) -> str:
            """Call *build_fig()* and return the chart div; on error return a notice."""
            try:
                fig = build_fig()
                return _div(fig)
            except Exception as exc:
                return f'<p class="chart-unavailable">Chart unavailable: {exc}</p>'

        snapshot_div = _try_div(pf.plots.snapshot)
        rolling_sharpe_div = _try_div(pf.plots.rolling_sharpe_plot)
        rolling_vol_div = _try_div(pf.plots.rolling_volatility_plot)
        annual_sharpe_div = _try_div(pf.plots.annual_sharpe_plot)
        monthly_heatmap_div = _try_div(pf.plots.monthly_returns_heatmap)
        corr_div = _try_div(pf.plots.correlation_heatmap)
        lead_lag_div = _try_div(pf.plots.lead_lag_ir_plot)
        trading_cost_div = _try_div(pf.plots.trading_cost_impact_plot)

        # ── Stats table ───────────────────────────────────────────────────────
        stats_table = _stats_table_html(pf.stats.summary())

        # ── Turnover table ────────────────────────────────────────────────────
        try:
            turnover_df = pf.turnover_summary()
            turnover_rows = "".join(
                f'<tr><td class="metric-name">{row["metric"].replace("_", " ").title()}</td>'
                f'<td class="metric-value">{row["value"]:.4f}</td></tr>'
                for row in turnover_df.iter_rows(named=True)
            )
            turnover_html = (
                '<table class="stats-table">'
                "<thead><tr>"
                '<th class="metric-header">Metric</th>'
                '<th class="asset-header">Value</th>'
                "</tr></thead>"
                f"<tbody>{turnover_rows}</tbody>"
                "</table>"
            )
        except Exception as exc:
            turnover_html = f'<p class="chart-unavailable">Turnover data unavailable: {exc}</p>'

        # ── Assemble HTML ─────────────────────────────────────────────────────
        footer_date = end_date if has_date else ""
        template = _env.get_template("portfolio_report.html")
        return template.render(
            title=title,
            period_info=period_info,
            assets_list=assets_list,
            aum=f"{pf.aum:,.0f}",
            footer_date=footer_date,
            snapshot_div=snapshot_div,
            rolling_sharpe_div=rolling_sharpe_div,
            rolling_vol_div=rolling_vol_div,
            annual_sharpe_div=annual_sharpe_div,
            monthly_heatmap_div=monthly_heatmap_div,
            corr_div=corr_div,
            lead_lag_div=lead_lag_div,
            trading_cost_div=trading_cost_div,
            stats_table=stats_table,
            turnover_html=turnover_html,
            container_max_width="1400px",
        )

    def save(self, path: str | Path, title: str = "JQuantStats Portfolio Report") -> Path:
        """Save the HTML report to a file.

        A ``.html`` suffix is appended automatically when *path* has no
        file extension.

        Args:
            path: Destination file path.
            title: HTML ``<title>`` text and visible page heading.

        Returns:
            The resolved `pathlib.Path` of the written file.
        """
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".html")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_html(title=title), encoding="utf-8")
        return p
