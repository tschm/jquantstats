"""Tests for jquantstats.analytics._report (Report facade and HTML generation)."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from jquantstats import Portfolio
from jquantstats._plots import PortfolioPlots
from jquantstats._reports import Report
from jquantstats._reports._portfolio import _fmt, _stats_table_html

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def multi_year_portfolio() -> Portfolio:
    """Three-year, 2-asset portfolio used for all report tests.

    Long enough for annual and monthly breakdowns to be populated.
    """
    n = 756  # ~3 years of trading days
    start = date(2020, 1, 1)
    end = start + timedelta(days=n - 1)
    dates = pl.date_range(start=start, end=end, interval="1d", eager=True).cast(pl.Date)

    a = pl.Series([100.0 * (1.001**i) for i in range(n)], dtype=pl.Float64)
    b = pl.Series([200.0 + 5.0 * np.sin(0.1 * i) for i in range(n)], dtype=pl.Float64)
    prices = pl.DataFrame({"date": dates, "A": a, "B": b})

    pos_a = pl.Series([1000.0 + float(i % 10) for i in range(n)], dtype=pl.Float64)
    pos_b = pl.Series([500.0 + float(i % 5) for i in range(n)], dtype=pl.Float64)
    positions = pl.DataFrame({"date": dates, "A": pos_a, "B": pos_b})

    return Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e6)


# ── _fmt helper ───────────────────────────────────────────────────────────────


def test_fmt_none_returns_na():
    """_fmt returns 'N/A' when passed None."""
    assert _fmt(None) == "N/A"


def test_fmt_nan_returns_na():
    """_fmt returns 'N/A' for NaN floats."""
    assert _fmt(float("nan")) == "N/A"


def test_fmt_inf_returns_na():
    """_fmt returns 'N/A' for infinite floats."""
    assert _fmt(float("inf")) == "N/A"


def test_fmt_valid_float():
    """_fmt formats a valid float with the given format string and suffix."""
    result = _fmt(0.12345, ".2f", " x")
    assert result == "0.12 x"


def test_fmt_integer_value():
    """_fmt formats an integer with the given format string."""
    result = _fmt(42, ".0f", "")
    assert result == "42"


# ── Report dataclass ──────────────────────────────────────────────────────────


def test_report_is_accessible_via_portfolio_property(multi_year_portfolio):
    """Portfolio.report should return a Report instance."""
    r = multi_year_portfolio.report
    assert isinstance(r, Report)
    assert r.portfolio is multi_year_portfolio


def test_report_is_dataclass():
    """Report is a dataclass as expected."""
    import dataclasses

    assert dataclasses.is_dataclass(Report)


# ── to_html ───────────────────────────────────────────────────────────────────


def test_to_html_returns_string(multi_year_portfolio):
    """to_html returns a non-trivial string."""
    html = multi_year_portfolio.report.to_html()
    assert isinstance(html, str)
    assert len(html) > 1000


def test_to_html_starts_with_doctype(multi_year_portfolio):
    """to_html output begins with a DOCTYPE declaration."""
    html = multi_year_portfolio.report.to_html()
    assert html.strip().startswith("<!DOCTYPE html>")


def test_to_html_includes_title(multi_year_portfolio):
    """Custom title is embedded in the HTML output."""
    html = multi_year_portfolio.report.to_html(title="My Test Report")
    assert "My Test Report" in html


def test_to_html_default_title_present(multi_year_portfolio):
    """Default title 'JQuantStats Portfolio Report' appears when no title is given."""
    html = multi_year_portfolio.report.to_html()
    assert "JQuantStats Portfolio Report" in html


def test_to_html_contains_section_ids(multi_year_portfolio):
    """All expected section ids are present in the HTML output."""
    html = multi_year_portfolio.report.to_html()
    for section_id in (
        "performance",
        "risk",
        "annual",
        "monthly",
        "stats-table",
        "correlation",
        "leadlag",
        "costs",
        "turnover",
    ):
        assert f'id="{section_id}"' in html, f"Missing section id: {section_id}"


def test_to_html_includes_asset_names(multi_year_portfolio):
    """Asset names from the portfolio appear in the HTML output."""
    html = multi_year_portfolio.report.to_html()
    for asset in multi_year_portfolio.assets:
        assert asset in html


def test_to_html_includes_aum(multi_year_portfolio):
    """AUM formatted with thousands separator appears in the HTML output."""
    html = multi_year_portfolio.report.to_html()
    # AUM = 1,000,000 formatted with commas
    assert "1,000,000" in html


def test_to_html_contains_plotlyjs_cdn(multi_year_portfolio):
    """HTML output references the Plotly.js library."""
    html = multi_year_portfolio.report.to_html()
    assert "plotly" in html.lower()


def test_to_html_contains_toc_links(multi_year_portfolio):
    """Table of contents links to key sections are present."""
    html = multi_year_portfolio.report.to_html()
    assert 'href="#performance"' in html
    assert 'href="#turnover"' in html
    assert 'href="#costs"' in html


def test_to_html_contains_stats_table(multi_year_portfolio):
    """The performance statistics table is embedded in the HTML."""
    html = multi_year_portfolio.report.to_html()
    assert "Sharpe Ratio" in html
    assert "Max Drawdown" in html
    assert "stats-table" in html


def test_to_html_contains_metric_values(multi_year_portfolio):
    """Key metric labels appear in the stats table section of the HTML."""
    html = multi_year_portfolio.report.to_html()
    # Stats table rows should contain formatted numeric values
    assert "Sharpe Ratio" in html
    assert "Volatility (ann.)" in html
    assert "Win Rate" in html


# ── save ─────────────────────────────────────────────────────────────────────


def test_save_writes_file(tmp_path: Path, multi_year_portfolio):
    """save() writes the HTML file to the specified path."""
    out = tmp_path / "report.html"
    result = multi_year_portfolio.report.save(out)
    assert result == out
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content


def test_save_appends_html_extension_when_missing(tmp_path: Path, multi_year_portfolio):
    """save() appends a .html extension when the path has no suffix."""
    out = tmp_path / "my_report"
    result = multi_year_portfolio.report.save(out)
    assert result.suffix == ".html"
    assert result.exists()


def test_save_accepts_string_path(tmp_path: Path, multi_year_portfolio):
    """save() accepts a plain string path and embeds the custom title."""
    out = str(tmp_path / "report2.html")
    result = multi_year_portfolio.report.save(out, title="Saved Report")
    assert result.exists()
    assert "Saved Report" in result.read_text(encoding="utf-8")


# ── monthly_returns_heatmap (in Plots) ───────────────────────────────────────


def test_monthly_returns_heatmap_returns_figure(multi_year_portfolio):
    """monthly_returns_heatmap returns a Plotly Figure."""
    import plotly.graph_objects as go

    fig = multi_year_portfolio.plots.monthly_returns_heatmap()
    assert isinstance(fig, go.Figure)


def test_monthly_returns_heatmap_has_heatmap_trace(multi_year_portfolio):
    """monthly_returns_heatmap figure contains exactly one Heatmap trace."""
    import plotly.graph_objects as go

    fig = multi_year_portfolio.plots.monthly_returns_heatmap()
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Heatmap)


def test_monthly_returns_heatmap_has_12_x_labels(multi_year_portfolio):
    """monthly_returns_heatmap x-axis labels are the 12 abbreviated month names."""
    fig = multi_year_portfolio.plots.monthly_returns_heatmap()
    assert list(fig.data[0].x) == [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]


def test_monthly_returns_heatmap_title(multi_year_portfolio):
    """monthly_returns_heatmap figure title contains the word 'Monthly'."""
    fig = multi_year_portfolio.plots.monthly_returns_heatmap()
    assert "Monthly" in fig.layout.title.text


def test_monthly_returns_heatmap_serializable(multi_year_portfolio):
    """monthly_returns_heatmap figure can be serialized to dict without error."""
    fig = multi_year_portfolio.plots.monthly_returns_heatmap()
    _ = fig.to_dict()  # must not raise


# ── _stats_table_html ─────────────────────────────────────────────────────────


def test_stats_table_html_returns_table_tag(multi_year_portfolio):
    """_stats_table_html returns an HTML string starting with a <table> tag."""
    summary = multi_year_portfolio.stats.summary()
    html = _stats_table_html(summary)
    assert html.strip().startswith("<table")
    assert "</table>" in html


def test_stats_table_html_contains_metric_labels(multi_year_portfolio):
    """_stats_table_html includes human-readable metric labels."""
    summary = multi_year_portfolio.stats.summary()
    html = _stats_table_html(summary)
    assert "Sharpe Ratio" in html
    assert "Max Drawdown" in html
    assert "Volatility" in html


def test_stats_table_html_contains_asset_headers(multi_year_portfolio):
    """_stats_table_html includes a header column per portfolio asset."""
    summary = multi_year_portfolio.stats.summary()
    html = _stats_table_html(summary)
    for asset in multi_year_portfolio.assets:
        assert asset in html


def test_stats_table_html_skips_missing_metric():
    """_stats_table_html skips metrics absent from the summary DataFrame (line 140).

    When a metric listed in _CATEGORIES is not present in the summary rows,
    the ``continue`` branch is taken and no row is rendered for that metric.
    """
    # Only 'sharpe' is present; all Return-category metrics are absent.
    summary = pl.DataFrame({"metric": ["sharpe"], "A": [1.5], "B": [1.2]})
    html = _stats_table_html(summary)
    assert "<table" in html
    # The present metric must appear.
    assert "Sharpe Ratio" in html
    # Absent metric must not appear.
    assert "Avg Return" not in html


def test_stats_table_html_return_metrics_formatted_as_pct():
    """Return metrics (avg_return, best, worst, VaR) display as percentages.

    Raw decimal values like 0.015 should be shown as '1.50%', not '0.015000'.
    """
    summary = pl.DataFrame({"metric": ["avg_return", "best", "worst", "value_at_risk"], "A": [0.015, 0.03, -0.02, -0.01]})
    html = _stats_table_html(summary)
    assert "1.50%" in html
    assert "3.00%" in html
    assert "-2.00%" in html
    assert "-1.00%" in html
    # Ensure raw decimal form is not present
    assert "0.015000" not in html


# ── dateless portfolio ────────────────────────────────────────────────────────


@pytest.fixture
def dateless_portfolio() -> Portfolio:
    """100-row, 2-asset portfolio whose prices frame has no 'date' column."""
    n = 100
    rng = np.random.default_rng(7)
    a = pl.Series((100.0 + np.cumsum(rng.normal(0, 0.5, n))).tolist(), dtype=pl.Float64)
    b = pl.Series((200.0 + np.cumsum(rng.normal(0, 0.7, n))).tolist(), dtype=pl.Float64)
    prices = pl.DataFrame({"A": a, "B": b})
    positions = pl.DataFrame(
        {
            "A": pl.Series([1000.0] * n, dtype=pl.Float64),
            "B": pl.Series([500.0] * n, dtype=pl.Float64),
        }
    )
    return Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e6)


def test_to_html_dateless_portfolio_shows_period_count(dateless_portfolio):
    """to_html uses '<n> periods' format when prices has no date column (lines 392-394)."""
    html = dateless_portfolio.report.to_html()
    assert isinstance(html, str)
    assert "100 periods" in html


def test_to_html_dateless_portfolio_omits_date_range(dateless_portfolio):
    """to_html omits the date-range arrow when prices has no date column."""
    html = dateless_portfolio.report.to_html()
    assert "\u2192" not in html  # → (the date-range arrow)


# ── _try_div exception handler ────────────────────────────────────────────────


def test_to_html_chart_unavailable_on_plot_error(multi_year_portfolio):
    """_try_div catches plot exceptions and embeds a notice (lines 413-414)."""
    with patch.object(PortfolioPlots, "snapshot", side_effect=RuntimeError("boom")):
        html = multi_year_portfolio.report.to_html()
    assert 'class="chart-unavailable"' in html
    assert "Chart unavailable" in html
    assert "boom" in html


# ── turnover_summary exception handler ───────────────────────────────────────


def test_to_html_turnover_unavailable_on_error(multi_year_portfolio):
    """to_html catches turnover_summary exceptions and embeds a notice (lines 445-446)."""
    with patch.object(Portfolio, "turnover_summary", side_effect=RuntimeError("no data")):
        html = multi_year_portfolio.report.to_html()
    assert 'class="chart-unavailable"' in html
    assert "Turnover data unavailable" in html
    assert "no data" in html
