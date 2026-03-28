"""Tests for the reports module functionality."""

import math
from datetime import date

import plotly.graph_objects as go
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from jquantstats._reports._data import (
    _build_full_html,
    _cagr_since,
    _comp_since,
    _drawdowns_section_html,
    _fmt,
    _is_finite,
    _metrics_table_html,
    _pct,
    _try_plotly_div,
)

# ── Module-level helper unit tests ────────────────────────────────────────────


def test_is_finite_accepts_int():
    """Integer values are finite."""
    assert _is_finite(1)


def test_is_finite_accepts_float():
    """Normal float values are finite."""
    assert _is_finite(0.5)


def test_is_finite_rejects_nan():
    """NaN is not finite."""
    assert not _is_finite(float("nan"))


def test_is_finite_rejects_inf():
    """Infinity is not finite."""
    assert not _is_finite(float("inf"))


def test_is_finite_rejects_non_numeric():
    """Non-numeric types are not finite."""
    assert not _is_finite("hello")
    assert not _is_finite(None)


def test_fmt_returns_na_for_nan():
    """Non-finite values produce 'N/A'."""
    assert _fmt(float("nan")) == "N/A"
    assert _fmt(float("inf")) == "N/A"
    assert _fmt("text") == "N/A"


def test_fmt_formats_finite_value():
    """Finite values are formatted with the given format string."""
    assert _fmt(1.2345) == "1.2345"
    assert _fmt(0.5, ".2%", "") == "50.00%"
    assert _fmt(7.0, ".1f", "x") == "7.0x"


def test_pct_multiplies_finite_values():
    """_pct multiplies finite values by 100."""
    result = _pct({"a": 0.1, "b": float("nan")})
    assert result["a"] == pytest.approx(10.0)
    assert math.isnan(result["b"])


def test_comp_since_empty_window():
    """Empty window after filtering returns nan."""
    df = pl.DataFrame({"Date": [date(2023, 1, 1)], "ret": [0.01]}).with_columns(pl.col("Date").cast(pl.Date))
    # cutoff beyond the data range → empty filter
    result = _comp_since(df, "Date", ["ret"], date(2025, 1, 1))
    assert math.isnan(result["ret"])


def test_cagr_since_single_observation():
    """Fewer than 2 observations returns nan."""
    df = pl.DataFrame({"Date": [date(2023, 1, 1)], "ret": [0.01]}).with_columns(pl.col("Date").cast(pl.Date))
    result = _cagr_since(df, "Date", ["ret"], date(2023, 1, 1), 252)
    assert math.isnan(result["ret"])


def test_metrics_table_html_contains_table_tag():
    """_metrics_table_html returns an HTML table string."""
    df = pl.DataFrame({"Metric": ["Sharpe", "Sortino"], "AAPL": [1.5, 2.0]})
    html = _metrics_table_html(df)
    assert "<table" in html
    assert "Sharpe" in html
    assert "Sortino" in html


def test_metrics_table_html_section_headers():
    """Known metrics are grouped under their section headers."""
    df = pl.DataFrame({"Metric": ["Sharpe"], "A": [1.0]})
    html = _metrics_table_html(df)
    assert "Risk-Adjusted Ratios" in html


def test_metrics_table_html_unrecognised_float_metric():
    """Metrics not in any section are still rendered as float cells."""
    df = pl.DataFrame({"Metric": ["Unknown Metric"], "A": [3.14]})
    html = _metrics_table_html(df)
    assert "Unknown Metric" in html


def test_metrics_table_html_unrecognised_string_metric():
    """Metrics with string values in unrecognised rows are rendered as-is."""
    df = pl.DataFrame({"Metric": ["Label"], "A": ["some text"]})
    html = _metrics_table_html(df)
    assert "some text" in html


def test_drawdowns_section_html_no_stats_attr():
    """Returns fallback message when data has no 'stats' attribute."""

    class _NoStats:
        """Stub with no stats attribute."""

    html = _drawdowns_section_html(_NoStats(), ["X"])
    assert "No drawdown data available" in html


def test_drawdowns_section_html_stats_raises():
    """Returns fallback when drawdown_details() raises."""

    class _BadStats:
        """Stats stub whose drawdown_details always raises."""

        def drawdown_details(self) -> dict:
            """Raise unconditionally."""
            raise RuntimeError("boom")

    class _FakeData:
        """Stub Data wrapping a BadStats."""

        stats = _BadStats()

    html = _drawdowns_section_html(_FakeData(), ["X"])
    assert "unavailable" in html


def test_drawdowns_section_html_empty_dataframe():
    """Empty DataFrame per asset shows 'No drawdown periods found'."""

    class _EmptyStats:
        """Stats stub that returns an empty drawdown DataFrame."""

        def drawdown_details(self) -> dict:
            """Return empty DataFrame for asset."""
            return {"X": pl.DataFrame()}

    class _FakeData:
        """Stub Data wrapping an EmptyStats."""

        stats = _EmptyStats()

    html = _drawdowns_section_html(_FakeData(), ["X"])
    assert "No drawdown periods found" in html


def test_drawdowns_section_html_with_real_data(data):
    """Renders an HTML table when real drawdown data is present."""
    assets = list(data.returns.columns)
    html = _drawdowns_section_html(data, assets)
    assert "<table" in html


def test_try_plotly_div_with_valid_figure():
    """Returns a non-empty HTML div for a valid Plotly figure."""
    fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
    result = _try_plotly_div(fig)
    assert isinstance(result, str)
    assert len(result) > 0


def test_try_plotly_div_with_cdn():
    """include_cdn=True embeds the Plotly JS CDN script."""
    fig = go.Figure()
    result = _try_plotly_div(fig, include_cdn=True)
    assert "plotly" in result.lower()


def test_try_plotly_div_with_invalid_object():
    """Returns empty string when conversion fails."""
    result = _try_plotly_div("not a figure")
    assert result == ""


def test_build_full_html_structure():
    """Produced document contains expected HTML landmarks."""
    html = _build_full_html("My Report", "2023 → 2024", "AAPL", "<table/>", "<dd/>", "<div/>")
    assert "<!DOCTYPE html>" in html
    assert "My Report" in html
    assert "2023 → 2024" in html
    assert "AAPL" in html
    assert "Generated by jquantstats" in html


# ── Reports fixture & snapshot test ──────────────────────────────────────────


@pytest.fixture
def reports(data):
    """Fixture that returns the reports property of the data fixture.

    Args:
        data: The data fixture containing a Data object.

    Returns:
        Reports: The reports property of the data fixture.

    """
    return data.reports


@pytest.fixture
def metrics(resource_dir):
    """Fixture that returns the metrics CSV file as a DataFrame.

    Args:
        resource_dir: The resource_dir fixture containing the path to the resources directory.

    Returns:
        pl.DataFrame: A DataFrame containing the metrics data.

    """
    return pl.read_csv(resource_dir / "metrics.csv")


def test_metric(reports, metrics):
    """Tests that the metrics method returns the correct metrics.

    Args:
        reports: The reports fixture containing a Reports object.
        metrics: The metrics fixture containing the expected metrics.

    Verifies:
        The metrics method returns a DataFrame that matches the expected metrics.

    """
    # reports.metrics().write_csv("metrics.csv")
    assert_frame_equal(metrics, reports.metrics())


# ── metrics() full mode ───────────────────────────────────────────────────────


def test_metrics_full_mode_returns_dataframe(reports):
    """metrics(mode='full') returns a non-empty DataFrame."""
    df = reports.metrics(mode="full")
    assert isinstance(df, pl.DataFrame)
    assert len(df) > 0


def test_metrics_full_mode_includes_extended_metrics(reports):
    """Full mode includes metrics absent from basic mode."""
    basic_labels = set(reports.metrics(mode="basic")["Metric"].to_list())
    full_labels = set(reports.metrics(mode="full")["Metric"].to_list())
    extended = full_labels - basic_labels
    # Smart Ratios, Risk, Averages, etc. should appear only in full mode
    assert len(extended) > 0
    assert "Volatility (ann.)" in extended or "Smart Sharpe" in extended


def test_metrics_full_mode_has_benchmark_greeks(reports):
    """Full mode includes Beta and Alpha when benchmark is present."""
    df = reports.metrics(mode="full")
    labels = df["Metric"].to_list()
    assert "Beta" in labels
    assert "Alpha" in labels


def test_metrics_full_mode_has_tail_risk(reports):
    """Full mode includes tail-risk metrics."""
    df = reports.metrics(mode="full")
    labels = df["Metric"].to_list()
    assert "Daily VaR" in labels
    assert "Expected Shortfall (cVaR)" in labels


def test_metrics_full_mode_has_streaks_and_best_worst(reports):
    """Full mode includes streak and best/worst metrics."""
    df = reports.metrics(mode="full")
    labels = df["Metric"].to_list()
    assert "Max Consecutive Wins" in labels
    assert "Best Day" in labels
    assert "Worst Day" in labels


# ── full() HTML report ────────────────────────────────────────────────────────


def test_full_returns_string(reports):
    """full() returns a non-empty string."""
    html = reports.full()
    assert isinstance(html, str)
    assert len(html) > 0


def test_full_is_valid_html_document(reports):
    """full() output starts with a proper HTML document declaration."""
    html = reports.full()
    assert "<!DOCTYPE html>" in html
    assert "<html" in html
    assert "</html>" in html


def test_full_contains_title(reports):
    """Custom title appears in the generated HTML."""
    html = reports.full(title="My Custom Report")
    assert "My Custom Report" in html


def test_full_contains_performance_metrics_section(reports):
    """Performance Metrics section is present in the report."""
    html = reports.full()
    assert "Performance Metrics" in html


def test_full_contains_drawdown_section(reports):
    """Worst drawdown section is present in the report."""
    html = reports.full()
    assert "Worst 5 Drawdown Periods" in html


def test_full_contains_charts_section(reports):
    """Charts section is present in the report."""
    html = reports.full()
    assert "Charts" in html


def test_full_contains_period_info(reports):
    """Period information (start → end) appears in the report header."""
    html = reports.full()
    assert "→" in html


def test_full_embeds_plotly(reports):
    """At least one Plotly chart div is embedded in the report."""
    html = reports.full()
    assert "plotly" in html.lower()


# ── Edge-case coverage for full-mode exception paths ─────────────────────────


def test_metrics_full_mode_greeks_raises(data):
    """metrics(mode='full') skips Beta/Alpha gracefully when greeks() raises."""
    from jquantstats._reports._data import Reports

    class _StatsProxy:
        """Proxy that delegates all stat calls except greeks()."""

        def __init__(self, real: object) -> None:
            """Wrap a real Stats object."""
            self._real = real

        def __getattr__(self, name: str) -> object:
            """Delegate to the wrapped Stats object."""
            return getattr(self._real, name)

        def greeks(self) -> dict:
            """Always raise to simulate missing benchmark."""
            raise AttributeError("no benchmark")  # noqa: TRY003

    class _DataProxy:
        """Proxy that replaces stats with the failing greeks proxy."""

        def __init__(self, real: object) -> None:
            """Wrap a real Data object."""
            self._real = real

        def __getattr__(self, name: str) -> object:
            """Delegate to the wrapped Data object."""
            return getattr(self._real, name)

        @property
        def stats(self) -> _StatsProxy:
            """Return a Stats proxy with greeks() overridden."""
            return _StatsProxy(self._real.stats)

        @property
        def all(self) -> pl.DataFrame:
            """Forward the all DataFrame."""
            return self._real.all  # type: ignore[union-attr]

    reports_proxy = Reports(data=_DataProxy(data))  # type: ignore[arg-type]
    df = reports_proxy.metrics(mode="full")
    labels = df["Metric"].to_list()
    assert "R²" in labels
    assert "Beta" not in labels


def test_metrics_full_mode_correlation_raises(data):
    """Full mode handles correlation failure gracefully."""
    from jquantstats._reports._data import Reports

    class _DataProxy:
        """Proxy that removes benchmark column from 'all' to break correlation."""

        def __init__(self, real: object) -> None:
            """Wrap a real Data object."""
            self._real = real

        def __getattr__(self, name: str) -> object:
            """Delegate to the wrapped Data object."""
            return getattr(self._real, name)

        @property
        def stats(self) -> object:
            """Forward the stats object."""
            return self._real.stats  # type: ignore[union-attr]

        @property
        def all(self) -> pl.DataFrame:
            """Return all DataFrame without benchmark column to break pl.corr."""
            real_all: pl.DataFrame = self._real.all  # type: ignore[union-attr]
            # Drop the benchmark column so selecting it later raises
            return real_all.select(["Date", "AAPL", "META"])

        @property
        def benchmark(self) -> pl.DataFrame:
            """Still return benchmark so the correlation block is entered."""
            return self._real.benchmark  # type: ignore[union-attr]

    reports_proxy = Reports(data=_DataProxy(data))  # type: ignore[arg-type]
    df = reports_proxy.metrics(mode="full")
    # Correlation row is absent but R² is still computed
    assert "R²" in df["Metric"].to_list()
    assert "Correlation" not in df["Metric"].to_list()


def test_full_skips_missing_plot_method(data):
    """full() skips chart methods that are absent on the plots object."""
    from jquantstats._reports._data import Reports

    class _PlotsProxy:
        """Plots proxy that hides plot_histogram to exercise the fn-is-None branch."""

        def __init__(self, real: object) -> None:
            """Wrap a real Plots object."""
            self._real = real

        def __getattr__(self, name: str) -> object:
            """Delegate all methods except plot_histogram."""
            if name == "plot_histogram":
                raise AttributeError(name)
            return getattr(self._real, name)

    class _DataProxy:
        """Proxy that swaps in the limited plots object."""

        def __init__(self, real: object) -> None:
            """Wrap a real Data object."""
            self._real = real

        def __getattr__(self, name: str) -> object:
            """Delegate to the wrapped Data object."""
            return getattr(self._real, name)

        @property
        def stats(self) -> object:
            """Forward the stats object."""
            return self._real.stats  # type: ignore[union-attr]

        @property
        def all(self) -> pl.DataFrame:
            """Forward the all DataFrame."""
            return self._real.all  # type: ignore[union-attr]

        @property
        def plots(self) -> _PlotsProxy:
            """Return a Plots proxy with plot_histogram removed."""
            return _PlotsProxy(self._real.plots)  # type: ignore[union-attr]

    reports_proxy = Reports(data=_DataProxy(data))  # type: ignore[arg-type]
    html = reports_proxy.full()
    assert "<!DOCTYPE html>" in html
