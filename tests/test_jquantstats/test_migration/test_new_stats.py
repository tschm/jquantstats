"""Migration tests for functions added after the initial quantstats port."""

import numpy as np
import pytest
import quantstats as qs


@pytest.fixture
def aligned(pandas_frame):
    """AAPL and SPY aligned on common dates."""
    return pandas_frame[["AAPL", "SPY -- Benchmark"]].dropna()


# ── r2 (alias for r_squared) ──────────────────────────────────────────────────


def test_r2_alias(stats, aligned):
    """r2() is an alias for r_squared() and matches quantstats."""
    x = stats.r2()
    y = qs.stats.r2(aligned["AAPL"], aligned["SPY -- Benchmark"])
    assert x["AAPL"] == pytest.approx(y, abs=1e-6)


# ── outliers / remove_outliers ────────────────────────────────────────────────


def test_outliers_count(stats, aapl):
    """outliers() returns the same values as quantstats."""
    jqs = stats.outliers()["AAPL"]
    qs_vals = qs.stats.outliers(aapl, quantile=0.95)
    assert len(jqs) == len(qs_vals)


def test_outliers_values(stats, aapl):
    """outliers() values match quantstats."""
    jqs = stats.outliers()["AAPL"].sort().to_list()
    qs_vals = sorted(qs.stats.outliers(aapl, quantile=0.95).tolist())
    np.testing.assert_allclose(jqs, qs_vals, atol=1e-12)


def test_remove_outliers_count(stats, aapl):
    """remove_outliers() returns the same number of values as quantstats."""
    jqs = stats.remove_outliers()["AAPL"]
    qs_vals = qs.stats.remove_outliers(aapl, quantile=0.95)
    assert len(jqs) == len(qs_vals)


def test_remove_outliers_values(stats, aapl):
    """remove_outliers() values match quantstats."""
    jqs = stats.remove_outliers()["AAPL"].sort().to_list()
    qs_vals = sorted(qs.stats.remove_outliers(aapl, quantile=0.95).tolist())
    np.testing.assert_allclose(jqs, qs_vals, atol=1e-12)


# ── expected_return ───────────────────────────────────────────────────────────


def test_expected_return_no_aggregate(stats, aapl):
    """expected_return() without aggregate matches quantstats geometric mean."""
    jqs = stats.expected_return()["AAPL"]
    qs_val = qs.stats.expected_return(aapl)
    assert jqs == pytest.approx(qs_val, abs=1e-12)


# ── drawdown_details ──────────────────────────────────────────────────────────
#
# qs.stats.drawdown_details takes a pre-computed drawdown series and includes a
# 1-period "drawdown" on the very first day of data when the initial return is
# negative (because the drawdown series is non-zero there).  jquantstats uses
# `nav < hwm` (strict) which is always False on day 1, so that 1-day stub is
# absent.  Tests below strip that row from the qs result before comparing.


def _qs_details(aapl):
    """Return qs drawdown details, dropping any 1-day stub on the first data date."""
    df = qs.stats.drawdown_details(qs.stats.to_drawdown_series(aapl))
    first_date = str(aapl.index[0].date())
    return df[df["start"] != first_date].reset_index(drop=True)


def test_drawdown_details_period_count(stats, aapl):
    """drawdown_details() finds the same number of drawdown periods as quantstats."""
    jqs_df = stats.drawdown_details()["AAPL"]
    qs_df = _qs_details(aapl)
    assert len(jqs_df) == len(qs_df)


def test_drawdown_details_start_dates(stats, aapl):
    """drawdown_details() start dates match quantstats."""
    import polars as pl

    jqs_df = stats.drawdown_details()["AAPL"]
    qs_df = _qs_details(aapl)

    jqs_starts = jqs_df.sort("start")["start"].cast(pl.String).to_list()
    qs_starts = sorted(qs_df["start"].tolist())
    assert jqs_starts == qs_starts


def test_drawdown_details_max_drawdown(stats, aapl):
    """drawdown_details() max drawdown values match quantstats (within 1e-4)."""
    jqs_df = stats.drawdown_details()["AAPL"].sort("start")
    qs_df = _qs_details(aapl).sort_values("start")

    # jqs returns fraction; qs returns percentage — multiply jqs by 100
    jqs_vals = (jqs_df["max_drawdown"] * 100).to_list()
    qs_vals = qs_df["max drawdown"].tolist()
    np.testing.assert_allclose(jqs_vals, qs_vals, atol=1e-4)


# ── rolling_greeks ────────────────────────────────────────────────────────────


def test_rolling_greeks_beta(stats, aapl, benchmark_pd):
    """rolling_greeks() beta matches quantstats on overlapping dates."""
    jqs_df = stats.rolling_greeks(rolling_period=126, periods_per_year=252)
    qs_df = qs.stats.rolling_greeks(aapl, benchmark_pd, periods=126)

    jqs_pd = jqs_df.to_pandas().set_index("Date")["AAPL_beta"].dropna()
    qs_beta = qs_df["beta"].dropna()
    common = jqs_pd.index.intersection(qs_beta.index)

    assert len(common) > 100
    np.testing.assert_allclose(jqs_pd[common].values, qs_beta[common].values, atol=1e-4)


# NOTE: rolling_greeks alpha is intentionally NOT compared against quantstats.
# qs.stats.rolling_greeks computes alpha as:
#   global_mean(returns) - rolling_beta * global_mean(benchmark)
# using the full-series mean instead of the rolling-window mean.  This is
# a known defect in quantstats.  jquantstats uses the correct OLS intercept:
#   rolling_mean(returns) - rolling_beta * rolling_mean(benchmark)


# ── treynor_ratio ─────────────────────────────────────────────────────────────


# NOTE: treynor_ratio is intentionally NOT compared against quantstats.
# qs.stats.treynor_ratio computes: comp(returns) / beta   (total compounded return)
# jquantstats computes:           cagr(returns) / beta    (annualised return)
# The jquantstats definition is the standard financial one.
