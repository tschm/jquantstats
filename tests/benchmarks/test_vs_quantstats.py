"""Pytest-benchmark tests: jquantstats vs quantstats.

These tests use pytest-benchmark to produce reproducible timing measurements
for the three canonical operations compared in the benchmark documentation:

    1. Sharpe ratio
    2. Maximum drawdown
    3. Full HTML report generation

Run via:
    make benchmark        # full benchmark suite (also generates report.html)
    uv run pytest tests/benchmarks/ -v --benchmark-disable-gc

The tests are deliberately excluded from the regular test run
(``make test`` ignores ``tests/benchmarks/``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest
import quantstats as qs

from jquantstats import Data

# ---------------------------------------------------------------------------
# Shared synthetic dataset  (~2520 daily returns, 10-year horizon)
# ---------------------------------------------------------------------------
_RNG = np.random.default_rng(42)
_N = 2520  # approx. 10 x 252 trading days
_DATES = pd.bdate_range("2014-01-02", periods=_N)
_RETURNS_ARR = _RNG.normal(loc=0.0003, scale=0.01, size=_N)

# quantstats expects a pandas Series with a DatetimeIndex
RETURNS_PD = pd.Series(_RETURNS_ARR, index=_DATES, name="Strategy")

# jquantstats expects a polars DataFrame with a Date column
RETURNS_PL = pl.DataFrame(
    {
        "Date": [d.date() for d in _DATES],
        "Strategy": _RETURNS_ARR.tolist(),
    }
).with_columns(pl.col("Date").cast(pl.Date))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def jqs_data() -> Data:
    """Return a jquantstats Data object built once per module."""
    return Data.from_returns(returns=RETURNS_PL)


# ---------------------------------------------------------------------------
# Sharpe ratio
# ---------------------------------------------------------------------------


def test_jqs_sharpe(benchmark, jqs_data: Data) -> None:
    """Benchmark jquantstats Sharpe ratio calculation.

    Args:
        benchmark: The pytest-benchmark fixture.
        jqs_data: The jquantstats Data fixture.

    """
    benchmark(jqs_data.stats.sharpe)


def test_qs_sharpe(benchmark) -> None:
    """Benchmark quantstats Sharpe ratio calculation.

    Args:
        benchmark: The pytest-benchmark fixture.

    """
    benchmark(qs.stats.sharpe, RETURNS_PD)


# ---------------------------------------------------------------------------
# Maximum drawdown
# ---------------------------------------------------------------------------


def test_jqs_max_drawdown(benchmark, jqs_data: Data) -> None:
    """Benchmark jquantstats max drawdown calculation.

    Args:
        benchmark: The pytest-benchmark fixture.
        jqs_data: The jquantstats Data fixture.

    """
    benchmark(jqs_data.stats.max_drawdown)


def test_qs_max_drawdown(benchmark) -> None:
    """Benchmark quantstats max drawdown calculation.

    Args:
        benchmark: The pytest-benchmark fixture.

    """
    benchmark(qs.stats.max_drawdown, RETURNS_PD)


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------


def test_jqs_report(benchmark, jqs_data: Data) -> None:
    """Benchmark jquantstats metrics report generation.

    Args:
        benchmark: The pytest-benchmark fixture.
        jqs_data: The jquantstats Data fixture.

    """
    benchmark(jqs_data.reports.metrics)


def test_qs_report(benchmark, tmp_path) -> None:
    """Benchmark quantstats HTML report generation.

    Args:
        benchmark: The pytest-benchmark fixture.
        tmp_path: Pytest tmp_path fixture for output file.

    """
    output = str(tmp_path / "report.html")

    def _gen() -> None:
        """Generate the quantstats HTML report to a temporary file."""
        qs.reports.html(RETURNS_PD, output=output)

    benchmark(_gen)
