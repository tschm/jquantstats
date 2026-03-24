#!/usr/bin/env python
"""Standalone benchmark: jquantstats vs quantstats.

Generates ~2520 synthetic daily returns (approx. 10 years) and measures
wall-clock time and peak RSS memory for three representative operations:

    1. Sharpe ratio
    2. Maximum drawdown
    3. Metrics report generation (jquantstats) vs HTML report (quantstats)

Run with:
    uv run python benchmarks/run.py

Dependencies (dev group):
    pandas, quantstats
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import tracemalloc
from collections.abc import Callable

import numpy as np
import pandas as pd
import polars as pl
import quantstats as qs

# ---------------------------------------------------------------------------
# Add the src directory to sys.path so this script can be run standalone
# from the repository root without an editable install.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from jquantstats import Data  # noqa: E402  (after sys.path manipulation)

# ---------------------------------------------------------------------------
# Shared synthetic dataset  (~2520 daily returns, 10-year horizon)
# ---------------------------------------------------------------------------
_RNG = np.random.default_rng(42)
_N = 2520  # approx. 10 x 252 trading days
_DATES = pd.bdate_range("2014-01-02", periods=_N)
_RETURNS_ARR = _RNG.normal(loc=0.0003, scale=0.01, size=_N)

# quantstats expects a pandas Series with a DatetimeIndex
_RETURNS_PD = pd.Series(_RETURNS_ARR, index=_DATES, name="Strategy")

# jquantstats expects a polars DataFrame with a Date column
_RETURNS_PL = pl.DataFrame(
    {
        "Date": [d.date() for d in _DATES],
        "Strategy": _RETURNS_ARR.tolist(),
    }
).with_columns(pl.col("Date").cast(pl.Date))


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

_REPEATS = 10


def _measure(fn: Callable[[], object]) -> tuple[float, float]:
    """Return (mean_ms, peak_kib) for *fn* called _REPEATS times.

    Args:
        fn: Zero-argument callable to benchmark.

    Returns:
        A tuple of (mean_ms, peak_kib) where mean_ms is the mean wall-clock
        time per call in milliseconds and peak_kib is the peak traced memory
        allocation in kibibytes.

    """
    # Warm-up - ensures imports / JIT etc. are not included in timing
    fn()

    tracemalloc.start()
    t0 = time.perf_counter()
    for _ in range(_REPEATS):
        fn()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    mean_ms = elapsed / _REPEATS * 1_000
    peak_kib = peak / 1_024
    return mean_ms, peak_kib


# ---------------------------------------------------------------------------
# jquantstats callables (Data object built once before measurement starts)
# ---------------------------------------------------------------------------

_JQS_DATA = Data.from_returns(returns=_RETURNS_PL)


def _jqs_sharpe() -> object:
    """Compute jquantstats Sharpe ratio on the shared dataset.

    Returns:
        A dict mapping asset names to their Sharpe ratios.

    """
    return _JQS_DATA.stats.sharpe()


def _jqs_max_drawdown() -> object:
    """Compute jquantstats maximum drawdown on the shared dataset.

    Returns:
        A dict mapping asset names to their maximum drawdown values.

    """
    return _JQS_DATA.stats.max_drawdown()


def _jqs_report() -> object:
    """Generate jquantstats metrics report on the shared dataset.

    Returns:
        A polars DataFrame containing all key financial metrics.

    """
    return _JQS_DATA.reports.metrics()


# ---------------------------------------------------------------------------
# quantstats callables
# ---------------------------------------------------------------------------


def _qs_sharpe() -> float:
    """Compute quantstats Sharpe ratio on the shared pandas Series.

    Returns:
        The Sharpe ratio as a float.

    """
    return qs.stats.sharpe(_RETURNS_PD)


def _qs_max_drawdown() -> float:
    """Compute quantstats maximum drawdown on the shared pandas Series.

    Returns:
        The maximum drawdown as a float.

    """
    return qs.stats.max_drawdown(_RETURNS_PD)


def _qs_report() -> None:
    """Generate quantstats HTML report to a temporary file.

    Writes the full matplotlib/seaborn HTML report to a temporary file
    to avoid stdout noise while still measuring the full generation cost.

    """
    with tempfile.NamedTemporaryFile(suffix=".html", delete=True) as tmp:
        qs.reports.html(_RETURNS_PD, output=tmp.name)


# ---------------------------------------------------------------------------
# Run and format results
# ---------------------------------------------------------------------------

_OPERATIONS: list[tuple[str, Callable, Callable]] = [
    ("Sharpe ratio", _jqs_sharpe, _qs_sharpe),
    ("Max drawdown", _jqs_max_drawdown, _qs_max_drawdown),
    ("Metrics report", _jqs_report, _qs_report),
]

_COL_WIDTHS = (22, 12, 12, 12, 12, 10)
_HEADERS = ("Operation", "jqs ms", "qs ms", "jqs KiB", "qs KiB", "x faster")


def _hline(widths: tuple[int, ...]) -> str:
    """Return a horizontal separator line for an ASCII table.

    Args:
        widths: Column widths for each column.

    Returns:
        A string containing the horizontal separator line.

    """
    return "+-" + "-+-".join("-" * w for w in widths) + "-+"


def _row(cells: tuple[str, ...], widths: tuple[int, ...]) -> str:
    """Return a formatted row string for an ASCII table.

    Args:
        cells: Cell values for each column.
        widths: Column widths for each column.

    Returns:
        A string containing the formatted row.

    """
    return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, widths, strict=False)) + " |"


def main() -> None:
    """Run all benchmarks and print a formatted summary table."""
    print(f"\nBenchmark: jquantstats vs quantstats  (n={_N} rows, {_REPEATS} repeats)")
    print(f"Python {sys.version.split()[0]}   polars {pl.__version__}   pandas {pd.__version__}\n")

    hl = _hline(_COL_WIDTHS)
    print(hl)
    print(_row(_HEADERS, _COL_WIDTHS))
    print(hl)

    for name, jqs_fn, qs_fn in _OPERATIONS:
        jqs_ms, jqs_kib = _measure(jqs_fn)
        qs_ms, qs_kib = _measure(qs_fn)
        speedup = qs_ms / jqs_ms if jqs_ms > 0 else float("inf")
        row = (name, f"{jqs_ms:.1f}", f"{qs_ms:.1f}", f"{jqs_kib:.0f}", f"{qs_kib:.0f}", f"{speedup:.1f}x")
        print(_row(row, _COL_WIDTHS))

    print(hl)
    print()
    print("Columns: ms = milliseconds per call | KiB = peak traced memory (kibibytes)")
    print("x faster = qs_ms / jqs_ms  (higher is better for jquantstats)")


if __name__ == "__main__":
    main()
