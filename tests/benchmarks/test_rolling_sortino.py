"""Benchmarks for rolling Sortino implementations."""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import numpy as np
import polars as pl
import pytest


def _build_10y_daily_returns() -> pl.DataFrame:
    """Generate a deterministic 10-year daily returns series."""
    rng = np.random.default_rng(7)
    return pl.DataFrame({"returns": rng.normal(loc=0.0003, scale=0.012, size=252 * 10)})


def _rolling_sortino_native(df: pl.DataFrame, rolling_period: int = 126, periods_per_year: int = 252) -> pl.DataFrame:
    """Rolling Sortino using native Polars expressions only."""
    scale = math.sqrt(periods_per_year)
    mean_ret = pl.col("returns").rolling_mean(window_size=rolling_period)
    downside = (
        pl.when(pl.col("returns") < 0)
        .then(pl.col("returns") ** 2)
        .otherwise(0.0)
        .rolling_mean(window_size=rolling_period)
    )
    return df.select(((mean_ret / downside.sqrt()) * scale).alias("returns"))


def _legacy_window_sortino(window: list[float], scale: float) -> float:
    """Sortino over one rolling window for the legacy map_elements path."""
    arr = np.asarray(window, dtype=np.float64)
    if np.isnan(arr).any():
        return float("nan")
    downside = np.sqrt(np.mean(np.where(arr < 0.0, arr**2, 0.0)))
    if downside == 0.0:
        mean = float(arr.mean())
        if mean > 0.0:
            return float("inf")
        if mean < 0.0:
            return float("-inf")
        return float("nan")
    return float((arr.mean() / downside) * scale)


def _rolling_sortino_legacy_map_elements(
    df: pl.DataFrame, rolling_period: int = 126, periods_per_year: int = 252
) -> pl.DataFrame:
    """Legacy rolling Sortino using Python-level map_elements UDF."""
    scale = math.sqrt(periods_per_year)
    window_values = pl.concat_list([pl.col("returns").shift(i) for i in range(rolling_period)])
    return df.select(
        window_values
        .map_elements(
            lambda window: _legacy_window_sortino(window, scale),
            return_dtype=pl.Float64,
            skip_nulls=False,
        )
        .alias("returns")
    )


@pytest.fixture
def ten_year_daily_returns() -> pl.DataFrame:
    """Deterministic benchmark input: 10 years of daily returns."""
    return _build_10y_daily_returns()


@pytest.mark.benchmark(group="rolling_sortino_10y_daily")
@pytest.mark.parametrize(
    ("implementation", "runner"),
    [
        ("legacy_map_elements", _rolling_sortino_legacy_map_elements),
        ("native_expressions", _rolling_sortino_native),
    ],
)
def test_rolling_sortino_benchmark(
    benchmark: pytest.BenchmarkFixture,
    ten_year_daily_returns: pl.DataFrame,
    implementation: str,
    runner: Callable[[pl.DataFrame], pl.DataFrame],
) -> None:
    """Benchmark old vs native rolling Sortino implementations."""
    result = benchmark(runner, ten_year_daily_returns)
    assert implementation in {"legacy_map_elements", "native_expressions"}
    assert result.shape == (252 * 10, 1)


def test_native_expression_speedup(ten_year_daily_returns: pl.DataFrame) -> None:
    """Confirm native expressions are faster than legacy map_elements approach."""

    def _median_runtime(fn: Callable[[pl.DataFrame], pl.DataFrame], repeats: int = 7) -> float:
        runtimes: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn(ten_year_daily_returns)
            runtimes.append(time.perf_counter() - start)
        return float(np.median(runtimes))

    legacy = _median_runtime(_rolling_sortino_legacy_map_elements)
    native = _median_runtime(_rolling_sortino_native)
    assert native < legacy
