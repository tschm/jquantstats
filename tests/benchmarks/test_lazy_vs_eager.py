"""Benchmarks: lazy (LazyData) vs eager (Data) for large return histories.

Scenario modelled after the original issue:
    "Benchmark lazy vs eager on large return histories
     (e.g. 20-year daily x 100 assets)"

Test matrix
-----------
1. **resample** — monthly resampling of the full dataset.
2. **truncate + resample** — filter to a 5-year window then resample monthly.
3. **scan_parquet + truncate + resample** — read from Parquet, filter, resample.

Each scenario is tested in both eager (Data) and lazy (LazyData) modes so
pytest-benchmark can compare wall-clock time side-by-side.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from jquantstats import Data, LazyData

# ---------------------------------------------------------------------------
# Dataset configuration (matches the issue specification)
# ---------------------------------------------------------------------------
N_ASSETS = 100
N_YEARS = 20
START_DATE = date(2004, 1, 1)

# Truncation window used for the "filter then process" scenario
TRUNC_START = date(2014, 1, 1)
TRUNC_END = date(2019, 12, 31)


# ---------------------------------------------------------------------------
# Fixtures - built once per session to avoid confounding setup cost
# ---------------------------------------------------------------------------


def _build_dates() -> list[date]:
    """Generate ~N_YEARS x 252 trading-day-like dates (Mon-Fri)."""
    dates: list[date] = []
    current = START_DATE
    total_needed = N_YEARS * 252
    while len(dates) < total_needed:
        if current.weekday() < 5:  # Mon-Fri
            dates.append(current)
        current += timedelta(days=1)
    return dates


@pytest.fixture(scope="session")
def large_data() -> Data:
    """20-year daily x 100-asset Data object built once for the whole session."""
    import random

    random.seed(42)
    dates = _build_dates()
    n = len(dates)
    cols: dict[str, object] = {"Date": dates}
    for i in range(N_ASSETS):
        cols[f"Asset{i:03d}"] = [round(random.gauss(0.0005, 0.01), 6) for _ in range(n)]
    df = pl.DataFrame(cols)
    return Data.from_returns(returns=df)


@pytest.fixture(scope="session")
def parquet_path(large_data: Data, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write *large_data* to a Parquet file and return its path."""
    path = tmp_path_factory.mktemp("bench") / "large_returns.parquet"
    large_data.all.write_parquet(str(path))
    return path


# ---------------------------------------------------------------------------
# Scenario 1 - resample only
# ---------------------------------------------------------------------------


def test_resample_eager(benchmark: BenchmarkFixture, large_data: Data) -> None:
    """Eagerly resample the full 20-year daily dataset to monthly returns."""
    result = benchmark(large_data.resample, "1mo")
    assert result.returns.shape[0] > 0


def test_resample_lazy(benchmark: BenchmarkFixture, large_data: Data) -> None:
    """Lazily resample the full 20-year daily dataset to monthly returns."""

    def _run() -> Data:
        """Build query plan and collect monthly returns."""
        return large_data.lazy().resample("1mo").collect()

    result = benchmark(_run)
    assert result.returns.shape[0] > 0


# ---------------------------------------------------------------------------
# Scenario 2 - truncate then resample
# ---------------------------------------------------------------------------


def test_truncate_resample_eager(benchmark: BenchmarkFixture, large_data: Data) -> None:
    """Eagerly truncate to 5-year window then resample to monthly."""

    def _run() -> Data:
        """Filter in-memory then aggregate to monthly returns."""
        return large_data.truncate(start=TRUNC_START, end=TRUNC_END).resample("1mo")

    result = benchmark(_run)
    assert result.returns.shape[0] > 0


def test_truncate_resample_lazy(benchmark: BenchmarkFixture, large_data: Data) -> None:
    """Lazily truncate to 5-year window then resample to monthly."""

    def _run() -> Data:
        """Build lazy query plan, collect after filtering and resampling."""
        return large_data.lazy().truncate(start=TRUNC_START, end=TRUNC_END).resample("1mo").collect()

    result = benchmark(_run)
    assert result.returns.shape[0] > 0


# ---------------------------------------------------------------------------
# Scenario 3 - scan Parquet + truncate + resample
# ---------------------------------------------------------------------------


def test_scan_parquet_truncate_resample_lazy(benchmark: BenchmarkFixture, parquet_path: Path) -> None:
    """Scan Parquet -> truncate -> resample lazily; Polars pushes predicates down."""

    def _run() -> Data:
        """Scan file, filter by date range, and resample; all deferred to collect."""
        return (
            LazyData.scan_parquet(str(parquet_path), date_col="Date")
            .truncate(start=TRUNC_START, end=TRUNC_END)
            .resample("1mo")
            .collect()
        )

    result = benchmark(_run)
    assert result.returns.shape[0] > 0


def test_scan_parquet_truncate_resample_eager(benchmark: BenchmarkFixture, parquet_path: Path) -> None:
    """Read full Parquet eagerly -> truncate -> resample; no pushdown optimisation."""

    def _run() -> Data:
        """Load entire Parquet into memory, then filter and resample eagerly."""
        df = pl.read_parquet(str(parquet_path))
        data = Data.from_returns(returns=df)
        return data.truncate(start=TRUNC_START, end=TRUNC_END).resample("1mo")

    result = benchmark(_run)
    assert result.returns.shape[0] > 0
