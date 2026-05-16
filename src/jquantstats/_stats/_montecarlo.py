"""Monte Carlo simulation statistics."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from ..data import Data


class _MonteCarloStatsMixin:
    """Mixin providing Monte Carlo simulation statistics for return series."""

    _data: Data
    all: pl.DataFrame

    if TYPE_CHECKING:
        from ._protocol import DataLike

        data: DataLike
        all: pl.DataFrame | None

    @staticmethod
    def _validate_positive_integer(name: str, value: int) -> int:
        """Validate that *value* is a positive integer."""
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    @staticmethod
    def _block_bootstrap_path(values: np.ndarray, period: int, block_size: int) -> np.ndarray:
        """Sample one return path via block bootstrap with replacement."""
        n_obs = values.size
        if n_obs == 0:
            return np.full(period, np.nan, dtype=np.float64)

        starts = np.random.randint(0, n_obs - block_size + 1, size=math.ceil(period / block_size))
        chunks = [values[start : start + block_size] for start in starts]
        return np.concatenate(chunks)[:period]

    def _simulate_distribution(self, n: int, period: int, metric: Callable[[np.ndarray], float]) -> pl.DataFrame:
        """Run *n* block-bootstrap simulations of length *period* and score each path."""
        n = self._validate_positive_integer("n", n)
        period = self._validate_positive_integer("period", period)

        result: dict[str, list[float]] = {}
        block_size = max(1, int(round(period**0.5)))

        for col, series in self._data.items():
            clean = series.cast(pl.Float64).drop_nulls().drop_nans()
            values = np.asarray(clean.to_numpy(), dtype=np.float64)
            if values.size == 0:
                result[col] = [float("nan")] * n
                continue

            block = min(block_size, values.size)
            sims = np.empty(n, dtype=np.float64)
            for i in range(n):
                sampled = self._block_bootstrap_path(values, period, block)
                sims[i] = metric(sampled)
            result[col] = sims.tolist()

        return pl.DataFrame(result)

    def montecarlo(self, n: int = 1000, period: int = 252) -> pl.DataFrame:
        """Simulate cumulative returns across *n* block-bootstrap paths."""
        return self._simulate_distribution(n=n, period=period, metric=lambda path: float(np.prod(1.0 + path) - 1.0))

    def montecarlo_sharpe(
        self,
        n: int = 1000,
        period: int = 252,
        periods_per_year: int | float | None = None,
    ) -> pl.DataFrame:
        """Simulate the Sharpe-ratio distribution across block-bootstrap paths."""
        ppy = self._data._periods_per_year if periods_per_year is None else periods_per_year
        if ppy <= 0:
            raise ValueError("periods_per_year must be positive")

        def _sharpe(path: np.ndarray) -> float:
            if path.size < 2:
                return float("nan")
            std = float(np.std(path, ddof=1))
            if std == 0.0:
                return float("nan")
            return float(path.mean() / std * math.sqrt(ppy))

        return self._simulate_distribution(n=n, period=period, metric=_sharpe)

    def montecarlo_drawdown(self, n: int = 1000, period: int = 252) -> pl.DataFrame:
        """Simulate the maximum-drawdown distribution across block-bootstrap paths."""

        def _max_drawdown(path: np.ndarray) -> float:
            if path.size == 0:
                return float("nan")
            nav = np.cumprod(1.0 + path)
            hwm = np.maximum.accumulate(nav)
            return float(np.min(nav / hwm - 1.0))

        return self._simulate_distribution(n=n, period=period, metric=_max_drawdown)

    def montecarlo_cagr(
        self,
        n: int = 1000,
        period: int = 252,
        periods_per_year: int | float | None = None,
    ) -> pl.DataFrame:
        """Simulate the CAGR distribution across block-bootstrap paths."""
        ppy = self._data._periods_per_year if periods_per_year is None else periods_per_year
        if ppy <= 0:
            raise ValueError("periods_per_year must be positive")

        def _cagr(path: np.ndarray) -> float:
            total = float(np.prod(1.0 + path))
            if total <= 0:
                return float("nan")
            years = period / ppy
            return float(total ** (1.0 / years) - 1.0)

        return self._simulate_distribution(n=n, period=period, metric=_cagr)
