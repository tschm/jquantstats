"""Module helpers and method decorators for statistical computations.

Provides:

- :func:`_drawdown_series` — drawdown series from a returns series.
- :func:`_to_float` — safe Polars aggregation result → Python float.
- :func:`columnwise_stat` — decorator: apply a metric to every asset column.
- :func:`to_frame` — decorator: build a per-column Polars DataFrame result.

These building blocks are shared across the stats mixin modules
(:mod:`~jquantstats._stats._basic`, :mod:`~jquantstats._stats._performance`,
:mod:`~jquantstats._stats._reporting`, :mod:`~jquantstats._stats._rolling`).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from functools import wraps
from typing import Any, cast

import polars as pl

# ── Module helpers ────────────────────────────────────────────────────────────


def _drawdown_series(series: pl.Series) -> pl.Series:
    """Compute the drawdown percentage series from a returns series.

    Builds a compound NAV (geometric cumulative product) from the returns
    series and expresses drawdown as the fraction below the running high-water
    mark.  This matches the quantstats convention.

    Args:
        series: A Polars Series of multiplicative daily returns.

    Returns:
        A Polars Float64 Series whose values are in [0, 1].  A value of 0
        means the NAV is at its all-time high; a value of 0.2 means the NAV
        is 20 % below its previous peak.

    Examples:
        >>> import polars as pl
        >>> s = pl.Series([0.0, -0.1, 0.2])
        >>> [round(x, 10) for x in _drawdown_series(s).to_list()]
        [0.0, 0.1, 0.0]
    """
    nav = (1.0 + series.cast(pl.Float64)).cum_prod()
    hwm = nav.cum_max()
    hwm_safe = hwm.clip(lower_bound=1e-10)
    return ((hwm - nav) / hwm_safe).clip(lower_bound=0.0)


def _to_float(value: Any) -> float:
    """Safely convert a Polars aggregation result to float.

    Examples:
        >>> _to_float(2.0)
        2.0
        >>> _to_float(None)
        0.0
    """
    if value is None:
        return 0.0
    if isinstance(value, timedelta):
        return value.total_seconds()
    return float(cast(float, value))


# ── Module-level decorators ──────────────────────────────────────────────────


def columnwise_stat(func: Callable[..., Any]) -> Callable[..., dict[str, float]]:
    """Apply a column-wise statistical function to all numeric columns.

    Args:
        func (Callable): The function to decorate.

    Returns:
        Callable: The decorated function.

    """

    @wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> dict[str, float]:
        """Apply *func* to every column and return a ``{column: value}`` mapping."""
        return {col: func(self, series, *args, **kwargs) for col, series in self.data.items()}

    return wrapper


def to_frame(func: Callable[..., Any]) -> Callable[..., pl.DataFrame]:
    """Apply per-column expressions and evaluates with .with_columns(...).

    Args:
        func (Callable): The function to decorate.

    Returns:
        Callable: The decorated function.

    """

    @wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> pl.DataFrame:
        """Apply *func* per column and return the result as a Polars DataFrame."""
        return cast(pl.DataFrame, self.all).select(
            [pl.col(name) for name in self.data.date_col]
            + [func(self, series, *args, **kwargs).alias(col) for col, series in self.data.items()]
        )

    return wrapper
