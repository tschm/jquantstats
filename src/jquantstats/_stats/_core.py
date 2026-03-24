"""Module helpers and method decorators for statistical computations.

Provides:

- :func:`_drawdown_series` — drawdown series from a returns series.
- :func:`_to_float` — safe Polars aggregation result → Python float.
- :func:`_lazy_columnwise` — run an ``_expr`` factory for all asset columns in
  one :py:meth:`polars.LazyFrame.select` call (no per-column Python loop).
- :func:`columnwise_stat` — decorator: apply a metric to every asset column
  (kept for stats not yet ported to the ``pl.Expr`` path).
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

    Treats ``series`` as additive daily returns and builds a normalised NAV
    starting at 1.0.  The high-water mark is the running maximum of that NAV;
    drawdown is expressed as the fraction below the high-water mark.

    Args:
        series: A Polars Series of additive returns (profit / AUM).

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
    nav = 1.0 + series.cast(pl.Float64).cum_sum()
    hwm = nav.cum_max()
    hwm_safe = hwm.clip(lower_bound=1e-10)
    return ((hwm - nav) / hwm_safe).clip(lower_bound=0.0)


def _to_float(value: object) -> float:
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


# ── Lazy helpers ─────────────────────────────────────────────────────────────


def _lazy_columnwise(
    lf: pl.LazyFrame,
    expr_fn: Callable[..., pl.Expr],
    asset_cols: list[str],
    **kwargs: Any,
) -> dict[str, float]:
    """Run *expr_fn* for every column in *asset_cols* in one LazyFrame.select().

    Builds a list of named ``pl.Expr`` objects — one per asset — and collects
    them in a single pass so the Polars query optimiser can fuse and parallelise
    the work rather than materialising each column individually in Python.

    Args:
        lf: LazyFrame containing all asset columns (e.g. ``data.lazy``).
        expr_fn: Callable with signature ``(col: str, **kwargs) -> pl.Expr``.
            The returned expression must be a scalar aggregation aliased to *col*.
        asset_cols: Ordered list of column names to compute the stat for.
        **kwargs: Extra keyword arguments forwarded verbatim to *expr_fn*.

    Returns:
        dict[str, float]: Mapping of asset name → computed scalar value, with
        the same key order as *asset_cols*.

    """
    exprs = [expr_fn(col, **kwargs) for col in asset_cols]
    row = lf.select(exprs).collect().row(0, named=True)
    return cast(dict[str, float], row)


# ── Module-level decorators ──────────────────────────────────────────────────


def columnwise_stat(func: Callable[..., Any]) -> Callable[..., dict[str, float]]:
    """Apply a column-wise statistical function to all numeric columns.

    This decorator is kept for stats that have **not** yet been ported to the
    :mod:`~jquantstats._stats._expr` lazy-path.  Ported stats call
    :func:`_lazy_columnwise` directly and no longer use this decorator.

    Args:
        func (Callable): The function to decorate.

    Returns:
        Callable: The decorated function.

    Note:
        A ``pl.Expr``-based lazy path is not yet available for stats decorated
        with this function.  See :func:`_lazy_columnwise` and
        :mod:`~jquantstats._stats._expr` for the new pattern.

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
