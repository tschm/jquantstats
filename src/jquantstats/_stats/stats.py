"""Statistical analysis tools for financial returns data.

This module provides the :class:`Stats` dataclass, which is the public-facing
class that combines four mixin classes:

- :class:`~jquantstats._stats._basic._BasicStatsMixin` — basic statistics,
  volatility, win/loss metrics, and risk metrics (VaR, Sharpe inputs, Kelly).
- :class:`~jquantstats._stats._performance._PerformanceStatsMixin` — Sharpe,
  Sortino, drawdown, benchmark/factor analytics (R², alpha, beta).
- :class:`~jquantstats._stats._reporting._ReportingStatsMixin` — temporal
  reporting, Calmar, recovery factor, capture ratios, annual breakdown, and
  summary.
- :class:`~jquantstats._stats._rolling._RollingStatsMixin` — rolling-window
  time-series metrics (rolling Sharpe, Sortino, and volatility).

Scalar statistics are computed via a single
:py:meth:`polars.LazyFrame.select` call (see
:func:`~jquantstats._stats._core._lazy_columnwise` and
:mod:`~jquantstats._stats._expr`) rather than a per-column Python loop.  This
lets the Polars query optimiser fuse and parallelise work across asset columns
and opens the door to streaming on large datasets.

Stats that cannot be cleanly expressed as a ``pl.Expr`` (e.g. ``greeks``,
``r_squared``, ``information_ratio``, ``prob_sharpe_ratio``,
``hhi_positive``, ``hhi_negative``, ``sharpe_variance``) remain on the
legacy ``@columnwise_stat`` path.

Module-level helpers and the ``columnwise_stat`` / ``to_frame`` decorators are
defined in :mod:`jquantstats._stats._core` and re-exported here for backwards
compatibility.  Pure ``pl.Expr`` factory functions are available in
:mod:`jquantstats._stats._expr`.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import polars as pl

from . import _expr
from ._basic import _BasicStatsMixin
from ._core import (
    _drawdown_series,
    _lazy_columnwise,
    _to_float,
    columnwise_stat,
    to_frame,
)
from ._performance import _PerformanceStatsMixin
from ._reporting import _ReportingStatsMixin
from ._rolling import _RollingStatsMixin

if TYPE_CHECKING:
    from ..data import Data

__all__ = [
    "Stats",
    "_drawdown_series",
    "_expr",
    "_lazy_columnwise",
    "_to_float",
    "columnwise_stat",
    "to_frame",
]


@dataclasses.dataclass(frozen=True)
class Stats(_BasicStatsMixin, _PerformanceStatsMixin, _ReportingStatsMixin, _RollingStatsMixin):
    """Statistical analysis tools for financial returns data.

    Provides a comprehensive set of methods for calculating various financial
    metrics and statistics on returns data, including:

    - Basic statistics (mean, skew, kurtosis)
    - Risk metrics (volatility, value-at-risk, drawdown)
    - Performance ratios (Sharpe, Sortino, information ratio)
    - Win/loss metrics (win rate, profit factor, payoff ratio)
    - Rolling calculations (rolling volatility, rolling Sharpe)
    - Factor analysis (alpha, beta, R-squared)

    Scalar statistics use :func:`~jquantstats._stats._core._lazy_columnwise`
    to compute all asset columns in a single :py:meth:`polars.LazyFrame.select`
    call instead of a per-column Python loop.  See
    :mod:`~jquantstats._stats._expr` for the ``pl.Expr`` factory functions.

    Stats that require multi-column access or complex scipy logic
    (``greeks``, ``r_squared``, ``information_ratio``, ``prob_sharpe_ratio``,
    ``hhi_positive``, ``hhi_negative``, ``sharpe_variance``) remain on the
    legacy ``@columnwise_stat`` path.

    Metrics are organised into focused modules:

    - :class:`~jquantstats._stats._basic._BasicStatsMixin`
    - :class:`~jquantstats._stats._performance._PerformanceStatsMixin`
    - :class:`~jquantstats._stats._reporting._ReportingStatsMixin`
    - :class:`~jquantstats._stats._rolling._RollingStatsMixin`

    Attributes:
        data: The :class:`~jquantstats._data.Data` object containing returns
            and benchmark data.
        all: A DataFrame combining all data (index, returns, benchmark) for
            easy column selection.
    """

    data: Data
    all: pl.DataFrame | None = None  # Default is None; will be set in __post_init__

    def __post_init__(self) -> None:
        object.__setattr__(self, "all", self.data.all)

    def __repr__(self) -> str:
        """Return a string representation of the Stats object."""
        return f"Stats(assets={self.data.assets})"
