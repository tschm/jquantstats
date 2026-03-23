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

Module-level helpers and the ``columnwise_stat`` / ``to_frame`` decorators are
defined in :mod:`jquantstats._stats._core` and re-exported here for backwards
compatibility.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import polars as pl

from ._basic import _BasicStatsMixin
from ._core import (
    _drawdown_series,
    _to_float,
    columnwise_stat,
    to_frame,
)
from ._performance import _PerformanceStatsMixin
from ._reporting import _ReportingStatsMixin
from ._rolling import _RollingStatsMixin

if TYPE_CHECKING:
    from .._data import Data

__all__ = [
    "Stats",
    "_drawdown_series",
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
