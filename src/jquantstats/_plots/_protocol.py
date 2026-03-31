"""Protocols describing the minimal interfaces required by the _plots subpackage."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import polars as pl

from jquantstats._cost_model import CostModel


@runtime_checkable
class DataLike(Protocol):  # pragma: no cover
    """Structural interface required by the :class:`~jquantstats._plots._data.DataPlots` class.

    Any object satisfying this protocol can be passed as ``data`` without a
    concrete dependency on :class:`~jquantstats._data.Data`.
    """

    @property
    def all(self) -> pl.DataFrame:
        """Combined DataFrame of date index and returns columns."""
        ...

    @property
    def assets(self) -> list[str]:
        """Names of the asset return columns."""
        ...


@runtime_checkable
class PortfolioLike(Protocol):  # pragma: no cover
    """Structural interface required by the :class:`~jquantstats._plots._portfolio.PortfolioPlots` class.

    Any object satisfying this protocol can be passed as ``portfolio`` without a
    concrete dependency on :class:`~jquantstats.portfolio.Portfolio`.
    """

    prices: pl.DataFrame
    aum: float
    cost_model: CostModel

    @property
    def nav_accumulated(self) -> pl.DataFrame:
        """Accumulated NAV series."""
        ...

    @property
    def tilt(self) -> PortfolioLike:
        """Tilt component portfolio."""
        ...

    @property
    def timing(self) -> PortfolioLike:
        """Timing component portfolio."""
        ...

    @property
    def net_cost_nav(self) -> pl.DataFrame:
        """Net-of-cost accumulated NAV series."""
        ...

    @property
    def drawdown(self) -> pl.DataFrame:
        """Drawdown series."""
        ...

    @property
    def assets(self) -> list[str]:
        """Asset names."""
        ...

    @property
    def monthly(self) -> pl.DataFrame:
        """Monthly returns grouped by year and month."""
        ...

    @property
    def profits(self) -> pl.DataFrame:
        """Per-period profit series."""
        ...

    @property
    def stats(self) -> Any:
        """Statistics facade (rolling_sharpe, rolling_volatility, annual_breakdown, sharpe)."""
        ...

    def lag(self, n: int) -> PortfolioLike:
        """Return a lagged copy of this portfolio."""
        ...

    def smoothed_holding(self, n: int) -> PortfolioLike:
        """Return a smoothed-holdings copy of this portfolio."""
        ...

    def trading_cost_impact(self, max_bps: int = 20) -> pl.DataFrame:
        """Return a DataFrame of Sharpe vs. one-way trading costs."""
        ...

    def correlation(self, frame: pl.DataFrame, name: str = "portfolio") -> pl.DataFrame:
        """Return the correlation matrix including the portfolio profit series."""
        ...
