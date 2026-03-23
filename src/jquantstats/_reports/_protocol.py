"""Protocols describing the minimal interfaces required by the _reports subpackage."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class StatsLike(Protocol):
    """Structural interface for the statistics methods used by :class:`~jquantstats._reports._data.Reports`."""

    def sharpe(self, periods: int | float | None = None) -> dict[str, float]:
        """Annualised Sharpe ratio per asset."""
        ...

    def sortino(self, periods: int | float | None = None) -> dict[str, float]:
        """Annualised Sortino ratio per asset."""
        ...

    def max_drawdown(self) -> dict[str, float]:
        """Maximum drawdown per asset."""
        ...

    def volatility(self, periods: int | float | None = None) -> dict[str, float]:
        """Annualised volatility per asset."""
        ...

    def value_at_risk(self, alpha: float = 0.05) -> dict[str, float]:
        """Value at Risk per asset."""
        ...

    def win_loss_ratio(self) -> dict[str, float]:
        """Win/loss ratio per asset."""
        ...

    def skew(self) -> dict[str, float]:
        """Skewness per asset."""
        ...

    def kurtosis(self) -> dict[str, float]:
        """Kurtosis per asset."""
        ...

    def summary(self) -> pl.DataFrame:
        """Full summary DataFrame (one row per metric, one column per asset)."""
        ...


@runtime_checkable
class DataLike(Protocol):
    """Structural interface required by the :class:`~jquantstats._reports._data.Reports` class.

    Any object satisfying this protocol can be passed as ``data`` without a
    concrete dependency on :class:`~jquantstats._data.Data`.
    """

    @property
    def stats(self) -> StatsLike:
        """Statistics facade."""
        ...


@runtime_checkable
class PlotsLike(Protocol):
    """Structural interface for the portfolio plots facade used by :class:`~jquantstats._reports._portfolio.Report`."""

    def snapshot(self) -> object:
        """NAV + drawdown snapshot figure."""
        ...

    def rolling_sharpe_plot(self) -> object:
        """Rolling Sharpe figure."""
        ...

    def rolling_volatility_plot(self) -> object:
        """Rolling volatility figure."""
        ...

    def annual_sharpe_plot(self) -> object:
        """Annual Sharpe figure."""
        ...

    def monthly_returns_heatmap(self) -> object:
        """Monthly returns heatmap figure."""
        ...

    def correlation_heatmap(self) -> object:
        """Correlation heatmap figure."""
        ...

    def lead_lag_ir_plot(self) -> object:
        """Lead/lag IR figure."""
        ...

    def trading_cost_impact_plot(self) -> object:
        """Trading cost impact figure."""
        ...


@runtime_checkable
class PortfolioLike(Protocol):
    """Structural interface required by the :class:`~jquantstats._reports._portfolio.Report` class.

    Any object satisfying this protocol can be passed as ``portfolio`` without a
    concrete dependency on :class:`~jquantstats.portfolio.Portfolio`.
    """

    prices: pl.DataFrame
    aum: float

    @property
    def assets(self) -> list[str]:
        """Asset names."""
        ...

    @property
    def plots(self) -> PlotsLike:
        """Portfolio plots facade."""
        ...

    @property
    def stats(self) -> StatsLike:
        """Statistics facade."""
        ...

    def turnover_summary(self) -> pl.DataFrame:
        """Turnover summary DataFrame."""
        ...
