"""Protocols describing the minimal interfaces required by the _reports subpackage."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import plotly.graph_objects as go
import polars as pl


@runtime_checkable
class StatsLike(Protocol):  # pragma: no cover
    """Structural interface for the statistics methods used by :class:`~jquantstats._reports._data.Reports`."""

    def sharpe(self, periods: int | float | None = None) -> dict[str, float]:
        """Annualised Sharpe ratio per asset."""
        ...

    def smart_sharpe(self, periods: int | float | None = None) -> dict[str, float]:
        """Smart Sharpe ratio per asset."""
        ...

    def sortino(self, periods: int | float | None = None) -> dict[str, float]:
        """Annualised Sortino ratio per asset."""
        ...

    def adjusted_sortino(self, periods: int | float | None = None) -> dict[str, float]:
        """Adjusted Sortino (Sortino / √2) per asset."""
        ...

    def smart_sortino(self, periods: int | float | None = None) -> dict[str, float]:
        """Smart Sortino ratio per asset."""
        ...

    def omega(self, periods: int | float | None = None) -> dict[str, float]:
        """Omega ratio per asset."""
        ...

    def probabilistic_sharpe_ratio(self) -> dict[str, float]:
        """Probabilistic Sharpe ratio per asset."""
        ...

    def cagr(self, periods: int | float | None = None) -> dict[str, float]:
        """Compound annual growth rate per asset."""
        ...

    def comp(self) -> dict[str, float]:
        """Total compounded return per asset."""
        ...

    def exposure(self) -> dict[str, float]:
        """Market exposure (time in market) per asset."""
        ...

    def max_drawdown(self) -> dict[str, float]:
        """Maximum drawdown per asset."""
        ...

    def avg_drawdown(self) -> dict[str, float]:
        """Average drawdown per asset."""
        ...

    def max_drawdown_duration(self) -> dict[str, float]:
        """Maximum drawdown duration per asset."""
        ...

    def recovery_factor(self) -> dict[str, float]:
        """Recovery factor per asset."""
        ...

    def ulcer_index(self) -> dict[str, float]:
        """Ulcer index per asset."""
        ...

    def serenity_index(self) -> dict[str, float]:
        """Serenity index per asset."""
        ...

    def ulcer_performance_index(self) -> dict[str, float]:
        """Ulcer performance index per asset."""
        ...

    def calmar(self, periods: int | float | None = None) -> dict[str, float]:
        """Calmar ratio per asset."""
        ...

    def rar(self, periods: int | float = 252) -> dict[str, float]:
        """Risk-adjusted return per asset."""
        ...

    def risk_return_ratio(self) -> dict[str, float]:
        """Risk-return ratio per asset."""
        ...

    def gain_to_pain_ratio(self, aggregate: str | None = None) -> dict[str, float]:
        """Gain-to-pain ratio per asset."""
        ...

    def payoff_ratio(self) -> dict[str, float]:
        """Payoff ratio per asset."""
        ...

    def profit_factor(self) -> dict[str, float]:
        """Profit factor per asset."""
        ...

    def profit_ratio(self) -> dict[str, float]:
        """Profit ratio per asset."""
        ...

    def common_sense_ratio(self) -> dict[str, float]:
        """Common sense ratio per asset."""
        ...

    def cpc_index(self) -> dict[str, float]:
        """CPC index per asset."""
        ...

    def tail_ratio(self) -> dict[str, float]:
        """Tail ratio per asset."""
        ...

    def outlier_win_ratio(self) -> dict[str, float]:
        """Outlier win ratio per asset."""
        ...

    def outlier_loss_ratio(self) -> dict[str, float]:
        """Outlier loss ratio per asset."""
        ...

    def volatility(self, periods: int | float | None = None) -> dict[str, float]:
        """Annualised volatility per asset."""
        ...

    def value_at_risk(self, alpha: float = 0.05) -> dict[str, float]:
        """Value at Risk per asset."""
        ...

    def conditional_value_at_risk(self, sigma: float = 1.0, alpha: float = 0.05, **kwargs: float) -> dict[str, float]:
        """Conditional Value at Risk per asset."""
        ...

    def win_loss_ratio(self) -> dict[str, float]:
        """Win/loss ratio per asset."""
        ...

    def win_rate(self) -> dict[str, float]:
        """Win rate per asset."""
        ...

    def monthly_win_rate(self) -> dict[str, float]:
        """Monthly win rate per asset."""
        ...

    def avg_return(self) -> dict[str, float]:
        """Average return per asset."""
        ...

    def avg_win(self) -> dict[str, float]:
        """Average win per asset."""
        ...

    def avg_loss(self) -> dict[str, float]:
        """Average loss per asset."""
        ...

    def best(self) -> dict[str, float]:
        """Best period return per asset."""
        ...

    def worst(self) -> dict[str, float]:
        """Worst period return per asset."""
        ...

    def skew(self) -> dict[str, float]:
        """Skewness per asset."""
        ...

    def kurtosis(self) -> dict[str, float]:
        """Kurtosis per asset."""
        ...

    def consecutive_wins(self) -> dict[str, float]:
        """Maximum consecutive wins per asset."""
        ...

    def consecutive_losses(self) -> dict[str, float]:
        """Maximum consecutive losses per asset."""
        ...

    def kelly_criterion(self) -> dict[str, float]:
        """Kelly criterion per asset."""
        ...

    def risk_of_ruin(self) -> dict[str, float]:
        """Risk of ruin per asset."""
        ...

    def expected_return(self, aggregate: str | None = None) -> dict[str, float]:
        """Expected return per asset."""
        ...

    def greeks(self) -> dict[str, dict[str, float]]:
        """Alpha and beta per asset."""
        ...

    def r2(self) -> dict[str, float]:
        """R-squared per asset versus benchmark."""
        ...

    def treynor_ratio(self, periods: int | float | None = None) -> dict[str, float]:
        """Treynor ratio per asset."""
        ...

    def drawdown_details(self) -> dict[str, pl.DataFrame]:
        """Drawdown period details per asset."""
        ...

    def summary(self) -> pl.DataFrame:
        """Full summary DataFrame (one row per metric, one column per asset)."""
        ...


@runtime_checkable
class DataLike(Protocol):  # pragma: no cover
    """Structural interface required by the :class:`~jquantstats._reports._data.Reports` class.

    Any object satisfying this protocol can be passed as ``data`` without a
    concrete dependency on :class:`~jquantstats._data.Data`.
    """

    @property
    def stats(self) -> StatsLike:
        """Statistics facade."""
        ...

    @property
    def all(self) -> pl.DataFrame:
        """Combined DataFrame of date index and all return columns."""
        ...


@runtime_checkable
class PlotsLike(Protocol):  # pragma: no cover
    """Structural interface for the portfolio plots facade used by :class:`~jquantstats._reports._portfolio.Report`."""

    def snapshot(self) -> go.Figure:
        """NAV + drawdown snapshot figure."""
        ...

    def rolling_sharpe_plot(self) -> go.Figure:
        """Rolling Sharpe figure."""
        ...

    def rolling_volatility_plot(self) -> go.Figure:
        """Rolling volatility figure."""
        ...

    def annual_sharpe_plot(self) -> go.Figure:
        """Annual Sharpe figure."""
        ...

    def monthly_returns_heatmap(self) -> go.Figure:
        """Monthly returns heatmap figure."""
        ...

    def correlation_heatmap(self) -> go.Figure:
        """Correlation heatmap figure."""
        ...

    def lead_lag_ir_plot(self) -> go.Figure:
        """Lead/lag IR figure."""
        ...

    def trading_cost_impact_plot(self) -> go.Figure:
        """Trading cost impact figure."""
        ...


@runtime_checkable
class PortfolioLike(Protocol):  # pragma: no cover
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
