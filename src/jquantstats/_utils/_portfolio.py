"""Utility methods for Portfolio objects."""

from __future__ import annotations

from collections.abc import Callable, Hashable

import numpy as np
import polars as pl

from ._data import DataUtils
from ._protocol import PortfolioLike

__all__ = ["PortfolioUtils"]


class PortfolioUtils:
    """Utility transforms and conversions for Portfolio objects.

    Exposes the same API as `DataUtils`
    but is initialised from a `Portfolio`
    and routes all calls through ``portfolio.data``.
    """

    __slots__ = ("_portfolio",)

    def __init__(self, portfolio: PortfolioLike) -> None:
        self._portfolio = portfolio

    def __repr__(self) -> str:
        """Return a string representation of the PortfolioUtils object."""
        return f"PortfolioUtils(assets={self._portfolio.assets})"

    def _du(self) -> DataUtils:
        """Return a DataUtils instance backed by this portfolio's data bridge."""
        return DataUtils(self._portfolio.data)

    # ── delegated API (mirrors DataUtils) ─────────────────────────────────────

    def to_prices(self, base: float = 1e5) -> pl.DataFrame:
        """Convert portfolio returns to a cumulative price series.

        See `to_prices` for full
        documentation.

        Args:
            base: Starting value for the price series.  Defaults to ``1e5``.

        Returns:
            DataFrame with date column (if present) and one price column per asset.

        """
        return self._du().to_prices(base=base)

    def to_log_returns(self) -> pl.DataFrame:
        """Convert portfolio returns to log returns: ``ln(1 + r)``.

        See `to_log_returns` for
        full documentation.

        Returns:
            DataFrame of log returns.

        """
        return self._du().to_log_returns()

    def log_returns(self) -> pl.DataFrame:
        """Alias for `to_log_returns`.

        Returns:
            DataFrame of log returns.

        """
        return self._du().log_returns()

    def rebase(self, base: float = 100.0) -> pl.DataFrame:
        """Normalise the portfolio's returns as a price series starting at *base*.

        See `rebase` for full
        documentation.

        Args:
            base: Target starting value.  Defaults to ``100.0``.

        Returns:
            DataFrame with price columns anchored to *base* at t = 0.

        """
        return self._du().rebase(base=base)

    def group_returns(self, period: str = "1mo", compounded: bool = True) -> pl.DataFrame:
        """Aggregate portfolio returns by a calendar period.

        See `group_returns` for
        full documentation.

        Args:
            period: Aggregation period.  Defaults to ``"1mo"`` (monthly).
            compounded: Whether to compound returns.  Defaults to ``True``.

        Returns:
            DataFrame with one row per period and one column per asset.

        """
        return self._du().group_returns(period=period, compounded=compounded)

    def aggregate_returns(self, period: str = "1mo", compounded: bool = True) -> pl.DataFrame:
        """Alias for `group_returns`.

        Args:
            period: Aggregation period.  Defaults to ``"1mo"`` (monthly).
            compounded: Whether to compound returns.  Defaults to ``True``.

        Returns:
            DataFrame with one row per period and one column per asset.

        """
        return self._du().aggregate_returns(period=period, compounded=compounded)

    def to_excess_returns(self, rf: float = 0.0, nperiods: int | None = None) -> pl.DataFrame:
        """Subtract a risk-free rate from portfolio returns.

        See `to_excess_returns`
        for full documentation.

        Args:
            rf: Annual risk-free rate as a decimal.  Defaults to ``0.0``.
            nperiods: Periods per year for rate conversion.  Defaults to ``None``.

        Returns:
            DataFrame of excess returns.

        """
        return self._du().to_excess_returns(rf=rf, nperiods=nperiods)

    def to_volatility_adjusted_returns(
        self,
        window: int = 60,
        vol_estimator: Callable[[pl.Expr], pl.Expr] | None = None,
    ) -> pl.DataFrame:
        """Convert portfolio returns to volatility-adjusted returns.

        See `to_volatility_adjusted_returns` for full documentation.

        Args:
            window: Rolling lookback for volatility.  Defaults to ``60``.
            vol_estimator: A callable ``(pl.Expr) -> pl.Expr`` that
                produces a volatility series.  Defaults to ``None``
                (uses ``rolling_std(window)``).

        Returns:
            DataFrame of volatility-adjusted returns.

        """
        return self._du().to_volatility_adjusted_returns(window=window, vol_estimator=vol_estimator)

    def exponential_stdev(self, window: int = 30, is_halflife: bool = False) -> pl.DataFrame:
        """Compute exponentially weighted standard deviation of portfolio returns.

        See `exponential_stdev`
        for full documentation.

        Args:
            window: Span or half-life of the EWMA decay.  Defaults to ``30``.
            is_halflife: Interpret *window* as half-life when ``True``.
                Defaults to ``False``.

        Returns:
            DataFrame of rolling EWMA standard deviations.

        """
        return self._du().exponential_stdev(window=window, is_halflife=is_halflife)

    def winsorise(self, window: int = 7, n_sigma: float = 3.0) -> pl.DataFrame:
        """Winsorise portfolio returns by clipping to within *n_sigma* rolling standard deviations.

        See `DataUtils.winsorise` for full
        documentation.

        Args:
            window: Rolling lookback for mean and standard deviation.
                Defaults to ``7``.
            n_sigma: Number of standard deviations for the clip bounds.
                Defaults to ``3.0``.

        Returns:
            DataFrame with the same columns as the input returns, extreme
            values clipped.

        """
        return self._du().winsorise(window=window, n_sigma=n_sigma)

    def exponential_cov(
        self, window: int = 30, is_halflife: bool = False, warmup: int = 0
    ) -> dict[Hashable, np.ndarray]:
        """Compute the exponentially weighted covariance matrix of portfolio returns.

        See `DataUtils.exponential_cov` for full
        documentation.

        Args:
            window: Span (default) or half-life (when *is_halflife* is
                ``True``) of the exponential decay.  Defaults to ``30``.
            is_halflife: When ``True`` *window* is interpreted as the
                half-life; otherwise it is the EWMA span.  Defaults to
                ``False``.
            warmup: Minimum number of common observations required before
                a pair's cell is non-NaN.  Defaults to ``0``.

        Returns:
            Dictionary keyed by index value (date or integer) mapping to
            a square symmetric ``numpy.ndarray`` whose dimensions match
            the return columns exposed by ``portfolio.data``. In this
            facade that is typically only the portfolio-level ``returns``
            column, so the matrices are usually ``(1, 1)`` even for
            multi-asset portfolios. Unavailable cells are ``NaN``.

        """
        return self._du().exponential_cov(window=window, is_halflife=is_halflife, warmup=warmup)
