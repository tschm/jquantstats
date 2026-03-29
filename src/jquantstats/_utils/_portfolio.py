"""Utility methods for Portfolio objects."""

from __future__ import annotations

import dataclasses

import polars as pl

from ._data import DataUtils
from ._protocol import PortfolioLike

__all__ = ["PortfolioUtils"]


@dataclasses.dataclass(frozen=True)
class PortfolioUtils:
    """Utility transforms and conversions for Portfolio objects.

    Exposes the same API as :class:`~jquantstats._utils._data.DataUtils`
    but is initialised from a :class:`~jquantstats.portfolio.Portfolio`
    and routes all calls through ``portfolio.data``.

    Attributes:
        portfolio: Any object satisfying the
            :class:`~jquantstats._utils._protocol.PortfolioLike` protocol —
            typically a :class:`~jquantstats.portfolio.Portfolio` instance.

    """

    portfolio: PortfolioLike

    def __repr__(self) -> str:
        """Return a string representation of the PortfolioUtils object."""
        return f"PortfolioUtils(assets={self.portfolio.assets})"

    def _du(self) -> DataUtils:
        """Return a DataUtils instance backed by this portfolio's data bridge."""
        return DataUtils(self.portfolio.data)

    # ── delegated API (mirrors DataUtils) ─────────────────────────────────────

    def to_prices(self, base: float = 1e5) -> pl.DataFrame:
        """Convert portfolio returns to a cumulative price series.

        See :meth:`~jquantstats._utils._data.DataUtils.to_prices` for full
        documentation.

        Args:
            base: Starting value for the price series.  Defaults to ``1e5``.

        Returns:
            DataFrame with date column (if present) and one price column per asset.

        """
        return self._du().to_prices(base=base)

    def to_log_returns(self) -> pl.DataFrame:
        """Convert portfolio returns to log returns: ``ln(1 + r)``.

        See :meth:`~jquantstats._utils._data.DataUtils.to_log_returns` for
        full documentation.

        Returns:
            DataFrame of log returns.

        """
        return self._du().to_log_returns()

    def log_returns(self) -> pl.DataFrame:
        """Alias for :meth:`to_log_returns`.

        Returns:
            DataFrame of log returns.

        """
        return self._du().log_returns()

    def rebase(self, base: float = 100.0) -> pl.DataFrame:
        """Normalise the portfolio's returns as a price series starting at *base*.

        See :meth:`~jquantstats._utils._data.DataUtils.rebase` for full
        documentation.

        Args:
            base: Target starting value.  Defaults to ``100.0``.

        Returns:
            DataFrame with price columns anchored to *base* at t = 0.

        """
        return self._du().rebase(base=base)

    def group_returns(self, period: str = "1mo", compounded: bool = True) -> pl.DataFrame:
        """Aggregate portfolio returns by a calendar period.

        See :meth:`~jquantstats._utils._data.DataUtils.group_returns` for
        full documentation.

        Args:
            period: Aggregation period.  Defaults to ``"1mo"`` (monthly).
            compounded: Whether to compound returns.  Defaults to ``True``.

        Returns:
            DataFrame with one row per period and one column per asset.

        """
        return self._du().group_returns(period=period, compounded=compounded)

    def aggregate_returns(self, period: str = "1mo", compounded: bool = True) -> pl.DataFrame:
        """Alias for :meth:`group_returns`.

        Args:
            period: Aggregation period.  Defaults to ``"1mo"`` (monthly).
            compounded: Whether to compound returns.  Defaults to ``True``.

        Returns:
            DataFrame with one row per period and one column per asset.

        """
        return self._du().aggregate_returns(period=period, compounded=compounded)

    def to_excess_returns(self, rf: float = 0.0, nperiods: int | None = None) -> pl.DataFrame:
        """Subtract a risk-free rate from portfolio returns.

        See :meth:`~jquantstats._utils._data.DataUtils.to_excess_returns`
        for full documentation.

        Args:
            rf: Annual risk-free rate as a decimal.  Defaults to ``0.0``.
            nperiods: Periods per year for rate conversion.  Defaults to ``None``.

        Returns:
            DataFrame of excess returns.

        """
        return self._du().to_excess_returns(rf=rf, nperiods=nperiods)

    def exponential_stdev(self, window: int = 30, is_halflife: bool = False) -> pl.DataFrame:
        """Compute exponentially weighted standard deviation of portfolio returns.

        See :meth:`~jquantstats._utils._data.DataUtils.exponential_stdev`
        for full documentation.

        Args:
            window: Span or half-life of the EWMA decay.  Defaults to ``30``.
            is_halflife: Interpret *window* as half-life when ``True``.
                Defaults to ``False``.

        Returns:
            DataFrame of rolling EWMA standard deviations.

        """
        return self._du().exponential_stdev(window=window, is_halflife=is_halflife)
