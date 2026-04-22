"""Utility methods for Data objects — the jquantstats equivalent of qs.utils."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable

import polars as pl

from ..exceptions import MissingDateColumnError
from ._protocol import DataLike

__all__ = ["DataUtils"]

# Maps human-readable aliases to Polars every-string format.
_PERIOD_ALIASES: dict[str, str] = {
    "daily": "1d",
    "weekly": "1w",
    "monthly": "1mo",
    "quarterly": "1q",
    "annual": "1y",
    "yearly": "1y",
}


@dataclasses.dataclass(frozen=True)
class DataUtils:
    """Utility transforms and conversions for financial returns data.

    Mirrors the public API of ``quantstats.utils`` but operates on Polars
    DataFrames and integrates with `Data` via the
    ``data.utils`` property.

    Attributes:
        data: Any object satisfying the `DataLike`
            protocol — typically a `Data` instance.

    """

    data: DataLike

    def __repr__(self) -> str:
        """Return a string representation of the DataUtils object."""
        return f"DataUtils(assets={list(self.data.returns.columns)})"

    # ── helpers ───────────────────────────────────────────────────────────────

    def _combined(self) -> pl.DataFrame:
        """Return index hstacked with returns (no benchmark)."""
        return pl.concat([self.data.index, self.data.returns], how="horizontal")

    def _asset_cols(self) -> list[str]:
        """Return the asset column names from returns (excluding benchmark)."""
        return list(self.data.returns.columns)

    def _require_temporal_index(self, method: str) -> str:
        """Raise MissingDateColumnError if the index is not temporal, else return date col name."""
        date_cols = self.data.date_col
        if not date_cols:
            raise MissingDateColumnError(method)  # pragma: no cover
        date_col = date_cols[0]
        if not self.data.index[date_col].dtype.is_temporal():
            raise MissingDateColumnError(method)
        return date_col

    # ── public API ────────────────────────────────────────────────────────────

    def to_prices(self, base: float = 1e5) -> pl.DataFrame:
        """Convert returns to a cumulative price series.

        Computes ``base * prod(1 + r_t)`` for each asset column, matching the
        behaviour of ``quantstats.utils.to_prices``.

        Args:
            base: Starting value for the price series.  Defaults to ``1e5``.

        Returns:
            DataFrame with the same date column (if present) and one price
            column per asset.

        """
        asset_cols = self._asset_cols()
        return self._combined().with_columns(
            [(pl.col(c).fill_null(0.0) + 1.0).cum_prod().mul(base).alias(c) for c in asset_cols]
        )

    def to_log_returns(self) -> pl.DataFrame:
        """Convert simple returns to log returns: ``ln(1 + r)``.

        Matches ``quantstats.utils.to_log_returns``.

        Returns:
            DataFrame with the same columns as the input returns, values
            replaced by their log-return equivalents.

        """
        asset_cols = self._asset_cols()
        return self._combined().with_columns(
            [(pl.col(c).fill_null(0.0) + 1.0).log(base=math.e).alias(c) for c in asset_cols]
        )

    def to_volatility_adjusted_returns(
        self,
        window: int = 60,
        vol_estimator: Callable[[pl.Expr], pl.Expr] | None = None,
    ) -> pl.DataFrame:
        """Convert simple returns to volatility adjusted returns.

        Divides each return by a volatility estimate:
        ``vol_adjusted_r_t = r_t / vol(r_t)``.

        By default the volatility estimate is
        ``pl.Expr.rolling_std(window)``.  Pass *vol_estimator* to
        override with any function that maps a ``pl.Expr`` to a
        ``pl.Expr`` (e.g. an EWMA standard deviation).

        Matches ``quantstats.utils.to_volatility_adjusted_returns``.

        Args:
            window: Rolling lookback for the default volatility
                estimator.  Ignored when *vol_estimator* is provided.
                Defaults to ``60``.
            vol_estimator: A callable ``(pl.Expr) -> pl.Expr`` that
                produces a volatility series from a returns expression.
                Defaults to ``None`` (uses ``rolling_std(window)``).

        Returns:
            DataFrame with the same columns as the input returns, values
            replaced by their volatility adjusted equivalents.

        """
        if vol_estimator is None:

            def vol_estimator(expr: pl.Expr) -> pl.Expr:
                """Return rolling standard deviation over *window*."""
                return expr.rolling_std(window)

        asset_cols = self._asset_cols()
        return self._combined().with_columns([pl.col(c) / vol_estimator(pl.col(c)) for c in asset_cols])

    def log_returns(self) -> pl.DataFrame:
        """Alias for `to_log_returns`.

        Matches ``quantstats.utils.log_returns``.

        Returns:
            DataFrame of log returns.

        """
        return self.to_log_returns()

    def rebase(self, base: float = 100.0) -> pl.DataFrame:
        """Normalise the returns as a price series that starts at *base*.

        Converts returns to prices via `to_prices` and then rescales
        each column so its first observation equals *base* exactly, matching
        the behaviour of ``quantstats.utils.rebase``.

        Args:
            base: Target starting value.  Defaults to ``100.0``.

        Returns:
            DataFrame with price columns anchored to *base* at t = 0.

        """
        prices_df = self.to_prices(base=1.0)
        asset_cols = self._asset_cols()
        return prices_df.with_columns([(pl.col(c) / pl.col(c).first() * base).alias(c) for c in asset_cols])

    def group_returns(self, period: str = "1mo", compounded: bool = True) -> pl.DataFrame:
        """Aggregate returns by a calendar period.

        Requires a temporal (Date/Datetime) index; raises
        `MissingDateColumnError` for integer-indexed data.

        Human-readable aliases are accepted alongside native Polars interval
        strings (``"1mo"``, ``"1q"``, ``"1y"``, ``"1w"``, ``"1d"``):

        ``"daily"``, ``"weekly"``, ``"monthly"``, ``"quarterly"``,
        ``"annual"`` / ``"yearly"``.

        Args:
            period: Aggregation period.  Defaults to ``"1mo"`` (monthly).
            compounded: When ``True`` (default) compound the returns
                ``prod(1 + r) - 1``; when ``False`` sum them.

        Returns:
            DataFrame with one row per period and one column per asset.

        """
        date_col = self._require_temporal_index("group_returns")
        polars_period = _PERIOD_ALIASES.get(period, period)
        asset_cols = self._asset_cols()

        if compounded:
            agg_exprs = [((pl.col(c).fill_null(0.0) + 1.0).product() - 1.0).alias(c) for c in asset_cols]
        else:
            agg_exprs = [pl.col(c).fill_null(0.0).sum().alias(c) for c in asset_cols]

        return (
            self._combined()
            .sort(date_col)
            .group_by_dynamic(date_col, every=polars_period)
            .agg(agg_exprs)
            .sort(date_col)
        )

    def aggregate_returns(self, period: str = "1mo", compounded: bool = True) -> pl.DataFrame:
        """Alias for `group_returns`.

        Matches ``quantstats.utils.aggregate_returns``.

        Args:
            period: Aggregation period.  See `group_returns` for accepted values.
            compounded: Whether to compound returns.  Defaults to ``True``.

        Returns:
            DataFrame with one row per period and one column per asset.

        """
        return self.group_returns(period=period, compounded=compounded)

    def to_excess_returns(self, rf: float = 0.0, nperiods: int | None = None) -> pl.DataFrame:
        """Subtract a risk-free rate from returns.

        When *nperiods* is supplied the annual *rf* is converted to a
        per-period rate via ``(1 + rf)^(1/nperiods) - 1``, matching
        ``quantstats.utils.to_excess_returns``.

        Args:
            rf: Annual risk-free rate as a decimal (e.g. ``0.05`` for 5 %).
                Defaults to ``0.0``.
            nperiods: Number of return periods per year used to convert *rf*
                to a per-period rate.  When ``None`` *rf* is applied as-is.

        Returns:
            DataFrame of excess returns with the same columns as the input.

        """
        rf_per_period = ((1.0 + rf) ** (1.0 / nperiods) - 1.0) if nperiods is not None else rf
        asset_cols = self._asset_cols()
        return self._combined().with_columns([(pl.col(c) - rf_per_period).alias(c) for c in asset_cols])

    def exponential_stdev(self, window: int = 30, is_halflife: bool = False) -> pl.DataFrame:
        """Compute the exponentially weighted standard deviation of returns.

        Matches ``quantstats.utils.exponential_stdev``.  Uses Polars
        ``ewm_std`` under the hood.

        Args:
            window: Span (default) or half-life (when *is_halflife* is
                ``True``) of the exponential decay.  Defaults to ``30``.
            is_halflife: When ``True`` *window* is interpreted as the
                half-life; otherwise it is the EWMA span.  Defaults to
                ``False``.

        Returns:
            DataFrame of rolling EWMA standard deviations with the same
            columns as the input returns.

        """
        asset_cols = self._asset_cols()
        if is_halflife:
            exprs = [pl.col(c).ewm_std(half_life=window, min_samples=1).alias(c) for c in asset_cols]
        else:
            exprs = [pl.col(c).ewm_std(span=window, min_samples=1).alias(c) for c in asset_cols]
        return self._combined().with_columns(exprs)
