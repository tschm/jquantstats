"""Basic statistical metrics for financial returns data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from ._core import _lazy_columnwise, columnwise_stat
from ._expr import (
    avg_loss_expr,
    avg_return_expr,
    avg_win_expr,
    best_expr,
    conditional_value_at_risk_expr,
    exposure_expr,
    gain_to_pain_ratio_expr,
    kurtosis_expr,
    payoff_ratio_expr,
    profit_factor_expr,
    profit_ratio_expr,
    risk_return_ratio_expr,
    skew_expr,
    value_at_risk_expr,
    volatility_expr,
    win_rate_expr,
    worst_expr,
)

# ── Basic statistics mixin ───────────────────────────────────────────────────


class _BasicStatsMixin:
    """Mixin providing basic return/risk and win/loss financial statistics.

    Covers: basic statistics (skew, kurtosis, avg return/win/loss), volatility,
    win/loss metrics (payoff ratio, profit factor), and risk metrics (VaR, CVaR,
    win rate, kelly criterion, best/worst, exposure).

    Scalar stats are computed via a single :py:meth:`polars.LazyFrame.select`
    call (using :func:`~jquantstats._stats._core._lazy_columnwise`) rather than
    a per-column Python loop, enabling the Polars query optimiser to fuse and
    parallelise work across asset columns.

    Attributes (provided by the concrete subclass):
        data: The :class:`~jquantstats._data.Data` object.
        all: Combined DataFrame for efficient column selection.
    """

    if TYPE_CHECKING:
        from ._protocol import DataLike

        data: DataLike
        all: pl.DataFrame | None

    # ── Basic statistics ──────────────────────────────────────────────────────

    def skew(self) -> dict[str, float]:
        """Calculate skewness (asymmetry) for each numeric column.

        Returns:
            dict[str, float]: Mapping of asset name → skewness value.

        """
        return _lazy_columnwise(self.data.lazy, skew_expr, self.data.assets)

    def kurtosis(self) -> dict[str, float]:
        """Calculate the kurtosis of returns.

        The degree to which a distribution peak compared to a normal distribution.

        Returns:
            dict[str, float]: Mapping of asset name → kurtosis value.

        """
        return _lazy_columnwise(self.data.lazy, kurtosis_expr, self.data.assets)

    def avg_return(self) -> dict[str, float]:
        """Calculate average return per non-zero, non-null value.

        Returns:
            dict[str, float]: Mapping of asset name → average return.

        """
        return _lazy_columnwise(self.data.lazy, avg_return_expr, self.data.assets)

    def avg_win(self) -> dict[str, float]:
        """Calculate the average winning return/trade for an asset.

        Returns:
            dict[str, float]: Mapping of asset name → average winning return.

        """
        return _lazy_columnwise(self.data.lazy, avg_win_expr, self.data.assets)

    def avg_loss(self) -> dict[str, float]:
        """Calculate the average loss return/trade for a period.

        Returns:
            dict[str, float]: Mapping of asset name → average loss.

        """
        return _lazy_columnwise(self.data.lazy, avg_loss_expr, self.data.assets)

    # ── Volatility & risk ─────────────────────────────────────────────────────

    def volatility(self, periods: int | float | None = None, annualize: bool = True) -> dict[str, float]:
        """Calculate the volatility of returns.

        - Std dev of returns
        - Annualized by sqrt(periods) if ``annualize`` is True.

        Args:
            periods (int, optional): Number of periods per year. Defaults to ``_periods_per_year``.
            annualize (bool, optional): Whether to annualize the result. Defaults to True.

        Returns:
            dict[str, float]: Mapping of asset name → volatility.

        Raises:
            TypeError: If *periods* is not numeric.

        """
        raw_periods = periods or self.data._periods_per_year

        # Ensure it's numeric
        if not isinstance(raw_periods, int | float):
            raise TypeError(f"Expected int or float for periods, got {type(raw_periods).__name__}")  # noqa: TRY003

        return _lazy_columnwise(
            self.data.lazy,
            volatility_expr,
            self.data.assets,
            periods=float(raw_periods),
            annualize=annualize,
        )

    # ── Win / loss metrics ────────────────────────────────────────────────────

    def payoff_ratio(self) -> dict[str, float]:
        """Measure the payoff ratio.

        The payoff ratio is calculated as average win / abs(average loss).

        Returns:
            dict[str, float]: Mapping of asset name → payoff ratio.

        """
        return _lazy_columnwise(self.data.lazy, payoff_ratio_expr, self.data.assets)

    def win_loss_ratio(self) -> dict[str, float]:
        """Shorthand for payoff_ratio().

        Returns:
            dict[str, float]: Dictionary mapping asset names to win/loss ratios.

        """
        return self.payoff_ratio()

    def profit_ratio(self) -> dict[str, float]:
        """Measure the profit ratio.

        The profit ratio is calculated as win ratio / loss ratio.

        Returns:
            dict[str, float]: Mapping of asset name → profit ratio.

        """
        return _lazy_columnwise(self.data.lazy, profit_ratio_expr, self.data.assets)

    def profit_factor(self) -> dict[str, float]:
        """Measure the profit factor.

        The profit factor is calculated as wins / loss.

        Returns:
            dict[str, float]: Mapping of asset name → profit factor.

        """
        return _lazy_columnwise(self.data.lazy, profit_factor_expr, self.data.assets)

    # ── Risk metrics ──────────────────────────────────────────────────────────

    def value_at_risk(self, sigma: float = 1.0, alpha: float = 0.05) -> dict[str, float]:
        """Calculate the daily value-at-risk.

        Uses variance-covariance calculation with confidence level.

        Args:
            alpha (float, optional): Confidence level. Defaults to 0.05.
            sigma (float, optional): Standard deviation multiplier. Defaults to 1.0.

        Returns:
            dict[str, float]: Mapping of asset name → VaR.

        """
        return _lazy_columnwise(
            self.data.lazy,
            value_at_risk_expr,
            self.data.assets,
            sigma=sigma,
            alpha=alpha,
        )

    def conditional_value_at_risk(self, sigma: float = 1.0, alpha: float = 0.05) -> dict[str, float]:
        """Calculate the conditional value-at-risk.

        Also known as CVaR or expected shortfall, calculated for each numeric column.

        Args:
            alpha (float, optional): Confidence level. Defaults to 0.05.
            sigma (float, optional): Standard deviation multiplier. Defaults to 1.0.

        Returns:
            dict[str, float]: Mapping of asset name → CVaR.

        """
        return _lazy_columnwise(
            self.data.lazy,
            conditional_value_at_risk_expr,
            self.data.assets,
            sigma=sigma,
            alpha=alpha,
        )

    def win_rate(self) -> dict[str, float]:
        """Calculate the win ratio for a period.

        Returns:
            dict[str, float]: Mapping of asset name → win rate.

        """
        return _lazy_columnwise(self.data.lazy, win_rate_expr, self.data.assets)

    def gain_to_pain_ratio(self) -> dict[str, float]:
        """Calculate Jack Schwager's Gain-to-Pain Ratio.

        The ratio is calculated as total return / sum of losses (in absolute value).

        Returns:
            dict[str, float]: Mapping of asset name → gain-to-pain ratio.

        """
        return _lazy_columnwise(self.data.lazy, gain_to_pain_ratio_expr, self.data.assets)

    def risk_return_ratio(self) -> dict[str, float]:
        """Calculate the return/risk ratio.

        This is equivalent to the Sharpe ratio without a risk-free rate.

        Returns:
            dict[str, float]: Mapping of asset name → risk-return ratio.

        """
        return _lazy_columnwise(self.data.lazy, risk_return_ratio_expr, self.data.assets)

    def kelly_criterion(self) -> dict[str, float]:
        """Calculate the optimal capital allocation per column.

        Uses the Kelly Criterion formula: f* = [(b * p) - q] / b
        where:
          - b = payoff ratio
          - p = win rate
          - q = 1 - p.

        Returns:
            dict[str, float]: Dictionary mapping asset names to Kelly criterion values.

        """
        b = self.payoff_ratio()
        p = self.win_rate()

        return {col: ((b[col] * p[col]) - (1 - p[col])) / b[col] for col in b}

    def best(self) -> dict[str, float]:
        """Find the maximum return per column (best period).

        Returns:
            dict[str, float]: Mapping of asset name → best return.

        """
        return _lazy_columnwise(self.data.lazy, best_expr, self.data.assets)

    def worst(self) -> dict[str, float]:
        """Find the minimum return per column (worst period).

        Returns:
            dict[str, float]: Mapping of asset name → worst return.

        """
        return _lazy_columnwise(self.data.lazy, worst_expr, self.data.assets)

    def exposure(self) -> dict[str, float]:
        """Calculate the market exposure time (returns != 0).

        Returns:
            dict[str, float]: Mapping of asset name → exposure fraction.

        """
        return _lazy_columnwise(self.data.lazy, exposure_expr, self.data.assets)


# Keep the decorator import in the module namespace for any code that does
# ``from jquantstats._stats._basic import columnwise_stat``.
__all__ = ["_BasicStatsMixin", "columnwise_stat"]
