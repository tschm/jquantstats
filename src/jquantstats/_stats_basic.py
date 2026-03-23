"""Basic statistical metrics for financial returns data."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import numpy as np
import polars as pl
from scipy.stats import norm

from ._stats_core import columnwise_stat

# ── Basic statistics mixin ───────────────────────────────────────────────────


class _BasicStatsMixin:
    """Mixin providing basic return/risk and win/loss financial statistics.

    Covers: basic statistics (skew, kurtosis, avg return/win/loss), volatility,
    win/loss metrics (payoff ratio, profit factor), and risk metrics (VaR, CVaR,
    win rate, kelly criterion, best/worst, exposure).

    Attributes (provided by the concrete subclass):
        data: The :class:`~jquantstats._data.Data` object.
        all: Combined DataFrame for efficient column selection.
    """

    @staticmethod
    def _mean_positive_expr(series: pl.Series) -> float:
        """Return the mean of all positive values in *series*, or NaN if none exist."""
        return cast(float, series.filter(series > 0).mean())

    @staticmethod
    def _mean_negative_expr(series: pl.Series) -> float:
        """Return the mean of all negative values in *series*, or NaN if none exist."""
        return cast(float, series.filter(series < 0).mean())

    # ── Basic statistics ──────────────────────────────────────────────────────

    @columnwise_stat
    def skew(self, series: pl.Series) -> int | float | None:
        """Calculate skewness (asymmetry) for each numeric column.

        Args:
            series (pl.Series): The series to calculate skewness for.

        Returns:
            float: The skewness value.

        """
        return series.skew(bias=False)

    @columnwise_stat
    def kurtosis(self, series: pl.Series) -> int | float | None:
        """Calculate the kurtosis of returns.

        The degree to which a distribution peak compared to a normal distribution.

        Args:
            series (pl.Series): The series to calculate kurtosis for.

        Returns:
            float: The kurtosis value.

        """
        return series.kurtosis(bias=False)

    @columnwise_stat
    def avg_return(self, series: pl.Series) -> float:
        """Calculate average return per non-zero, non-null value.

        Args:
            series (pl.Series): The series to calculate average return for.

        Returns:
            float: The average return value.

        """
        return cast(float, series.filter(series.is_not_null() & (series != 0)).mean())

    @columnwise_stat
    def avg_win(self, series: pl.Series) -> float:
        """Calculate the average winning return/trade for an asset.

        Args:
            series (pl.Series): The series to calculate average win for.

        Returns:
            float: The average winning return.

        """
        return self._mean_positive_expr(series)

    @columnwise_stat
    def avg_loss(self, series: pl.Series) -> float:
        """Calculate the average loss return/trade for a period.

        Args:
            series (pl.Series): The series to calculate average loss for.

        Returns:
            float: The average loss return.

        """
        return self._mean_negative_expr(series)

    # ── Volatility & risk ─────────────────────────────────────────────────────

    @columnwise_stat
    def volatility(self, series: pl.Series, periods: int | float | None = None, annualize: bool = True) -> float:
        """Calculate the volatility of returns.

        - Std dev of returns
        - Annualized by sqrt(periods) if `annualize` is True.

        Args:
            series (pl.Series): The series to calculate volatility for.
            periods (int, optional): Number of periods per year. Defaults to 252.
            annualize (bool, optional): Whether to annualize the result. Defaults to True.

        Returns:
            float: The volatility value.

        """
        raw_periods = periods or self.data._periods_per_year

        # Ensure it's numeric
        if not isinstance(raw_periods, int | float):
            raise TypeError(f"Expected int or float for periods, got {type(raw_periods).__name__}")  # noqa: TRY003

        factor = float(np.sqrt(raw_periods)) if annualize else 1.0
        std_val = cast(float, series.std())
        return (std_val if std_val is not None else 0.0) * factor

    # ── Win / loss metrics ────────────────────────────────────────────────────

    @columnwise_stat
    def payoff_ratio(self, series: pl.Series) -> float:
        """Measure the payoff ratio.

        The payoff ratio is calculated as average win / abs(average loss).

        Args:
            series (pl.Series): The series to calculate payoff ratio for.

        Returns:
            float: The payoff ratio value.

        """
        avg_win = cast(float, series.filter(series > 0).mean())
        avg_loss = float(np.abs(cast(float, series.filter(series < 0).mean())))
        return avg_win / avg_loss

    def win_loss_ratio(self) -> dict[str, float]:
        """Shorthand for payoff_ratio().

        Returns:
            dict[str, float]: Dictionary mapping asset names to win/loss ratios.

        """
        return self.payoff_ratio()

    @columnwise_stat
    def profit_ratio(self, series: pl.Series) -> float:
        """Measure the profit ratio.

        The profit ratio is calculated as win ratio / loss ratio.

        Args:
            series (pl.Series): The series to calculate profit ratio for.

        Returns:
            float: The profit ratio value.

        """
        wins = series.filter(series >= 0)
        losses = series.filter(series < 0)

        try:
            win_mean = cast(float, wins.mean())
            loss_mean = cast(float, losses.mean())
            win_ratio = float(np.abs(win_mean / wins.count()))
            loss_ratio = float(np.abs(loss_mean / losses.count()))

            return win_ratio / loss_ratio

        except TypeError:
            return float(np.nan)

    @columnwise_stat
    def profit_factor(self, series: pl.Series) -> float:
        """Measure the profit factor.

        The profit factor is calculated as wins / loss.

        Args:
            series (pl.Series): The series to calculate profit factor for.

        Returns:
            float: The profit factor value.

        """
        wins = series.filter(series > 0)
        losses = series.filter(series < 0)
        wins_sum = wins.sum()
        losses_sum = losses.sum()

        return float(np.abs(wins_sum / losses_sum))

    # ── Risk metrics ──────────────────────────────────────────────────────────

    @columnwise_stat
    def value_at_risk(self, series: pl.Series, sigma: float = 1.0, alpha: float = 0.05) -> float:
        """Calculate the daily value-at-risk.

        Uses variance-covariance calculation with confidence level.

        Args:
            series (pl.Series): The series to calculate value at risk for.
            alpha (float, optional): Confidence level. Defaults to 0.05.
            sigma (float, optional): Standard deviation multiplier. Defaults to 1.0.

        Returns:
            float: The value at risk.

        """
        mean_val = cast(float, series.mean())
        std_val = cast(float, series.std())
        mu = mean_val if mean_val is not None else 0.0
        sigma *= std_val if std_val is not None else 0.0

        return float(norm.ppf(alpha, mu, sigma))

    @columnwise_stat
    def conditional_value_at_risk(self, series: pl.Series, sigma: float = 1.0, alpha: float = 0.05) -> float:
        """Calculate the conditional value-at-risk.

        Also known as CVaR or expected shortfall, calculated for each numeric column.

        Args:
            series (pl.Series): The series to calculate conditional value at risk for.
            alpha (float, optional): Confidence level. Defaults to 0.05.
            sigma (float, optional): Standard deviation multiplier. Defaults to 1.0.

        Returns:
            float: The conditional value at risk.

        """
        mean_val = cast(float, series.mean())
        std_val = cast(float, series.std())
        mu = mean_val if mean_val is not None else 0.0
        sigma *= std_val if std_val is not None else 0.0

        var = norm.ppf(alpha, mu, sigma)

        # Compute mean of returns less than or equal to VaR
        # Cast to Any or pl.Series to suppress Ty error
        # Cast the mask to pl.Expr to satisfy type checker
        mask = cast(Iterable[bool], series < var)
        return cast(float, series.filter(mask).mean())

    @columnwise_stat
    def win_rate(self, series: pl.Series) -> float:
        """Calculate the win ratio for a period.

        Args:
            series (pl.Series): The series to calculate win rate for.

        Returns:
            float: The win rate value.

        """
        num_pos = series.filter(series > 0).count()
        num_nonzero = series.filter(series != 0).count()
        return float(num_pos / num_nonzero)

    @columnwise_stat
    def gain_to_pain_ratio(self, series: pl.Series) -> float:
        """Calculate Jack Schwager's Gain-to-Pain Ratio.

        The ratio is calculated as total return / sum of losses (in absolute value).

        Args:
            series (pl.Series): The series to calculate gain to pain ratio for.

        Returns:
            float: The gain to pain ratio value.

        """
        total_gain = series.sum()
        total_pain = series.filter(series < 0).abs().sum()
        try:
            return float(total_gain / total_pain)
        except ZeroDivisionError:
            return float(np.nan)

    @columnwise_stat
    def risk_return_ratio(self, series: pl.Series) -> float:
        """Calculate the return/risk ratio.

        This is equivalent to the Sharpe ratio without a risk-free rate.

        Args:
            series (pl.Series): The series to calculate risk return ratio for.

        Returns:
            float: The risk return ratio value.

        """
        mean_val = cast(float, series.mean())
        std_val = cast(float, series.std())
        return (mean_val if mean_val is not None else 0.0) / (std_val if std_val is not None else 1.0)

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

    @columnwise_stat
    def best(self, series: pl.Series) -> float | None:
        """Find the maximum return per column (best period).

        Args:
            series (pl.Series): The series to find the best return for.

        Returns:
            float: The maximum return value.

        """
        val = cast(float, series.max())
        return val if val is not None else None

    @columnwise_stat
    def worst(self, series: pl.Series) -> float | None:
        """Find the minimum return per column (worst period).

        Args:
            series (pl.Series): The series to find the worst return for.

        Returns:
            float: The minimum return value.

        """
        val = cast(float, series.min())
        return val if val is not None else None

    @columnwise_stat
    def exposure(self, series: pl.Series) -> float:
        """Calculate the market exposure time (returns != 0).

        Args:
            series (pl.Series): The series to calculate exposure for.

        Returns:
            float: The exposure value.

        """
        all_data = cast(pl.DataFrame, self.all)
        return float(np.round((series.filter(series != 0).count() / all_data.height), decimals=2))
