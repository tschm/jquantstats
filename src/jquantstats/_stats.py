"""Statistical analysis tools for financial returns data."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable
from datetime import timedelta
from functools import wraps
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import polars as pl
from scipy.stats import norm

if TYPE_CHECKING:
    from ._data import Data


# ── Module helpers ────────────────────────────────────────────────────────────


def _drawdown_series(series: pl.Series) -> pl.Series:
    """Compute the drawdown percentage series from a returns series.

    Treats ``series`` as additive daily returns and builds a normalised NAV
    starting at 1.0.  The high-water mark is the running maximum of that NAV;
    drawdown is expressed as the fraction below the high-water mark.

    Args:
        series: A Polars Series of additive returns (profit / AUM).

    Returns:
        A Polars Float64 Series whose values are in [0, 1].  A value of 0
        means the NAV is at its all-time high; a value of 0.2 means the NAV
        is 20 % below its previous peak.

    Examples:
        >>> import polars as pl
        >>> s = pl.Series([0.0, -0.1, 0.2])
        >>> [round(x, 10) for x in _drawdown_series(s).to_list()]
        [0.0, 0.1, 0.0]
    """
    nav = 1.0 + series.cast(pl.Float64).cum_sum()
    hwm = nav.cum_max()
    hwm_safe = hwm.clip(lower_bound=1e-10)
    return ((hwm - nav) / hwm_safe).clip(lower_bound=0.0)


def _to_float(value: object) -> float:
    """Safely convert a Polars aggregation result to float.

    Examples:
        >>> _to_float(2.0)
        2.0
        >>> _to_float(None)
        0.0
    """
    if value is None:
        return 0.0
    if isinstance(value, timedelta):
        return value.total_seconds()
    return float(cast(float, value))


@dataclasses.dataclass(frozen=True)
class Stats:
    """Statistical analysis tools for financial returns data.

    This class provides a comprehensive set of methods for calculating various
    financial metrics and statistics on returns data, including:

    - Basic statistics (mean, skew, kurtosis)
    - Risk metrics (volatility, value-at-risk, drawdown)
    - Performance ratios (Sharpe, Sortino, information ratio)
    - Win/loss metrics (win rate, profit factor, payoff ratio)
    - Rolling calculations (rolling volatility, rolling Sharpe)
    - Factor analysis (alpha, beta, R-squared)

    The class is designed to work with the _Data class and operates on Polars DataFrames
    for efficient computation.

    Attributes:
        data: The _Data object containing returns and benchmark data.
        all: A DataFrame combining all data (index, returns, benchmark) for easy access.

    """

    data: Data
    all: pl.DataFrame | None = None  # Default is None; will be set in __post_init__

    def __post_init__(self) -> None:
        object.__setattr__(self, "all", self.data.all)

    def __repr__(self) -> str:
        """Return a string representation of the Stats object."""
        return f"Stats(assets={self.data.assets})"

    @staticmethod
    def _mean_positive_expr(series: pl.Series) -> float:
        """Return the mean of all positive values in *series*, or NaN if none exist."""
        return cast(float, series.filter(series > 0).mean())

    @staticmethod
    def _mean_negative_expr(series: pl.Series) -> float:
        """Return the mean of all negative values in *series*, or NaN if none exist."""
        return cast(float, series.filter(series < 0).mean())

    # ── Decorators ────────────────────────────────────────────────────────────

    @staticmethod
    def columnwise_stat(func: Callable[..., Any]) -> Callable[..., dict[str, float]]:
        """Apply a column-wise statistical function to all numeric columns.

        Args:
            func (Callable): The function to decorate.

        Returns:
            Callable: The decorated function.

        """

        @wraps(func)
        def wrapper(self: Stats, *args: Any, **kwargs: Any) -> dict[str, float]:
            """Apply *func* to every column and return a ``{column: value}`` mapping."""
            return {col: func(self, series, *args, **kwargs) for col, series in self.data.items()}

        return wrapper

    @staticmethod
    def to_frame(func: Callable[..., Any]) -> Callable[..., pl.DataFrame]:
        """Apply per-column expressions and evaluates with .with_columns(...).

        Args:
            func (Callable): The function to decorate.

        Returns:
            Callable: The decorated function.

        """

        @wraps(func)
        def wrapper(self: Stats, *args: Any, **kwargs: Any) -> pl.DataFrame:
            """Apply *func* per column and return the result as a Polars DataFrame."""
            return cast(pl.DataFrame, self.all).select(
                [pl.col(name) for name in self.data.date_col]
                + [func(self, series, *args, **kwargs).alias(col) for col, series in self.data.items()]
            )

        return wrapper

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

        # periods = periods or self.data._periods_per_year
        # factor = np.sqrt(periods) if annualize else 1
        # return series.std() * factor

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

        # filtered_series = cast(pl.Series, series.filter(series < var))
        # return filtered_series.mean()

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

        return {
            col: ((b[col] * p[col]) - (1 - p[col])) / b[col]
            # if b[col] not in (None, 0) and p[col] is not None else None
            for col in b
        }

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

    # ── Sharpe & Sortino ──────────────────────────────────────────────────────

    @columnwise_stat
    def sharpe(self, series: pl.Series, periods: int | float | None = None) -> float:
        """Calculate the Sharpe ratio of asset returns.

        Args:
            series (pl.Series): The series to calculate Sharpe ratio for.
            periods (int, optional): Number of periods per year. Defaults to 252.

        Returns:
            float: The Sharpe ratio value.

        """
        periods = periods or self.data._periods_per_year

        std_val = cast(float, series.std(ddof=1))
        mean_val = cast(float, series.mean())
        divisor = std_val if std_val is not None else 0.0
        mean_f = mean_val if mean_val is not None else 0.0

        _eps = np.finfo(np.float64).eps
        if divisor <= _eps * max(abs(mean_f), _eps) * 10:
            return float("nan")

        res = mean_f / divisor
        factor = periods or 1
        return float(res * np.sqrt(factor))

    @columnwise_stat
    def sharpe_variance(self, series: pl.Series, periods: int | float | None = None) -> float:
        r"""Calculate the asymptotic variance of the Sharpe Ratio.

        .. math::
            \text{Var}(SR) = \frac{1 + \frac{S \cdot SR}{2} + \frac{(K - 3) \cdot SR^2}{4}}{T}

        where:
            - \(S\) is the skewness of returns
            - \(K\) is the kurtosis of returns
            - \(SR\) is the Sharpe ratio (unannualized)
            - \(T\) is the number of observations

        Args:
            series (pl.Series): The series to calculate Sharpe ratio variance for.
            periods (int | float, optional): Number of periods per year. Defaults to data periods.

        Returns:
            float: The asymptotic variance of the Sharpe ratio.
            If number of periods per year is provided or inferred from the data, the result is annualized.

        """
        t = series.count()
        mean_val = cast(float, series.mean())
        std_val = cast(float, series.std(ddof=1))
        if mean_val is None or std_val is None or std_val == 0:
            return float(np.nan)
        sr = mean_val / std_val

        skew_val = series.skew(bias=False)
        kurt_val = series.kurtosis(bias=False)

        if skew_val is None or kurt_val is None:
            return float(np.nan)
        # Base variance calculation using unannualized Sharpe ratio
        # Formula: (1 + skew*SR/2 + (kurt-3)*SR²/4) / T
        base_variance = (1 + (float(skew_val) * sr) / 2 + ((float(kurt_val) - 3) / 4) * sr**2) / t
        # Annualize by scaling with the number of periods
        periods = periods or self.data._periods_per_year
        factor = periods or 1
        return float(base_variance * factor)

    @columnwise_stat
    def prob_sharpe_ratio(self, series: pl.Series, benchmark_sr: float) -> float:
        r"""Calculate the probabilistic sharpe ratio (PSR).

        Args:
            series (pl.Series): The series to calculate probabilistic Sharpe ratio for.
            benchmark_sr (float): The target Sharpe ratio to compare against. This should be unannualized.

        Returns:
            float: Probabilistic Sharpe Ratio.

        Note:
            PSR is the probability that the observed Sharpe ratio is greater than a
            given benchmark Sharpe ratio.

        """
        t = series.count()

        # Calculate observed unannualized Sharpe ratio
        mean_val = cast(float, series.mean())
        std_val = cast(float, series.std(ddof=1))
        if mean_val is None or std_val is None or std_val == 0:
            return float(np.nan)
        # Unannualized observed Sharpe ratio
        observed_sr = mean_val / std_val

        skew_val = series.skew(bias=False)
        kurt_val = series.kurtosis(bias=False)

        if skew_val is None or kurt_val is None:
            return float(np.nan)

        # Calculate variance using unannualized benchmark Sharpe ratio
        var_bench_sr = (1 + (float(skew_val) * benchmark_sr) / 2 + ((float(kurt_val) - 3) / 4) * benchmark_sr**2) / t

        if var_bench_sr <= 0:
            return float(np.nan)
        return float(norm.cdf((observed_sr - benchmark_sr) / np.sqrt(var_bench_sr)))

    @columnwise_stat
    def hhi_positive(self, series: pl.Series) -> float:
        r"""Calculate the Herfindahl-Hirschman Index (HHI) for positive returns.

        This quantifies how concentrated the positive returns are in a series.

        .. math::
            w^{\plus} = \frac{r_{t}^{\plus}}{\sum{r_{t}^{\plus}}} \\
            HHI^{\plus} = \frac{N_{\plus} \sum{(w^{\plus})^2} - 1}{N_{\plus} - 1}

        where:
            - \(r_{t}^{\plus}\) are the positive returns
            - \(N_{\plus}\) is the number of positive returns
            - \(w^{\plus}\) are the weights of positive returns

        Args:
            series (pl.Series): The series to calculate HHI for.

        Returns:
            float: The HHI value for positive returns. Returns NaN if fewer than 3
                positive returns are present.

        Note:
            Values range from 0 (perfectly diversified gains) to 1 (all gains
            concentrated in a single period).
        """
        positive_returns = series.filter(series > 0).drop_nans()
        if positive_returns.len() <= 2:
            return float(np.nan)
        weight = positive_returns / positive_returns.sum()
        return float((weight.len() * (weight**2).sum() - 1) / (weight.len() - 1))

    @columnwise_stat
    def hhi_negative(self, series: pl.Series) -> float:
        r"""Calculate the Herfindahl-Hirschman Index (HHI) for negative returns.

        This quantifies how concentrated the negative returns are in a series.

        .. math::
            w^{\minus} = \frac{r_{t}^{\minus}}{\sum{r_{t}^{\minus}}} \\
            HHI^{\minus} = \frac{N_{\minus} \sum{(w^{\minus})^2} - 1}{N_{\minus} - 1}

        where:
            - \(r_{t}^{\minus}\) are the negative returns
            - \(N_{\minus}\) is the number of negative returns
            - \(w^{\minus}\) are the weights of negative returns

        Args:
            series (pl.Series): The returns series to calculate HHI for.

        Returns:
            float: The HHI value for negative returns. Returns NaN if fewer than 3
                negative returns are present.

        Note:
            Values range from 0 (perfectly diversified losses) to 1 (all losses
            concentrated in a single period).
        """
        negative_returns = series.filter(series < 0).drop_nans()
        if negative_returns.len() <= 2:
            return float(np.nan)
        weight = negative_returns / negative_returns.sum()
        return float((weight.len() * (weight**2).sum() - 1) / (weight.len() - 1))

    @columnwise_stat
    def sortino(self, series: pl.Series, periods: int | float | None = None) -> float:
        """Calculate the Sortino ratio.

        The Sortino ratio is the mean return divided by downside deviation.
        Based on Red Rock Capital's Sortino ratio paper.

        Args:
            series (pl.Series): The series to calculate Sortino ratio for.
            periods (int, optional): Number of periods per year. Defaults to 252.

        Returns:
            float: The Sortino ratio value.

        """
        periods = periods or self.data._periods_per_year
        downside_sum = ((series.filter(series < 0)) ** 2).sum()
        downside_deviation = float(np.sqrt(downside_sum / series.count()))
        mean_val = cast(float, series.mean())
        mean_f = mean_val if mean_val is not None else 0.0
        if downside_deviation == 0.0:
            if mean_f > 0:
                return float("inf")
            elif mean_f < 0:
                return float("-inf")
            else:
                return float("nan")
        ratio = mean_f / downside_deviation
        return float(ratio * np.sqrt(periods))

    # ── Rolling windows ───────────────────────────────────────────────────────

    @to_frame
    def rolling_sortino(
        self, series: pl.Expr, rolling_period: int = 126, periods_per_year: int | float | None = None
    ) -> pl.Expr:
        """Calculate the rolling Sortino ratio.

        Args:
            series (pl.Expr): The expression to calculate rolling Sortino ratio for.
            rolling_period (int, optional): The rolling window size. Defaults to 126.
            periods_per_year (int, optional): Number of periods per year. Defaults to 252.

        Returns:
            pl.Expr: The rolling Sortino ratio expression.

        """
        ppy = periods_per_year or self.data._periods_per_year

        mean_ret = series.rolling_mean(window_size=rolling_period)

        # Rolling downside deviation (squared negative returns averaged over window)
        downside = series.map_elements(lambda x: x**2 if x < 0 else 0.0, return_dtype=pl.Float64).rolling_mean(
            window_size=rolling_period
        )

        # Avoid division by zero
        sortino = mean_ret / downside.sqrt().fill_nan(0).fill_null(0)
        return cast(pl.Expr, sortino * (ppy**0.5))

    def rolling_sharpe(
        self,
        window: int | None = None,
        periods: int | float | None = None,
        rolling_period: int | None = None,
        periods_per_year: int | float | None = None,
    ) -> pl.DataFrame:
        """Calculate the rolling Sharpe ratio.

        Accepts both the analytics-style (``window``, ``periods``) and the
        legacy-style (``rolling_period``, ``periods_per_year``) keyword
        arguments so that callers using either convention continue to work.

        Args:
            window: Rolling window size (analytics style). Defaults to 126.
            periods: Periods per year for annualisation (analytics style).
            rolling_period: Alias for ``window`` (legacy style).
            periods_per_year: Alias for ``periods`` (legacy style).

        Returns:
            pl.DataFrame: Date column(s) plus one annualised rolling Sharpe
            column per asset.

        Raises:
            ValueError: If the effective window size is not a positive integer.

        """
        actual_window = window if window is not None else (rolling_period if rolling_period is not None else 126)
        actual_periods = periods or periods_per_year or self.data._periods_per_year
        if not isinstance(actual_window, int) or actual_window <= 0:
            raise ValueError("window must be a positive integer")  # noqa: TRY003
        scale = float(np.sqrt(actual_periods))
        return cast(pl.DataFrame, self.all).select(
            [pl.col(name) for name in self.data.date_col]
            + [
                (
                    pl.col(col).rolling_mean(window_size=actual_window)
                    / pl.col(col).rolling_std(window_size=actual_window)
                    * scale
                ).alias(col)
                for col, _ in self.data.items()
            ]
        )

    def rolling_volatility(
        self,
        window: int | None = None,
        periods: int | float | None = None,
        annualize: bool = True,
        rolling_period: int | None = None,
        periods_per_year: int | float | None = None,
    ) -> pl.DataFrame:
        """Calculate the rolling volatility of returns.

        Accepts both the analytics-style (``window``, ``periods``,
        ``annualize``) and the legacy-style (``rolling_period``,
        ``periods_per_year``) keyword arguments.

        Args:
            window: Rolling window size (analytics style). Defaults to 126.
            periods: Periods per year for annualisation (analytics style).
            annualize: Multiply by ``sqrt(periods)`` when True (default).
            rolling_period: Alias for ``window`` (legacy style).
            periods_per_year: Alias for ``periods`` (legacy style).

        Returns:
            pl.DataFrame: Date column(s) plus one rolling volatility column
            per asset.

        Raises:
            ValueError: If the effective window size is not a positive integer.
            TypeError: If the effective periods value is not numeric.

        """
        actual_window = window if window is not None else (rolling_period if rolling_period is not None else 126)
        actual_periods = periods or periods_per_year or self.data._periods_per_year
        if not isinstance(actual_window, int) or actual_window <= 0:
            raise ValueError("window must be a positive integer")  # noqa: TRY003
        if not isinstance(actual_periods, int | float):
            raise TypeError
        factor = float(np.sqrt(actual_periods)) if annualize else 1.0
        return cast(pl.DataFrame, self.all).select(
            [pl.col(name) for name in self.data.date_col]
            + [(pl.col(col).rolling_std(window_size=actual_window) * factor).alias(col) for col, _ in self.data.items()]
        )

    # ── Drawdown ──────────────────────────────────────────────────────────────

    @to_frame
    def drawdown(self, series: pl.Series) -> pl.Series:
        """Calculate the drawdown series for returns.

        Args:
            series (pl.Series): The series to calculate drawdown for.

        Returns:
            pl.Series: The drawdown series.

        """
        equity = self.prices(series)
        d = (equity / equity.cum_max()) - 1
        return -d

    @staticmethod
    def prices(series: pl.Series) -> pl.Series:
        """Convert returns series to price series.

        Args:
            series (pl.Series): The returns series to convert.

        Returns:
            pl.Series: The price series.

        """
        return (1.0 + series).cum_prod()

    @staticmethod
    def max_drawdown_single_series(series: pl.Series) -> float:
        """Compute the maximum drawdown for a single returns series.

        Args:
            series: A Polars Series of returns values.

        Returns:
            The maximum drawdown as a positive float (e.g. 0.25 means 25% drawdown).

        """
        price = Stats.prices(series)
        peak = price.cum_max()
        drawdown = price / peak - 1
        dd_min = cast(float, drawdown.min())
        return -dd_min if dd_min is not None else 0.0

    @columnwise_stat
    def max_drawdown(self, series: pl.Series) -> float:
        """Calculate the maximum drawdown for each column.

        Args:
            series (pl.Series): The series to calculate maximum drawdown for.

        Returns:
            float: The maximum drawdown value.

        """
        return Stats.max_drawdown_single_series(series)

    def adjusted_sortino(self, periods: int | float | None = None) -> dict[str, float]:
        """Calculate Jack Schwager's adjusted Sortino ratio.

        This adjustment allows for direct comparison to Sharpe ratio.
        See: https://archive.is/wip/2rwFW.

        Args:
            periods (int, optional): Number of periods per year. Defaults to 252.

        Returns:
            dict[str, float]: Dictionary mapping asset names to adjusted Sortino ratios.

        """
        sortino_data = self.sortino(periods=periods)
        return {k: v / np.sqrt(2) for k, v in sortino_data.items()}

    # ── Benchmark & factor ────────────────────────────────────────────────────

    @columnwise_stat
    def r_squared(self, series: pl.Series, benchmark: str | None = None) -> float:
        """Measure the straight line fit of the equity curve.

        Args:
            series (pl.Series): The series to calculate R-squared for.
            benchmark (str, optional): The benchmark column name. Defaults to None.

        Returns:
            float: The R-squared value.

        Raises:
            AttributeError: If no benchmark data is available.

        """
        if self.data.benchmark is None:
            raise AttributeError("No benchmark data available")  # noqa: TRY003

        benchmark_col = benchmark or self.data.benchmark.columns[0]

        # Evaluate both series and benchmark as Series
        all_data = cast(pl.DataFrame, self.all)
        dframe = all_data.select([series, pl.col(benchmark_col).alias("benchmark")])

        # Drop nulls
        dframe = dframe.drop_nulls()

        matrix = dframe.to_numpy()
        # Get actual Series

        strategy_np = matrix[:, 0]
        benchmark_np = matrix[:, 1]

        corr_matrix = np.corrcoef(strategy_np, benchmark_np)
        r = corr_matrix[0, 1]
        return float(r**2)

    def r2(self) -> dict[str, float]:
        """Shorthand for r_squared().

        Returns:
            dict[str, float]: Dictionary mapping asset names to R-squared values.

        """
        return self.r_squared()

    @columnwise_stat
    def information_ratio(
        self, series: pl.Series, periods_per_year: int | float | None = None, benchmark: str | None = None
    ) -> float:
        """Calculate the information ratio.

        This is essentially the risk return ratio of the net profits.

        Args:
            series (pl.Series): The series to calculate information ratio for.
            periods_per_year (int, optional): Number of periods per year. Defaults to 252.
            benchmark (str, optional): The benchmark column name. Defaults to None.

        Returns:
            float: The information ratio value.

        """
        ppy = periods_per_year or self.data._periods_per_year

        benchmark_data = cast(pl.DataFrame, self.data.benchmark)
        benchmark_col = benchmark or benchmark_data.columns[0]

        active = series - benchmark_data[benchmark_col]

        mean_val = cast(float, active.mean())
        std_val = cast(float, active.std())

        try:
            mean_f = mean_val if mean_val is not None else 0.0
            std_f = std_val if std_val is not None else 1.0
            return float((mean_f / std_f) * (ppy**0.5))
        except ZeroDivisionError:
            return 0.0

    @columnwise_stat
    def greeks(
        self, series: pl.Series, periods_per_year: int | float | None = None, benchmark: str | None = None
    ) -> dict[str, float]:
        """Calculate alpha and beta of the portfolio.

        Args:
            series (pl.Series): The series to calculate greeks for.
            periods_per_year (int, optional): Number of periods per year. Defaults to 252.
            benchmark (str, optional): The benchmark column name. Defaults to None.

        Returns:
            dict[str, float]: Dictionary containing alpha and beta values.

        """
        ppy = periods_per_year or self.data._periods_per_year

        benchmark_data = cast(pl.DataFrame, self.data.benchmark)
        benchmark_col = benchmark or benchmark_data.columns[0]

        # Evaluate both series and benchmark as Series
        all_data = cast(pl.DataFrame, self.all)
        dframe = all_data.select([series, pl.col(benchmark_col).alias("benchmark")])

        # Drop nulls
        dframe = dframe.drop_nulls()
        matrix = dframe.to_numpy()

        # Get actual Series
        strategy_np = matrix[:, 0]
        benchmark_np = matrix[:, 1]

        # 2x2 covariance matrix: [[var_strategy, cov], [cov, var_benchmark]]
        cov_matrix = np.cov(strategy_np, benchmark_np)

        cov = cov_matrix[0, 1]
        var_benchmark = cov_matrix[1, 1]

        beta = float(cov / var_benchmark) if var_benchmark != 0 else float("nan")
        alpha = float(np.mean(strategy_np) - beta * np.mean(benchmark_np))

        return {"alpha": float(alpha * ppy), "beta": beta}

    # ── Temporal & reporting ──────────────────────────────────────────────────

    @property
    def periods_per_year(self) -> float:
        """Estimate the number of periods per year from the data index spacing.

        Returns:
            float: Estimated number of observations per calendar year.
        """
        return self.data._periods_per_year

    @columnwise_stat
    def avg_drawdown(self, series: pl.Series) -> float:
        """Average drawdown across all underwater periods.

        Returns 0.0 when there are no underwater periods.

        Args:
            series (pl.Series): Series of additive daily returns.

        Returns:
            float: Mean drawdown in [0, 1].
        """
        dd = _drawdown_series(series)
        in_dd = dd.filter(dd > 0)
        if in_dd.is_empty():
            return 0.0
        return _to_float(in_dd.mean())

    @columnwise_stat
    def calmar(self, series: pl.Series, periods: int | float | None = None) -> float:
        """Calmar ratio (annualised return divided by maximum drawdown).

        Returns ``nan`` when the maximum drawdown is zero.

        Args:
            series (pl.Series): Series of additive daily returns.
            periods: Annualisation factor. Defaults to ``periods_per_year``.

        Returns:
            float: Calmar ratio, or ``nan`` if max drawdown is zero.
        """
        raw_periods = periods or self.data._periods_per_year
        max_dd = _to_float(_drawdown_series(series).max())
        if max_dd <= 0:
            return float("nan")
        ann_return = _to_float(series.mean()) * raw_periods
        return ann_return / max_dd

    @columnwise_stat
    def recovery_factor(self, series: pl.Series) -> float:
        """Recovery factor (total return divided by maximum drawdown).

        Returns ``nan`` when the maximum drawdown is zero.

        Args:
            series (pl.Series): Series of additive daily returns.

        Returns:
            float: Recovery factor, or ``nan`` if max drawdown is zero.
        """
        max_dd = _to_float(_drawdown_series(series).max())
        if max_dd <= 0:
            return float("nan")
        total_return = _to_float(series.sum())
        return total_return / max_dd

    def max_drawdown_duration(self) -> dict[str, float | int | None]:
        """Maximum drawdown duration in calendar days (or periods) per asset.

        When the index is a temporal column (``Date`` / ``Datetime``) the
        duration is expressed as calendar days spanned by the longest
        underwater run.  For integer-indexed data each row counts as one
        period.

        Returns:
            dict[str, float | int | None]: Asset → max drawdown duration.
            Returns 0 when there are no underwater periods.
        """
        all_df = cast(pl.DataFrame, self.all)
        date_col_name = self.data.date_col[0] if self.data.date_col else None
        has_date = date_col_name is not None and all_df[date_col_name].dtype.is_temporal()
        result: dict[str, float | int | None] = {}
        for col, series in self.data.items():
            nav = 1.0 + series.cast(pl.Float64).cum_sum()
            hwm = nav.cum_max()
            in_dd = nav < hwm

            if not in_dd.any():
                result[col] = 0
                continue

            if has_date and date_col_name is not None:
                frame = pl.DataFrame({"date": all_df[date_col_name], "in_dd": in_dd})
            else:
                frame = pl.DataFrame({"date": pl.Series(list(range(len(series))), dtype=pl.Int64), "in_dd": in_dd})

            frame = frame.with_columns(pl.col("in_dd").rle_id().alias("run_id"))
            dd_runs = (
                frame.filter(pl.col("in_dd"))
                .group_by("run_id")
                .agg([pl.col("date").min().alias("start"), pl.col("date").max().alias("end")])
            )

            if has_date:
                dd_runs = dd_runs.with_columns(
                    ((pl.col("end") - pl.col("start")).dt.total_days() + 1).alias("duration")
                )
            else:
                dd_runs = dd_runs.with_columns((pl.col("end") - pl.col("start") + 1).alias("duration"))

            result[col] = int(_to_float(dd_runs["duration"].max()))
        return result

    def monthly_win_rate(self) -> dict[str, float]:
        """Fraction of calendar months with a positive compounded return per asset.

        Requires a temporal (Date / Datetime) index.  Returns ``nan`` per
        asset when no temporal index is present.

        Returns:
            dict[str, float]: Monthly win rate in [0, 1] per asset.
        """
        all_df = cast(pl.DataFrame, self.all)
        date_col_name = self.data.date_col[0] if self.data.date_col else None
        if date_col_name is None or not all_df[date_col_name].dtype.is_temporal():
            return {col: float("nan") for col, _ in self.data.items()}

        result: dict[str, float] = {}
        for col, _ in self.data.items():
            df = (
                all_df.select([date_col_name, col])
                .drop_nulls()
                .with_columns(
                    [
                        pl.col(date_col_name).dt.year().alias("_year"),
                        pl.col(date_col_name).dt.month().alias("_month"),
                    ]
                )
            )
            monthly = (
                df.group_by(["_year", "_month"])
                .agg((pl.col(col) + 1.0).product().alias("gross"))
                .with_columns((pl.col("gross") - 1.0).alias("monthly_return"))
            )
            n_total = len(monthly)
            if n_total == 0:
                result[col] = float("nan")
            else:
                n_positive = int((monthly["monthly_return"] > 0).sum())
                result[col] = n_positive / n_total
        return result

    def worst_n_periods(self, n: int = 5) -> dict[str, list[float | None]]:
        """Return the N worst return periods per asset.

        If a series has fewer than ``n`` non-null observations the list is
        padded with ``None`` on the right.

        Args:
            n: Number of worst periods to return. Defaults to 5.

        Returns:
            dict[str, list[float | None]]: Sorted worst returns per asset.
        """
        result: dict[str, list[float | None]] = {}
        for col, series in self.data.items():
            nonnull = series.drop_nulls()
            worst: list[float | None] = nonnull.sort(descending=False).head(n).to_list()
            while len(worst) < n:
                worst.append(None)
            result[col] = worst
        return result

    # ── Capture ratios ────────────────────────────────────────────────────────

    def up_capture(self, benchmark: pl.Series) -> dict[str, float]:
        """Up-market capture ratio relative to an explicit benchmark series.

        Measures the fraction of the benchmark's upside that the strategy
        captures.  A value greater than 1.0 means the strategy outperformed
        the benchmark in rising markets.

        Args:
            benchmark: Benchmark return series aligned row-by-row with the data.

        Returns:
            dict[str, float]: Up capture ratio per asset.
        """
        up_mask = benchmark > 0
        bench_up = benchmark.filter(up_mask).drop_nulls()
        if bench_up.is_empty():
            return {col: float("nan") for col, _ in self.data.items()}
        bench_geom = float((bench_up + 1.0).product()) ** (1.0 / len(bench_up)) - 1.0
        if bench_geom == 0.0:  # pragma: no cover
            return {col: float("nan") for col, _ in self.data.items()}
        result: dict[str, float] = {}
        for col, series in self.data.items():
            strat_up = series.filter(up_mask).drop_nulls()
            if strat_up.is_empty():
                result[col] = float("nan")
            else:
                strat_geom = float((strat_up + 1.0).product()) ** (1.0 / len(strat_up)) - 1.0
                result[col] = strat_geom / bench_geom
        return result

    def down_capture(self, benchmark: pl.Series) -> dict[str, float]:
        """Down-market capture ratio relative to an explicit benchmark series.

        A value less than 1.0 means the strategy lost less than the benchmark
        in falling markets (a desirable property).

        Args:
            benchmark: Benchmark return series aligned row-by-row with the data.

        Returns:
            dict[str, float]: Down capture ratio per asset.
        """
        down_mask = benchmark < 0
        bench_down = benchmark.filter(down_mask).drop_nulls()
        if bench_down.is_empty():
            return {col: float("nan") for col, _ in self.data.items()}
        bench_geom = float((bench_down + 1.0).product()) ** (1.0 / len(bench_down)) - 1.0
        if bench_geom == 0.0:  # pragma: no cover
            return {col: float("nan") for col, _ in self.data.items()}
        result: dict[str, float] = {}
        for col, series in self.data.items():
            strat_down = series.filter(down_mask).drop_nulls()
            if strat_down.is_empty():
                result[col] = float("nan")
            else:
                strat_geom = float((strat_down + 1.0).product()) ** (1.0 / len(strat_down)) - 1.0
                result[col] = strat_geom / bench_geom
        return result

    # ── Summary & breakdown ────────────────────────────────────────────────────

    def annual_breakdown(self) -> pl.DataFrame:
        """Summary statistics broken down by calendar year.

        Groups the data by calendar year using the date index, computes a
        full :py:meth:`summary` for each year, and stacks the results with an
        additional ``year`` column.

        Returns:
            pl.DataFrame: Columns ``year``, ``metric``, one per asset, sorted
            by ``year``.

        Raises:
            ValueError: If the data has no date index.
        """
        all_df = cast(pl.DataFrame, self.all)
        date_col_name = self.data.date_col[0] if self.data.date_col else None
        has_temporal = date_col_name is not None and all_df[date_col_name].dtype.is_temporal()

        from ._data import Data

        if not has_temporal:
            # Integer-index fallback: group by chunks of ~_periods_per_year rows
            chunk = round(self.data._periods_per_year)
            total = all_df.height
            frames_int: list[pl.DataFrame] = []
            for i, start in enumerate(range(0, total, chunk), start=1):
                chunk_all = all_df.slice(start, chunk)
                if chunk_all.height < max(5, chunk // 4):
                    continue
                chunk_index = chunk_all.select(self.data.date_col)
                chunk_returns = chunk_all.select(self.data.returns.columns)
                chunk_benchmark = (
                    chunk_all.select(self.data.benchmark.columns) if self.data.benchmark is not None else None
                )
                chunk_data = Data(returns=chunk_returns, index=chunk_index, benchmark=chunk_benchmark)
                chunk_summary = Stats(chunk_data).summary()
                chunk_summary = chunk_summary.with_columns(pl.lit(i).alias("year"))
                frames_int.append(chunk_summary)
            if not frames_int:
                return pl.DataFrame()
            result_int = pl.concat(frames_int)
            ordered_int = ["year", "metric", *[c for c in result_int.columns if c not in ("year", "metric")]]
            return result_int.select(ordered_int)

        if date_col_name is None:  # unreachable: has_temporal guarantees non-None  # pragma: no cover
            return pl.DataFrame()  # pragma: no cover
        years = all_df[date_col_name].dt.year().unique().sort().to_list()

        frames: list[pl.DataFrame] = []
        for year in years:
            year_all = all_df.filter(pl.col(date_col_name).dt.year() == year)
            if year_all.height < 2:
                continue
            year_index = year_all.select([date_col_name])
            year_returns = year_all.select(self.data.returns.columns)
            year_benchmark = year_all.select(self.data.benchmark.columns) if self.data.benchmark is not None else None
            year_data = Data(returns=year_returns, index=year_index, benchmark=year_benchmark)
            year_summary = Stats(year_data).summary()
            year_summary = year_summary.with_columns(pl.lit(year).alias("year"))
            frames.append(year_summary)

        if not frames:
            asset_cols = list(self.data.returns.columns)
            schema: dict[str, type[pl.DataType]] = {
                "year": pl.Int32,
                "metric": pl.String,
                **dict.fromkeys(asset_cols, pl.Float64),
            }
            return pl.DataFrame(schema=schema)

        result = pl.concat(frames)
        ordered = ["year", "metric", *[c for c in result.columns if c not in ("year", "metric")]]
        return result.select(ordered)

    def summary(self) -> pl.DataFrame:
        """Summary statistics for each asset as a tidy DataFrame.

        Each row is one metric; each column beyond ``metric`` is one asset.

        Returns:
            pl.DataFrame: A DataFrame with a ``metric`` column followed by one
            column per asset.
        """
        assets = [col for col, _ in self.data.items()]

        def _safe(fn: Any) -> dict[str, Any]:
            """Call *fn()* and return its result; return NaN for each asset on any exception."""
            try:
                return fn()
            except Exception:
                return dict.fromkeys(assets, float("nan"))

        metrics: dict[str, dict[str, Any]] = {
            "avg_return": _safe(self.avg_return),
            "avg_win": _safe(self.avg_win),
            "avg_loss": _safe(self.avg_loss),
            "win_rate": _safe(self.win_rate),
            "profit_factor": _safe(self.profit_factor),
            "payoff_ratio": _safe(self.payoff_ratio),
            "monthly_win_rate": _safe(self.monthly_win_rate),
            "best": _safe(self.best),
            "worst": _safe(self.worst),
            "volatility": _safe(self.volatility),
            "sharpe": _safe(self.sharpe),
            "skew": _safe(self.skew),
            "kurtosis": _safe(self.kurtosis),
            "value_at_risk": _safe(self.value_at_risk),
            "conditional_value_at_risk": _safe(self.conditional_value_at_risk),
            "max_drawdown": _safe(self.max_drawdown),
            "avg_drawdown": _safe(self.avg_drawdown),
            "max_drawdown_duration": _safe(self.max_drawdown_duration),
            "calmar": _safe(self.calmar),
            "recovery_factor": _safe(self.recovery_factor),
        }

        rows: list[dict[str, object]] = [
            {"metric": name, **{asset: values.get(asset) for asset in assets}} for name, values in metrics.items()
        ]
        return pl.DataFrame(rows)
