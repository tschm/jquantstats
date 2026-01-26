from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable
from functools import wraps
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import polars as pl
from scipy.stats import norm

if TYPE_CHECKING:
    from ._data import Data


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

    @staticmethod
    def _mean_positive_expr(series: pl.Series) -> float:
        return cast(float, series.filter(series > 0).mean())

    @staticmethod
    def _mean_negative_expr(series: pl.Series) -> float:
        return cast(float, series.filter(series < 0).mean())

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
            return cast(pl.DataFrame, self.all).select(
                [pl.col(name) for name in self.data.date_col]
                + [func(self, series, *args, **kwargs).alias(col) for col, series in self.data.items()]
            )

        return wrapper

    @columnwise_stat
    def skew(self, series: pl.Series) -> int | float | None:
        """Calculate skewness (asymmetry) for each numeric column.

        Args:
            series (pl.Series): The series to calculate skewness for.

        Returns:
            float: The skewness value.

        """
        return cast("int | float | None", series.skew(bias=False))

    @columnwise_stat
    def kurtosis(self, series: pl.Series) -> int | float | None:
        """Calculate the kurtosis of returns.

        The degree to which a distribution peak compared to a normal distribution.

        Args:
            series (pl.Series): The series to calculate kurtosis for.

        Returns:
            float: The kurtosis value.

        """
        return cast("int | float | None", series.kurtosis(bias=False))

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
        wins_sum = cast(float, wins.sum())
        losses_sum = cast(float, losses.sum())

        return float(np.abs(wins_sum / losses_sum))

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
        divisor = std_val if std_val is not None else 1.0

        res = (mean_val if mean_val is not None else 0.0) / divisor
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
        downside_sum = cast(float, ((series.filter(series < 0)) ** 2).sum())
        downside_deviation = float(np.sqrt(downside_sum / series.count()))
        mean_val = cast(float, series.mean())
        ratio = (mean_val if mean_val is not None else 0.0) / downside_deviation
        return float(ratio * np.sqrt(periods))

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

    @to_frame
    def rolling_sharpe(
        self, series: pl.Expr, rolling_period: int = 126, periods_per_year: int | float | None = None
    ) -> pl.Expr:
        """Calculate the rolling Sharpe ratio.

        Args:
            series (pl.Expr): The expression to calculate rolling Sharpe ratio for.
            rolling_period (int, optional): The rolling window size. Defaults to 126.
            periods_per_year (int, optional): Number of periods per year. Defaults to 252.

        Returns:
            pl.Expr: The rolling Sharpe ratio expression.

        """
        ppy = periods_per_year or self.data._periods_per_year
        res = series.rolling_mean(window_size=rolling_period) / series.rolling_std(window_size=rolling_period)
        return cast(pl.Expr, res * np.sqrt(ppy))

    @to_frame
    def rolling_volatility(
        self, series: pl.Expr, rolling_period: int = 126, periods_per_year: int | float | None = None
    ) -> pl.Expr:
        """Calculate the rolling volatility of returns.

        Args:
            series (pl.Expr): The expression to calculate rolling volatility for.
            rolling_period (int, optional): The rolling window size. Defaults to 126.
            periods_per_year (float, optional): Number of periods per year. Defaults to None.

        Returns:
            pl.Expr: The rolling volatility expression.

        """
        ppy = periods_per_year or self.data._periods_per_year
        return cast(pl.Expr, series.rolling_std(window_size=rolling_period) * np.sqrt(ppy))

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
