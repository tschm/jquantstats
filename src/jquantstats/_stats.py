import dataclasses
from collections.abc import Callable
from functools import wraps

import numpy as np
import polars as pl
from scipy.stats import norm


@dataclasses.dataclass(frozen=True)
class Stats:
    """
    Statistical analysis tools for financial returns data.

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

    data: "Data"  # type: ignore
    all: pl.DataFrame = None  # Default is None; will be set in __post_init__

    def __post_init__(self):
        object.__setattr__(self, "all", self.data.all)

    @staticmethod
    def _mean_positive_expr(series: pl.Series) -> float:
        return series.filter(series > 0).mean()

    @staticmethod
    def _mean_negative_expr(series: pl.Series) -> float:
        return series.filter(series < 0).mean()

    @staticmethod
    def columnwise_stat(func: Callable) -> Callable:
        """
        Decorator that applies a column-wise statistical function to all numeric columns
        of `self.data` and returns a dictionary with keys named appropriately.

        Args:
            func (Callable): The function to decorate.

        Returns:
            Callable: The decorated function.
        """

        @wraps(func)
        def wrapper(self, *args, **kwargs) -> dict[str, float]:
            return {col: func(self, series, *args, **kwargs) for col, series in self.data.items()}

        return wrapper

    @staticmethod
    def to_frame(func: Callable) -> Callable:
        """
        Decorator: Applies per-column expressions and evaluates with .with_columns(...)

        Args:
            func (Callable): The function to decorate.

        Returns:
            Callable: The decorated function.
        """

        @wraps(func)
        def wrapper(self, *args, **kwargs) -> pl.DataFrame:
            return self.all.select(
                [pl.col(name) for name in self.data.date_col]
                + [func(self, series, *args, **kwargs).alias(col) for col, series in self.data.items()]
            )

        return wrapper

    @columnwise_stat
    def skew(self, series: pl.Series) -> float:
        """
        Calculates skewness (asymmetry) for each numeric column.

        Args:
            series (pl.Series): The series to calculate skewness for.

        Returns:
            float: The skewness value.
        """
        return series.skew(bias=False)

    @columnwise_stat
    def kurtosis(self, series: pl.Series) -> float:
        """
        Calculates returns' kurtosis
        (the degree to which a distribution peak compared to a normal distribution)

        Args:
            series (pl.Series): The series to calculate kurtosis for.

        Returns:
            float: The kurtosis value.
        """
        return series.kurtosis(bias=False)

    @columnwise_stat
    def avg_return(self, series: pl.Series) -> float:
        """
        Average return per non-zero, non-null value.

        Args:
            series (pl.Series): The series to calculate average return for.

        Returns:
            float: The average return value.
        """
        return series.filter(series.is_not_null() & (series != 0)).mean()

    @columnwise_stat
    def avg_win(self, series: pl.Series) -> float:
        """
        Calculates the average winning return/trade for an asset.

        Args:
            series (pl.Series): The series to calculate average win for.

        Returns:
            float: The average winning return.
        """
        return self._mean_positive_expr(series)

    @columnwise_stat
    def avg_loss(self, series: pl.Series) -> float:
        """
        Calculates the average loss return/trade for a period.

        Args:
            series (pl.Series): The series to calculate average loss for.

        Returns:
            float: The average loss return.
        """
        return self._mean_negative_expr(series)

    @columnwise_stat
    def volatility(self, series: pl.Series, periods: float = None, annualize: bool = True) -> float:
        """
        Calculates the volatility of returns:
        - Std dev of returns
        - Annualized by sqrt(periods) if `annualize` is True

        Args:
            series (pl.Series): The series to calculate volatility for.
            periods (int, optional): Number of periods per year. Defaults to 252.
            annualize (bool, optional): Whether to annualize the result. Defaults to True.

        Returns:
            float: The volatility value.
        """
        periods = periods or self.data._periods_per_year
        factor = np.sqrt(periods) if annualize else 1
        return series.std() * factor

    @columnwise_stat
    def payoff_ratio(self, series: pl.Series) -> float:
        """
        Measures the payoff ratio: average win / abs(average loss).

        Args:
            series (pl.Series): The series to calculate payoff ratio for.

        Returns:
            float: The payoff ratio value.
        """
        avg_win = series.filter(series > 0).mean()
        # avg_win = self.avg_win(series)
        avg_loss = np.abs(series.filter(series < 0).mean())
        return avg_win / avg_loss

    def win_loss_ratio(self) -> dict[str, float]:
        """
        Shorthand for payoff_ratio()

        Returns:
            dict[str, float]: Dictionary mapping asset names to win/loss ratios.
        """
        return self.payoff_ratio()

    @columnwise_stat
    def profit_ratio(self, series: pl.Series) -> float:
        """
        Measures the profit ratio (win ratio / loss ratio)

        Args:
            series (pl.Series): The series to calculate profit ratio for.

        Returns:
            float: The profit ratio value.
        """
        wins = series.filter(series >= 0)
        losses = series.filter(series < 0)

        try:
            win_ratio = np.abs(wins.mean() / wins.count())
            loss_ratio = np.abs(losses.mean() / losses.count())

            return win_ratio / loss_ratio

        except TypeError:
            return np.nan

    @columnwise_stat
    def profit_factor(self, series: pl.Series) -> float:
        """
        Measures the profit ratio (wins / loss)

        Args:
            series (pl.Series): The series to calculate profit factor for.

        Returns:
            float: The profit factor value.
        """
        wins = series.filter(series > 0)
        losses = series.filter(series < 0)

        return np.abs(wins.sum() / losses.sum())

    @columnwise_stat
    def value_at_risk(self, series: pl.Series, sigma=1, alpha: float = 0.05) -> float:
        """
        Calculates the daily value-at-risk
        (variance-covariance calculation with confidence level)

        Args:
            series (pl.Series): The series to calculate value at risk for.
            alpha (float, optional): Confidence level. Defaults to 0.05.

        Returns:
            float: The value at risk.
        """
        mu = series.mean()
        sigma *= series.std()

        return norm.ppf(alpha, mu, sigma)

    @columnwise_stat
    def conditional_value_at_risk(self, series: pl.Series, sigma=1, alpha: float = 0.05) -> float:
        """
        Calculates the conditional value-at-risk (CVaR / expected shortfall)
        for each numeric column.

        Args:
            series (pl.Series): The series to calculate conditional value at risk for.
            alpha (float, optional): Confidence level. Defaults to 0.05.

        Returns:
            float: The conditional value at risk.
        """
        mu = series.mean()
        sigma *= series.std()

        var = norm.ppf(alpha, mu, sigma)

        # Compute mean of returns less than or equal to VaR
        return series.filter(series < var).mean()

    @columnwise_stat
    def win_rate(self, series: pl.Series) -> float:
        """
        Calculates the win ratio for a period.

        Args:
            series (pl.Series): The series to calculate win rate for.

        Returns:
            float: The win rate value.
        """
        num_pos = series.filter(series > 0).count()
        num_nonzero = series.filter(series != 0).count()
        return num_pos / num_nonzero

    @columnwise_stat
    def gain_to_pain_ratio(self, series: pl.Series) -> float:
        """
        Jack Schwager's Gain-to-Pain Ratio:
        total return / sum of losses (in absolute value).

        Args:
            series (pl.Series): The series to calculate gain to pain ratio for.

        Returns:
            float: The gain to pain ratio value.
        """
        total_gain = series.sum()
        total_pain = series.filter(series < 0).abs().sum()
        try:
            return total_gain / total_pain
        except ZeroDivisionError:
            return np.nan

    @columnwise_stat
    def risk_return_ratio(self, series: pl.Series) -> float:
        """
        Calculates the return/risk ratio (Sharpe ratio w/o risk-free rate).

        Args:
            series (pl.Series): The series to calculate risk return ratio for.

        Returns:
            float: The risk return ratio value.
        """
        return series.mean() / series.std()

    def kelly_criterion(self) -> dict[str, float]:
        """
        Calculates the optimal capital allocation (Kelly Criterion) per column:
        f* = [(b * p) - q] / b
        where:
          - b = payoff ratio
          - p = win rate
          - q = 1 - p

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
    def best(self, series: pl.Series) -> float:
        """
        Returns the maximum return per column (best period).

        Args:
            series (pl.Series): The series to find the best return for.

        Returns:
            float: The maximum return value.
        """
        return series.max()  # .alias(series.meta.output_name)

    @columnwise_stat
    def worst(self, series: pl.Series) -> float:
        """
        Returns the minimum return per column (worst period).

        Args:
            series (pl.Series): The series to find the worst return for.

        Returns:
            float: The minimum return value.
        """
        return series.min()  # .alias(series.meta.output_name)

    @columnwise_stat
    def exposure(self, series: pl.Series) -> float:
        """
        Returns the market exposure time (returns != 0)

        Args:
            series (pl.Series): The series to calculate exposure for.

        Returns:
            float: The exposure value.
        """
        return np.round((series.filter(series != 0).count() / self.all.height), decimals=2)

    @columnwise_stat
    def sharpe(self, series: pl.Series, periods: float = None) -> float:
        """
        Calculates the Sharpe ratio of asset returns.

        Args:
            series (pl.Series): The series to calculate Sharpe ratio for.
            periods (int, optional): Number of periods per year. Defaults to 252.

        Returns:
            float: The Sharpe ratio value.
        """
        periods = periods or self.data._periods_per_year

        divisor = series.std(ddof=1)

        res = series.mean() / divisor
        factor = periods or 1
        return res * np.sqrt(factor)

    @columnwise_stat
    def sortino(self, series: pl.Series, periods: float = None) -> float:
        """
        Calculates the Sortino ratio: mean return divided by downside deviation.
        Based on Red Rock Capital's Sortino ratio paper.

        Args:
            series (pl.Series): The series to calculate Sortino ratio for.
            periods (int, optional): Number of periods per year. Defaults to 252.

        Returns:
            float: The Sortino ratio value.
        """
        periods = periods or self.data._periods_per_year
        downside_deviation = np.sqrt(((series.filter(series < 0)) ** 2).sum() / series.count())
        ratio = series.mean() / downside_deviation
        return ratio * np.sqrt(periods)

    @to_frame
    def rolling_sortino(self, series: pl.Expr, rolling_period: int = 126, periods_per_year: float = None) -> pl.Expr:
        """
        Calculates the rolling Sortino ratio.

        Args:
            series (pl.Expr): The expression to calculate rolling Sortino ratio for.
            rolling_period (int, optional): The rolling window size. Defaults to 126.
            periods_per_year (int, optional): Number of periods per year. Defaults to 252.

        Returns:
            pl.Expr: The rolling Sortino ratio expression.
        """
        periods_per_year = periods_per_year or self.data._periods_per_year

        mean_ret = series.rolling_mean(window_size=rolling_period)

        # Rolling downside deviation (squared negative returns averaged over window)
        downside = series.map_elements(lambda x: x**2 if x < 0 else 0.0).rolling_mean(window_size=rolling_period)

        # Avoid division by zero
        sortino = mean_ret / downside.sqrt().fill_nan(0).fill_null(0)
        return sortino * (periods_per_year**0.5)

    @to_frame
    def rolling_sharpe(self, series: pl.Expr, rolling_period: int = 126, periods_per_year: float = None) -> pl.Expr:
        """
        Calculates the rolling Sharpe ratio.

        Args:
            series (pl.Expr): The expression to calculate rolling Sharpe ratio for.
            rolling_period (int, optional): The rolling window size. Defaults to 126.
            periods_per_year (int, optional): Number of periods per year. Defaults to 252.

        Returns:
            pl.Expr: The rolling Sharpe ratio expression.
        """
        periods_per_year = periods_per_year or self.data._periods_per_year
        res = series.rolling_mean(window_size=rolling_period) / series.rolling_std(window_size=rolling_period)
        return res * np.sqrt(periods_per_year)

    @to_frame
    def rolling_volatility(self, series: pl.Expr, rolling_period=126, periods_per_year: float = None) -> pl.Expr:
        return series.rolling_std(window_size=rolling_period) * np.sqrt(periods_per_year)

    @to_frame
    def drawdown(self, series: pl.Series) -> pl.Series:
        equity = self.prices(series)
        d = (equity / equity.cum_max()) - 1
        return -d

    @staticmethod
    def prices(series: pl.Series) -> pl.Series:
        return (1.0 + series).cum_prod()

    @staticmethod
    def max_drawdown_single_series(series: pl.Series) -> float:
        price = Stats.prices(series)
        peak = price.cum_max()
        drawdown = price / peak - 1
        return -drawdown.min()

    @columnwise_stat
    def max_drawdown(self, series: pl.Expr) -> float:
        return Stats.max_drawdown_single_series(series)

    # @columnwise_stat
    # def calmar(self, series):
    #    dd = Stats.max_drawdown_single_series(series)
    #    #dd = self.max_drawdown(series)
    #    return 100*series.mean() / dd if dd > 0 else np.nan

    def adjusted_sortino(self, periods: float = None) -> dict[str, float]:
        """
        Jack Schwager's adjusted Sortino ratio for direct comparison to Sharpe.
        See: https://archive.is/wip/2rwFW

        Args:
            periods (int, optional): Number of periods per year. Defaults to 252.

        Returns:
            dict[str, float]: Dictionary mapping asset names to adjusted Sortino ratios.
        """
        sortino_data = self.sortino(periods=periods)
        return {k: v / np.sqrt(2) for k, v in sortino_data.items()}

    @columnwise_stat
    def r_squared(self, series: pl.Series, benchmark: str = None) -> float:
        """
        Measures the straight line fit of the equity curve

        Args:
            series (pl.Series): The series to calculate R-squared for.
            benchmark (str, optional): The benchmark column name. Defaults to None.

        Returns:
            float: The R-squared value.

        Raises:
            AttributeError: If no benchmark data is available.
        """
        if self.data.benchmark is None:
            raise AttributeError("No benchmark data available")

        benchmark_col = benchmark or self.data.benchmark.columns[0]

        # if self.data.benchmark is None:
        #    raise AttributeError("No benchmark data available")
        # Evaluate both series and benchmark as Series
        df = self.all.select([series, pl.col(benchmark_col).alias("benchmark")])

        # Drop nulls
        df = df.drop_nulls()

        matrix = df.to_numpy()
        # Get actual Series

        strategy_np = matrix[:, 0]
        benchmark_np = matrix[:, 1]

        corr_matrix = np.corrcoef(strategy_np, benchmark_np)
        r = corr_matrix[0, 1]
        return r**2

    def r2(self) -> dict[str, float]:
        """
        Shorthand for r_squared()

        Returns:
            dict[str, float]: Dictionary mapping asset names to R-squared values.
        """
        return self.r_squared()

    @columnwise_stat
    def information_ratio(self, series: pl.Series, periods_per_year: float = None, benchmark: str = None) -> float:
        """
        Calculates the information ratio
        (basically the risk return ratio of the net profits)

        Args:
            series (pl.Series): The series to calculate information ratio for.
            periods_per_year (int, optional): Number of periods per year. Defaults to 252.
            benchmark (str, optional): The benchmark column name. Defaults to None.

        Returns:
            float: The information ratio value.
        """
        periods_per_year = periods_per_year or self.data.periods_per_year

        benchmark_col = benchmark or self.data.benchmark.columns[0]

        active = series - self.data.benchmark[benchmark_col]

        mean = active.mean()
        std = active.std()

        try:
            return (mean / std) * (periods_per_year**0.5)
        except ZeroDivisionError:
            return 0.0

    @columnwise_stat
    def greeks(self, series: pl.Series, periods_per_year: float = None, benchmark: str = None) -> dict[str, float]:
        """
        Calculates alpha and beta of the portfolio

        Args:
            series (pl.Series): The series to calculate greeks for.
            periods_per_year (int, optional): Number of periods per year. Defaults to 252.
            benchmark (str, optional): The benchmark column name. Defaults to None.

        Returns:
            dict[str, float]: Dictionary containing alpha and beta values.
        """
        periods_per_year = periods_per_year or self.data._periods_per_year

        # period_col = benchmark or self.data.benchmark.columns[0]

        # find covariance
        benchmark_col = benchmark or self.data.benchmark.columns[0]

        # Evaluate both series and benchmark as Series
        df = self.all.select([series, pl.col(benchmark_col).alias("benchmark")])

        # Drop nulls
        df = df.drop_nulls()
        matrix = df.to_numpy()

        # Get actual Series
        strategy_np = matrix[:, 0]
        benchmark_np = matrix[:, 1]

        # 2x2 covariance matrix: [[var_strategy, cov], [cov, var_benchmark]]
        cov_matrix = np.cov(strategy_np, benchmark_np)

        cov = cov_matrix[0, 1]
        var_benchmark = cov_matrix[1, 1]

        beta = cov / var_benchmark if var_benchmark != 0 else float("nan")
        alpha = np.mean(strategy_np) - beta * np.mean(benchmark_np)

        return {"alpha": alpha * periods_per_year, "beta": beta}
