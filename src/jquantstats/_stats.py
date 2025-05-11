import dataclasses
from collections.abc import Callable
from functools import wraps

import numpy as np
import polars as pl


@dataclasses.dataclass(frozen=True)
class Stats:
    data: "_Data"  # type: ignore
    all: pl.DataFrame = None  # Default is None; will be set in __post_init__

    def __post_init__(self):
        object.__setattr__(self, "all", self.data.all)

    @staticmethod
    def _quantile_expr(series, q):
        return series.quantile(q)

    @staticmethod
    def _mean_positive_expr(series):
        return series.filter(series >= 0).mean()

    @staticmethod
    def _mean_negative_expr(series):
        return series.filter(series < 0).mean()

    @staticmethod
    def _quantile_expr(series, cutoff):
        return series.quantile(cutoff)

    @staticmethod
    def columnwise_stat(func):
        """
        Decorator that applies a column-wise statistical function to all numeric columns
        of `self.data` and returns a dictionary with keys named appropriately.
        """

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return {col: func(self, self.all[col], *args, **kwargs) for col in self.data.assets}

        return wrapper

    @staticmethod
    def to_frame(func: Callable) -> Callable:
        """Decorator: Applies per-column expressions and evaluates with .with_columns(...)"""

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return self.all.select(
                [pl.col(name) for name in self.data.date_col]
                + [func(self, pl.col(name), *args, **kwargs).alias(name) for name in self.data.assets]
            )

        return wrapper

    @columnwise_stat
    def skew(self, series):
        """
        Calculates skewness (asymmetry) for each numeric column.
        """
        return series.skew(bias=False)

    @columnwise_stat
    def kurtosis(self, series):
        """
        Calculates returns' kurtosis
        (the degree to which a distribution peak compared to a normal distribution)
        """
        return series.kurtosis(bias=False)

    @columnwise_stat
    def avg_return(self, series):
        """Average return per non-zero, non-null value."""
        return series.filter(series.is_not_null() & (series != 0)).mean()

    @columnwise_stat
    def avg_win(self, series):
        """
        Calculates the average winning
        return/trade for an asset
        """
        return self._mean_positive_expr(series)

    @columnwise_stat
    def avg_loss(self, series):
        """
        Calculates the average low if
        return/trade return for a period
        """
        return self._mean_negative_expr(series)

    @columnwise_stat
    def volatility(self, series, periods=252, annualize=True):
        """
        Calculates the volatility of returns:
        - Std dev of returns
        - Annualized by sqrt(periods) if `annualize` is True
        """
        factor = np.sqrt(periods) if annualize else 1
        return series.std() * factor

    @to_frame
    def rolling_volatility(self, series: pl.Expr, rolling_period=126, periods_per_year=252) -> pl.Expr:
        return series.rolling_std(window_size=rolling_period) * np.sqrt(periods_per_year)

    @to_frame
    def price(self, series: pl.Expr, compounded=False, initial=1.0) -> pl.Expr:
        if compounded:
            # First compute cumulative compounded returns
            cum = initial * (1 + series).cum_prod()
        else:
            # Simple cumulative sum of returns
            cum = initial + series.cum_sum()

        return cum

    @to_frame
    def drawdown(self, series: pl.Expr, compounded=False, initial=1.0) -> pl.Expr:
        """
        Computes drawdown from the high-water mark.

        Args:
            series (pl.Expr): Polars expression for the return series.
            compounded (bool): Whether to use compounded returns.
            initial (float): Initial portfolio value (default is 1).

        Returns:
            pl.Expr: A Polars expression representing the drawdown.
        """
        if compounded:
            # First compute cumulative compounded returns
            equity = initial * (1 + series).cum_prod()
        else:
            # Simple cumulative sum of returns
            equity = initial + series.cum_sum()

        # equity = self.price(series, compounded, initial=initial)
        return -100 * ((equity / equity.cum_max()) - 1)

    @columnwise_stat
    def autocorr(self, series: pl.Series):
        """
        Metric to account for autocorrelation.
        Applies autocorrelation penalty to each numeric series (column).
        """
        corr = pl.corr(series, series.shift(1)).cast(pl.Float64)
        return corr

    @columnwise_stat
    def payoff_ratio(self, series):
        """
        Measures the payoff ratio: average win / abs(average loss).
        """
        avg_win = series.filter(series > 0).mean()
        # avg_win = self.avg_win(series)
        avg_loss = np.abs(series.filter(series < 0).mean())
        return avg_win / avg_loss

    def win_loss_ratio(self):
        """Shorthand for payoff_ratio()"""
        return self.payoff_ratio()

    @columnwise_stat
    def profit_ratio(self, series):
        """Measures the profit ratio (win ratio / loss ratio)"""
        wins = series.filter(series > 0)
        losses = series.filter(series < 0)

        try:
            win_ratio = np.abs(wins.mean() / wins.count())
            loss_ratio = np.abs(losses.mean() / losses.count())

            return win_ratio / loss_ratio

        except TypeError:
            return np.nan

    @columnwise_stat
    def profit_factor(self, series):
        """Measures the profit ratio (wins / loss)"""
        wins = series.filter(series > 0)
        losses = series.filter(series < 0)

        return np.abs(wins.sum() / losses.sum())

    def common_sense_ratio(self):
        """Measures the common sense ratio (profit factor * tail ratio)"""
        profit_factor = self.profit_factor()
        tail_ratio = self.tail_ratio()

        return {name: profit_factor[name] * tail_ratio[name] for name in profit_factor.keys()}

    @columnwise_stat
    def value_at_risk(self, series: pl.Series, alpha: float = 0.05):
        """
        Calculates the daily value-at-risk
        (variance-covariance calculation with confidence level)
        """
        # Ensure returns are sorted and drop nulls
        cleaned_returns = series.drop_nulls()

        # Compute VaR using quantile; note that VaR is typically a negative number (i.e. loss)
        return cleaned_returns.quantile(alpha, interpolation="nearest")

    def var(self, alpha: float = 0.05):
        """Shorthand for value_at_risk()"""
        return self.value_at_risk(alpha)

    @columnwise_stat
    def conditional_value_at_risk(self, series, alpha=0.05):
        """
        Calculates the conditional value-at-risk (CVaR / expected shortfall)
        for each numeric column.
        """
        # Ensure returns are sorted and drop nulls
        cleaned_returns = series.drop_nulls()

        # Compute VaR using quantile; note that VaR is typically a negative number (i.e. loss)
        var = cleaned_returns.quantile(alpha, interpolation="nearest")

        # Compute mean of returns less than or equal to VaR
        cvar = cleaned_returns.filter(cleaned_returns <= var).mean()

        return cvar

        # Compute CVaR: mean of values less than the VaR threshold
        # return pl.when(series < var_expr).then(series).otherwise(None).mean()

    def cvar(self, alpha=0.05):
        """Shorthand for conditional_value_at_risk()"""
        return self.conditional_value_at_risk(alpha)

    def expected_shortfall(self, alpha=0.05):
        """Shorthand for conditional_value_at_risk()"""
        return self.conditional_value_at_risk(alpha)

    @columnwise_stat
    def tail_ratio(self, series, cutoff=0.95):
        """Calculates the ratio of the right (95%) and left (5%) tails."""
        left_tail = self._quantile_expr(series, 1 - cutoff)
        right_tail = self._quantile_expr(series, cutoff)
        return abs(right_tail / left_tail)  # .alias(series.meta.output_name)

    @columnwise_stat
    def win_rate(self, series):
        """Calculates the win ratio for a period."""
        num_pos = series.filter(series > 0).count()
        num_nonzero = series.filter(series != 0).count()
        return num_pos / num_nonzero

    @columnwise_stat
    def gain_to_pain_ratio(self, series):
        """
        Jack Schwager's Gain-to-Pain Ratio:
        total return / sum of losses (in absolute value).
        """
        total_gain = series.sum()
        total_pain = series.filter(series < 0).abs().sum()
        try:
            return total_gain / total_pain
        except ZeroDivisionError:
            return np.nan

    @columnwise_stat
    def outlier_win_ratio(self, series: pl.Series, quantile: float = 0.99):
        """
        Calculates the outlier winners ratio:
        99th percentile of returns / mean positive return
        """
        q = series.quantile(quantile, interpolation="nearest")
        mean_positive = series.filter(series > 0).mean()
        return q / mean_positive

    @columnwise_stat
    def outlier_loss_ratio(self, series: pl.Series, quantile: float = 0.01):
        """
        Calculates the outlier losers ratio
        1st percentile of returns / mean negative return
        """
        q = series.quantile(quantile, interpolation="nearest")
        mean_negative = series.filter(series < 0).mean()
        return q / mean_negative

    @columnwise_stat
    def risk_return_ratio(self, series):
        """
        Calculates the return/risk ratio (Sharpe ratio w/o risk-free rate).
        """
        return series.mean() / series.std()

    def kelly_criterion(self):
        """
        Calculates the optimal capital allocation (Kelly Criterion) per column:
        f* = [(b * p) - q] / b
        where:
          - b = payoff ratio
          - p = win rate
          - q = 1 - p
        """
        b = self.payoff_ratio()
        p = self.win_rate()

        return {
            col: ((b[col] * p[col]) - (1 - p[col])) / b[col]
            # if b[col] not in (None, 0) and p[col] is not None else None
            for col in b
        }

    @columnwise_stat
    def best(self, series):
        """Returns the maximum return per column (best period)."""
        return series.max()  # .alias(series.meta.output_name)

    @columnwise_stat
    def worst(self, series):
        """Returns the minimum return per column (worst period)."""
        return series.min()  # .alias(series.meta.output_name)

    @columnwise_stat
    def exposure(self, series):
        """Returns the market exposure time (returns != 0)"""
        return np.round((series.filter(series != 0).count() / self.all.height), decimals=2)

    @columnwise_stat
    def sharpe(self, series, periods=252):
        """
        Calculates the Sharpe ratio of asset returns.
        """
        divisor = series.std(ddof=1)

        res = series.mean() / divisor
        factor = periods or 1
        return res * np.sqrt(factor)

    @to_frame
    def rolling_sharpe(self, series: pl.Expr, rolling_period=126, periods_per_year=252) -> pl.Expr:
        res = series.rolling_mean(window_size=rolling_period) / series.rolling_std(window_size=rolling_period)
        factor = periods_per_year or 1
        return res * np.sqrt(factor)

    @columnwise_stat
    def sortino(self, series: pl.Series, periods=252):
        """
        Calculates the Sortino ratio: mean return divided by downside deviation.
        Based on Red Rock Capital's Sortino ratio paper.
        """

        downside_deviation = np.sqrt(((series.filter(series < 0)) ** 2).mean())

        ratio = series.mean() / downside_deviation
        return ratio * np.sqrt(periods)

    @to_frame
    def rolling_sortino(self, series, rolling_period=126, periods_per_year=252):
        mean_ret = series.rolling_mean(window_size=rolling_period)

        # Rolling downside deviation (squared negative returns averaged over window)
        downside = series.map_elements(lambda x: x**2 if x < 0 else 0.0).rolling_mean(window_size=rolling_period)

        # Avoid division by zero
        sortino = mean_ret / downside.sqrt().fill_nan(0).fill_null(0)
        return sortino * (periods_per_year**0.5)

    def adjusted_sortino(self, periods=252):
        """
        Jack Schwager's adjusted Sortino ratio for direct comparison to Sharpe.
        See: https://archive.is/wip/2rwFW
        """
        sortino_data = self.sortino(periods=periods)
        return {k: v / np.sqrt(2) for k, v in sortino_data.items()}

    @columnwise_stat
    def r_squared(self, series, benchmark=None):
        """Measures the straight line fit of the equity curve"""
        if self.data.benchmark is None:
            raise AttributeError("No benchmark data available")

        benchmark_col = benchmark or self.data.benchmark.columns[0]

        # if self.data.benchmark is None:
        #    raise AttributeError("No benchmark data available")
        # Evaluate both series and benchmark as Series
        df = self.all.select([series, pl.col(benchmark_col).alias("benchmark")])

        # Drop nulls
        df = df.drop_nulls()
        print(df)

        matrix = df.to_numpy()
        # Get actual Series

        strategy_np = matrix[:, 0]
        benchmark_np = matrix[:, 1]

        corr_matrix = np.corrcoef(strategy_np, benchmark_np)
        r = corr_matrix[0, 1]
        return r**2

    def r2(self):
        """Shorthand for r_squared()"""
        return self.r_squared()

    @columnwise_stat
    def information_ratio(self, series, periods_per_year=252, benchmark=None):
        """
        Calculates the information ratio
        (basically the risk return ratio of the net profits)
        """
        benchmark_col = benchmark or self.data.benchmark.columns[0]

        active = series - self.data.benchmark[benchmark_col]

        mean = active.mean()
        std = active.std()

        try:
            return (mean / std) * (periods_per_year**0.5)
        except ZeroDivisionError:
            return 0.0

    @columnwise_stat
    def greeks(self, series, periods_per_year=252, benchmark=None):
        """Calculates alpha and beta of the portfolio"""
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
