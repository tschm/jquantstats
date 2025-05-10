# QuantStats: Portfolio analytics for quants
# https://github.com/tschm/jquantstats
#
# Copyright 2019-2024 Ran Aroussi
# Copyright 2025 Thomas Schmelzer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
QuantStats API module.

This module provides the core API for the QuantStats library,
including the Data class
for handling financial returns data and benchmarks.
"""

import calendar
import dataclasses
from functools import wraps
from math import ceil as _ceil

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots
from scipy.stats import linregress


def build_data(
    returns: pl.DataFrame, rf: float | pl.DataFrame = 0.0, benchmark: pl.DataFrame = None, date_col: str = "Date"
) -> "_Data":
    """
    Build a _Data object from returns and optional benchmark using Polars.

    Parameters:
        returns (pl.DataFrame): Financial returns.
        rf (float | pl.DataFrame): Risk-free rate (scalar or time series).
        benchmark (pl.DataFrame, optional): Benchmark returns.
        date_col (str): Name of the date column.

    Returns:
        _Data: Object containing excess returns and benchmark (if any).
    """

    def subtract_risk_free(df: pl.DataFrame, rf: float | pl.DataFrame, date_col: str) -> pl.DataFrame:
        if df is None:
            return None

        # Handle scalar rf case
        if isinstance(rf, float):
            rf_df = df.select([pl.col(date_col), pl.lit(rf).alias("rf")])
        else:
            rf_df = rf.rename({rf.columns[1]: "rf"}) if rf.columns[1] != "rf" else rf

        # Join and subtract
        df = df.join(rf_df, on=date_col, how="inner")
        return df.select(
            [pl.col(date_col)]
            + [
                (pl.col(col) - pl.col("rf")).alias(col)
                for col in df.columns
                if col not in {date_col, "rf"} and df.schema[col] in pl.NUMERIC_DTYPES
            ]
        )

    # Align returns and benchmark if both provided
    if benchmark is not None:
        joined_dates = returns.join(benchmark, on=date_col, how="inner").select(date_col)
        if joined_dates.is_empty():
            raise ValueError("No overlapping dates between returns and benchmark.")
        returns = returns.join(joined_dates, on=date_col, how="inner")
        benchmark = benchmark.join(joined_dates, on=date_col, how="inner")

    # Subtract risk-free rate
    index = returns.select(date_col)
    excess_returns = subtract_risk_free(returns, rf, date_col).drop(date_col)
    excess_benchmark = subtract_risk_free(benchmark, rf, date_col).drop(date_col) if benchmark is not None else None

    return _Data(returns=excess_returns, benchmark=excess_benchmark, index=index)


@dataclasses.dataclass(frozen=True)
class _Data:
    """
    A container for financial returns data and an optional benchmark.

    This class provides methods for analyzing and manipulating financial returns data,
    including converting returns to prices, calculating drawdowns, and resampling data
    to different time periods.

    Attributes:
        returns (pd.DataFrame): DataFrame containing returns data, typically with dates as index
                               and assets as columns.
        #benchmark (pd.Series, optional): Series containing benchmark returns data with the same
        #                                index as returns. Defaults to None.
    """

    returns: pl.DataFrame
    benchmark: pl.DataFrame | None = None
    index: pd.DataFrame | None = None

    @property
    def plots(self):
        return _Plots(self)

    @property
    def stats(self):
        return _Stats(self)

    def __post_init__(self) -> None:
        """
        Validates that the benchmark index matches the returns index if benchmark is provided.

        Raises:
            AssertionError: If benchmark is provided and its index doesn't match returns index.
        """

    @property
    def all(self) -> pl.DataFrame:
        """
        Combines returns and benchmark data into a single DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing all returns data and benchmark (if available),
                         with NaN values filled with 0.0.
        """
        return pl.concat([self.index, self.returns, self.benchmark], how="horizontal")

    def prices(self, compounded: bool = False, initial_value: float = 100.0) -> pl.DataFrame:
        """
        Converts returns to prices.

        Args:
            compounded (bool, optional): If True, uses compounded returns (cumprod).
                                         If False, uses simple returns (cumsum).
            initial_value (float, optional): Starting price value. Defaults to 100.0.

        Returns:
            pl.DataFrame: Price data derived from returns.
        """

        def to_prices(data: pl.DataFrame) -> pl.DataFrame:
            if data is None:
                return None
            df = data.fill_null(0.0)
            if compounded:
                return df.select(
                    [(pl.lit(initial_value) * (pl.col(col) + 1).cumprod()).alias(col) for col in df.columns]
                )
            else:
                return df.select([(pl.lit(initial_value) + pl.col(col).cum_sum()).alias(col) for col in df.columns])

        parts = [self.index, to_prices(self.returns)]
        if self.benchmark is not None:
            parts.append(to_prices(self.benchmark))

        return pl.concat(parts, how="horizontal")

    def resample(self, every: str = "1mo", compounded: bool = False) -> "_Data":
        """
        Resamples returns and benchmark to a different frequency using Polars.

        Args:
            every (str, optional): Resampling frequency (e.g., '1mo', '1y'). Defaults to '1mo'.
            compounded (bool, optional): Whether to compound returns. Defaults to False.

        Returns:
            _Data: Resampled data.
        """

        def resample_frame(df: pl.DataFrame) -> pl.DataFrame:
            if df is None:
                return None

            df = self.index.hstack(df)  # Add the date column for resampling
            # agg_fn = (
            #    (pl.col(col) + 1.0).product() - 1.0 if compounded else pl.col(col).sum()
            #    for col in df.columns
            #    if col != self.index.columns[0]
            # )

            return df.group_by_dynamic(
                index_column=self.index.columns[0], every=every, period=every, closed="right", label="right"
            ).agg(
                [
                    pl.col(col).sum().alias(col) if not compounded else (pl.col(col) + 1.0).product().alias(col)
                    for col in df.columns
                    if col != self.index.columns[0]
                ]
            )

        resampled_returns = resample_frame(self.returns)
        resampled_benchmark = resample_frame(self.benchmark) if self.benchmark is not None else None
        resampled_index = resampled_returns.select(self.index.columns[0])

        return _Data(
            returns=resampled_returns.drop(self.index.columns[0]),
            benchmark=resampled_benchmark.drop(self.index.columns[0]) if resampled_benchmark is not None else None,
            index=resampled_index,
        )

    # def apply(self, fct: Callable, **kwargs: Any) -> Any:
    #     """
    #     Applies a function to the returns DataFrame.
    #
    #     Args:
    #         fct (Callable): Function to apply to the returns DataFrame.
    #         **kwargs: Additional keyword arguments to pass to the function.
    #
    #     Returns:
    #         Any: The result of applying the function to the returns DataFrame.
    #     """
    #     return fct(self.returns, **kwargs)

    def copy(self) -> "_Data":
        """
        Creates a deep copy of the Data object.

        Returns:
            _Data: A new Data object with copies of the returns and benchmark.
        """
        try:
            return _Data(returns=self.returns.clone(), benchmark=self.benchmark.clone(), index=self.index.clone())
        except AttributeError:
            # Handle case where benchmark is None
            return _Data(returns=self.returns.clone(), index=self.index.clone())

    # def numeric(self):
    #     return self.returns.select(pl.col(pl.NUMERIC_DTYPES))

    # def highwater_mark(self, compounded: bool = False) -> pl.DataFrame:
    #     """
    #     Calculates the running maximum (high-water mark) of prices.
    #
    #     Args:
    #         compounded (bool, optional): If True, uses compounded returns to calculate prices.
    #                                     If False, uses simple returns. Defaults to False.
    #
    #     Returns:
    #         pd.DataFrame: A DataFrame containing the high-water mark for each asset.
    #     """
    #     def to_highwater(data: pl.DataFrame) -> pl.DataFrame:
    #         if compounded:
    #             # Calculate cumulative product for compounded returns
    #             return data.select([
    #                 (1 + pl.col(col)).cum_prod().alias(col)
    #                 for col in data.columns
    #             ])
    #         else:
    #             # Calculate cumulative sum for simple returns
    #             return data.select([
    #                 pl.col(col).cum_sum().cum_max().alias(col)
    #                 for col in data.columns
    #             ])
    #
    #
    #     parts = [self.index, to_highwater(self.returns)]
    #     if self.benchmark is not None:
    #         parts.append(to_highwater(self.benchmark))
    #
    #     return pl.concat(parts, how="horizontal")

    def head(self, n: int = 5) -> "_Data":
        """
        Returns the first n rows of the combined returns and benchmark data.

        Args:
            n (int, optional): Number of rows to return. Defaults to 5.

        Returns:
            _Data: A new Data object containing the first n rows of the combined data.
        """
        return _Data(returns=self.returns.head(n), benchmark=self.benchmark.head(n), index=self.index.head(n))

    def tail(self, n: int = 5) -> "_Data":
        """
        Returns the last n rows of the combined returns and benchmark data.

        Args:
            n (int, optional): Number of rows to return. Defaults to 5.

        Returns:
            _Data: A new Data object containing the last n rows of the combined data.
        """
        return _Data(returns=self.returns.tail(n), benchmark=self.benchmark.tail(n), index=self.index.tail(n))

    # def drawdown(self, compounded: bool = False) -> pl.DataFrame:
    #     """
    #     Calculates drawdowns from prices.
    #
    #     Args:
    #         compounded (bool, optional): If True, calculates drawdowns as percentage change
    #                                     from high-water mark. If False, calculates drawdowns
    #                                     as absolute difference from high-water mark.
    #                                     Defaults to False.
    #
    #     Returns:
    #         pl.DataFrame: A DataFrame containing drawdowns for each asset (and benchmark if available).
    #     """
    #     prices = self.prices(compounded)
    #
    #     # def calculate_drawdown(data: pl.DataFrame) -> pl.DataFrame:
    #     #     if compounded:
    #     #         # Calculate cumulative product for compounded returns
    #     #         peak =
    #     #         return data.select([
    #     #             pl.col(col)).cum_prod().alias(col)
    #     #             for col in data.columns
    #     #         ])
    #     #     else:
    #     #         # Calculate cumulative sum for simple returns
    #     #         return data.select([
    #     #             pl.col(col).cum_sum().cum_max().alias(col)
    #     #             for col in data.columns
    #     #         ])
    #
    #     def calculate_drawdown(data: pl.DataFrame) -> pl.DataFrame:
    #         peak = data.cum_max()
    #         if compounded:
    #             return data / peak - 1.0
    #         return peak - data
    #
    #     parts = [self.index]
    #     parts.append(calculate_drawdown(prices))
    #
    #     if self.benchmark is not None:
    #         benchmark_prices = self.benchmark_prices(compounded)  # Assuming this method exists
    #         parts.append(calculate_drawdown(benchmark_prices))
    #
    #     return pl.concat(parts, how="horizontal")


@dataclasses.dataclass  # (frozen=True)
class _Stats:
    data: _Data
    numeric_columns: list[str] = None
    all: pl.DataFrame = None

    def __post_init__(self):
        self.all = self.data.all
        numeric_dtypes = {pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.UInt32, pl.UInt64}
        self.numeric_columns = [name for name, dtype in self.all.schema.items() if dtype in numeric_dtypes]

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
    def per_numeric_column(expr_func):
        """Decorator: Apply a given column-wise expr_func across all numeric columns."""

        @wraps(expr_func)
        def wrapper(self, *args, **kwargs):
            return [
                # pl.col(name).drop_nans()
                expr_func(self, pl.col(name).drop_nans(), *args, **kwargs).alias(name)
                for name in self.numeric_columns
            ]

        return wrapper

    @staticmethod
    def to_dict(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            expressions = func(self, *args, **kwargs)
            frame = self.all.select(expressions)
            return dict(zip(frame.columns, frame.row(0)))

        return wrapper

    @staticmethod
    def to_frame(func):
        """Decorator: Applies per-column expressions and evaluates with .with_columns(...)"""

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            expressions = func(self, *args, **kwargs)
            return self.all_pl.with_columns(expressions)

        return wrapper

    @to_dict
    @per_numeric_column
    def skew(self, series):
        """
        Calculates skewness (asymmetry) for each numeric column.
        """
        return series.skew(bias=False)

    @to_dict
    @per_numeric_column
    def kurtosis(self, series):
        """
        Calculates returns' kurtosis
        (the degree to which a distribution peak compared to a normal distribution)
        """
        return series.kurtosis(bias=False)

    @to_dict
    @per_numeric_column
    def avg_return(self, series):
        """Average return per non-zero, non-null value."""
        return series.filter(series.is_not_null() & (series != 0)).mean()

    @to_dict
    @per_numeric_column
    def avg_win(self, series):
        """
        Calculates the average winning
        return/trade for an asset
        """
        return self._mean_positive_expr(series)

    @to_dict
    @per_numeric_column
    def avg_loss(self, series):
        """
        Calculates the average low if
        return/trade return for a period
        """
        return self._mean_negative_expr(series)

    @to_dict
    @per_numeric_column
    def volatility(self, series, periods=252, annualize=True):
        """
        Calculates the volatility of returns:
        - Std dev of returns
        - Annualized by sqrt(periods) if `annualize` is True
        """
        factor = np.sqrt(periods) if annualize else 1
        return (series.std() * factor).cast(pl.Float64)

    def rolling_volatility(self, rolling_period=126, periods_per_year=252):
        return self.all.rolling(rolling_period).std() * np.sqrt(periods_per_year)

    # def log_returns(self):
    #     """Shorthand for to_log_returns"""
    #     return self.to_log_returns()
    #
    # def to_log_returns(self):
    #     """Converts returns series to log returns"""
    #     return np.log(self.all + 1).replace([np.inf, np.inf], float("NaN"))
    #
    # def implied_volatility(self, periods=252):
    #     """Calculates the implied volatility of returns for a period"""
    #     logret = self.log_returns()
    #     factor = periods or 1
    #     return logret.std() * np.sqrt(factor)

    @to_dict
    @per_numeric_column
    def autocorr(self, series: pl.Series):
        """
        Metric to account for autocorrelation.
        Applies autocorrelation penalty to each numeric series (column).
        """
        corr = pl.corr(series, series.shift(1)).cast(pl.Float64)
        return corr

    @to_dict
    @per_numeric_column
    def payoff_ratio(self, series):
        """
        Measures the payoff ratio: average win / abs(average loss).
        """
        avg_win = series.filter(series > 0).mean()
        # avg_win = self.avg_win(series)
        avg_loss = series.filter(series < 0).mean().abs()
        return (avg_win / avg_loss).cast(pl.Float64)

    def win_loss_ratio(self):
        """Shorthand for payoff_ratio()"""
        return self.payoff_ratio()

    @to_dict
    @per_numeric_column
    def profit_ratio(self, series):
        """Measures the profit ratio (win ratio / loss ratio)"""
        wins = series.filter(series > 0)
        losses = series.filter(series < 0)

        win_ratio = (wins.mean() / wins.count()).abs()
        loss_ratio = (losses.mean() / losses.count()).abs()

        return (win_ratio / loss_ratio).fill_null(np.nan).cast(pl.Float64)

    @to_dict
    @per_numeric_column
    def profit_factor(self, series):
        """Measures the profit ratio (wins / loss)"""
        wins = series.filter(series > 0)
        losses = series.filter(series < 0)

        return (wins.sum() / losses.sum()).abs()

    def common_sense_ratio(self):
        """Measures the common sense ratio (profit factor * tail ratio)"""
        profit_factor = self.profit_factor()
        tail_ratio = self.tail_ratio()

        return {name: profit_factor[name] * tail_ratio[name] for name in profit_factor.keys()}
        # return profit_factor * tail_ratio

    @to_dict
    @per_numeric_column
    def value_at_risk(self, series: pl.Series, alpha: float = 0.05):
        """
        Calculates the daily value-at-risk
        (variance-covariance calculation with confidence level)
        """
        # Ensure returns are sorted and drop nulls
        cleaned_returns = series.drop_nulls()

        # Compute VaR using quantile; note that VaR is typically a negative number (i.e. loss)
        var = cleaned_returns.quantile(alpha, interpolation="nearest")

        return var.cast(pl.Float64)

    def var(self, alpha: float = 0.05):
        """Shorthand for value_at_risk()"""
        return self.value_at_risk(alpha)

    @to_dict
    @per_numeric_column
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

    @to_dict
    @per_numeric_column
    def tail_ratio(self, series, cutoff=0.95):
        """Calculates the ratio of the right (95%) and left (5%) tails."""
        left_tail = self._quantile_expr(series, 1 - cutoff)
        right_tail = self._quantile_expr(series, cutoff)
        return abs(right_tail / left_tail)  # .alias(series.meta.output_name)

    @to_dict
    @per_numeric_column
    def win_rate(self, series):
        """Calculates the win ratio for a period."""
        num_pos = series.filter(series > 0).count()
        num_nonzero = series.filter(series != 0).count()
        return (num_pos / num_nonzero).cast(pl.Float64)

    @to_dict
    @per_numeric_column
    def gain_to_pain_ratio(self, series):
        """
        Jack Schwager's Gain-to-Pain Ratio:
        total return / sum of losses (in absolute value).
        """
        total_gain = series.sum()
        total_pain = series.filter(series < 0).abs().sum()
        return (total_gain / total_pain).cast(pl.Float64)

    @to_dict
    @per_numeric_column
    def outlier_win_ratio(self, series: pl.Series, quantile: float = 0.99):
        """
        Calculates the outlier winners ratio:
        99th percentile of returns / mean positive return
        """
        q = series.quantile(quantile, interpolation="nearest")
        mean_positive = series.filter(series > 0).mean()
        return q / mean_positive

    @to_dict
    @per_numeric_column
    def outlier_loss_ratio(self, series: pl.Series, quantile: float = 0.01):
        """
        Calculates the outlier losers ratio
        1st percentile of returns / mean negative return
        """
        q = series.quantile(quantile, interpolation="nearest")
        mean_negative = series.filter(series < 0).mean()
        return q / mean_negative

    # def recovery_factor(self, rf=0.0):
    #    """Measures how fast the strategy recovers from drawdowns"""
    #    total_returns = returns.sum() - rf
    #    max_dd = max_drawdown()
    #    return abs(total_returns) / abs(max_dd)

    @to_dict
    @per_numeric_column
    def risk_return_ratio(self, series):
        """
        Calculates the return/risk ratio (Sharpe ratio w/o risk-free rate).
        """
        return (series.mean() / series.std()).cast(pl.Float64)

    # def kelly_criterion(self):
    #     """
    #     Calculates the recommended maximum amount of capital that
    #     should be allocated to the given strategy, based on the
    #     Kelly Criterion (http://en.wikipedia.org/wiki/Kelly_criterion)
    #     """
    #     win_loss_ratio = self.payoff_ratio()
    #     win_prob = self.win_rate()
    #     lose_prob = 1 - win_prob
    #
    #     return ((win_loss_ratio * win_prob) - lose_prob) / win_loss_ratio

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

    @to_dict
    @per_numeric_column
    def best(self, series):
        """Returns the maximum return per column (best period)."""
        return series.max()  # .alias(series.meta.output_name)

    @to_dict
    @per_numeric_column
    def worst(self, series):
        """Returns the minimum return per column (worst period)."""
        return series.min()  # .alias(series.meta.output_name)

    def exposure(self):
        """Returns the market exposure time (returns != 0)"""

        def _exposure(ret):
            ex = len(ret[(~np.isnan(ret)) & (ret != 0)]) / len(ret)
            return _ceil(ex * 100) / 100

        return self.all.apply(_exposure)

    @to_dict
    @per_numeric_column
    def sharpe(self, series, periods=252):
        """
        Calculates the Sharpe ratio of asset returns.
        """
        divisor = series.std(ddof=1)

        res = series.mean() / divisor
        factor = periods or 1
        return res * np.sqrt(factor)

    def rolling_sharpe(self, rolling_period=126, periods_per_year=252):
        res = self.all.rolling(rolling_period).mean() / self.all.rolling(rolling_period).std()
        factor = periods_per_year or 1
        return res * np.sqrt(factor)

    @to_dict
    @per_numeric_column
    def sortino(self, series: pl.Series, periods=252):
        """
        Calculates the Sortino ratio: mean return divided by downside deviation.
        Based on Red Rock Capital's Sortino ratio paper.
        """

        downside_deviation = np.sqrt(((series.filter(series < 0)) ** 2).mean())

        ratio = series.mean() / downside_deviation
        return ratio * np.sqrt(periods)

    def rolling_sortino(self, rolling_period=126, periods_per_year=252):
        downside = (
            self.all.rolling(rolling_period).apply(lambda x: (x.values[x.values < 0] ** 2).sum()) / rolling_period
        )

        res = self.all.rolling(rolling_period).mean() / np.sqrt(downside)
        factor = periods_per_year or 1
        return res * np.sqrt(factor)

    def adjusted_sortino(self, periods=252):
        """
        Jack Schwager's adjusted Sortino ratio for direct comparison to Sharpe.
        See: https://archive.is/wip/2rwFW
        """
        sortino_data = self.sortino(periods=periods)
        return {k: v / np.sqrt(2) for k, v in sortino_data.items()}

    def r_squared(self):
        """Measures the straight line fit of the equity curve"""
        # slope, intercept, r_val, p_val, std_err = _linregress(
        if self.data.benchmark is None:
            raise AttributeError("No benchmark data available")

        def f(returns):
            _, _, r_val, _, _ = linregress(returns, self.all["Benchmark"])
            return r_val**2

        return self.all.apply(f)

    def r2(self):
        """Shorthand for r_squared()"""
        return self.r_squared()

    @to_dict
    @per_numeric_column
    def information_ratio(self, series):
        """
        Calculates the information ratio
        (basically the risk return ratio of the net profits)
        """
        active = series - self.data.benchmark
        print(active)

        mean_active = active.mean()
        std_active = active.std()

        print(mean_active, std_active)

        returns = self.all  # [self.data.names]
        diff = returns.sub(self.data.benchmark, axis=0)

        return diff.mean() / returns.std()

    def greeks(self, periods=252.0):
        """Calculates alpha and beta of the portfolio"""
        # find covariance
        # if not isinstance(returns, pd.Series):
        #    returns = returns[returns.columns[0]]

        def f(returns, benchmark):
            matrix = np.cov(returns, benchmark)
            beta = matrix[0, 1] / matrix[1, 1]

            # calculates measures now
            alpha = returns.mean() - beta * benchmark.mean()
            alpha = alpha * periods

            return pd.Series(
                {
                    "beta": beta,
                    "alpha": alpha,
                    # "vol": _np.sqrt(matrix[0, 0]) * _np.sqrt(periods)
                }
            ).fillna(0)

        return self.all.apply(f, benchmark=self.all["Benchmark"])


@dataclasses.dataclass(frozen=True)
class _Plots:
    data: _Data

    _FLATUI_COLORS = [
        "#FEDD78",  # Yellow
        "#348DC1",  # Blue
        "#BA516B",  # Rose
        "#4FA487",  # Green
        "#9B59B6",  # Purple
        "#613F66",  # Dark Purple
        "#84B082",  # Light Green
        "#DC136C",  # Pink
        "#559CAD",  # Light Blue
        "#4A5899",  # Navy Blue
    ]

    @staticmethod
    def _get_colors():
        """
        Returns the default color palette and styling parameters for plots.

        Returns:
            tuple: A tuple containing:
                - colors (list): List of hex color codes
                - ls (str): Line style ("-" for solid)
                - alpha (float): Opacity value (0.8)
        """
        colors = _Plots._FLATUI_COLORS
        ls = "-"  # Line style
        alpha = 0.8  # Opacity
        return colors, ls, alpha

    @staticmethod
    def _compsum(returns):
        """Calculates rolling compounded returns"""
        return returns.add(1).cumprod(axis=0) - 1

    def plot_returns_bars(self):
        """
        Creates a bar chart of returns for each asset in the data.

        This function visualizes the returns of each asset as bars, making it easy
        to compare performance across different time periods.

        Args:
            data (_Data): A Data object containing returns data to plot.

        Returns:
            plotly.graph_objects.Figure: A Plotly figure object containing the bar chart.
                The figure shows returns for each asset with a horizontal line at y=0.

        Example:
            >>> from quantstats.api import _Data
            >>> import pandas as pd
            >>> returns = pd.DataFrame(...)
            >>> data = _Data(returns=returns)
            >>> fig = data.plots.plot_returns_bars()
            >>> fig.show()
        """
        # Get color palette
        colors, _, _ = _Plots._get_colors()

        # Create figure
        fig = go.Figure()

        # Add a bar trace for each asset
        for idx, col in enumerate(self.data.returns.columns):
            fig.add_trace(
                go.Bar(
                    x=self.data.index,
                    y=self.data.returns[col],
                    name=col,
                    marker_color=colors[idx % len(colors)],  # Cycle through colors if more assets than colors
                )
            )

        # Update layout for better readability
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(
                tickformat="%Y",  # Format x-axis as years
                showgrid=False,
            ),
            yaxis=dict(
                tickformat=".0%",  # Format y-axis as percentages
                showgrid=True,
                gridcolor="lightgray",
            ),
        )

        # Add horizontal line at y=0 to distinguish positive and negative returns
        fig.add_hline(y=0, line=dict(color="black", width=1, dash="dash"))

        return fig

    def plot_snapshot(self, title="Portfolio Summary", compounded=True, log_scale=False):
        """
        Creates a comprehensive dashboard with multiple plots for portfolio analysis.

        This function generates a three-panel plot showing:
        1. Cumulative returns over time
        2. Drawdowns over time
        3. Daily returns over time

        This provides a complete visual summary of portfolio performance.

        Args:
            data (_Data): A Data object containing returns data.
            title (str, optional): Title of the plot. Defaults to "Portfolio Summary".
            compounded (bool, optional): Whether to use compounded returns. Defaults to True.
            log_scale (bool, optional): Whether to use logarithmic scale for cumulative returns.
                Defaults to False.

        Returns:
            plotly.graph_objects.Figure: A Plotly figure object containing the dashboard.

        Example:
            >>> from quantstats.api import _Data
            >>> import pandas as pd
            >>> returns = pd.DataFrame(...)
            >>> data = _Data(returns=returns)
            >>> fig = snapshot_plotly(data, title="My Portfolio Performance")
            >>> fig.show()
        """
        # Calculate drawdowns
        dd = self.data.drawdown(compounded=compounded)

        # Create subplot structure
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,  # Share x-axis across all subplots
            row_heights=[0.5, 0.25, 0.25],  # Allocate more space to cumulative returns
            vertical_spacing=0.03,
            subplot_titles=["Cumulative Return", "Drawdown", "Daily Return"],
        )

        # Plot cumulative returns for each asset
        for col in self.returns.columns:
            cum_returns = 100 * ((1 + self.data.returns[col]).cumprod())  # Convert to percentage
            fig.add_trace(
                go.Scatter(
                    x=cum_returns.index,
                    y=cum_returns,
                    name=col,
                    mode="lines",
                ),
                row=1,
                col=1,
            )

        # Plot drawdowns for each asset
        for col in self.data.returns.columns:
            fig.add_trace(
                go.Scatter(
                    x=dd[col].index,
                    y=dd[col],
                    name=f"DD: {col}",
                    mode="lines",
                ),
                row=2,
                col=1,
            )

        # Plot daily returns for each asset
        for col in self.data.names:
            fig.add_trace(
                go.Scatter(
                    x=self.data.returns[col].index,
                    y=self.data.returns[col] * 100,  # Convert to percentage
                    name=f"{col} Return",
                    mode="lines",
                ),
                row=3,
                col=1,
            )

        # Configure layout
        fig.update_layout(
            height=800,  # Taller figure for better visibility
            title_text=title,
            showlegend=True,
            template="plotly_white",  # Clean white template
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        # Apply log scale to cumulative returns if requested
        if log_scale:
            fig.update_yaxes(type="log", row=1, col=1)

        # Format y-axes
        fig.update_yaxes(title="Cumulative Return (%)", row=1, col=1)
        fig.update_yaxes(title="Drawdown", tickformat=".1%", row=2, col=1)
        fig.update_yaxes(title="Daily Return (%)", row=3, col=1)

        return fig

    def monthly_heatmap(
        self,
        annot_size=13,
        cbar=True,
        returns_label="Strategy",
        compounded=True,
        eoy=False,
        fontname="Arial",
        ylabel=True,
    ):
        """
        Creates a heatmap of monthly returns by year.

        This function visualizes monthly returns in a calendar-like heatmap format,
        with years on the y-axis and months on the x-axis. Positive returns are shown
        in green, negative in red, with color intensity proportional to return magnitude.

        Args:
            data (_Data): A Data object containing returns data.
            annot_size (int, optional): Font size for annotations. Defaults to 13.
            cbar (bool, optional): Whether to show the color bar. Defaults to True.
            returns_label (str, optional): Label for the returns data. Defaults to "Strategy".
            compounded (bool, optional): Whether to use compounded returns. Defaults to True.
            eoy (bool, optional): Whether to include end-of-year summary. Defaults to False.
            fontname (str, optional): Font family to use. Defaults to "Arial".
            ylabel (bool, optional): Whether to show y-axis label. Defaults to True.

        Returns:
            plotly.graph_objects.Figure: A Plotly figure object containing the heatmap.

        Example:
            >>> from quantstats.api import _Data
            >>> import pandas as pd
            >>> returns = pd.DataFrame(...)
            >>> data = _Data(returns=returns)
            >>> fig = data.plots.monthly_heatmap(returns_label="My Portfolio")
            >>> fig.show()
        """
        # Define color map (Red-Yellow-Green)
        cmap = "RdYlGn"

        # Prepare monthly returns as percentage
        returns = self.data.resample(every="1m", compounded=compounded).returns * 100

        # Extract returns for the first asset
        returns = returns[self.data.returns.columns[0]]
        # returns.index = pd.to_datetime(returns.index)

        # Convert to DataFrame for manipulation
        # returns = returns.to_frame()

        # Add Year and Month columns
        returns["Year"] = returns.index.year
        returns["Month"] = returns.index.month

        # Create pivot table with years as rows and months as columns
        returns = returns.pivot(index="Year", columns="Month", values=self.data.names[0]).fillna(0)

        # Rename month numbers to month abbreviations
        returns = returns.rename(columns=lambda x: calendar.month_abbr[x])

        # Calculate color scale limits to ensure symmetry around zero
        zmin = -max(abs(returns.min().min()), abs(returns.max().max()))
        zmax = max(abs(returns.min().min()), abs(returns.max().max()))

        # Set index name and reverse order (most recent years on top)
        returns.index.name = "Year"
        returns.columns.name = None
        returns = returns.iloc[::-1]

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=returns.values,
                x=[col for col in returns.columns],  # Month abbreviations
                y=returns.index.astype(str),  # Years
                text=np.round(returns.values, 2),
                texttemplate="%{text:.2f}%",  # Annotate cells with return values
                colorscale=cmap,
                zmid=0,  # Center color scale at zero
                zmin=zmin,
                zmax=zmax,
                colorbar=dict(
                    title="Return (%)",
                    ticksuffix="%",
                    tickfont=dict(size=annot_size),
                )
                if cbar
                else None,
                hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%",
            )
        )

        # Configure layout
        fig.update_layout(
            title={
                "text": f"{returns_label} - Monthly Returns (%)",
                "y": 0.95,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
                "font": dict(family=fontname, size=16, color="black"),
            },
            xaxis=dict(
                title="",
                side="top",  # Show months at the top
                showgrid=False,
                tickfont=dict(family=fontname, size=annot_size),
            ),
            yaxis=dict(
                title="Years" if ylabel else "",
                autorange="reversed",  # Most recent years at the top
                showgrid=False,
                tickfont=dict(family=fontname, size=annot_size),
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=0, r=0, t=80, b=0),
        )

        return fig

    def plot_distribution(self, fontname="Arial", compounded=True, title=None):
        """
        Creates a box plot showing the distribution of returns across different time periods.

        This function visualizes the distribution of returns at daily, weekly, monthly,
        quarterly, and yearly frequencies, making it easy to compare volatility and
        central tendency across different time horizons.

        Args:
            data (_Data): A Data object containing returns data.
            fontname (str, optional): Font family to use. Defaults to "Arial".
            compounded (bool, optional): Whether to use compounded returns. Defaults to True.
            title (str, optional): Title of the plot. If None, defaults to "Return Quantiles".

        Returns:
            plotly.graph_objects.Figure: A Plotly figure object containing the distribution plot.

        Example:
            >>> from quantstats.api import _Data
            >>> import pandas as pd
            >>> returns = pd.DataFrame(...)
            >>> data = _Data(returns=returns)
            >>> fig = data.plots.plot_distribution(title="Portfolio Returns Distribution")
            >>> fig.show()
        """
        # Get color palette
        colors = _Plots._FLATUI_COLORS

        # Extract returns for the first asset and ensure DataFrame format
        port = pd.DataFrame(self.data.returns[self.data.returns.columns[0]]).fillna(0)
        port.columns = ["Daily"]

        # Define function to apply when resampling (compound or sum)
        apply_fnc = _Plots._compsum if compounded else np.sum

        # Resample returns to different frequencies
        port["Weekly"] = port["Daily"].resample("W-MON").apply(apply_fnc).ffill()
        port["Monthly"] = port["Daily"].resample("ME").apply(apply_fnc).ffill()
        port["Quarterly"] = port["Daily"].resample("QE").apply(apply_fnc).ffill()
        port["Yearly"] = port["Daily"].resample("YE").apply(apply_fnc).ffill()

        # Create figure
        fig = go.Figure()

        # Add box plots for each time frequency
        for i, col in enumerate(["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]):
            fig.add_trace(
                go.Box(
                    y=port[col],
                    name=col,
                    marker_color=colors[i],
                    boxmean="sd",  # Show mean and standard deviation
                )
            )

        # Set title
        if not title:
            title = "Return Quantiles"
        else:
            title = f"{title} - Return Quantiles"

        # Create date range string for subtitle
        date_range = f"{self.data.index.min():%d %b '%y} - {self.data.returns.index.max():%d %b '%y}"

        # Configure layout
        fig.update_layout(
            title={
                "text": f"<b>{title}</b><br><sub>{date_range}</sub>",
                "y": 0.9,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
            font=dict(family=fontname, size=14),
            yaxis_title="Returns (%)",
            yaxis_tickformat=".0%",
            boxmode="group",
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        return fig
