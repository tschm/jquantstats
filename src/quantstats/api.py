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
from collections.abc import Callable
from math import ceil as _ceil
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import linregress, norm


def build_data(returns: pd.DataFrame | pd.Series, rf=0.0, benchmark=None, nperiods=None) -> "_Data":
    """
    Build a Data object from returns and benchmark data.

    This function processes financial returns data and an optional benchmark,
    calculating excess returns by subtracting a risk-free rate. It ensures that
    returns and benchmark data are properly aligned by date.

    Args:
        returns (pd.DataFrame | pd.Series): Financial returns data. If a Series is provided,
            it will be converted to a DataFrame.
        rf (float | pd.Series, optional): Risk-free rate. Can be a constant float value
            or a Series with the same index as returns. Defaults to 0.0.
        benchmark (pd.Series, optional): Benchmark returns data. If provided, will be
            aligned with the 'returns' data. Defaults to None.
        nperiods (int, optional): Number of periods per year, used to deannualize
            the risk-free rate if it's an annual rate. Defaults to None.

    Returns:
        _Data: A Data object containing processed returns and benchmark data.

    Examples:
        >>> import pandas as pd
        >>> from quantstats.api import build_data
        >>>
        >>> # With a constant risk-free rate
        >>> data = build_data(returns=df, rf=0.02/252)
        >>>
        >>> # With a Series as risk-free rate
        >>> rf_series = pd.Series(index=df.index, data=0.001)
        >>> data = build_data(returns=df, rf=rf_series)
        >>>
        >>> # With a benchmark
        >>> data = build_data(returns=df, benchmark=benchmark_series)
    """

    def _calculate_excess_returns(data, rf=0.0, nperiods=None):
        """
        Calculate excess returns by subtracting the risk-free rate.

        Args:
            data (pd.DataFrame | pd.Series): Returns data.
            rf (float | pd.Series): Risk-free rate.
            nperiods (int, optional): Number of periods for deannualization.

        Returns:
            pd.DataFrame | pd.Series: Excess returns.
        """
        # If rf is a Series, filter it to match data's index
        if not isinstance(rf, float):
            rf = rf[rf.index.isin(data.index)]

        # Deannualize risk-free rate if nperiods is provided
        if nperiods is not None:
            rf = np.power(1 + rf, 1.0 / nperiods) - 1.0

        # Calculate excess returns and remove timezone info
        excess_returns = data - rf
        excess_returns = excess_returns.tz_localize(None)

        return excess_returns

    # Convert Series to DataFrame if necessary
    if isinstance(returns, pd.Series):
        returns_df = returns.copy().to_frame(name="returns")
    else:
        returns_df = returns.copy()

    # Process returns without benchmark
    if benchmark is None:
        return _Data(returns=_calculate_excess_returns(returns_df, rf, nperiods=nperiods))

    # Process returns with benchmark
    else:
        # Find common dates between returns and benchmark
        common_dates = sorted(list(set(returns_df.index) & set(benchmark.index)))

        # Calculate excess returns for both returns and benchmark
        excess_returns = _calculate_excess_returns(returns_df.loc[common_dates], rf, nperiods=nperiods)
        excess_benchmark = _calculate_excess_returns(benchmark.loc[common_dates], rf, nperiods=nperiods)

        return _Data(returns=excess_returns, benchmark=excess_benchmark)


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
        benchmark (pd.Series, optional): Series containing benchmark returns data with the same
                                        index as returns. Defaults to None.
    """

    returns: pd.DataFrame
    benchmark: pd.Series | None = None

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
        if self.benchmark is not None:
            pd.testing.assert_index_equal(self.benchmark.index, self.returns.index)

    def all(self) -> pd.DataFrame:
        """
        Combines returns and benchmark data into a single DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing all returns data and benchmark (if available),
                         with NaN values filled with 0.0.
        """
        # Copy to avoid mutating the original returns
        result = self.returns.copy()

        # Only add 'Benchmark' column if benchmark data is available
        if self.benchmark is not None:
            result["Benchmark"] = self.benchmark

        # Return the combined DataFrame with NaN values filled with 0.0
        return result.fillna(0.0)

    @property
    def index(self) -> pd.Index:
        """
        Returns the index of the returns DataFrame.

        Returns:
            pd.Index: The index of the returns DataFrame, typically dates.
        """
        return self.returns.index

    @property
    def names(self) -> pd.Index:
        """
        Returns the column names of the returns DataFrame.

        Returns:
            pd.Index: The column names of the returns DataFrame, typically asset names.
        """
        return self.returns.columns

    def prices(self, compounded: bool = False) -> pd.DataFrame:
        """
        Converts returns to prices.

        Args:
            compounded (bool, optional): If True, uses compounded returns (cumprod).
                                        If False, uses simple returns (cumsum).
                                        Defaults to False.

        Returns:
            pd.DataFrame: A DataFrame containing price data derived from returns.
        """
        if compounded:
            return self.all().fillna(0.0).add(1).cumprod(axis=0)
        else:
            return self.all().fillna(0.0).cumsum()

    def resample(self, resample: str = "YE", compounded: bool = False) -> "_Data":
        """
        Resamples returns data to a different frequency.

        Args:
            resample (str, optional): Pandas resample rule (e.g., 'YE' for year-end,
                                     'ME' for month-end). Defaults to "YE".
            compounded (bool, optional): If True, compounds returns when resampling.
                                        If False, sums returns. Defaults to False.

        Returns:
            _Data: A new Data object containing the resampled returns.
        """

        def comp(x: pd.Series) -> float:
            """Compounds returns: (1 + r1) * (1 + r2) * ... * (1 + rn) - 1"""
            return (1 + x).prod() - 1.0

        frame = self.all().fillna(0.0)

        if compounded:
            frame = frame.resample(resample).apply(comp)
        else:
            frame = frame.resample(resample).sum()

        return _Data(returns=frame[self.names], benchmark=frame.get("Benchmark", None))

    def apply(self, fct: Callable, **kwargs: Any) -> Any:
        """
        Applies a function to the returns DataFrame.

        Args:
            fct (Callable): Function to apply to the returns DataFrame.
            **kwargs: Additional keyword arguments to pass to the function.

        Returns:
            Any: The result of applying the function to the returns DataFrame.
        """
        return fct(self.returns, **kwargs)

    def copy(self) -> "_Data":
        """
        Creates a deep copy of the Data object.

        Returns:
            _Data: A new Data object with copies of the returns and benchmark.
        """
        try:
            return _Data(returns=self.returns.copy(), benchmark=self.benchmark.copy())
        except AttributeError:
            # Handle case where benchmark is None
            return _Data(returns=self.returns.copy())

    def highwater_mark(self, compounded: bool = False) -> pd.DataFrame:
        """
        Calculates the running maximum (high-water mark) of prices.

        Args:
            compounded (bool, optional): If True, uses compounded returns to calculate prices.
                                        If False, uses simple returns. Defaults to False.

        Returns:
            pd.DataFrame: A DataFrame containing the high-water mark for each asset.
        """
        prices = self.prices(compounded)
        return prices.cummax()

    def head(self, n: int = 5) -> "_Data":
        """
        Returns the first n rows of the combined returns and benchmark data.

        Args:
            n (int, optional): Number of rows to return. Defaults to 5.

        Returns:
            _Data: A new Data object containing the first n rows of the combined data.
        """
        all_data = self.all().head(n=n)
        return _Data(returns=all_data[self.names], benchmark=all_data.get("Benchmark", None))

    def tail(self, n: int = 5) -> "_Data":
        """
        Returns the last n rows of the combined returns and benchmark data.

        Args:
            n (int, optional): Number of rows to return. Defaults to 5.

        Returns:
            _Data: A new Data object containing the last n rows of the combined data.
        """
        all_data = self.all().tail(n=n)
        return _Data(returns=all_data[self.names], benchmark=all_data.get("Benchmark", None))

    def drawdown(self, compounded: bool = False) -> pd.DataFrame:
        """
        Calculates drawdowns from prices.

        Args:
            compounded (bool, optional): If True, calculates drawdowns as percentage change
                                        from high-water mark. If False, calculates drawdowns
                                        as absolute difference from high-water mark.
                                        Defaults to False.

        Returns:
            pd.DataFrame: A DataFrame containing drawdowns for each asset.
        """
        prices = self.prices(compounded)

        if compounded:
            # Percentage drawdown: (current_price / peak_price) - 1
            drawdown = prices / prices.cummax() - 1.0
        else:
            # Absolute drawdown: peak_price - current_price
            drawdown = prices.cummax() - prices

        return drawdown


@dataclasses.dataclass  # (frozen=True)
class _Stats:
    data: _Data
    all: pd.DataFrame = None

    def __post_init__(self):
        self.all = self.data.all()

    def skew(self):
        """
        Calculates returns' skewness
        (the degree of asymmetry of a distribution around its mean)
        """
        return self.all.skew()

    def kurtosis(self):
        """
        Calculates returns' kurtosis
        (the degree to which a distribution peak compared to a normal distribution)
        """
        return self.all.kurtosis()

    def avg_return(self):
        """Calculates the average return/trade return for a period"""
        return self.all[self.all != 0].dropna().mean()

    def avg_win(self):
        """
        Calculates the average winning
        return/trade return for a period
        """
        return self.all[self.all > 0].dropna().mean()

    def avg_loss(self):
        """
        Calculates the average low if
        return/trade return for a period
        """
        return self.all[self.all < 0].dropna().mean()

    def volatility(self, periods=252, annualize=True):
        """Calculates the volatility of returns for a period"""
        std = self.all.std()
        factor = np.sqrt(periods) if annualize else 1
        return std * factor

    def rolling_volatility(self, rolling_period=126, periods_per_year=252):
        return self.all.rolling(rolling_period).std() * np.sqrt(periods_per_year)

    def log_returns(self):
        """Shorthand for to_log_returns"""
        return self.to_log_returns()

    def to_log_returns(self):
        """Converts returns series to log returns"""
        return np.log(self.all + 1).replace([np.inf, np.inf], float("NaN"))

    def implied_volatility(self, periods=252):
        """Calculates the implied volatility of returns for a period"""
        logret = self.log_returns()
        factor = periods or 1
        return logret.std() * np.sqrt(factor)

    def autocorr_penalty(self):
        """Metric to account for auto correlation"""

        # if isinstance(returns, pd.DataFrame):
        #    returns = returns[returns.columns[0]]
        def f(series):
            num = len(series)
            coef = np.abs(np.corrcoef(series[:-1], series[1:])[0, 1])
            corr = [((num - x) / num) * coef**x for x in range(1, num)]
            return np.sqrt(1 + 2 * np.sum(corr))

        return self.all.apply(f)

    def payoff_ratio(self):
        """Measures the payoff ratio (average win/average loss)"""
        return self.avg_win() / abs(self.avg_loss())

    def win_loss_ratio(self):
        """Shorthand for payoff_ratio()"""
        return self.payoff_ratio()

    def profit_ratio(self):
        """Measures the profit ratio (win ratio / loss ratio)"""
        wins = self.all[self.all >= 0]
        loss = self.all[self.all < 0]

        win_ratio = abs(wins.mean() / wins.count())
        loss_ratio = abs(loss.mean() / loss.count())
        return win_ratio / loss_ratio

    def profit_factor(self):
        """Measures the profit ratio (wins/loss)"""
        return abs(self.all[self.all >= 0].sum() / self.all[self.all < 0].sum())

    def cpc_index(self):
        """
        Measures the cpc ratio
        (profit factor * win % * win loss ratio)
        """
        return self.profit_factor() * self.win_rate() * self.win_loss_ratio()

    def common_sense_ratio(self):
        """Measures the common sense ratio (profit factor * tail ratio)"""
        return self.profit_factor() * self.tail_ratio()

    def value_at_risk(self, confidence=0.95):
        """
        Calculates the daily value-at-risk
        (variance-covariance calculation with confidence n)
        """

        def f(series):
            mu = series.mean()
            sigma = series.std()
            return norm.ppf(1 - confidence, mu, sigma)

        return self.all.apply(f)

        # return _norm.ppf(1 - confidence, mu, sigma)

    def var(self, confidence=0.95):
        """Shorthand for value_at_risk()"""
        return self.value_at_risk(confidence)

    def conditional_value_at_risk(self, confidence=0.95):
        """
        Calculats the conditional daily value-at-risk (aka expected shortfall)
        quantifies the amount of tail risk an investment
        """
        var = self.value_at_risk(confidence)
        c_var = self.all[self.all < var].mean()
        return c_var  # if ~np.isnan(c_var) else var

    def cvar(self, confidence=0.95):
        """Shorthand for conditional_value_at_risk()"""
        return self.conditional_value_at_risk(confidence)

    def expected_shortfall(self, confidence=0.95):
        """Shorthand for conditional_value_at_risk()"""
        return self.conditional_value_at_risk(confidence)

    def tail_ratio(self, cutoff=0.95):
        """
        Measures the ratio between the right
        (95%) and left tail (5%).
        """
        return abs(self.all.quantile(cutoff) / self.all.quantile(1 - cutoff))

    def win_rate(self):
        """Calculates the win ratio for a period"""

        def _win_rate(series):
            try:
                return len(series[series > 0]) / len(series[series != 0])
            except ZeroDivisionError:
                return np.nan

        return self.all.apply(_win_rate)

    def gain_to_pain_ratio(self):
        """
        Jack Schwager's GPR. See here for more info:
        https://archive.is/wip/2rwFW
        """
        # returns = returns.resample(resolution).sum()
        downside = abs(self.all[self.all < 0].sum())
        return self.all.sum() / downside

    # def risk_of_ruin(self):
    #     """
    #     Calculates the risk of ruin
    #     (the likelihood of losing all one's investment capital)
    #     """
    #     wins = self.win_rate()
    #     return ((1 - wins) / (1 + wins)) ** len(returns)
    #
    #
    # def ror(self):
    #     """Shorthand for risk_of_ruin()"""
    #     return self.risk_of_ruin()
    #

    def outlier_win_ratio(self, quantile=0.99):
        """
        Calculates the outlier winners ratio
        99th percentile of returns / mean positive return
        """
        return self.all.quantile(quantile).mean() / self.all[self.all >= 0].mean()

    def outlier_loss_ratio(self, quantile=0.01):
        """
        Calculates the outlier losers ratio
        1st percentile of returns / mean negative return
        """
        return self.all.quantile(quantile).mean() / self.all[self.all < 0].mean()

    # def recovery_factor(self, rf=0.0):
    #    """Measures how fast the strategy recovers from drawdowns"""
    #    total_returns = returns.sum() - rf
    #    max_dd = max_drawdown()
    #    return abs(total_returns) / abs(max_dd)

    def risk_return_ratio(self):
        """
        Calculates the return / risk ratio
        (sharpe ratio without factoring in the risk-free rate)
        """
        return self.all.mean() / self.all.std()

    # def max_drawdown(self):
    #    """Calculates the maximum drawdown"""
    #    return prices / prices.cummax().min() - 1

    def kelly_criterion(self):
        """
        Calculates the recommended maximum amount of capital that
        should be allocated to the given strategy, based on the
        Kelly Criterion (http://en.wikipedia.org/wiki/Kelly_criterion)
        """
        win_loss_ratio = self.payoff_ratio()
        win_prob = self.win_rate()
        lose_prob = 1 - win_prob

        return ((win_loss_ratio * win_prob) - lose_prob) / win_loss_ratio

    def expected_return(self):
        """
        Returns the expected return for a given period
        by calculating the geometric holding period return
        """

        def f(series):
            return np.prod(1 + series) ** (1 / len(series)) - 1

        return self.all.apply(f)

        # return np.prod(1 + returns, axis=0) ** (1 / len(returns)) - 1

    def geometric_mean(self):
        """Shorthand for expected_return()"""
        return self.expected_return()

    def ghpr(self):
        """Shorthand for expected_return()"""
        return self.expected_return()

    def outliers(self, quantile=0.95):
        """Returns a frame of outliers"""
        return self.all[self.all > self.all.quantile(quantile)].dropna(how="all")

    #
    # def remove_outliers(returns, quantile=0.95):
    #     """Returns series of returns without the outliers"""
    #     return returns[returns < returns.quantile(quantile)]

    def best(self):
        """Returns the best day/month/week/quarter/year's return"""
        return self.all.max()

    def worst(self):
        """Returns the worst day/month/week/quarter/year's return"""
        return self.all.min()

    def exposure(self):
        """Returns the market exposure time (returns != 0)"""

        def _exposure(ret):
            ex = len(ret[(~np.isnan(ret)) & (ret != 0)]) / len(ret)
            return _ceil(ex * 100) / 100

        return self.all.apply(_exposure)

    def sharpe(self, periods=252, smart=False):
        """
        Calculates the sharpe ratio of access returns

        If rf is non-zero, you must specify periods.
        In this case, rf is assumed to be expressed in yearly (annualized) terms

        Args:
            * periods (int): Freq. of returns (252/365 for daily, 12 for monthly)
            * annualize: return annualize sharpe?
            * smart: return smart sharpe ratio
        """
        divisor = self.all.std(ddof=1)
        if smart:
            # penalize sharpe with auto correlation
            divisor = divisor * self.autocorr_penalty()
        res = self.all.mean() / divisor
        factor = periods or 1

        return res * np.sqrt(factor)

    def rolling_sharpe(self, rolling_period=126, periods_per_year=252):
        res = self.all.rolling(rolling_period).mean() / self.all.rolling(rolling_period).std()
        factor = periods_per_year or 1
        return res * np.sqrt(factor)

    def sortino(self, periods=252, smart=False):
        """
        Calculates the sortino ratio of access returns

        If rf is non-zero, you must specify periods.
        In this case, rf is assumed to be expressed in yearly (annualized) terms

        Calculation is based on this paper by Red Rock Capital
        http://www.redrockcapital.com/Sortino__A__Sharper__Ratio_Red_Rock_Capital.pdf
        """

        def f(series):
            return np.sqrt((series[series < 0] ** 2).sum() / len(series.dropna()))

        downside = self.all.apply(f)

        #    np.sqrt((self.all[self.all < 0] ** 2).sum() / len(returns)))

        if smart:
            # penalize sortino with auto correlation
            downside = downside * self.autocorr_penalty()

        res = self.all.mean() / downside
        factor = periods or 1
        return res * np.sqrt(factor)

    def rolling_sortino(self, rolling_period=126, periods_per_year=252):
        downside = (
            self.all.rolling(rolling_period).apply(lambda x: (x.values[x.values < 0] ** 2).sum()) / rolling_period
        )

        res = self.all.rolling(rolling_period).mean() / np.sqrt(downside)
        factor = periods_per_year or 1
        return res * np.sqrt(factor)

    def adjusted_sortino(self, periods=252, smart=False):
        """
        Jack Schwager's version of the Sortino ratio allows for
        direct comparisons to the Sharpe. See here for more info:
        https://archive.is/wip/2rwFW
        """
        data = self.sortino(periods=periods, smart=smart)
        return data / np.sqrt(2)

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

    def information_ratio(self):
        """
        Calculates the information ratio
        (basically the risk return ratio of the net profits)
        """
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
        for idx, col in enumerate(self.data.names):
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
        for col in self.data.names:
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
        for col in self.data.names:
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
        returns = self.data.resample(resample="ME", compounded=compounded).returns * 100

        # Extract returns for the first asset
        returns = returns[self.data.names[0]]
        returns.index = pd.to_datetime(returns.index)

        # Convert to DataFrame for manipulation
        returns = returns.to_frame()

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
        port = pd.DataFrame(self.data.returns[self.data.names[0]]).fillna(0)
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
