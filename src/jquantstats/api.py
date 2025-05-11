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

import dataclasses

import polars as pl

from ._plots import Plots
from ._stats import Stats


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
    index: pl.DataFrame | None = None

    @property
    def plots(self):
        return Plots(self)

    @property
    def stats(self):
        return Stats(self)

    @property
    def date_col(self):
        return self.index.columns

    @property
    def assets(self):
        try:
            return self.returns.columns + self.benchmark.columns
        except AttributeError:
            return self.returns.columns

    def __post_init__(self) -> None:
        """
        Validates that the benchmark index matches the returns index if benchmark is provided.

        Raises:
            AssertionError: If benchmark is provided and its index doesn't match returns index.
        """

    @property
    def numeric_columns(self):
        numeric_dtypes = {pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.UInt32, pl.UInt64}
        return [name for name, dtype in self.all.schema.items() if dtype in numeric_dtypes]

    @property
    def all(self) -> pl.DataFrame:
        """
        Combines returns and benchmark data into a single DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing all returns data and benchmark (if available),
                         with NaN values filled with 0.0.
        """
        if self.benchmark is None:
            return pl.concat([self.index, self.returns], how="horizontal")
        else:
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
