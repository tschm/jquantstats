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

This module provides the core API for the QuantStats library, including the _Data class
for handling financial returns data and benchmarks. The main entry point is the build_data
function, which creates a _Data object from returns and optional benchmark data.

The _Data class provides methods for analyzing and manipulating financial returns data,
including accessing statistical metrics through the stats property and visualization
through the plots property.
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
    to different time periods. It also provides access to statistical metrics through
    the stats property and visualization through the plots property.

    Attributes:
        returns (pl.DataFrame): DataFrame containing returns data with assets as columns.
        benchmark (pl.DataFrame, optional): DataFrame containing benchmark returns data.
                                           Defaults to None.
        index (pl.DataFrame): DataFrame containing the date index for the returns data.
    """

    returns: pl.DataFrame
    benchmark: pl.DataFrame | None = None
    index: pl.DataFrame | None = None

    @property
    def plots(self) -> "Plots":
        """
        Provides access to visualization methods for the financial data.

        Returns:
            Plots: An instance of the Plots class initialized with this data.
        """
        return Plots(self)

    @property
    def stats(self) -> "Stats":
        """
        Provides access to statistical analysis methods for the financial data.

        Returns:
            Stats: An instance of the Stats class initialized with this data.
        """
        return Stats(self)

    @property
    def date_col(self) -> list[str]:
        """
        Returns the column names of the index DataFrame.

        Returns:
            list[str]: List of column names in the index DataFrame, typically containing
                      the date column name.
        """
        return self.index.columns

    @property
    def assets(self) -> list[str]:
        """
        Returns the combined list of asset column names from returns and benchmark.

        Returns:
            list[str]: List of all asset column names from both returns and benchmark
                      (if available).
        """
        try:
            return self.returns.columns + self.benchmark.columns
        except AttributeError:
            return self.returns.columns

    def __post_init__(self) -> None:
        """
        Post-initialization hook for the dataclass.

        This method is automatically called after the object is initialized.
        In the current implementation, this method is a placeholder for potential
        validation logic, such as ensuring that benchmark and returns have matching indices.

        Note:
            The method body is currently empty, but the method signature is maintained
            for future implementation of validation logic.
        """

    @property
    def all(self) -> pl.DataFrame:
        """
        Combines index, returns, and benchmark data into a single DataFrame.

        This property provides a convenient way to access all data in a single DataFrame,
        which is useful for analysis and visualization.

        Returns:
            pl.DataFrame: A DataFrame containing the index, all returns data, and benchmark data
                         (if available) combined horizontally.
        """
        if self.benchmark is None:
            return pl.concat([self.index, self.returns], how="horizontal")
        else:
            return pl.concat([self.index, self.returns, self.benchmark], how="horizontal")

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
            df = self.index.hstack(df)  # Add the date column for resampling

            return df.group_by_dynamic(
                index_column=self.index.columns[0], every=every, period=every, closed="right", label="right"
            ).agg(
                [
                    pl.col(col).sum().alias(col) if not compounded else ((pl.col(col) + 1.0).product() - 1.0).alias(col)
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
