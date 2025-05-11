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

import polars as pl

from ._data import Data


def build_data(
    returns: pl.DataFrame, rf: float | pl.DataFrame = 0.0, benchmark: pl.DataFrame = None, date_col: str = "Date"
) -> "Data":
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

    return Data(returns=excess_returns, benchmark=excess_benchmark, index=index)
