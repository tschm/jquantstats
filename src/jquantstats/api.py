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
jQuantStats API module.

This module provides the core API for the jQuantStats library, including the Data class
for handling financial returns data and benchmarks.

Overview
--------
The main entry point is the `build_data` function, which creates a Data object from
returns and optional benchmark data. The Data class provides methods for analyzing and
manipulating financial returns data, including accessing statistical metrics through
the `stats` property and visualization through the `plots` property.

Features
--------
- Support for both pandas and polars DataFrames as input
- Automatic conversion to polars for efficient data processing
- Handling of risk-free rate adjustments
- Benchmark comparison capabilities
- Date alignment between returns and benchmark data

Example
-------
```python
import polars as pl
from jquantstats.api import build_data

# Create a Data object from returns
returns = pl.DataFrame({
    "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
    "Asset1": [0.01, -0.02, 0.03]
}).with_columns(pl.col("Date").str.to_date())

data = build_data(returns=returns)

# With benchmark and risk-free rate
benchmark = pl.DataFrame({
    "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
    "Market": [0.005, -0.01, 0.02]
}).with_columns(pl.col("Date").str.to_date())

data = build_data(
    returns=returns,
    benchmark=benchmark,
    rf=0.0002,  # risk-free rate (e.g., 0.02% per day)
)
```
"""

import pandas as pd
import polars as pl

from ._data import Data


def build_data(
    returns: pl.DataFrame | pd.DataFrame | pd.Series,
    rf: float | pl.DataFrame | pd.DataFrame | pd.Series = 0.0,
    benchmark: pl.DataFrame | pd.DataFrame | pd.Series = None,
    date_col: str = "Date",
) -> Data:
    """
    Build a Data object from returns and optional benchmark using Polars.

    This function is the main entry point for creating a Data object, which is the core
    container for financial returns data in jQuantStats.

    Description
    -----------
    The `build_data` function handles the conversion of pandas DataFrames and Series to
    polars DataFrames, aligns dates between returns and benchmark data, and subtracts
    the risk-free rate to calculate excess returns.

    Parameters
    ----------
    returns : pl.DataFrame | pd.DataFrame | pd.Series
        Financial returns data.

        - If pl.DataFrame: First column should be the date column, remaining columns are asset returns.
        - If pd.DataFrame: Index can be dates (will be included) or a date column should be present.
        - If pd.Series: Index should be dates, values are returns for a single asset.

    rf : float | pl.DataFrame | pd.DataFrame | pd.Series, optional
        Risk-free rate. Default is 0.0 (no risk-free rate adjustment).

        - If float: Constant risk-free rate applied to all dates.
        - If DataFrame/Series: Time-varying risk-free rate with dates matching returns.

    benchmark : pl.DataFrame | pd.DataFrame | pd.Series, optional
        Benchmark returns. Default is None (no benchmark).

        - If pl.DataFrame: First column should be the date column, remaining columns are benchmark returns.
        - If pd.DataFrame: Index can be dates (will be included) or a date column should be present.
        - If pd.Series: Index should be dates, values are returns for a single benchmark.

    date_col : str, optional
        Name of the date column in the DataFrames. Default is "Date".

    Returns
    -------
    Data
        Object containing excess returns and benchmark (if any), with methods for
        analysis and visualization through the `stats` and `plots` properties.

    Raises
    ------
    ValueError
        If there are no overlapping dates between returns and benchmark.

    Examples
    --------
    Basic usage with polars DataFrame:

    ```python
    import polars as pl
    from jquantstats.api import build_data

    returns = pl.DataFrame({
        "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "Asset1": [0.01, -0.02, 0.03],
        "Asset2": [0.02, 0.01, -0.01]
    }).with_columns(pl.col("Date").str.to_date())

    data = build_data(returns=returns)
    ```

    With pandas DataFrame:

    ```python
    import pandas as pd
    from jquantstats.api import build_data

    returns_pd = pd.DataFrame({
        "Date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
        "Asset1": [0.01, -0.02, 0.03],
        "Asset2": [0.02, 0.01, -0.01]
    })

    data = build_data(returns=returns_pd)
    ```

    With benchmark and risk-free rate:

    ```python
    benchmark = pl.DataFrame({
        "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "Market": [0.005, -0.01, 0.02]
    }).with_columns(pl.col("Date").str.to_date())

    data = build_data(returns=returns, benchmark=benchmark, rf=0.0002)
    ```
    """

    def subtract_risk_free(df: pl.DataFrame, rf: float | pl.DataFrame, date_col: str) -> pl.DataFrame:
        """
        Subtract the risk-free rate from all numeric columns in the DataFrame.

        Description
        -----------
        This function handles both scalar risk-free rates and time series risk-free rates.
        For scalar rates, it creates a constant column with the risk-free rate value.
        For time series, it joins the risk-free rate DataFrame with the returns DataFrame
        on the date column and then subtracts the risk-free rate from each numeric column.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame containing returns data with a date column
            and one or more numeric columns representing asset returns.

        rf : float | pl.DataFrame
            Risk-free rate to subtract from returns.

            - If float: A constant risk-free rate applied to all dates.
            - If pl.DataFrame: A DataFrame with a date column and a second column
              containing time-varying risk-free rates.

        date_col : str
            Name of the date column in both DataFrames for joining
            when rf is a DataFrame.

        Returns
        -------
        pl.DataFrame
            DataFrame with the risk-free rate subtracted from all numeric columns,
            preserving the original column names. The resulting DataFrame includes the
            date column and all numeric columns from the input DataFrame.

        Notes
        -----
        - The function performs an inner join when rf is a DataFrame, which means
          only dates present in both DataFrames will be included in the result.
        - Only columns with numeric data types will have the risk-free rate subtracted.
        - The date column and any non-numeric columns are preserved in the output.
        """
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

    if isinstance(returns, pd.Series):
        returns = pl.from_pandas(returns.to_frame(), include_index=True)

    if isinstance(returns, pd.DataFrame):
        returns = pl.from_pandas(returns, include_index=True)

    if isinstance(rf, pd.Series):
        rf = pl.from_pandas(rf.to_frame(), include_index=True)

    if isinstance(rf, pd.DataFrame):
        rf = pl.from_pandas(rf, include_index=True)

    if isinstance(benchmark, pd.Series):
        benchmark = pl.from_pandas(benchmark.to_frame(), include_index=True)

    if isinstance(benchmark, pd.DataFrame):
        benchmark = pl.from_pandas(benchmark, include_index=True)

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
