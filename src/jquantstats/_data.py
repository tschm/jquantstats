import dataclasses
from datetime import timedelta

import polars as pl

from ._plots import Plots
from ._stats import Stats


@dataclasses.dataclass(frozen=True)
class Data:
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
    index: pl.DataFrame
    benchmark: pl.DataFrame | None = None

    def __post_init__(self):
        # You need at least two points
        print(self.index.shape)
        if self.index.shape[0] < 2:
            raise ValueError("Index must contain at least two timestamps.")

        # Check index is monotonically increasing
        datetime_col = self.index[self.index.columns[0]]
        if not datetime_col.is_sorted():
            raise ValueError("Index must be monotonically increasing.")

        # Check row count matches returns
        if self.returns.shape[0] != self.index.shape[0]:
            raise ValueError("Returns and index must have the same number of rows.")

        # Check row count matches benchmark (if provided)
        if self.benchmark is not None and self.benchmark.shape[0] != self.index.shape[0]:
            raise ValueError("Benchmark and index must have the same number of rows.")

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

    def resample(self, every: str = "1mo", compounded: bool = False) -> "Data":
        """
        Resamples returns and benchmark to a different frequency using Polars.

        Args:
            every (str, optional): Resampling frequency (e.g., '1mo', '1y'). Defaults to '1mo'.
            compounded (bool, optional): Whether to compound returns. Defaults to False.

        Returns:
            Data: Resampled data.
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

        return Data(
            returns=resampled_returns.drop(self.index.columns[0]),
            benchmark=resampled_benchmark.drop(self.index.columns[0]) if resampled_benchmark is not None else None,
            index=resampled_index,
        )

    def copy(self) -> "Data":
        """
        Creates a deep copy of the Data object.

        Returns:
            Data: A new Data object with copies of the returns and benchmark.
        """
        try:
            return Data(returns=self.returns.clone(), benchmark=self.benchmark.clone(), index=self.index.clone())
        except AttributeError:
            # Handle case where benchmark is None
            return Data(returns=self.returns.clone(), index=self.index.clone())

    def head(self, n: int = 5) -> "Data":
        """
        Returns the first n rows of the combined returns and benchmark data.

        Args:
            n (int, optional): Number of rows to return. Defaults to 5.

        Returns:
            Data: A new Data object containing the first n rows of the combined data.
        """
        return Data(returns=self.returns.head(n), benchmark=self.benchmark.head(n), index=self.index.head(n))

    def tail(self, n: int = 5) -> "Data":
        """
        Returns the last n rows of the combined returns and benchmark data.

        Args:
            n (int, optional): Number of rows to return. Defaults to 5.

        Returns:
            Data: A new Data object containing the last n rows of the combined data.
        """
        return Data(returns=self.returns.tail(n), benchmark=self.benchmark.tail(n), index=self.index.tail(n))

    @property
    def _periods_per_year(self) -> int:
        """
        Estimate the number of periods per year based on average frequency in the index.
        Assumes `self.index` is a Polars DataFrame with a single datetime column.
        """
        # Extract the datetime column (assuming only one)
        datetime_col = self.index[self.index.columns[0]]

        # Ensure it's sorted
        sorted_dt = datetime_col.sort()

        # Compute differences
        diffs = sorted_dt.diff().drop_nulls()

        # Mean difference (Duration)
        mean_diff = diffs.mean()

        # if mean_diff is None:
        #    raise ValueError("Cannot compute mean frequency: result is None.")

        # Convert Duration (timedelta) to seconds
        seconds = mean_diff.total_seconds() if isinstance(mean_diff, timedelta) else mean_diff / timedelta(seconds=1)

        periods_per_year = round((365 * 24 * 60 * 60) / seconds)
        return int(periods_per_year)
