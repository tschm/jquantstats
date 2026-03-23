"""Financial returns data container and manipulation utilities."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from datetime import timedelta
from typing import TYPE_CHECKING, cast

import polars as pl

if TYPE_CHECKING:
    from ._plots import Plots
    from ._reports import Reports
    from ._stats import Stats


@dataclasses.dataclass(frozen=True)
class Data:
    """A container for financial returns data and an optional benchmark.

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

    def __post_init__(self) -> None:
        # You need at least two points
        if self.index.shape[0] < 2:
            raise ValueError("Index must contain at least two timestamps.")  # noqa: TRY003

        # Check index is monotonically increasing
        datetime_col = self.index[self.index.columns[0]]
        if not datetime_col.is_sorted():
            raise ValueError("Index must be monotonically increasing.")  # noqa: TRY003

        # Check row count matches returns
        if self.returns.shape[0] != self.index.shape[0]:
            raise ValueError("Returns and index must have the same number of rows.")  # noqa: TRY003

        # Check row count matches benchmark (if provided)
        if self.benchmark is not None and self.benchmark.shape[0] != self.index.shape[0]:
            raise ValueError("Benchmark and index must have the same number of rows.")  # noqa: TRY003

    def __repr__(self) -> str:
        """Return a string representation of the Data object."""
        return f"Data(assets={self.assets}, rows={len(self.index)})"

    @property
    def plots(self) -> Plots:
        """Provides access to visualization methods for the financial data.

        Returns:
            Plots: An instance of the Plots class initialized with this data.

        """
        from ._plots import Plots

        return Plots(self)

    @property
    def stats(self) -> Stats:
        """Provides access to statistical analysis methods for the financial data.

        Returns:
            Stats: An instance of the Stats class initialized with this data.

        """
        from ._stats import Stats

        return Stats(self)

    @property
    def reports(self) -> Reports:
        """Provides access to reporting methods for the financial data.

        Returns:
            Reports: An instance of the Reports class initialized with this data.

        """
        from ._reports import Reports

        return Reports(self)

    @property
    def date_col(self) -> list[str]:
        """Return the column names of the index DataFrame.

        Returns:
            list[str]: List of column names in the index DataFrame, typically containing
                      the date column name.

        """
        return list(self.index.columns)

    @property
    def assets(self) -> list[str]:
        """Return the combined list of asset column names from returns and benchmark.

        Returns:
            list[str]: List of all asset column names from both returns and benchmark
                      (if available).

        """
        if self.benchmark is not None:
            return list(self.returns.columns) + list(self.benchmark.columns)
        return list(self.returns.columns)

    @property
    def all(self) -> pl.DataFrame:
        """Combine index, returns, and benchmark data into a single DataFrame.

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

    def resample(self, every: str = "1mo") -> Data:
        """Resamples returns and benchmark to a different frequency using Polars.

        Args:
            every (str, optional): Resampling frequency (e.g., '1mo', '1y'). Defaults to '1mo'.

        Returns:
            Data: Resampled data.

        """

        def resample_frame(dframe: pl.DataFrame) -> pl.DataFrame:
            """Resample a single DataFrame to the target frequency using compound returns."""
            dframe = self.index.hstack(dframe)  # Add the date column for resampling

            return dframe.group_by_dynamic(
                index_column=self.index.columns[0], every=every, period=every, closed="right", label="right"
            ).agg(
                [
                    ((pl.col(col) + 1.0).product() - 1.0).alias(col)
                    for col in dframe.columns
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

    def copy(self) -> Data:
        """Create a deep copy of the Data object.

        Returns:
            Data: A new Data object with copies of the returns and benchmark.

        """
        if self.benchmark is not None:
            return Data(returns=self.returns.clone(), benchmark=self.benchmark.clone(), index=self.index.clone())
        return Data(returns=self.returns.clone(), index=self.index.clone())

    def head(self, n: int = 5) -> Data:
        """Return the first n rows of the combined returns and benchmark data.

        Args:
            n (int, optional): Number of rows to return. Defaults to 5.

        Returns:
            Data: A new Data object containing the first n rows of the combined data.

        """
        benchmark_head = self.benchmark.head(n) if self.benchmark is not None else None
        return Data(returns=self.returns.head(n), benchmark=benchmark_head, index=self.index.head(n))

    def tail(self, n: int = 5) -> Data:
        """Return the last n rows of the combined returns and benchmark data.

        Args:
            n (int, optional): Number of rows to return. Defaults to 5.

        Returns:
            Data: A new Data object containing the last n rows of the combined data.

        """
        benchmark_tail = self.benchmark.tail(n) if self.benchmark is not None else None
        return Data(returns=self.returns.tail(n), benchmark=benchmark_tail, index=self.index.tail(n))

    def truncate(self, start: object = None, end: object = None) -> Data:
        """Return a new Data object truncated to the inclusive [start, end] range.

        When the index is temporal (Date/Datetime), truncation is performed by
        comparing the date column against ``start`` and ``end`` values.

        When the index is integer-based, row slicing is used instead, and
        ``start`` and ``end`` must be non-negative integers.  Passing
        non-integer bounds to an integer-indexed Data raises :exc:`TypeError`.

        Args:
            start: Optional lower bound (inclusive).  A date/datetime value
                when the index is temporal; a non-negative :class:`int` row
                index when the data has no temporal index.
            end: Optional upper bound (inclusive).  Same type rules as
                ``start``.

        Returns:
            Data: A new Data object filtered to the specified range.

        Raises:
            TypeError: When the index is not temporal and a non-integer bound
                is supplied.

        """
        date_column = self.index.columns[0]
        is_temporal = self.index[date_column].dtype.is_temporal()

        if is_temporal:
            cond = pl.lit(True)
            if start is not None:
                cond = cond & (pl.col(date_column) >= pl.lit(start))
            if end is not None:
                cond = cond & (pl.col(date_column) <= pl.lit(end))
            mask = self.index.select(cond.alias("mask"))["mask"]
            new_index = self.index.filter(mask)
            new_returns = self.returns.filter(mask)
            new_benchmark = self.benchmark.filter(mask) if self.benchmark is not None else None
        else:
            if start is not None and not isinstance(start, int):
                raise TypeError(f"start must be an integer, got {type(start).__name__}.")  # noqa: TRY003
            if end is not None and not isinstance(end, int):
                raise TypeError(f"end must be an integer, got {type(end).__name__}.")  # noqa: TRY003
            row_start = start if start is not None else 0
            row_end = end + 1 if end is not None else self.index.height
            length = max(0, row_end - row_start)
            new_index = self.index.slice(row_start, length)
            new_returns = self.returns.slice(row_start, length)
            new_benchmark = self.benchmark.slice(row_start, length) if self.benchmark is not None else None

        return Data(returns=new_returns, benchmark=new_benchmark, index=new_index)

    @property
    def _periods_per_year(self) -> float:
        """Estimate the number of periods per year based on average frequency in the index.

        For temporal (Date/Datetime) indices, computes the mean gap between observations
        and converts to an annualised period count (e.g. ~252 for daily, ~52 for weekly).

        For integer indices (date-free portfolios), falls back to 252 trading days per year
        because integer diffs have no time meaning.
        """
        datetime_col = self.index[self.index.columns[0]]

        if not datetime_col.dtype.is_temporal():
            return 252.0

        sorted_dt = datetime_col.sort()
        diffs = sorted_dt.diff().drop_nulls()
        mean_diff = diffs.mean()

        if isinstance(mean_diff, timedelta):
            seconds = mean_diff.total_seconds()
        else:  # pragma: no cover  # Polars always returns timedelta for temporal diff
            seconds = cast(float, mean_diff) if mean_diff is not None else 1.0

        return (365 * 24 * 60 * 60) / seconds

    def items(self) -> Iterator[tuple[str, pl.Series]]:
        """Iterate over all assets and their corresponding data series.

        This method provides a convenient way to iterate over all assets in the data,
        yielding each asset name and its corresponding data series.

        Yields:
            tuple[str, pl.Series]: A tuple containing the asset name and its data series.

        """
        matrix = self.all

        for col in self.assets:
            yield col, matrix.get_column(col)
