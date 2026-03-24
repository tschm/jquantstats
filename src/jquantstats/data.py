"""Financial returns data container and manipulation utilities."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from datetime import timedelta
from typing import TYPE_CHECKING, cast

import narwhals as nw
import polars as pl

from ._types import NativeFrame, NativeFrameOrScalar

if TYPE_CHECKING:
    from ._plots import DataPlots
    from ._reports import Reports
    from ._stats import Stats

__all__ = ["Data", "LazyData"]


def _to_polars(df: NativeFrame) -> pl.DataFrame:
    """Convert any narwhals-compatible DataFrame to a polars DataFrame."""
    if isinstance(df, pl.DataFrame):
        return df
    return nw.from_native(df, eager_only=True).to_polars()


def _subtract_risk_free(dframe: pl.DataFrame, rf: float | pl.DataFrame, date_col: str) -> pl.DataFrame:
    """Subtract the risk-free rate from all numeric columns in the DataFrame.

    Parameters
    ----------
    dframe : pl.DataFrame
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

    Returns:
    -------
    pl.DataFrame
        DataFrame with the risk-free rate subtracted from all numeric columns,
        preserving the original column names.

    """
    if isinstance(rf, float):
        rf_dframe = dframe.select([pl.col(date_col), pl.lit(rf).alias("rf")])
    else:
        if not isinstance(rf, pl.DataFrame):
            raise TypeError("rf must be a float or DataFrame")  # noqa: TRY003
        rf_dframe = rf.rename({rf.columns[1]: "rf"}) if rf.columns[1] != "rf" else rf

    dframe = dframe.join(rf_dframe, on=date_col, how="inner")
    return dframe.select(
        [pl.col(date_col)]
        + [(pl.col(col) - pl.col("rf")).alias(col) for col in dframe.columns if col not in {date_col, "rf"}]
    )


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
        """Validate the Data object after initialization."""
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

    @classmethod
    def from_returns(
        cls,
        returns: NativeFrame,
        rf: NativeFrameOrScalar = 0.0,
        benchmark: NativeFrame | None = None,
        date_col: str = "Date",
    ) -> Data:
        """Create a Data object from returns and optional benchmark.

        Parameters
        ----------
        returns : NativeFrame
            Financial returns data. First column should be the date column,
            remaining columns are asset returns.

        rf : float | NativeFrame, optional
            Risk-free rate. Default is 0.0 (no risk-free rate adjustment).

            - If float: Constant risk-free rate applied to all dates.
            - If NativeFrame: Time-varying risk-free rate with dates matching returns.

        benchmark : NativeFrame | None, optional
            Benchmark returns. Default is None (no benchmark).
            First column should be the date column, remaining columns are benchmark returns.

        date_col : str, optional
            Name of the date column in the DataFrames. Default is "Date".

        Returns:
        -------
        Data
            Object containing excess returns and benchmark (if any), with methods for
            analysis and visualization through the ``stats`` and ``plots`` properties.

        Raises:
        ------
        ValueError
            If there are no overlapping dates between returns and benchmark.

        Examples:
        --------
        Basic usage:

        ```python
        from jquantstats import Data
        import polars as pl

        returns = pl.DataFrame({
            "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "Asset1": [0.01, -0.02, 0.03]
        }).with_columns(pl.col("Date").str.to_date())

        data = Data.from_returns(returns=returns)
        ```

        With benchmark and risk-free rate:

        ```python
        benchmark = pl.DataFrame({
            "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "Market": [0.005, -0.01, 0.02]
        }).with_columns(pl.col("Date").str.to_date())

        data = Data.from_returns(returns=returns, benchmark=benchmark, rf=0.0002)
        ```

        """
        returns_pl = _to_polars(returns)
        benchmark_pl = _to_polars(benchmark) if benchmark is not None else None
        rf_converted: float | pl.DataFrame
        if isinstance(rf, pl.DataFrame) or (not isinstance(rf, float) and not isinstance(rf, int)):
            rf_converted = _to_polars(rf)
        else:
            rf_converted = rf  # int: _subtract_risk_free raises TypeError (see test_subtract_rf_invalid_type)

        if benchmark_pl is not None:
            joined_dates = returns_pl.join(benchmark_pl, on=date_col, how="inner").select(date_col)
            if joined_dates.is_empty():
                raise ValueError("No overlapping dates between returns and benchmark.")  # noqa: TRY003
            returns_pl = returns_pl.join(joined_dates, on=date_col, how="inner")
            benchmark_pl = benchmark_pl.join(joined_dates, on=date_col, how="inner")

        index = returns_pl.select(date_col)
        excess_returns = _subtract_risk_free(returns_pl, rf_converted, date_col).drop(date_col)
        excess_benchmark = (
            _subtract_risk_free(benchmark_pl, rf_converted, date_col).drop(date_col)
            if benchmark_pl is not None
            else None
        )

        return cls(returns=excess_returns, benchmark=excess_benchmark, index=index)

    def __repr__(self) -> str:
        """Return a string representation of the Data object."""
        rows = len(self.index)
        date_cols = self.date_col
        if date_cols:
            date_column = date_cols[0]
            start = self.index[date_column].min()
            end = self.index[date_column].max()
            return f"Data(assets={self.assets}, rows={rows}, start={start}, end={end})"
        return f"Data(assets={self.assets}, rows={rows})"

    @property
    def plots(self) -> DataPlots:
        """Provides access to visualization methods for the financial data.

        Returns:
            DataPlots: An instance of the DataPlots class initialized with this data.

        """
        from ._plots import DataPlots

        return DataPlots(self)

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

    def describe(self) -> pl.DataFrame:
        """Return a tidy summary of shape, date range and asset names.

        Returns:
        -------
        pl.DataFrame
            One row per asset with columns: asset, start, end, rows, has_benchmark.

        """
        date_column = self.date_col[0]
        start = self.index[date_column].min()
        end = self.index[date_column].max()
        rows = len(self.index)
        return pl.DataFrame(
            {
                "asset": self.returns.columns,
                "start": [start] * len(self.returns.columns),
                "end": [end] * len(self.returns.columns),
                "rows": [rows] * len(self.returns.columns),
                "has_benchmark": [self.benchmark is not None] * len(self.returns.columns),
            }
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

    def lazy(self) -> LazyData:
        """Return a lazy-backed version of this Data object.

        Wraps the underlying Polars DataFrames as LazyFrames so that subsequent
        operations (``resample``, ``truncate``, ``head``, ``tail``) build a query
        plan rather than executing immediately.  Call :meth:`LazyData.collect` on
        the result to materialise the final :class:`Data` object.

        This is the recommended entry point when chaining multiple transformations
        on large datasets, as Polars can optimise and fuse operations before any
        computation takes place.

        Returns:
        -------
        LazyData
            A lazy wrapper backed by the same data as this object.

        Examples:
        --------
        ```python
        lazy = data.lazy().truncate(start=date(2020, 1, 1)).resample("1mo")
        result = lazy.collect()   # single optimised execution
        ```

        """
        return LazyData(
            returns=self.returns.lazy(),
            index=self.index.lazy(),
            benchmark=self.benchmark.lazy() if self.benchmark is not None else None,
        )


@dataclasses.dataclass(frozen=True)
class LazyData:
    """Lazy-evaluation container for financial returns data.

    Wraps Polars :class:`~polars.LazyFrame` objects to enable query optimisation
    and streaming for large datasets.  Operations such as :meth:`resample` and
    :meth:`truncate` add steps to a lazy query plan; no computation happens until
    :meth:`collect` is called.

    Prefer :meth:`Data.lazy` to convert an existing eager :class:`Data` object,
    or use the file-scanning constructors :meth:`scan_parquet` / :meth:`scan_csv`
    to read large datasets without loading them fully into memory.

    Attributes:
        returns: LazyFrame of asset returns (no date column).
        index: LazyFrame with a single date column.
        benchmark: Optional LazyFrame of benchmark returns (no date column).

    Examples:
    --------
    Convert an eager Data object to lazy and chain transformations:

    ```python
    from datetime import date
    lazy = data.lazy().truncate(start=date(2020, 1, 1)).resample("1mo")
    result = lazy.collect()
    ```

    Scan a large Parquet file without loading it into memory:

    ```python
    lazy = LazyData.scan_parquet("returns.parquet", benchmark="bench.parquet")
    result = lazy.truncate(start=date(2020, 1, 1)).collect()
    ```

    """

    returns: pl.LazyFrame
    index: pl.LazyFrame
    benchmark: pl.LazyFrame | None = None

    def collect(self) -> Data:
        """Materialise all lazy frames and return an eager :class:`Data` object.

        Triggers Polars query optimisation and evaluation across all pending
        operations.

        Returns:
        -------
        Data
            An eager Data object with all computations applied.

        Examples:
        --------
        ```python
        result: Data = data.lazy().resample("1mo").collect()
        ```

        """
        return Data(
            returns=cast(pl.DataFrame, self.returns.collect()),
            index=cast(pl.DataFrame, self.index.collect()),
            benchmark=cast(pl.DataFrame, self.benchmark.collect()) if self.benchmark is not None else None,
        )

    def resample(self, every: str = "1mo") -> LazyData:
        """Lazily resample returns to a lower frequency using compound returns.

        Adds a ``group_by_dynamic`` aggregation step to the query plan.  The
        actual resampling is deferred until :meth:`collect` is called, allowing
        Polars to fuse this step with any preceding filters or projections.

        Args:
        ----
        every : str
            Polars duration string, e.g. ``"1mo"``, ``"1y"``, ``"1w"``.
            Defaults to ``"1mo"``.

        Returns:
        -------
        LazyData
            A new LazyData with the resampling step added to the query plan.

        """
        date_col = self.index.collect_schema().names()[0]
        returns_cols = self.returns.collect_schema().names()

        def _resample(lf: pl.LazyFrame, asset_cols: list[str]) -> pl.LazyFrame:
            """Combine index with *lf* and aggregate using compound returns."""
            combined = pl.concat([self.index, lf], how="horizontal")
            return combined.group_by_dynamic(
                index_column=date_col,
                every=every,
                period=every,
                closed="right",
                label="right",
            ).agg([((pl.col(col) + 1.0).product() - 1.0).alias(col) for col in asset_cols])

        resampled_returns = _resample(self.returns, returns_cols)
        resampled_benchmark: pl.LazyFrame | None = None
        if self.benchmark is not None:
            bench_cols = self.benchmark.collect_schema().names()
            resampled_benchmark = _resample(self.benchmark, bench_cols)

        return LazyData(
            returns=resampled_returns.drop(date_col),
            index=resampled_returns.select(date_col),
            benchmark=resampled_benchmark.drop(date_col) if resampled_benchmark is not None else None,
        )

    def truncate(self, start: object = None, end: object = None) -> LazyData:
        """Lazily filter to the inclusive ``[start, end]`` date range.

        Adds a filter predicate to the query plan.  The actual filtering is
        deferred until :meth:`collect` is called, enabling Polars' predicate
        pushdown optimisation when reading from files via :meth:`scan_parquet`
        or :meth:`scan_csv`.

        Args:
        ----
        start : optional
            Lower-bound date (inclusive).  Pass ``None`` to keep the earliest
            available date.
        end : optional
            Upper-bound date (inclusive).  Pass ``None`` to keep the latest
            available date.

        Returns:
        -------
        LazyData
            A new LazyData with the filter step added to the query plan.

        """
        date_col = self.index.collect_schema().names()[0]

        cond = pl.lit(True)
        if start is not None:
            cond = cond & (pl.col(date_col) >= pl.lit(start))
        if end is not None:
            cond = cond & (pl.col(date_col) <= pl.lit(end))

        new_index = self.index.filter(cond)

        # Re-attach the date column so the filter can reference it, then drop it again.
        returns_cols = self.returns.collect_schema().names()
        new_returns = pl.concat([self.index, self.returns], how="horizontal").filter(cond).select(returns_cols)

        new_benchmark = None
        if self.benchmark is not None:
            bench_cols = self.benchmark.collect_schema().names()
            new_benchmark = pl.concat([self.index, self.benchmark], how="horizontal").filter(cond).select(bench_cols)

        return LazyData(returns=new_returns, index=new_index, benchmark=new_benchmark)

    def head(self, n: int = 5) -> LazyData:
        """Return the first *n* rows lazily.

        Args:
        ----
        n : int
            Number of rows to keep. Defaults to 5.

        Returns:
        -------
        LazyData
            A new LazyData limited to the first n rows.

        """
        return LazyData(
            returns=self.returns.head(n),
            index=self.index.head(n),
            benchmark=self.benchmark.head(n) if self.benchmark is not None else None,
        )

    def tail(self, n: int = 5) -> LazyData:
        """Return the last *n* rows lazily.

        Args:
        ----
        n : int
            Number of rows to keep. Defaults to 5.

        Returns:
        -------
        LazyData
            A new LazyData limited to the last n rows.

        """
        return LazyData(
            returns=self.returns.tail(n),
            index=self.index.tail(n),
            benchmark=self.benchmark.tail(n) if self.benchmark is not None else None,
        )

    @classmethod
    def scan_parquet(
        cls,
        returns: str,
        benchmark: str | None = None,
        rf: float = 0.0,
        date_col: str = "Date",
    ) -> LazyData:
        """Scan Parquet file(s) without loading them fully into memory.

        Uses :func:`polars.scan_parquet` which enables predicate pushdown and
        projection pushdown — only the rows and columns needed for the final
        query are read from disk when :meth:`collect` is called.

        Args:
        ----
        returns : str
            Path (or glob pattern) to the returns Parquet file(s).
        benchmark : str | None, optional
            Path (or glob pattern) to the benchmark Parquet file(s).
        rf : float, optional
            Constant risk-free rate to subtract from all return columns.
            Defaults to ``0.0`` (no adjustment).
        date_col : str, optional
            Name of the date column in the files. Defaults to ``"Date"``.

        Returns:
        -------
        LazyData
            Lazy container backed by Parquet file scans.

        Examples:
        --------
        ```python
        lazy = LazyData.scan_parquet("returns.parquet", benchmark="bench.parquet")
        result = lazy.truncate(start=date(2020, 1, 1)).collect()
        ```

        """
        returns_lf = pl.scan_parquet(returns)
        asset_cols = [c for c in returns_lf.collect_schema().names() if c != date_col]

        index = returns_lf.select(date_col)
        if rf != 0.0:
            returns_only: pl.LazyFrame = returns_lf.select([(pl.col(c) - rf).alias(c) for c in asset_cols])
        else:
            returns_only = returns_lf.select(asset_cols)

        benchmark_only: pl.LazyFrame | None = None
        if benchmark is not None:
            bench_lf = pl.scan_parquet(benchmark)
            bench_cols = [c for c in bench_lf.collect_schema().names() if c != date_col]
            if rf != 0.0:
                benchmark_only = bench_lf.select([(pl.col(c) - rf).alias(c) for c in bench_cols])
            else:
                benchmark_only = bench_lf.select(bench_cols)

        return cls(returns=returns_only, index=index, benchmark=benchmark_only)

    @classmethod
    def scan_csv(
        cls,
        returns: str,
        benchmark: str | None = None,
        rf: float = 0.0,
        date_col: str = "Date",
    ) -> LazyData:
        """Scan CSV file(s) without loading them fully into memory.

        Uses :func:`polars.scan_csv` which enables predicate pushdown and
        projection pushdown.

        Args:
        ----
        returns : str
            Path (or glob pattern) to the returns CSV file(s).
        benchmark : str | None, optional
            Path (or glob pattern) to the benchmark CSV file(s).
        rf : float, optional
            Constant risk-free rate to subtract from all return columns.
            Defaults to ``0.0`` (no adjustment).
        date_col : str, optional
            Name of the date column in the files. Defaults to ``"Date"``.

        Returns:
        -------
        LazyData
            Lazy container backed by CSV file scans.

        Examples:
        --------
        ```python
        lazy = LazyData.scan_csv("returns.csv", benchmark="bench.csv", rf=0.0002)
        result = lazy.resample("1mo").collect()
        ```

        """
        returns_lf = pl.scan_csv(returns, try_parse_dates=True)
        asset_cols = [c for c in returns_lf.collect_schema().names() if c != date_col]

        index = returns_lf.select(date_col)
        if rf != 0.0:
            returns_only: pl.LazyFrame = returns_lf.select([(pl.col(c) - rf).alias(c) for c in asset_cols])
        else:
            returns_only = returns_lf.select(asset_cols)

        benchmark_only: pl.LazyFrame | None = None
        if benchmark is not None:
            bench_lf = pl.scan_csv(benchmark, try_parse_dates=True)
            bench_cols = [c for c in bench_lf.collect_schema().names() if c != date_col]
            if rf != 0.0:
                benchmark_only = bench_lf.select([(pl.col(c) - rf).alias(c) for c in bench_cols])
            else:
                benchmark_only = bench_lf.select(bench_cols)

        return cls(returns=returns_only, index=index, benchmark=benchmark_only)
