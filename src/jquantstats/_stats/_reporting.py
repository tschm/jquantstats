"""Temporal reporting, capture ratios, and summary statistics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import polars as pl

from ._core import _drawdown_series, _to_float, columnwise_stat

# ── Reporting statistics mixin ───────────────────────────────────────────────


class _ReportingStatsMixin:
    """Mixin providing temporal, capture, and summary reporting metrics.

    Covers: periods per year, average drawdown, Calmar ratio, recovery factor,
    max drawdown duration, monthly win rate, worst-N periods, up/down capture
    ratios, annual breakdown, and summary statistics table.

    Attributes (provided by the concrete subclass):
        data: The :class:`~jquantstats._data.Data` object.
        all: Combined DataFrame for efficient column selection.
    """

    if TYPE_CHECKING:
        from .._data import Data

        data: Data
        all: pl.DataFrame | None

        def avg_return(self) -> dict[str, float]:
            """Defined on _BasicStatsMixin."""

        def avg_win(self) -> dict[str, float]:
            """Defined on _BasicStatsMixin."""

        def avg_loss(self) -> dict[str, float]:
            """Defined on _BasicStatsMixin."""

        def win_rate(self) -> dict[str, float]:
            """Defined on _BasicStatsMixin."""

        def profit_factor(self) -> dict[str, float]:
            """Defined on _BasicStatsMixin."""

        def payoff_ratio(self) -> dict[str, float]:
            """Defined on _BasicStatsMixin."""

        def best(self) -> dict[str, float]:
            """Defined on _BasicStatsMixin."""

        def worst(self) -> dict[str, float]:
            """Defined on _BasicStatsMixin."""

        def volatility(self) -> dict[str, float]:
            """Defined on _BasicStatsMixin."""

        def sharpe(self) -> dict[str, float]:
            """Defined on _PerformanceStatsMixin."""

        def skew(self) -> dict[str, float]:
            """Defined on _BasicStatsMixin."""

        def kurtosis(self) -> dict[str, float]:
            """Defined on _BasicStatsMixin."""

        def value_at_risk(self) -> dict[str, float]:
            """Defined on _BasicStatsMixin."""

        def conditional_value_at_risk(self) -> dict[str, float]:
            """Defined on _BasicStatsMixin."""

        def max_drawdown(self) -> dict[str, float]:
            """Defined on _PerformanceStatsMixin."""

    # ── Temporal & reporting ──────────────────────────────────────────────────

    @property
    def periods_per_year(self) -> float:
        """Estimate the number of periods per year from the data index spacing.

        Returns:
            float: Estimated number of observations per calendar year.
        """
        return self.data._periods_per_year

    @columnwise_stat
    def avg_drawdown(self, series: pl.Series) -> float:
        """Average drawdown across all underwater periods.

        Returns 0.0 when there are no underwater periods.

        Args:
            series (pl.Series): Series of additive daily returns.

        Returns:
            float: Mean drawdown in [0, 1].
        """
        dd = _drawdown_series(series)
        in_dd = dd.filter(dd > 0)
        if in_dd.is_empty():
            return 0.0
        return _to_float(in_dd.mean())

    @columnwise_stat
    def calmar(self, series: pl.Series, periods: int | float | None = None) -> float:
        """Calmar ratio (annualised return divided by maximum drawdown).

        Returns ``nan`` when the maximum drawdown is zero.

        Args:
            series (pl.Series): Series of additive daily returns.
            periods: Annualisation factor. Defaults to ``periods_per_year``.

        Returns:
            float: Calmar ratio, or ``nan`` if max drawdown is zero.
        """
        raw_periods = periods or self.data._periods_per_year
        max_dd = _to_float(_drawdown_series(series).max())
        if max_dd <= 0:
            return float("nan")
        ann_return = _to_float(series.mean()) * raw_periods
        return ann_return / max_dd

    @columnwise_stat
    def recovery_factor(self, series: pl.Series) -> float:
        """Recovery factor (total return divided by maximum drawdown).

        Returns ``nan`` when the maximum drawdown is zero.

        Args:
            series (pl.Series): Series of additive daily returns.

        Returns:
            float: Recovery factor, or ``nan`` if max drawdown is zero.
        """
        max_dd = _to_float(_drawdown_series(series).max())
        if max_dd <= 0:
            return float("nan")
        total_return = _to_float(series.sum())
        return total_return / max_dd

    def max_drawdown_duration(self) -> dict[str, float | int | None]:
        """Maximum drawdown duration in calendar days (or periods) per asset.

        When the index is a temporal column (``Date`` / ``Datetime``) the
        duration is expressed as calendar days spanned by the longest
        underwater run.  For integer-indexed data each row counts as one
        period.

        Returns:
            dict[str, float | int | None]: Asset → max drawdown duration.
            Returns 0 when there are no underwater periods.
        """
        all_df = cast(pl.DataFrame, self.all)
        date_col_name = self.data.date_col[0] if self.data.date_col else None
        has_date = date_col_name is not None and all_df[date_col_name].dtype.is_temporal()
        result: dict[str, float | int | None] = {}
        for col, series in self.data.items():
            nav = 1.0 + series.cast(pl.Float64).cum_sum()
            hwm = nav.cum_max()
            in_dd = nav < hwm

            if not in_dd.any():
                result[col] = 0
                continue

            if has_date and date_col_name is not None:
                frame = pl.DataFrame({"date": all_df[date_col_name], "in_dd": in_dd})
            else:
                frame = pl.DataFrame({"date": pl.Series(list(range(len(series))), dtype=pl.Int64), "in_dd": in_dd})

            frame = frame.with_columns(pl.col("in_dd").rle_id().alias("run_id"))
            dd_runs = (
                frame.filter(pl.col("in_dd"))
                .group_by("run_id")
                .agg([pl.col("date").min().alias("start"), pl.col("date").max().alias("end")])
            )

            if has_date:
                dd_runs = dd_runs.with_columns(
                    ((pl.col("end") - pl.col("start")).dt.total_days() + 1).alias("duration")
                )
            else:
                dd_runs = dd_runs.with_columns((pl.col("end") - pl.col("start") + 1).alias("duration"))

            result[col] = int(_to_float(dd_runs["duration"].max()))
        return result

    def monthly_win_rate(self) -> dict[str, float]:
        """Fraction of calendar months with a positive compounded return per asset.

        Requires a temporal (Date / Datetime) index.  Returns ``nan`` per
        asset when no temporal index is present.

        Returns:
            dict[str, float]: Monthly win rate in [0, 1] per asset.
        """
        all_df = cast(pl.DataFrame, self.all)
        date_col_name = self.data.date_col[0] if self.data.date_col else None
        if date_col_name is None or not all_df[date_col_name].dtype.is_temporal():
            return {col: float("nan") for col, _ in self.data.items()}

        result: dict[str, float] = {}
        for col, _ in self.data.items():
            df = (
                all_df.select([date_col_name, col])
                .drop_nulls()
                .with_columns(
                    [
                        pl.col(date_col_name).dt.year().alias("_year"),
                        pl.col(date_col_name).dt.month().alias("_month"),
                    ]
                )
            )
            monthly = (
                df.group_by(["_year", "_month"])
                .agg((pl.col(col) + 1.0).product().alias("gross"))
                .with_columns((pl.col("gross") - 1.0).alias("monthly_return"))
            )
            n_total = len(monthly)
            if n_total == 0:
                result[col] = float("nan")
            else:
                n_positive = int((monthly["monthly_return"] > 0).sum())
                result[col] = n_positive / n_total
        return result

    def worst_n_periods(self, n: int = 5) -> dict[str, list[float | None]]:
        """Return the N worst return periods per asset.

        If a series has fewer than ``n`` non-null observations the list is
        padded with ``None`` on the right.

        Args:
            n: Number of worst periods to return. Defaults to 5.

        Returns:
            dict[str, list[float | None]]: Sorted worst returns per asset.
        """
        result: dict[str, list[float | None]] = {}
        for col, series in self.data.items():
            nonnull = series.drop_nulls()
            worst: list[float | None] = nonnull.sort(descending=False).head(n).to_list()
            while len(worst) < n:
                worst.append(None)
            result[col] = worst
        return result

    # ── Capture ratios ────────────────────────────────────────────────────────

    def up_capture(self, benchmark: pl.Series) -> dict[str, float]:
        """Up-market capture ratio relative to an explicit benchmark series.

        Measures the fraction of the benchmark's upside that the strategy
        captures.  A value greater than 1.0 means the strategy outperformed
        the benchmark in rising markets.

        Args:
            benchmark: Benchmark return series aligned row-by-row with the data.

        Returns:
            dict[str, float]: Up capture ratio per asset.
        """
        up_mask = benchmark > 0
        bench_up = benchmark.filter(up_mask).drop_nulls()
        if bench_up.is_empty():
            return {col: float("nan") for col, _ in self.data.items()}
        bench_geom = float((bench_up + 1.0).product()) ** (1.0 / len(bench_up)) - 1.0
        if bench_geom == 0.0:  # pragma: no cover
            return {col: float("nan") for col, _ in self.data.items()}
        result: dict[str, float] = {}
        for col, series in self.data.items():
            strat_up = series.filter(up_mask).drop_nulls()
            if strat_up.is_empty():
                result[col] = float("nan")
            else:
                strat_geom = float((strat_up + 1.0).product()) ** (1.0 / len(strat_up)) - 1.0
                result[col] = strat_geom / bench_geom
        return result

    def down_capture(self, benchmark: pl.Series) -> dict[str, float]:
        """Down-market capture ratio relative to an explicit benchmark series.

        A value less than 1.0 means the strategy lost less than the benchmark
        in falling markets (a desirable property).

        Args:
            benchmark: Benchmark return series aligned row-by-row with the data.

        Returns:
            dict[str, float]: Down capture ratio per asset.
        """
        down_mask = benchmark < 0
        bench_down = benchmark.filter(down_mask).drop_nulls()
        if bench_down.is_empty():
            return {col: float("nan") for col, _ in self.data.items()}
        bench_geom = float((bench_down + 1.0).product()) ** (1.0 / len(bench_down)) - 1.0
        if bench_geom == 0.0:  # pragma: no cover
            return {col: float("nan") for col, _ in self.data.items()}
        result: dict[str, float] = {}
        for col, series in self.data.items():
            strat_down = series.filter(down_mask).drop_nulls()
            if strat_down.is_empty():
                result[col] = float("nan")
            else:
                strat_geom = float((strat_down + 1.0).product()) ** (1.0 / len(strat_down)) - 1.0
                result[col] = strat_geom / bench_geom
        return result

    # ── Summary & breakdown ────────────────────────────────────────────────────

    def annual_breakdown(self) -> pl.DataFrame:
        """Summary statistics broken down by calendar year.

        Groups the data by calendar year using the date index, computes a
        full :py:meth:`summary` for each year, and stacks the results with an
        additional ``year`` column.

        Returns:
            pl.DataFrame: Columns ``year``, ``metric``, one per asset, sorted
            by ``year``.

        Raises:
            ValueError: If the data has no date index.
        """
        all_df = cast(pl.DataFrame, self.all)
        date_col_name = self.data.date_col[0] if self.data.date_col else None
        has_temporal = date_col_name is not None and all_df[date_col_name].dtype.is_temporal()

        from .._data import Data

        if not has_temporal:
            # Integer-index fallback: group by chunks of ~_periods_per_year rows
            chunk = round(self.data._periods_per_year)
            total = all_df.height
            frames_int: list[pl.DataFrame] = []
            for i, start in enumerate(range(0, total, chunk), start=1):
                chunk_all = all_df.slice(start, chunk)
                if chunk_all.height < max(5, chunk // 4):
                    continue
                chunk_index = chunk_all.select(self.data.date_col)
                chunk_returns = chunk_all.select(self.data.returns.columns)
                chunk_benchmark = (
                    chunk_all.select(self.data.benchmark.columns) if self.data.benchmark is not None else None
                )
                chunk_data = Data(returns=chunk_returns, index=chunk_index, benchmark=chunk_benchmark)
                chunk_summary = cast(Any, type(self))(chunk_data).summary()
                chunk_summary = chunk_summary.with_columns(pl.lit(i).alias("year"))
                frames_int.append(chunk_summary)
            if not frames_int:
                return pl.DataFrame()
            result_int = pl.concat(frames_int)
            ordered_int = ["year", "metric", *[c for c in result_int.columns if c not in ("year", "metric")]]
            return result_int.select(ordered_int)

        if date_col_name is None:  # unreachable: has_temporal guarantees non-None  # pragma: no cover
            return pl.DataFrame()  # pragma: no cover
        years = all_df[date_col_name].dt.year().unique().sort().to_list()

        frames: list[pl.DataFrame] = []
        for year in years:
            year_all = all_df.filter(pl.col(date_col_name).dt.year() == year)
            if year_all.height < 2:
                continue
            year_index = year_all.select([date_col_name])
            year_returns = year_all.select(self.data.returns.columns)
            year_benchmark = year_all.select(self.data.benchmark.columns) if self.data.benchmark is not None else None
            year_data = Data(returns=year_returns, index=year_index, benchmark=year_benchmark)
            year_summary = cast(Any, type(self))(year_data).summary()
            year_summary = year_summary.with_columns(pl.lit(year).alias("year"))
            frames.append(year_summary)

        if not frames:
            asset_cols = list(self.data.returns.columns)
            schema: dict[str, type[pl.DataType]] = {
                "year": pl.Int32,
                "metric": pl.String,
                **dict.fromkeys(asset_cols, pl.Float64),
            }
            return pl.DataFrame(schema=schema)

        result = pl.concat(frames)
        ordered = ["year", "metric", *[c for c in result.columns if c not in ("year", "metric")]]
        return result.select(ordered)

    def summary(self) -> pl.DataFrame:
        """Summary statistics for each asset as a tidy DataFrame.

        Each row is one metric; each column beyond ``metric`` is one asset.

        Returns:
            pl.DataFrame: A DataFrame with a ``metric`` column followed by one
            column per asset.
        """
        assets = [col for col, _ in self.data.items()]

        def _safe(fn: Any) -> dict[str, Any]:
            """Call *fn()* and return its result; return NaN for each asset on any exception."""
            try:
                return fn()
            except Exception:
                return dict.fromkeys(assets, float("nan"))

        metrics: dict[str, dict[str, Any]] = {
            "avg_return": _safe(self.avg_return),
            "avg_win": _safe(self.avg_win),
            "avg_loss": _safe(self.avg_loss),
            "win_rate": _safe(self.win_rate),
            "profit_factor": _safe(self.profit_factor),
            "payoff_ratio": _safe(self.payoff_ratio),
            "monthly_win_rate": _safe(self.monthly_win_rate),
            "best": _safe(self.best),
            "worst": _safe(self.worst),
            "volatility": _safe(self.volatility),
            "sharpe": _safe(self.sharpe),
            "skew": _safe(self.skew),
            "kurtosis": _safe(self.kurtosis),
            "value_at_risk": _safe(self.value_at_risk),
            "conditional_value_at_risk": _safe(self.conditional_value_at_risk),
            "max_drawdown": _safe(self.max_drawdown),
            "avg_drawdown": _safe(self.avg_drawdown),
            "max_drawdown_duration": _safe(self.max_drawdown_duration),
            "calmar": _safe(self.calmar),
            "recovery_factor": _safe(self.recovery_factor),
        }

        rows: list[dict[str, object]] = [
            {"metric": name, **{asset: values.get(asset) for asset in assets}} for name, values in metrics.items()
        ]
        return pl.DataFrame(rows)
