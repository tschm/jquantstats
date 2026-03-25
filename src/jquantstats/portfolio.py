"""Portfolio analytics class for quant finance.

This module provides :class:`Portfolio`, a single frozen dataclass that
stores the raw portfolio inputs (prices, cash positions, AUM) and exposes
both the derived data series and the full analytics / visualisation suite:

- Derived data series — :attr:`profits`, :attr:`profit`, :attr:`nav_accumulated`,
  :attr:`returns`, :attr:`monthly`, :attr:`nav_compounded`, :attr:`highwater`,
  :attr:`drawdown`, :attr:`all`
- Lazy composition accessors — :attr:`stats`, :attr:`plots`, :attr:`report`
- Portfolio transforms — :meth:`truncate`, :meth:`lag`, :meth:`smoothed_holding`
- Attribution — :attr:`tilt`, :attr:`timing`, :attr:`tilt_timing_decomp`
- Turnover analysis — :attr:`turnover`, :attr:`turnover_weekly`, :meth:`turnover_summary`
- Cost analysis — :meth:`cost_adjusted_returns`, :meth:`trading_cost_impact`
- Utility — :meth:`correlation`
"""

import dataclasses
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from ._stats import Stats as Stats
    from .data import Data as Data

import polars as pl
import polars.selectors as cs

from ._cost_model import CostModel
from ._plots import PortfolioPlots
from ._reports import Report
from .exceptions import (
    IntegerIndexBoundError,
    InvalidCashPositionTypeError,
    InvalidPricesTypeError,
    MissingDateColumnError,
    NonPositiveAumError,
    RowCountMismatchError,
)


@dataclasses.dataclass(frozen=True)
class Portfolio:
    """Portfolio analytics class for quant finance.

    Stores the three raw inputs — cash positions, prices, and AUM — and
    exposes the standard derived data series, analytics facades, transforms,
    and attribution tools.

    Derived data series:

    - :attr:`profits` — per-asset daily cash P&L
    - :attr:`profit` — aggregate daily portfolio profit
    - :attr:`nav_accumulated` — cumulative additive NAV
    - :attr:`nav_compounded` — compounded NAV
    - :attr:`returns` — daily returns (profit / AUM)
    - :attr:`monthly` — monthly compounded returns
    - :attr:`highwater` — running high-water mark
    - :attr:`drawdown` — drawdown from high-water mark
    - :attr:`all` — merged view of all derived series

    - Lazy composition accessors: :attr:`stats`, :attr:`plots`, :attr:`report`
    - Portfolio transforms: :meth:`truncate`, :meth:`lag`,
      :meth:`smoothed_holding`
    - Attribution: :attr:`tilt`, :attr:`timing`, :attr:`tilt_timing_decomp`
    - Turnover: :attr:`turnover`, :attr:`turnover_weekly`,
      :meth:`turnover_summary`
    - Cost analysis: :meth:`cost_adjusted_returns`,
      :meth:`trading_cost_impact`
    - Utility: :meth:`correlation`

    Attributes:
        cashposition: Polars DataFrame of positions per asset over time
            (includes date column if present).
        prices: Polars DataFrame of prices per asset over time (includes date
            column if present).
        aum: Assets under management used as base NAV offset.

    Analytics facades
    -----------------
    - ``.stats``   : delegates to the legacy ``Stats`` pipeline via ``.data``; all 50+ metrics available.
    - ``.plots``   : portfolio-specific ``Plots``; NAV overlays, lead-lag IR, rolling Sharpe/vol, heatmaps.
    - ``.report``  : HTML ``Report``; self-contained portfolio performance report.
    - ``.data``    : bridge to the legacy ``Data`` / ``Stats`` / ``DataPlots`` pipeline.

    ``.plots`` and ``.report`` are intentionally *not* delegated to the legacy path: the legacy
    path operates on a bare returns series, while the analytics path has access to raw prices,
    positions, and AUM for richer portfolio-specific visualisations.

    Cost models
    -----------
    Two independent cost models are provided. They are not interchangeable:

    **Model A — position-delta (stateful, set at construction):**
        ``cost_per_unit: float``  — one-way cost per unit of position change (e.g. 0.01 per share).
        Used by ``.position_delta_costs`` and ``.net_cost_nav``.
        Best for: equity portfolios where cost scales with shares traded.

    **Model B — turnover-bps (stateless, passed at call time):**
        ``cost_bps: float``  — one-way cost in basis points of AUM turnover (e.g. 5 bps).
        Used by ``.cost_adjusted_returns(cost_bps)`` and ``.trading_cost_impact(max_bps)``.
        Best for: macro / fund-of-funds portfolios where cost scales with notional traded.

    To sweep a range of cost assumptions use ``trading_cost_impact(max_bps=20)`` (Model B).
    To compute a net-NAV curve set ``cost_per_unit`` at construction and read ``.net_cost_nav`` (Model A).

    Date column requirement
    -----------------------
    Most analytics work with or without a ``date`` column. The following features require a
    temporal ``date`` column (``pl.Date`` or ``pl.Datetime``):

    - ``portfolio.plots.correlation_heatmap()``
    - ``portfolio.plots.lead_lag_ir_plot()``
    - ``stats.monthly_win_rate()``      — returns NaN per column when no date is present
    - ``stats.annual_breakdown()``      — raises ``ValueError`` when no date is present
    - ``stats.max_drawdown_duration()`` — returns period count (int) instead of days

    Portfolios without a ``date`` column (integer-indexed) are fully supported for
    NAV, returns, Sharpe, drawdown, cost analytics, and most rolling metrics.

    Examples:
        >>> import polars as pl
        >>> from datetime import date
        >>> prices = pl.DataFrame({"date": [date(2020, 1, 1), date(2020, 1, 2)], "A": [100.0, 110.0]})
        >>> pos = pl.DataFrame({"date": [date(2020, 1, 1), date(2020, 1, 2)], "A": [1000.0, 1000.0]})
        >>> pf = Portfolio(prices=prices, cashposition=pos, aum=1e6)
        >>> pf.assets
        ['A']
    """

    cashposition: pl.DataFrame
    prices: pl.DataFrame
    aum: float
    cost_per_unit: float = 0.0
    cost_bps: float = 0.0
    _data_bridge: "Data | None" = dataclasses.field(init=False, repr=False, compare=False, hash=False)
    _stats_cache: "Stats | None" = dataclasses.field(init=False, repr=False, compare=False, hash=False)
    _plots_cache: "PortfolioPlots | None" = dataclasses.field(init=False, repr=False, compare=False, hash=False)
    _report_cache: "Report | None" = dataclasses.field(init=False, repr=False, compare=False, hash=False)

    @staticmethod
    def _build_data_bridge(ret: pl.DataFrame) -> "Data":
        """Build a :class:`~jquantstats._data.Data` bridge from a returns frame.

        Splits out the ``'date'`` column (if present) into an index and passes
        the remaining numeric columns as returns.  Used internally to populate
        ``_data_bridge`` at construction time so the ``data`` property is O(1).

        Args:
            ret: Returns DataFrame, optionally with a leading ``'date'`` column.

        Returns:
            A :class:`~jquantstats._data.Data` instance backed by *ret*.
        """
        from .data import Data

        if "date" in ret.columns:
            return Data(returns=ret.drop("date"), index=ret.select("date"))
        return Data(returns=ret, index=pl.DataFrame({"index": list(range(ret.height))}))

    def __post_init__(self) -> None:
        """Validate input types, shapes, and parameters post-initialization."""
        if not isinstance(self.prices, pl.DataFrame):
            raise InvalidPricesTypeError(type(self.prices).__name__)
        if not isinstance(self.cashposition, pl.DataFrame):
            raise InvalidCashPositionTypeError(type(self.cashposition).__name__)
        if self.cashposition.shape[0] != self.prices.shape[0]:
            raise RowCountMismatchError(self.prices.shape[0], self.cashposition.shape[0])
        if self.aum <= 0.0:
            raise NonPositiveAumError(self.aum)
        object.__setattr__(self, "_data_bridge", None)
        object.__setattr__(self, "_stats_cache", None)
        object.__setattr__(self, "_plots_cache", None)
        object.__setattr__(self, "_report_cache", None)

    def _date_range(self) -> tuple[int, object, object]:
        """Return (rows, start, end) for the portfolio's returns series.

        ``start`` and ``end`` are ``None`` when there is no ``'date'`` column.
        """
        ret = self.returns
        rows = ret.height
        if "date" in ret.columns:
            return rows, ret["date"].min(), ret["date"].max()
        return rows, None, None

    def __repr__(self) -> str:
        """Return a string representation of the Portfolio object."""
        rows, start, end = self._date_range()
        if start is not None:
            return f"Portfolio(assets={self.assets}, rows={rows}, start={start}, end={end})"
        return f"Portfolio(assets={self.assets}, rows={rows})"

    def describe(self) -> pl.DataFrame:
        """Return a tidy summary of shape, date range and asset names.

        Returns:
        -------
        pl.DataFrame
            One row per asset with columns: asset, start, end, rows.

        Examples:
            >>> import polars as pl
            >>> from datetime import date
            >>> prices = pl.DataFrame({"date": [date(2020, 1, 1), date(2020, 1, 2)], "A": [100.0, 110.0]})
            >>> pos = pl.DataFrame({"date": [date(2020, 1, 1), date(2020, 1, 2)], "A": [1000.0, 1000.0]})
            >>> pf = Portfolio(prices=prices, cashposition=pos, aum=1e6)
            >>> df = pf.describe()
            >>> list(df.columns)
            ['asset', 'start', 'end', 'rows']
        """
        rows, start, end = self._date_range()
        return pl.DataFrame(
            {
                "asset": self.assets,
                "start": [start] * len(self.assets),
                "end": [end] * len(self.assets),
                "rows": [rows] * len(self.assets),
            }
        )

    # ── Factory classmethods ──────────────────────────────────────────────────

    @classmethod
    def from_risk_position(
        cls,
        prices: pl.DataFrame,
        risk_position: pl.DataFrame,
        aum: float,
        vola: int | dict[str, int] = 32,
        cost_per_unit: float = 0.0,
        cost_bps: float = 0.0,
        cost_model: CostModel | None = None,
    ) -> Self:
        """Create a Portfolio from per-asset risk positions.

        De-volatizes each risk position using an EWMA volatility estimate
        derived from the corresponding price series.

        Args:
            prices: Price levels per asset over time (may include a date column).
            risk_position: Risk units per asset aligned with prices.
            vola: EWMA lookback (span-equivalent) used to estimate volatility.
                Pass an ``int`` to apply the same span to every asset, or a
                ``dict[str, int]`` to set a per-asset span (assets absent from
                the dict default to ``32``).
            aum: Assets under management used as the base NAV offset.
            cost_per_unit: One-way trading cost per unit of position change.
                Defaults to 0.0 (no cost).  Ignored when *cost_model* is given.
            cost_bps: One-way trading cost in basis points of AUM turnover.
                Defaults to 0.0 (no cost).  Ignored when *cost_model* is given.
            cost_model: Optional :class:`~jquantstats.CostModel`
                instance.  When supplied, its ``cost_per_unit`` and
                ``cost_bps`` values take precedence over the individual
                parameters above.

        Returns:
            A Portfolio instance whose cash positions are risk_position
            divided by EWMA volatility.
        """
        if cost_model is not None:
            cost_per_unit = cost_model.cost_per_unit
            cost_bps = cost_model.cost_bps
        assets = [col for col, dtype in prices.schema.items() if dtype.is_numeric()]

        def _span(asset: str) -> int:
            """Return the EWMA span for *asset*, falling back to 32 if not specified."""
            if isinstance(vola, dict):
                return int(vola.get(asset, 32))
            return int(vola)

        cash_position = risk_position.with_columns(
            (
                pl.col(asset)
                / prices[asset].pct_change().ewm_std(com=_span(asset) - 1, adjust=True, min_samples=_span(asset))
            ).alias(asset)
            for asset in assets
        )
        return cls(prices=prices, cashposition=cash_position, aum=aum, cost_per_unit=cost_per_unit, cost_bps=cost_bps)

    @classmethod
    def from_cash_position(
        cls,
        prices: pl.DataFrame,
        cash_position: pl.DataFrame,
        aum: float,
        cost_per_unit: float = 0.0,
        cost_bps: float = 0.0,
        cost_model: CostModel | None = None,
    ) -> Self:
        """Create a Portfolio directly from cash positions aligned with prices.

        Args:
            prices: Price levels per asset over time (may include a date column).
            cash_position: Cash exposure per asset over time.
            aum: Assets under management used as the base NAV offset.
            cost_per_unit: One-way trading cost per unit of position change.
                Defaults to 0.0 (no cost).  Ignored when *cost_model* is given.
            cost_bps: One-way trading cost in basis points of AUM turnover.
                Defaults to 0.0 (no cost).  Ignored when *cost_model* is given.
            cost_model: Optional :class:`~jquantstats.CostModel`
                instance.  When supplied, its ``cost_per_unit`` and
                ``cost_bps`` values take precedence over the individual
                parameters above.

        Returns:
            A Portfolio instance with the provided cash positions.
        """
        if cost_model is not None:
            cost_per_unit = cost_model.cost_per_unit
            cost_bps = cost_model.cost_bps
        return cls(prices=prices, cashposition=cash_position, aum=aum, cost_per_unit=cost_per_unit, cost_bps=cost_bps)

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _assert_clean_series(series: pl.Series, name: str = "") -> None:
        """Raise ValueError if *series* contains nulls or non-finite values."""
        if series.null_count() != 0:
            raise ValueError
        if not series.is_finite().all():
            raise ValueError

    # ── Core data properties ───────────────────────────────────────────────────

    @property
    def assets(self) -> list[str]:
        """List the asset column names from prices (numeric columns).

        Returns:
            list[str]: Names of numeric columns in prices; typically excludes
            ``'date'``.
        """
        return [c for c in self.prices.columns if self.prices[c].dtype.is_numeric()]

    @property
    def profits(self) -> pl.DataFrame:
        """Compute per-asset daily cash profits, preserving non-numeric columns.

        Returns:
            pl.DataFrame: Per-asset daily profit series along with any
            non-numeric columns (e.g., ``'date'``).

        Examples:
            >>> import polars as pl
            >>> prices = pl.DataFrame({"A": [100.0, 110.0, 105.0]})
            >>> pos = pl.DataFrame({"A": [1000.0, 1000.0, 1000.0]})
            >>> pf = Portfolio(prices=prices, cashposition=pos, aum=1e6)
            >>> pf.profits.columns
            ['A']
        """
        assets = [c for c in self.prices.columns if self.prices[c].dtype.is_numeric()]

        result = self.prices.with_columns(
            (self.prices[asset].pct_change().fill_null(0.0) * self.cashposition[asset].shift(n=1).fill_null(0.0)).alias(
                asset
            )
            for asset in assets
        )

        if assets:
            result = result.with_columns(
                pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(0.0).fill_null(0.0).alias(c) for c in assets
            )
        return result

    @property
    def profit(self) -> pl.DataFrame:
        """Return total daily portfolio profit including the ``'date'`` column.

        Aggregates per-asset profits into a single ``'profit'`` column and
        validates that no day's total profit is NaN/null.
        """
        df_profits = self.profits
        assets = [c for c in df_profits.columns if df_profits[c].dtype.is_numeric()]

        if not assets:
            raise ValueError

        non_assets = [c for c in df_profits.columns if c not in set(assets)]

        portfolio_daily_profit = pl.sum_horizontal([pl.col(c).fill_null(0.0) for c in assets]).alias("profit")
        result = df_profits.select([*non_assets, portfolio_daily_profit])

        self._assert_clean_series(series=result["profit"])
        return result

    @property
    def nav_accumulated(self) -> pl.DataFrame:
        """Compute cumulative additive NAV of the portfolio, preserving ``'date'``."""
        return self.profit.with_columns((pl.col("profit").cum_sum() + self.aum).alias("NAV_accumulated"))

    @property
    def returns(self) -> pl.DataFrame:
        """Return daily returns as profit scaled by AUM, preserving ``'date'``.

        The returned DataFrame contains the original ``'date'`` column with the
        ``'profit'`` column scaled by AUM (i.e., per-period returns), and also
        an additional convenience column named ``'returns'`` with the same
        values for downstream consumers.
        """
        return self.nav_accumulated.with_columns(
            (pl.col("profit") / self.aum).alias("returns"),
        )

    @property
    def monthly(self) -> pl.DataFrame:
        """Return monthly compounded returns and calendar columns.

        Aggregates daily returns (profit/AUM) by calendar month and computes
        the compounded monthly return: prod(1 + r_d) - 1. The resulting frame
        includes:

        - ``date``: month-end label as a Polars Date (end of the grouping window)
        - ``returns``: compounded monthly return
        - ``NAV_accumulated``: last NAV within the month
        - ``profit``: summed profit within the month
        - ``year``: integer year (e.g., 2020)
        - ``month``: integer month number (1-12)
        - ``month_name``: abbreviated month name (e.g., ``"Jan"``, ``"Feb"``)

        Raises:
            MissingDateColumnError: If the portfolio data has no ``'date'``
                column.
        """
        if "date" not in self.prices.columns:
            raise MissingDateColumnError("monthly")
        daily = self.returns.select(["date", "returns", "profit", "NAV_accumulated"])
        monthly = (
            daily.group_by_dynamic(
                "date",
                every="1mo",
                period="1mo",
                label="left",
                closed="right",
            )
            .agg(
                [
                    pl.col("profit").sum().alias("profit"),
                    pl.col("NAV_accumulated").last().alias("NAV_accumulated"),
                    (pl.col("returns") + 1.0).product().alias("gross"),
                ]
            )
            .with_columns((pl.col("gross") - 1.0).alias("returns"))
            .select(["date", "returns", "NAV_accumulated", "profit"])
            .with_columns(
                [
                    pl.col("date").dt.year().alias("year"),
                    pl.col("date").dt.month().alias("month"),
                    pl.col("date").dt.strftime("%b").alias("month_name"),
                ]
            )
            .sort("date")
        )
        return monthly

    @property
    def nav_compounded(self) -> pl.DataFrame:
        """Compute compounded NAV from returns (profit/AUM), preserving ``'date'``."""
        return self.returns.with_columns(((pl.col("returns") + 1.0).cum_prod() * self.aum).alias("NAV_compounded"))

    @property
    def highwater(self) -> pl.DataFrame:
        """Return the cumulative maximum of NAV as the high-water mark series.

        The resulting DataFrame preserves the ``'date'`` column and adds a
        ``'highwater'`` column computed as the cumulative maximum of
        ``'NAV_accumulated'``.
        """
        return self.returns.with_columns(pl.col("NAV_accumulated").cum_max().alias("highwater"))

    @property
    def drawdown(self) -> pl.DataFrame:
        """Return drawdown as the distance from high-water mark to current NAV.

        Computes ``'drawdown'`` = ``'highwater'`` - ``'NAV_accumulated'`` and
        preserves the ``'date'`` column alongside the intermediate columns.
        """
        return self.highwater.with_columns(
            (pl.col("highwater") - pl.col("NAV_accumulated")).alias("drawdown"),
            ((pl.col("highwater") - pl.col("NAV_accumulated")) / pl.col("highwater")).alias("drawdown_pct"),
        )

    @property
    def all(self) -> pl.DataFrame:
        """Return a merged view of drawdown and compounded NAV.

        When a ``'date'`` column is present the two frames are joined on that
        column to ensure temporal alignment.  When the data is integer-indexed
        (no ``'date'`` column) the frames are stacked horizontally — they are
        guaranteed to have identical row counts because both are derived from
        the same source portfolio.
        """
        left = self.drawdown
        if "date" in left.columns:
            right = self.nav_compounded.select(["date", "NAV_compounded"])
            return left.join(right, on="date", how="inner")
        else:
            right = self.nav_compounded.select(["NAV_compounded"])
            return left.hstack(right)

    # ── Lazy composition accessors ─────────────────────────────────────────────

    @property
    def data(self) -> "Data":
        """Build a legacy :class:`~jquantstats._data.Data` object from this portfolio's returns.

        This bridges the two entry points: ``Portfolio`` compiles the NAV curve from
        prices and positions; the returned :class:`~jquantstats._data.Data` object
        gives access to the full legacy analytics pipeline (``data.stats``,
        ``data.plots``, ``data.reports``).

        Returns:
            :class:`~jquantstats._data.Data`: A Data object whose ``returns`` column
            is the portfolio's daily return series and whose ``index`` holds the date
            column (or a synthetic integer index for date-free portfolios).

        Examples:
            >>> import polars as pl
            >>> from datetime import date
            >>> prices = pl.DataFrame({"date": [date(2020, 1, 1), date(2020, 1, 2)], "A": [100.0, 110.0]})
            >>> pos = pl.DataFrame({"date": [date(2020, 1, 1), date(2020, 1, 2)], "A": [1000.0, 1000.0]})
            >>> pf = Portfolio(prices=prices, cashposition=pos, aum=1e6)
            >>> d = pf.data
            >>> "returns" in d.returns.columns
            True
        """
        if self._data_bridge is not None:
            return self._data_bridge
        bridge = Portfolio._build_data_bridge(self.returns)
        object.__setattr__(self, "_data_bridge", bridge)
        return bridge

    @property
    def stats(self) -> "Stats":
        """Return a Stats object built from the portfolio's daily returns.

        Delegates to the legacy :class:`~jquantstats._stats.Stats` pipeline via
        :attr:`data`, so all analytics (Sharpe, drawdown, summary, etc.) are
        available through the shared implementation.

        The result is cached after first access so repeated calls are O(1).
        """
        if self._stats_cache is None:
            object.__setattr__(self, "_stats_cache", self.data.stats)
        return self._stats_cache  # type: ignore[return-value]

    @property
    def plots(self) -> PortfolioPlots:
        """Convenience accessor returning a PortfolioPlots facade for this portfolio.

        Use this to create Plotly visualizations such as snapshots, lagged
        performance curves, and lead/lag IR charts.

        Returns:
            :class:`~jquantstats._plots.PortfolioPlots`: Helper object with
            plotting methods.

        The result is cached after first access so repeated calls are O(1).
        """
        if self._plots_cache is None:
            object.__setattr__(self, "_plots_cache", PortfolioPlots(self))
        return self._plots_cache  # type: ignore[return-value]

    @property
    def report(self) -> Report:
        """Convenience accessor returning a Report facade for this portfolio.

        Use this to generate a self-contained HTML performance report
        containing statistics tables and interactive charts.

        Returns:
            :class:`~jquantstats._reports.Report`: Helper object with
            report methods.

        The result is cached after first access so repeated calls are O(1).
        """
        if self._report_cache is None:
            object.__setattr__(self, "_report_cache", Report(self))
        return self._report_cache  # type: ignore[return-value]

    # ── Portfolio transforms ───────────────────────────────────────────────────

    def truncate(self, start: object = None, end: object = None) -> "Portfolio":
        """Return a new Portfolio truncated to the inclusive [start, end] range.

        When a ``'date'`` column is present in both prices and cash positions,
        truncation is performed by comparing the ``'date'`` column against
        ``start`` and ``end`` (which should be date/datetime values or strings
        parseable by Polars).

        When the ``'date'`` column is absent, integer-based row slicing is
        used instead.  In this case ``start`` and ``end`` must be non-negative
        integers representing 0-based row indices.  Passing non-integer bounds
        to an integer-indexed portfolio raises :exc:`TypeError`.

        In all cases the ``aum`` value is preserved.

        Args:
            start: Optional lower bound (inclusive). A date/datetime or
                Polars-parseable string when a ``'date'`` column exists; a
                non-negative int row index when the data has no ``'date'``
                column.
            end: Optional upper bound (inclusive). Same type rules as
                ``start``.

        Returns:
            A new Portfolio instance with prices and cash positions filtered
            to the specified range.

        Raises:
            TypeError: When the portfolio has no ``'date'`` column and a
                non-integer bound is supplied.
        """
        has_date = "date" in self.prices.columns
        if has_date:
            cond = pl.lit(True)
            if start is not None:
                cond = cond & (pl.col("date") >= pl.lit(start))
            if end is not None:
                cond = cond & (pl.col("date") <= pl.lit(end))
            pr = self.prices.filter(cond)
            cp = self.cashposition.filter(cond)
        else:
            if start is not None and not isinstance(start, int):
                raise IntegerIndexBoundError("start", type(start).__name__)
            if end is not None and not isinstance(end, int):
                raise IntegerIndexBoundError("end", type(end).__name__)
            row_start = int(start) if start is not None else 0
            row_end = int(end) + 1 if end is not None else self.prices.height
            length = max(0, row_end - row_start)
            pr = self.prices.slice(row_start, length)
            cp = self.cashposition.slice(row_start, length)
        return Portfolio(
            prices=pr,
            cashposition=cp,
            aum=self.aum,
            cost_per_unit=self.cost_per_unit,
            cost_bps=self.cost_bps,
        )

    def lag(self, n: int) -> "Portfolio":
        """Return a new Portfolio with cash positions lagged by ``n`` steps.

        This method shifts the numeric asset columns in the cashposition
        DataFrame by ``n`` rows, preserving the ``'date'`` column and any
        non-numeric columns unchanged.  Positive ``n`` delays weights (moves
        them down); negative ``n`` leads them (moves them up); ``n == 0``
        returns the current portfolio unchanged.

        Notes:
            Missing values introduced by the shift are left as nulls;
            downstream profit computation already guards and treats nulls as
            zero when multiplying by returns.

        Args:
            n: Number of rows to shift (can be negative, zero, or positive).

        Returns:
            A new Portfolio instance with lagged cash positions and the same
            prices/AUM as the original.
        """
        if not isinstance(n, int):
            raise TypeError
        if n == 0:
            return self

        assets = [c for c in self.cashposition.columns if c != "date" and self.cashposition[c].dtype.is_numeric()]
        cp_lagged = self.cashposition.with_columns(pl.col(c).shift(n) for c in assets)
        return Portfolio(
            prices=self.prices,
            cashposition=cp_lagged,
            aum=self.aum,
            cost_per_unit=self.cost_per_unit,
            cost_bps=self.cost_bps,
        )

    def smoothed_holding(self, n: int) -> "Portfolio":
        """Return a new Portfolio with cash positions smoothed by a rolling mean.

        Applies a trailing window average over the last ``n`` steps for each
        numeric asset column (excluding ``'date'``). The window length is
        ``n + 1`` so that:

        - n=0 returns the original weights (no smoothing),
        - n=1 averages the current and previous weights,
        - n=k averages the current and last k weights.

        Args:
            n: Non-negative integer specifying how many previous steps to
                include.

        Returns:
            A new Portfolio with smoothed cash positions and the same
            prices/AUM.
        """
        if not isinstance(n, int):
            raise TypeError
        if n < 0:
            raise ValueError
        if n == 0:
            return self

        assets = [c for c in self.cashposition.columns if c != "date" and self.cashposition[c].dtype.is_numeric()]
        window = n + 1
        cp_smoothed = self.cashposition.with_columns(
            pl.col(c).rolling_mean(window_size=window, min_samples=1).alias(c) for c in assets
        )
        return Portfolio(
            prices=self.prices,
            cashposition=cp_smoothed,
            aum=self.aum,
            cost_per_unit=self.cost_per_unit,
            cost_bps=self.cost_bps,
        )

    # ── Attribution ────────────────────────────────────────────────────────────

    @property
    def tilt(self) -> "Portfolio":
        """Return the 'tilt' portfolio with constant average weights.

        Computes the time-average of each asset's cash position (ignoring
        nulls/NaNs) and builds a new Portfolio with those constant weights
        applied across time. Prices and AUM are preserved.
        """
        const_position = self.cashposition.with_columns(
            pl.col(col).drop_nulls().drop_nans().mean().alias(col) for col in self.assets
        )
        return Portfolio.from_cash_position(
            self.prices,
            const_position,
            aum=self.aum,
            cost_per_unit=self.cost_per_unit,
            cost_bps=self.cost_bps,
        )

    @property
    def timing(self) -> "Portfolio":
        """Return the 'timing' portfolio capturing deviations from the tilt.

        Constructs weights as original cash positions minus the tilt's
        constant positions, per asset. This isolates timing (alloc-demeaned)
        effects. Prices and AUM are preserved.
        """
        const_position = self.tilt.cashposition
        position = self.cashposition.with_columns((pl.col(col) - const_position[col]).alias(col) for col in self.assets)
        return Portfolio.from_cash_position(
            self.prices,
            position,
            aum=self.aum,
            cost_per_unit=self.cost_per_unit,
            cost_bps=self.cost_bps,
        )

    @property
    def tilt_timing_decomp(self) -> pl.DataFrame:
        """Return the portfolio's tilt/timing NAV decomposition.

        When a ``'date'`` column is present the three NAV series are joined on
        it. When data is integer-indexed the frames are stacked horizontally.
        """
        if "date" in self.nav_accumulated.columns:
            nav_portfolio = self.nav_accumulated.select(["date", "NAV_accumulated"])
            nav_tilt = self.tilt.nav_accumulated.select(["date", "NAV_accumulated"])
            nav_timing = self.timing.nav_accumulated.select(["date", "NAV_accumulated"])

            merged_df = nav_portfolio.join(nav_tilt, on="date", how="inner", suffix="_tilt").join(
                nav_timing, on="date", how="inner", suffix="_timing"
            )
        else:
            nav_portfolio = self.nav_accumulated.select(["NAV_accumulated"])
            nav_tilt = self.tilt.nav_accumulated.select(["NAV_accumulated"]).rename(
                {"NAV_accumulated": "NAV_accumulated_tilt"}
            )
            nav_timing = self.timing.nav_accumulated.select(["NAV_accumulated"]).rename(
                {"NAV_accumulated": "NAV_accumulated_timing"}
            )
            merged_df = nav_portfolio.hstack(nav_tilt).hstack(nav_timing)

        merged_df = merged_df.rename(
            {"NAV_accumulated_tilt": "tilt", "NAV_accumulated_timing": "timing", "NAV_accumulated": "portfolio"}
        )
        return merged_df

    # ── Turnover ───────────────────────────────────────────────────────────────

    @property
    def turnover(self) -> pl.DataFrame:
        """Daily one-way portfolio turnover as a fraction of AUM.

        Computes the sum of absolute position changes across all assets for
        each period, normalised by AUM.  The first row is always zero because
        there is no prior position to form a difference against.

        Returns:
            pl.DataFrame: Frame with an optional ``'date'`` column and a
            ``'turnover'`` column (dimensionless fraction of AUM).

        Examples:
            >>> import polars as pl
            >>> from datetime import date
            >>> _d = [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)]
            >>> prices = pl.DataFrame({"date": _d, "A": [100.0, 110.0, 121.0]})
            >>> pos = pl.DataFrame({"date": prices["date"], "A": [1000.0, 1200.0, 900.0]})
            >>> pf = Portfolio(prices=prices, cashposition=pos, aum=1e5)
            >>> pf.turnover["turnover"].to_list()
            [0.0, 0.002, 0.003]
        """
        assets = [c for c in self.cashposition.columns if c != "date" and self.cashposition[c].dtype.is_numeric()]
        daily_abs_chg = (
            pl.sum_horizontal(pl.col(c).diff().abs().fill_null(0.0).fill_nan(0.0) for c in assets) / self.aum
        ).alias("turnover")
        cols: list[str | pl.Expr] = []
        if "date" in self.cashposition.columns:
            cols.append("date")
        cols.append(daily_abs_chg)
        return self.cashposition.select(cols)

    @property
    def turnover_weekly(self) -> pl.DataFrame:
        """Weekly aggregated one-way portfolio turnover as a fraction of AUM.

        When a ``'date'`` column is present, sums the daily turnover within
        each calendar week (Monday-based ``group_by_dynamic``).  Without a
        date column, a rolling 5-period sum with ``min_samples=5`` is returned
        (the first four rows will be ``null``).

        Returns:
            pl.DataFrame: Frame with an optional ``'date'`` column (week
            start) and a ``'turnover'`` column (fraction of AUM, summed over
            the week).
        """
        daily = self.turnover
        if "date" not in daily.columns or not daily["date"].dtype.is_temporal():
            return daily.with_columns(pl.col("turnover").rolling_sum(window_size=5, min_samples=5))
        return daily.group_by_dynamic("date", every="1w").agg(pl.col("turnover").sum()).sort("date")

    def turnover_summary(self) -> pl.DataFrame:
        """Return a summary DataFrame of turnover statistics.

        Computes three metrics from the daily turnover series:

        - ``mean_daily_turnover``: mean of daily one-way turnover (fraction
          of AUM).
        - ``mean_weekly_turnover``: mean of weekly-aggregated turnover
          (fraction of AUM).
        - ``turnover_std``: standard deviation of daily turnover (fraction of
          AUM); complements the mean to detect regime switches.

        Returns:
            pl.DataFrame: One row per metric with columns ``'metric'`` and
            ``'value'``.

        Examples:
            >>> import polars as pl
            >>> from datetime import date, timedelta
            >>> import numpy as np
            >>> start = date(2020, 1, 1)
            >>> dates = pl.date_range(start=start, end=start + timedelta(days=9), interval="1d", eager=True)
            >>> prices = pl.DataFrame({"date": dates, "A": pl.Series(np.ones(10) * 100.0)})
            >>> pos = pl.DataFrame({"date": dates, "A": pl.Series([float(i) * 100 for i in range(10)])})
            >>> pf = Portfolio(prices=prices, cashposition=pos, aum=1e4)
            >>> summary = pf.turnover_summary()
            >>> list(summary["metric"])
            ['mean_daily_turnover', 'mean_weekly_turnover', 'turnover_std']
        """
        daily_col = self.turnover["turnover"]
        _mean = daily_col.mean()
        mean_daily = float(_mean) if isinstance(_mean, (int, float)) else 0.0
        _std = daily_col.std()
        std_daily = float(_std) if isinstance(_std, (int, float)) else 0.0
        weekly_col = self.turnover_weekly["turnover"].drop_nulls()
        _weekly_mean = weekly_col.mean()
        mean_weekly = (
            float(_weekly_mean) if weekly_col.len() > 0 and isinstance(_weekly_mean, (int, float)) else float("nan")
        )
        return pl.DataFrame(
            {
                "metric": ["mean_daily_turnover", "mean_weekly_turnover", "turnover_std"],
                "value": [mean_daily, mean_weekly, std_daily],
            }
        )

    # ── Cost analysis ──────────────────────────────────────────────────────────

    @property
    def position_delta_costs(self) -> pl.DataFrame:
        """Daily trading cost using the position-delta model.

        Computes the per-period cost as::

            cost_t = sum_i( |x_{i,t} - x_{i,t-1}| ) * cost_per_unit

        where ``x_{i,t}`` is the cash position in asset *i* at time *t* and
        ``cost_per_unit`` is the one-way cost per unit of traded notional.
        The first row is always zero because there is no prior position to
        form a difference against.

        Returns:
            pl.DataFrame: Frame with an optional ``'date'`` column and a
            ``'cost'`` column (absolute cash cost per period).

        Examples:
            >>> import polars as pl
            >>> from datetime import date
            >>> _d = [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)]
            >>> prices = pl.DataFrame({"date": _d, "A": [100.0, 110.0, 121.0]})
            >>> pos = pl.DataFrame({"date": _d, "A": [1000.0, 1200.0, 900.0]})
            >>> pf = Portfolio(prices=prices, cashposition=pos, aum=1e5, cost_per_unit=0.01)
            >>> pf.position_delta_costs["cost"].to_list()
            [0.0, 2.0, 3.0]
        """
        assets = [c for c in self.cashposition.columns if c != "date" and self.cashposition[c].dtype.is_numeric()]
        abs_position_changes = pl.sum_horizontal(pl.col(c).diff().abs().fill_null(0.0).fill_nan(0.0) for c in assets)
        daily_cost = (abs_position_changes * self.cost_per_unit).alias("cost")
        cols: list[str | pl.Expr] = []
        if "date" in self.cashposition.columns:
            cols.append("date")
        cols.append(daily_cost)
        return self.cashposition.select(cols)

    @property
    def net_cost_nav(self) -> pl.DataFrame:
        """Net-of-cost cumulative additive NAV using the position-delta cost model.

        Deducts :attr:`position_delta_costs` from daily portfolio profit and
        computes the running cumulative sum offset by AUM.  The result
        represents the realised NAV path a strategy would achieve after paying
        ``cost_per_unit`` on every unit of position change.

        When ``cost_per_unit`` is zero the result equals :attr:`nav_accumulated`.

        Returns:
            pl.DataFrame: Frame with an optional ``'date'`` column,
            ``'profit'``, ``'cost'``, and ``'NAV_accumulated_net'`` columns.

        Examples:
            >>> import polars as pl
            >>> from datetime import date
            >>> _d = [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)]
            >>> prices = pl.DataFrame({"date": _d, "A": [100.0, 110.0, 121.0]})
            >>> pos = pl.DataFrame({"date": _d, "A": [1000.0, 1200.0, 900.0]})
            >>> pf = Portfolio(prices=prices, cashposition=pos, aum=1e5, cost_per_unit=0.0)
            >>> net = pf.net_cost_nav
            >>> list(net.columns)
            ['date', 'profit', 'cost', 'NAV_accumulated_net']
        """
        profit_df = self.profit
        cost_df = self.position_delta_costs
        if "date" in profit_df.columns:
            df = profit_df.join(cost_df, on="date", how="left")
        else:
            df = profit_df.hstack(cost_df.select(["cost"]))
        return df.with_columns(((pl.col("profit") - pl.col("cost")).cum_sum() + self.aum).alias("NAV_accumulated_net"))

    def cost_adjusted_returns(self, cost_bps: float | None = None) -> pl.DataFrame:
        """Return daily portfolio returns net of estimated one-way trading costs.

        Trading costs are modelled as a linear function of daily one-way
        turnover: for every unit of AUM traded, the strategy incurs
        ``cost_bps`` basis points (i.e. ``cost_bps / 10_000`` fractional
        cost). The daily cost deduction is therefore::

            daily_cost = turnover * (cost_bps / 10_000)

        where ``turnover`` is the fraction-of-AUM one-way turnover already
        computed by :attr:`turnover`.  The deduction is applied to the
        ``returns`` column of :attr:`returns`, leaving all other columns
        (including ``date``) untouched.

        Args:
            cost_bps: One-way trading cost in basis points per unit of AUM
                traded.  Must be non-negative.  Defaults to ``self.cost_bps``
                set at construction time.

        Returns:
            pl.DataFrame: Same schema as :attr:`returns` but with the
            ``returns`` column reduced by the per-period trading cost.

        Raises:
            ValueError: If ``cost_bps`` is negative.

        Examples:
            >>> import polars as pl
            >>> from datetime import date
            >>> _d = [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)]
            >>> prices = pl.DataFrame({"date": _d, "A": [100.0, 110.0, 121.0]})
            >>> pos = pl.DataFrame({"date": _d, "A": [1000.0, 1200.0, 900.0]})
            >>> pf = Portfolio(prices=prices, cashposition=pos, aum=1e5)
            >>> adj = pf.cost_adjusted_returns(0.0)
            >>> float(adj["returns"][1]) == float(pf.returns["returns"][1])
            True
        """
        effective_bps = cost_bps if cost_bps is not None else self.cost_bps
        if effective_bps < 0:
            raise ValueError
        base = self.returns
        daily_cost = self.turnover["turnover"] * (effective_bps / 10_000.0)
        return base.with_columns((pl.col("returns") - daily_cost).alias("returns"))

    def trading_cost_impact(self, max_bps: int = 20) -> pl.DataFrame:
        """Estimate the impact of trading costs on the Sharpe ratio.

        Computes the annualised Sharpe ratio of cost-adjusted returns for
        each integer cost level from 0 up to and including ``max_bps`` basis
        points (1 bp = 0.01 %).  The result lets you quickly assess at what
        cost level the strategy's edge is eroded.

        Args:
            max_bps: Maximum one-way trading cost to evaluate, in basis
                points.  Defaults to 20 (i.e., evaluates 0, 1, 2, …, 20
                bps).  Must be a positive integer.

        Returns:
            pl.DataFrame: Frame with columns ``'cost_bps'`` (Int64) and
            ``'sharpe'`` (Float64), one row per cost level from 0 to
            ``max_bps`` inclusive.

        Raises:
            ValueError: If ``max_bps`` is not a positive integer.

        Examples:
            >>> import polars as pl
            >>> from datetime import date, timedelta
            >>> import numpy as np
            >>> start = date(2020, 1, 1)
            >>> dates = pl.date_range(
            ...     start=start, end=start + timedelta(days=99), interval="1d", eager=True
            ... )
            >>> rng = np.random.default_rng(0)
            >>> prices = pl.DataFrame({
            ...     "date": dates,
            ...     "A": pl.Series(np.cumprod(1 + rng.normal(0.001, 0.01, 100)) * 100),
            ... })
            >>> pos = pl.DataFrame({"date": dates, "A": pl.Series(np.ones(100) * 1000.0)})
            >>> pf = Portfolio(prices=prices, cashposition=pos, aum=1e5)
            >>> impact = pf.trading_cost_impact(max_bps=5)
            >>> list(impact["cost_bps"])
            [0, 1, 2, 3, 4, 5]
        """
        if not isinstance(max_bps, int) or max_bps < 1:
            raise ValueError
        import numpy as np

        periods = self.data._periods_per_year  # one Data object, outside the loop
        _eps = np.finfo(np.float64).eps
        sqrt_periods = float(np.sqrt(periods))
        cost_levels = list(range(0, max_bps + 1))

        # Extract base returns and turnover once — O(1) allocations regardless of max_bps
        base_rets = self.returns["returns"]
        turnover_s = self.turnover["turnover"]

        # Build all cost-adjusted return columns in one vectorised DataFrame construction,
        # then compute means and stds in a single aggregate pass (no per-iteration allocation).
        sweep = pl.DataFrame({str(bps): base_rets - turnover_s * (bps / 10_000.0) for bps in cost_levels})
        means_row = sweep.mean().row(0)
        stds_row = sweep.std(ddof=1).row(0)

        sharpe_values: list[float] = []
        for mean_raw, std_raw in zip(means_row, stds_row, strict=False):
            mean_val = 0.0 if mean_raw is None else float(mean_raw)
            if std_raw is None or float(std_raw) <= _eps * max(abs(mean_val), _eps) * 10:
                sharpe_values.append(float("nan"))
            else:
                sharpe_values.append(mean_val / float(std_raw) * sqrt_periods)
        return pl.DataFrame({"cost_bps": pl.Series(cost_levels, dtype=pl.Int64), "sharpe": pl.Series(sharpe_values)})

    # ── Utility ────────────────────────────────────────────────────────────────

    def correlation(self, frame: pl.DataFrame, name: str = "portfolio") -> pl.DataFrame:
        """Compute a correlation matrix of asset returns plus the portfolio.

        Computes percentage changes for all numeric columns in ``frame``,
        appends the portfolio profit series under the provided ``name``, and
        returns the Pearson correlation matrix across all numeric columns.

        Args:
            frame: A Polars DataFrame containing at least the asset price
                columns (and a date column which will be ignored if
                non-numeric).
            name: The column name to use when adding the portfolio profit
                series to the input frame.

        Returns:
            A square Polars DataFrame where each cell is the correlation
            between a pair of series (values in [-1, 1]).
        """
        p = frame.with_columns(cs.by_dtype(pl.Float32, pl.Float64).pct_change())
        p = p.with_columns(pl.Series(name, self.profit["profit"]))
        corr_matrix = p.select(cs.numeric()).fill_null(0.0).corr()
        return corr_matrix
