"""Pure data container for portfolio P&L computations.

This module provides :class:`PortfolioData`, a frozen dataclass that holds the
raw portfolio inputs (prices, cash positions, AUM) and computes the derived
data series (profits, NAV, returns, drawdown) without any analytics or
visualisation dependencies.

:class:`PortfolioData` is held by composition inside
:class:`~jquantstats.analytics.Portfolio`, which adds analytics facades (Stats,
Plots, Report), portfolio transforms (lag, truncate, smoothed_holding), and
attribution tools (tilt/timing decomposition, turnover, cost analysis).
"""

import dataclasses
from typing import Self

import polars as pl

from .exceptions import (
    InvalidCashPositionTypeError,
    InvalidPricesTypeError,
    MissingDateColumnError,
    NonPositiveAumError,
    RowCountMismatchError,
)


@dataclasses.dataclass(frozen=True)
class PortfolioData:
    """Pure data holder for portfolio inputs and derived series.

    Stores the three raw inputs — cash positions, prices, and AUM — and
    exposes the standard derived data series as lazy properties:

    - :attr:`profits` — per-asset daily cash P&L
    - :attr:`profit` — aggregate daily portfolio profit
    - :attr:`nav_accumulated` — cumulative additive NAV
    - :attr:`nav_compounded` — compounded NAV
    - :attr:`returns` — daily returns (profit / AUM)
    - :attr:`monthly` — monthly compounded returns
    - :attr:`highwater` — running high-water mark
    - :attr:`drawdown` — drawdown from high-water mark
    - :attr:`all` — merged view of all derived series

    This class deliberately carries no analytics, visualisation, or transform
    logic; those live in :class:`~jquantstats.analytics.Portfolio`, which holds a
    ``PortfolioData`` instance via composition.

    Attributes:
        cashposition: Polars DataFrame of per-asset cash positions over time
            (may include a ``date`` column).
        prices: Polars DataFrame of per-asset price levels over time
            (may include a ``date`` column).
        aum: Assets under management used as the base NAV offset.

    Examples:
        >>> import polars as pl
        >>> from datetime import date
        >>> prices = pl.DataFrame({"date": [date(2020, 1, 1), date(2020, 1, 2)], "A": [100.0, 110.0]})
        >>> pos = pl.DataFrame({"date": [date(2020, 1, 1), date(2020, 1, 2)], "A": [1000.0, 1000.0]})
        >>> pd = PortfolioData(prices=prices, cashposition=pos, aum=1e6)
        >>> pd.assets
        ['A']
    """

    cashposition: pl.DataFrame
    prices: pl.DataFrame
    aum: float

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

    # ── Factory class methods ──────────────────────────────────────────────────

    @classmethod
    def from_risk_position(
        cls,
        prices: pl.DataFrame,
        risk_position: pl.DataFrame,
        aum: float,
        vola: int | dict[str, int] = 32,
    ) -> Self:
        """Create a PortfolioData from per-asset risk positions.

        De-volatizes each risk position using an EWMA volatility estimate
        derived from the corresponding price series.

        Args:
            prices: Price levels per asset over time (may include a date column).
            risk_position: Risk units per asset (e.g., target risk exposure)
                aligned with prices.
            vola: EWMA lookback (span-equivalent) used to estimate volatility
                in trading days.  Pass an ``int`` to apply the same span to
                every asset, or a ``dict[str, int]`` to set a per-asset span
                (assets absent from the dict default to ``32``).
            aum: Assets under management used as the base NAV offset.

        Returns:
            A PortfolioData instance whose cash positions are risk_position
            divided by EWMA volatility.
        """
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
        return cls(prices=prices, cashposition=cash_position, aum=aum)

    @classmethod
    def from_cash_position(cls, prices: pl.DataFrame, cash_position: pl.DataFrame, aum: float) -> Self:
        """Create a PortfolioData directly from cash positions aligned with prices.

        Args:
            prices: Price levels per asset over time (may include a date column).
            cash_position: Cash exposure per asset over time (same shape/index
                as prices).
            aum: Assets under management used as the base NAV offset.

        Returns:
            A PortfolioData instance with the provided cash positions.
        """
        return cls(prices=prices, cashposition=cash_position, aum=aum)

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
            >>> pd = PortfolioData(prices=prices, cashposition=pos, aum=1e6)
            >>> pd.profits.columns
            ['A']
        """
        assets = [c for c in self.prices.columns if self.prices[c].dtype.is_numeric()]

        profits = self.prices.with_columns(
            (self.prices[asset].pct_change().fill_null(0.0) * self.cashposition[asset].shift(n=1).fill_null(0.0)).alias(
                asset
            )
            for asset in assets
        )

        if assets:
            profits = profits.with_columns(
                pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(0.0).fill_null(0.0).alias(c) for c in assets
            )
        return profits

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
