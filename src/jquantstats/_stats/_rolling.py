"""Rolling-window statistical metrics for financial returns data."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl

from ._core import to_frame

# ── Rolling statistics mixin ─────────────────────────────────────────────────


class _RollingStatsMixin:
    """Mixin class providing rolling-window financial statistics methods.

    Separates rolling-window computations from the core point-in-time metrics
    in :mod:`~jquantstats._stats._core`.  The concrete
    :class:`~jquantstats._stats.Stats` dataclass inherits from both.

    Attributes (provided by the concrete subclass):
        data: The :class:`~jquantstats._data.Data` object.
        all: Combined DataFrame for efficient column selection.
    """

    if TYPE_CHECKING:
        from ._protocol import DataLike

        data: DataLike
        all: pl.DataFrame | None

    @to_frame
    def rolling_sortino(
        self, series: pl.Expr, rolling_period: int = 126, periods_per_year: int | float | None = None
    ) -> pl.Expr:
        """Calculate the rolling Sortino ratio.

        Args:
            series (pl.Expr): The expression to calculate rolling Sortino ratio for.
            rolling_period (int, optional): The rolling window size. Defaults to 126.
            periods_per_year (int, optional): Number of periods per year. Defaults to 252.

        Returns:
            pl.Expr: The rolling Sortino ratio expression.

        """
        ppy = periods_per_year or self.data._periods_per_year

        mean_ret = series.rolling_mean(window_size=rolling_period)

        # Rolling downside deviation (squared negative returns averaged over window)
        downside = series.map_elements(lambda x: x**2 if x < 0 else 0.0, return_dtype=pl.Float64).rolling_mean(
            window_size=rolling_period
        )

        # Avoid division by zero
        sortino = mean_ret / downside.sqrt().fill_nan(0).fill_null(0)
        return cast(pl.Expr, sortino * (ppy**0.5))

    def rolling_sharpe(
        self,
        rolling_period: int = 126,
        periods_per_year: int | float | None = None,
    ) -> pl.DataFrame:
        """Calculate the rolling Sharpe ratio.

        Args:
            rolling_period: Rolling window size. Defaults to 126.
            periods_per_year: Periods per year for annualisation.

        Returns:
            pl.DataFrame: Date column(s) plus one annualised rolling Sharpe
            column per asset.

        Raises:
            ValueError: If rolling_period is not a positive integer.

        """
        actual_window = rolling_period
        actual_periods = periods_per_year or self.data._periods_per_year
        if not isinstance(actual_window, int) or actual_window <= 0:
            raise ValueError("rolling_period must be a positive integer")  # noqa: TRY003
        scale = float(np.sqrt(actual_periods))
        return cast(pl.DataFrame, self.all).select(
            [pl.col(name) for name in self.data.date_col]
            + [
                (
                    pl.col(col).rolling_mean(window_size=actual_window)
                    / pl.col(col).rolling_std(window_size=actual_window)
                    * scale
                ).alias(col)
                for col, _ in self.data.items()
            ]
        )

    def rolling_volatility(
        self,
        rolling_period: int = 126,
        periods_per_year: int | float | None = None,
        annualize: bool = True,
    ) -> pl.DataFrame:
        """Calculate the rolling volatility of returns.

        Args:
            rolling_period: Rolling window size. Defaults to 126.
            periods_per_year: Periods per year for annualisation.
            annualize: Multiply by ``sqrt(periods_per_year)`` when True (default).

        Returns:
            pl.DataFrame: Date column(s) plus one rolling volatility column
            per asset.

        Raises:
            ValueError: If rolling_period is not a positive integer.
            TypeError: If periods_per_year is not numeric.

        """
        actual_window = rolling_period
        actual_periods = periods_per_year or self.data._periods_per_year
        if not isinstance(actual_window, int) or actual_window <= 0:
            raise ValueError("rolling_period must be a positive integer")  # noqa: TRY003
        if not isinstance(actual_periods, int | float):
            raise TypeError
        factor = float(np.sqrt(actual_periods)) if annualize else 1.0
        return cast(pl.DataFrame, self.all).select(
            [pl.col(name) for name in self.data.date_col]
            + [(pl.col(col).rolling_std(window_size=actual_window) * factor).alias(col) for col, _ in self.data.items()]
        )
