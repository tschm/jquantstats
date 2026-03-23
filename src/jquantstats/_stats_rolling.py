"""Rolling-window statistical metrics for financial returns data."""

from __future__ import annotations

from typing import cast

import numpy as np
import polars as pl

from ._stats_core import to_frame

# ── Rolling statistics mixin ─────────────────────────────────────────────────


class _RollingStatsMixin:
    """Mixin class providing rolling-window financial statistics methods.

    Separates rolling-window computations from the core point-in-time metrics
    in :class:`~jquantstats._stats_core._CoreStatsMixin`.  The concrete
    :class:`~jquantstats._stats.Stats` dataclass inherits from both.

    Attributes (provided by the concrete subclass):
        data: The :class:`~jquantstats._data.Data` object.
        all: Combined DataFrame for efficient column selection.
    """

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
        window: int | None = None,
        periods: int | float | None = None,
        rolling_period: int | None = None,
        periods_per_year: int | float | None = None,
    ) -> pl.DataFrame:
        """Calculate the rolling Sharpe ratio.

        Accepts both the analytics-style (``window``, ``periods``) and the
        legacy-style (``rolling_period``, ``periods_per_year``) keyword
        arguments so that callers using either convention continue to work.

        Args:
            window: Rolling window size (analytics style). Defaults to 126.
            periods: Periods per year for annualisation (analytics style).
            rolling_period: Alias for ``window`` (legacy style).
            periods_per_year: Alias for ``periods`` (legacy style).

        Returns:
            pl.DataFrame: Date column(s) plus one annualised rolling Sharpe
            column per asset.

        Raises:
            ValueError: If the effective window size is not a positive integer.

        """
        actual_window = window if window is not None else (rolling_period if rolling_period is not None else 126)
        actual_periods = periods or periods_per_year or self.data._periods_per_year
        if not isinstance(actual_window, int) or actual_window <= 0:
            raise ValueError("window must be a positive integer")  # noqa: TRY003
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
        window: int | None = None,
        periods: int | float | None = None,
        annualize: bool = True,
        rolling_period: int | None = None,
        periods_per_year: int | float | None = None,
    ) -> pl.DataFrame:
        """Calculate the rolling volatility of returns.

        Accepts both the analytics-style (``window``, ``periods``,
        ``annualize``) and the legacy-style (``rolling_period``,
        ``periods_per_year``) keyword arguments.

        Args:
            window: Rolling window size (analytics style). Defaults to 126.
            periods: Periods per year for annualisation (analytics style).
            annualize: Multiply by ``sqrt(periods)`` when True (default).
            rolling_period: Alias for ``window`` (legacy style).
            periods_per_year: Alias for ``periods`` (legacy style).

        Returns:
            pl.DataFrame: Date column(s) plus one rolling volatility column
            per asset.

        Raises:
            ValueError: If the effective window size is not a positive integer.
            TypeError: If the effective periods value is not numeric.

        """
        actual_window = window if window is not None else (rolling_period if rolling_period is not None else 126)
        actual_periods = periods or periods_per_year or self.data._periods_per_year
        if not isinstance(actual_window, int) or actual_window <= 0:
            raise ValueError("window must be a positive integer")  # noqa: TRY003
        if not isinstance(actual_periods, int | float):
            raise TypeError
        factor = float(np.sqrt(actual_periods)) if annualize else 1.0
        return cast(pl.DataFrame, self.all).select(
            [pl.col(name) for name in self.data.date_col]
            + [(pl.col(col).rolling_std(window_size=actual_window) * factor).alias(col) for col, _ in self.data.items()]
        )
