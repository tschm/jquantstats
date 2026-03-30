"""Shared computational helpers for statistics mixin modules.

Contains pure, reusable sub-computations that are used by two or more of
the stats mixin modules (:mod:`~jquantstats._stats._basic`,
:mod:`~jquantstats._stats._performance`,
:mod:`~jquantstats._stats._reporting`,
:mod:`~jquantstats._stats._rolling`).

Helpers
-------
:func:`_comp_return`
    Total compounded return: ``∏(1 + rᵢ) - 1``.
:func:`_nav_series`
    Cumulative NAV (price) series: ``cum_prod(1 + rᵢ)``.
:func:`_annualization_factor`
    ``sqrt(periods)`` or ``periods`` for annualizing rates and ratios.
:func:`_downside_deviation`
    Downside semi-deviation used by the Sortino ratio family.

All functions operate on a :class:`polars.Series` of returns and are
intentionally free of side-effects so that they are easy to test and
compose.
"""

from __future__ import annotations

import math

import polars as pl


def _comp_return(series: pl.Series) -> float:
    """Compute the total compounded return over a full period.

    Computed as ``∏(1 + rᵢ) - 1`` after dropping null values and casting
    to ``Float64``.

    Args:
        series: A Polars Series of per-period returns.

    Returns:
        float: Total compounded return.

    Examples:
        >>> import polars as pl
        >>> s = pl.Series([0.1, -0.05, 0.2])
        >>> round(_comp_return(s), 4)
        0.254

        >>> _comp_return(pl.Series([], dtype=pl.Float64))
        0.0
    """
    return float((1.0 + series.drop_nulls().cast(pl.Float64)).product()) - 1.0


def _nav_series(series: pl.Series) -> pl.Series:
    """Convert a returns series to a cumulative NAV (price) series.

    Computed as ``cum_prod(1 + rᵢ)``, which gives the value of one unit of
    currency invested at the start of the series.

    Args:
        series: A Polars Series of per-period returns.

    Returns:
        pl.Series: Float64 series of cumulative NAV values (starts at ``1 + r₁``).

    Examples:
        >>> import polars as pl
        >>> s = pl.Series([0.0, 0.1, -0.1])
        >>> [round(x, 10) for x in _nav_series(s).to_list()]
        [1.0, 1.1, 0.99]
    """
    return (1.0 + series.cast(pl.Float64)).cum_prod()


def _annualization_factor(periods: int | float, sqrt: bool = True) -> float:
    """Return the annualization factor for a given number of periods per year.

    When ``sqrt=True`` (the default) returns ``sqrt(periods)``, which is used
    to annualize ratios such as Sharpe and Sortino.  When ``sqrt=False``
    returns ``periods`` directly, which is used for linear scaling.

    Args:
        periods: Number of observations per calendar year (e.g. 252 for daily
            equity returns).
        sqrt: Whether to return the square-root scaling factor.  Defaults to
            ``True``.

    Returns:
        float: ``sqrt(periods)`` when *sqrt* is ``True``, else ``periods``.

    Raises:
        ValueError: If *periods* is not a positive finite number.

    Examples:
        >>> _annualization_factor(252)
        15.874507866387544

        >>> _annualization_factor(252, sqrt=False)
        252.0

        >>> _annualization_factor(12)
        3.4641016151377544
    """
    if periods <= 0 or not math.isfinite(periods):
        raise ValueError(f"periods must be a positive finite number, got {periods!r}")  # noqa: TRY003
    return math.sqrt(periods) if sqrt else float(periods)


def _downside_deviation(series: pl.Series) -> float:
    r"""Compute the downside semi-deviation of a returns series.

    Calculates the root-mean-square of all *negative* returns:

    .. math::

        \text{downside\_dev} = \sqrt{\frac{\sum_{r_t < 0} r_t^2}{N}}

    where *N* is the total number of observations (not just the negative ones).
    This is the convention used in the Red Rock Capital Sortino ratio paper and
    matches quantstats' implementation.

    Args:
        series: A Polars Series of per-period returns.

    Returns:
        float: Downside semi-deviation (always non-negative).

    Examples:
        >>> import polars as pl
        >>> s = pl.Series([0.05, -0.02, 0.03, -0.01, 0.0])
        >>> round(_downside_deviation(s), 10)
        0.01

        >>> _downside_deviation(pl.Series([0.1, 0.2, 0.3]))
        0.0
    """
    downside_sum = float(((series.filter(series < 0)) ** 2).sum())
    n = series.count()
    if n == 0:
        return 0.0
    return math.sqrt(downside_sum / n)
