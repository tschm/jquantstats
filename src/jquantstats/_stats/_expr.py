"""Pure ``pl.Expr`` factory functions for scalar financial statistics.

Each function takes a column name string and optional scalar parameters and
returns a named ``pl.Expr`` (i.e. the expression already carries ``.alias(col)``).
These functions contain no ``self``, no ``pl.Series``, and perform no eager
materialisation — they are designed to be passed to
:func:`~jquantstats._stats._core._lazy_columnwise` so that all per-asset
aggregations are fused into a single :py:meth:`polars.LazyFrame.select` call.

Usage example::

    from jquantstats._stats._expr import sharpe_expr
    from jquantstats._stats._core import _lazy_columnwise

    result = _lazy_columnwise(data.lazy, sharpe_expr, data.assets, periods=252.0)
    # result → {"AAPL": 1.23, "META": 0.87}

Stats that cannot be expressed as a pure ``pl.Expr`` (e.g. those requiring
multi-column access, complex scipy calls, or conditional Python logic such as
``greeks``, ``r_squared``, ``information_ratio``, ``prob_sharpe_ratio``,
``hhi_positive``, ``hhi_negative``, and ``sharpe_variance``) are **not** present
here and remain on the old ``@columnwise_stat`` path.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import norm as scipy_norm

# ── Basic statistics ─────────────────────────────────────────────────────────


def skew_expr(col: str) -> pl.Expr:
    """Return an expression computing the unbiased skewness of *col*.

    Args:
        col: Column name.

    Returns:
        pl.Expr: Named expression aliased to *col*.

    """
    return pl.col(col).skew(bias=False).alias(col)


def kurtosis_expr(col: str) -> pl.Expr:
    """Return an expression computing the unbiased excess kurtosis of *col*.

    Args:
        col: Column name.

    Returns:
        pl.Expr: Named expression aliased to *col*.

    """
    return pl.col(col).kurtosis(bias=False).alias(col)


def avg_return_expr(col: str) -> pl.Expr:
    """Mean of non-null, non-zero values in *col*.

    Args:
        col: Column name.

    Returns:
        pl.Expr: Named expression aliased to *col*.

    """
    return pl.col(col).filter(pl.col(col).is_not_null() & (pl.col(col) != 0)).mean().alias(col)


def avg_win_expr(col: str) -> pl.Expr:
    """Mean of strictly positive values in *col*.

    Args:
        col: Column name.

    Returns:
        pl.Expr: Named expression aliased to *col*.

    """
    return pl.col(col).filter(pl.col(col) > 0).mean().alias(col)


def avg_loss_expr(col: str) -> pl.Expr:
    """Mean of strictly negative values in *col*.

    Args:
        col: Column name.

    Returns:
        pl.Expr: Named expression aliased to *col*.

    """
    return pl.col(col).filter(pl.col(col) < 0).mean().alias(col)


# ── Volatility ────────────────────────────────────────────────────────────────


def volatility_expr(col: str, periods: float, annualize: bool = True) -> pl.Expr:
    """Standard deviation of *col*, optionally annualised by ``sqrt(periods)``.

    Args:
        col: Column name.
        periods: Number of observations per year used for annualisation.
        annualize: Multiply by ``sqrt(periods)`` when ``True`` (default).

    Returns:
        pl.Expr: Named expression aliased to *col*.

    """
    factor = float(np.sqrt(periods)) if annualize else 1.0
    return (pl.col(col).std() * factor).alias(col)


# ── Win / loss metrics ────────────────────────────────────────────────────────


def payoff_ratio_expr(col: str) -> pl.Expr:
    """``avg_win / |avg_loss|`` for *col*.

    Args:
        col: Column name.

    Returns:
        pl.Expr: Named expression aliased to *col*.

    """
    avg_win = pl.col(col).filter(pl.col(col) > 0).mean()
    avg_loss = pl.col(col).filter(pl.col(col) < 0).mean().abs()
    return (avg_win / avg_loss).alias(col)


def profit_factor_expr(col: str) -> pl.Expr:
    """``sum(positives) / |sum(negatives)|`` for *col*.

    Args:
        col: Column name.

    Returns:
        pl.Expr: Named expression aliased to *col*.

    """
    wins_sum = pl.col(col).filter(pl.col(col) > 0).sum()
    losses_sum = pl.col(col).filter(pl.col(col) < 0).sum().abs()
    return (wins_sum / losses_sum).alias(col)


def profit_ratio_expr(col: str) -> pl.Expr:
    r"""``(|mean_win| / count_win) / (|mean_loss| / count_loss)`` for *col*.

    Returns ``NaN`` when there are no wins or no losses.

    Args:
        col: Column name.

    Returns:
        pl.Expr: Named expression aliased to *col*.

    """
    win_mean = pl.col(col).filter(pl.col(col) >= 0).mean()
    loss_mean = pl.col(col).filter(pl.col(col) < 0).mean()
    win_count = pl.col(col).filter(pl.col(col) >= 0).count().cast(pl.Float64)
    loss_count = pl.col(col).filter(pl.col(col) < 0).count().cast(pl.Float64)

    win_ratio = win_mean.abs() / win_count
    loss_ratio = loss_mean.abs() / loss_count

    return (
        pl.when((win_count > 0) & (loss_count > 0)).then(win_ratio / loss_ratio).otherwise(pl.lit(float("nan")))
    ).alias(col)


def win_rate_expr(col: str) -> pl.Expr:
    """``count(>0) / count(!=0)`` for *col*.

    Args:
        col: Column name.

    Returns:
        pl.Expr: Named expression aliased to *col*.

    """
    num_pos = (pl.col(col) > 0).sum()
    num_nonzero = (pl.col(col) != 0).sum()
    return (num_pos / num_nonzero).alias(col)


def gain_to_pain_ratio_expr(col: str) -> pl.Expr:
    """``sum / |sum(negatives)|`` for *col*, or ``NaN`` when sum of negatives is zero.

    Args:
        col: Column name.

    Returns:
        pl.Expr: Named expression aliased to *col*.

    """
    total_gain = pl.col(col).sum()
    total_pain = pl.col(col).filter(pl.col(col) < 0).abs().sum()
    return (pl.when(total_pain != 0).then(total_gain / total_pain).otherwise(pl.lit(float("nan")))).alias(col)


def risk_return_ratio_expr(col: str) -> pl.Expr:
    """``mean / std`` for *col*.

    Args:
        col: Column name.

    Returns:
        pl.Expr: Named expression aliased to *col*.

    """
    return (pl.col(col).mean() / pl.col(col).std()).alias(col)


def best_expr(col: str) -> pl.Expr:
    """Maximum value of *col*.

    Args:
        col: Column name.

    Returns:
        pl.Expr: Named expression aliased to *col*.

    """
    return pl.col(col).max().alias(col)


def worst_expr(col: str) -> pl.Expr:
    """Minimum value of *col*.

    Args:
        col: Column name.

    Returns:
        pl.Expr: Named expression aliased to *col*.

    """
    return pl.col(col).min().alias(col)


def exposure_expr(col: str) -> pl.Expr:
    """``count(!=0) / total_rows`` rounded to 2 decimal places for *col*.

    Args:
        col: Column name.

    Returns:
        pl.Expr: Named expression aliased to *col*.

    """
    return ((pl.col(col) != 0).sum() / pl.len()).round(2).alias(col)


# ── Risk metrics ──────────────────────────────────────────────────────────────


def _norm_ppf(alpha: float) -> float:
    """Return the standard-normal quantile for *alpha* (cached via scipy)."""
    return float(scipy_norm.ppf(alpha))


def value_at_risk_expr(col: str, sigma: float = 1.0, alpha: float = 0.05) -> pl.Expr:
    """Variance-covariance Value-at-Risk for *col*.

    Uses ``norm.ppf(alpha, mu, sigma * std)`` expressed as a pure ``pl.Expr``:
    ``mu + sigma * std * z`` where ``z = norm.ppf(alpha)`` is precomputed.

    Args:
        col: Column name.
        sigma: Standard deviation multiplier. Defaults to ``1.0``.
        alpha: Confidence level. Defaults to ``0.05``.

    Returns:
        pl.Expr: Named expression aliased to *col*.

    """
    z = _norm_ppf(alpha)
    return (pl.col(col).mean() + sigma * pl.col(col).std() * z).alias(col)


def conditional_value_at_risk_expr(col: str, sigma: float = 1.0, alpha: float = 0.05) -> pl.Expr:
    """Conditional Value-at-Risk (CVaR / Expected Shortfall) for *col*.

    Computes ``E[X | X < VaR]`` where VaR is the variance-covariance estimate.

    Args:
        col: Column name.
        sigma: Standard deviation multiplier. Defaults to ``1.0``.
        alpha: Confidence level. Defaults to ``0.05``.

    Returns:
        pl.Expr: Named expression aliased to *col*.

    """
    z = _norm_ppf(alpha)
    # VaR threshold (scalar aggregation, broadcast for comparison)
    var_e = pl.col(col).mean() + sigma * pl.col(col).std() * z
    # E[X | X < VaR] — null-fill for values above VaR, then take mean (nulls ignored)
    return pl.when(pl.col(col) < var_e).then(pl.col(col)).otherwise(None).mean().alias(col)


# ── Performance ratios ────────────────────────────────────────────────────────


def sharpe_expr(col: str, periods: float) -> pl.Expr:
    """Annualised Sharpe ratio for *col*, ``NaN`` when std ≈ 0.

    Args:
        col: Column name.
        periods: Number of observations per year.

    Returns:
        pl.Expr: Named expression aliased to *col*.

    """
    mean_e = pl.col(col).mean()
    std_e = pl.col(col).std(ddof=1)
    scale = float(np.sqrt(periods))
    eps = float(np.finfo(np.float64).eps) * 10
    return (pl.when(std_e.abs() > eps).then(mean_e / std_e * scale).otherwise(pl.lit(float("nan")))).alias(col)


def sortino_expr(col: str, periods: float) -> pl.Expr:
    """Annualised Sortino ratio for *col*.

    Returns ``inf`` / ``-inf`` / ``NaN`` when downside deviation is zero,
    following Red Rock Capital's convention.

    Args:
        col: Column name.
        periods: Number of observations per year.

    Returns:
        pl.Expr: Named expression aliased to *col*.

    """
    scale = float(np.sqrt(periods))
    mean_e = pl.col(col).mean()
    downside_sum = (pl.col(col).filter(pl.col(col) < 0) ** 2).sum()
    n = pl.col(col).count()
    downside_dev = (downside_sum / n).sqrt()

    return (
        pl.when(downside_dev > 0)
        .then(mean_e / downside_dev * scale)
        .otherwise(
            pl.when(mean_e > 0)
            .then(pl.lit(float("inf")))
            .when(mean_e < 0)
            .then(pl.lit(float("-inf")))
            .otherwise(pl.lit(float("nan")))
        )
    ).alias(col)


# ── Drawdown helpers ──────────────────────────────────────────────────────────


def _cum_sum_nav_drawdown_expr(col: str) -> pl.Expr:
    """Per-row drawdown using additive cumsum NAV (matches ``_drawdown_series``).

    Args:
        col: Column name.

    Returns:
        pl.Expr: Per-row drawdown expression (not aliased, for internal reuse).

    """
    nav = 1.0 + pl.col(col).cast(pl.Float64).cum_sum()
    hwm = nav.cum_max()
    hwm_safe = hwm.clip(lower_bound=1e-10)
    return ((hwm - nav) / hwm_safe).clip(lower_bound=0.0)


def max_drawdown_expr(col: str) -> pl.Expr:
    """Maximum drawdown using multiplicative ``cum_prod`` NAV for *col*.

    Returns the max drawdown as a positive fraction (e.g. ``0.2`` for 20 %).

    Args:
        col: Column name.

    Returns:
        pl.Expr: Named expression aliased to *col*.

    """
    price = (1.0 + pl.col(col)).cum_prod()
    peak = price.cum_max()
    dd_min = (price / peak - 1.0).min()
    return (-dd_min).alias(col)


def avg_drawdown_expr(col: str) -> pl.Expr:
    """Mean of positive drawdown values (additive NAV), ``0.0`` when no drawdown.

    Args:
        col: Column name.

    Returns:
        pl.Expr: Named expression aliased to *col*.

    """
    dd = _cum_sum_nav_drawdown_expr(col)
    return pl.when(dd > 0).then(dd).otherwise(None).mean().fill_null(0.0).alias(col)


def calmar_expr(col: str, periods: float) -> pl.Expr:
    """Calmar ratio (annualised return / max drawdown) for *col*.

    Returns ``NaN`` when max drawdown is zero.

    Args:
        col: Column name.
        periods: Number of observations per year.

    Returns:
        pl.Expr: Named expression aliased to *col*.

    """
    dd = _cum_sum_nav_drawdown_expr(col)
    max_dd = dd.max()
    ann_return = pl.col(col).mean() * periods
    return (pl.when(max_dd > 0).then(ann_return / max_dd).otherwise(pl.lit(float("nan")))).alias(col)


def recovery_factor_expr(col: str) -> pl.Expr:
    """Recovery factor (total return / max drawdown) for *col*.

    Returns ``NaN`` when max drawdown is zero.

    Args:
        col: Column name.

    Returns:
        pl.Expr: Named expression aliased to *col*.

    """
    dd = _cum_sum_nav_drawdown_expr(col)
    max_dd = dd.max()
    total_return = pl.col(col).sum()
    return (pl.when(max_dd > 0).then(total_return / max_dd).otherwise(pl.lit(float("nan")))).alias(col)
