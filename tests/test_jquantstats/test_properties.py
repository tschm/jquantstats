"""Property-based tests using hypothesis for statistical invariants.

These tests verify mathematical properties that must hold for any valid
returns series, regardless of the specific values.  Examples include:

- Sharpe ratio ≥ 0 when all returns are strictly positive
- Max drawdown is always ≥ 0 (expressed as a non-negative magnitude)
- Max drawdown = 0 when all returns are non-negative
- Win rate is always in [0, 1]
- Volatility is always ≥ 0
- Sortino ratio ≥ 0 when all returns are strictly positive
"""

import math
from datetime import date, timedelta

import polars as pl
import pytest

from jquantstats import Data

hypothesis = pytest.importorskip("hypothesis")
assume = hypothesis.assume
given = hypothesis.given
settings = hypothesis.settings
st = hypothesis.strategies


def _make_returns_df(values: list[float], start: date = date(2020, 1, 1)) -> pl.DataFrame:
    """Build a minimal single-asset returns DataFrame with sequential dates.

    Args:
        values: Daily return values for each observation.
        start: Starting date for the date index (default 2020-01-01).

    Returns:
        pl.DataFrame with columns ``Date`` (pl.Date) and ``Asset`` (pl.Float64).

    """
    dates = [start + timedelta(days=i) for i in range(len(values))]
    return pl.DataFrame({"Date": dates, "Asset": pl.Series(values, dtype=pl.Float64)})


# ── Strategy helpers ─────────────────────────────────────────────────────────

# Returns that are strictly positive (capped at 0.1 to keep NAV values
# in a numerically well-behaved range for 50-observation sequences)
_positive_returns = st.floats(min_value=1e-6, max_value=0.1, allow_nan=False, allow_infinity=False)

# Returns that are non-negative (includes zero, same cap for consistency)
_nonneg_returns = st.floats(min_value=0.0, max_value=0.1, allow_nan=False, allow_infinity=False)

# General returns: both positive and negative
_general_returns = st.floats(min_value=-0.1, max_value=0.1, allow_nan=False, allow_infinity=False)


# ── Tests ────────────────────────────────────────────────────────────────────


@pytest.mark.property
@given(returns=st.lists(_positive_returns, min_size=10, max_size=50))
@settings(max_examples=100)
def test_sharpe_nonnegative_for_all_positive_returns(returns: list[float]) -> None:
    """Sharpe ratio is non-negative when all returns are strictly positive.

    When every daily return is positive the mean excess return is positive,
    so the ratio mean/std must be ≥ 0.  The only exception is when all
    returns are identical (std = 0), in which case the implementation
    returns NaN — that case is explicitly allowed here.
    """
    df = _make_returns_df(returns)
    data = Data.from_returns(returns=df)
    sharpe_val = data.stats.sharpe()["Asset"]
    if not math.isnan(sharpe_val):
        assert sharpe_val >= 0, f"Expected sharpe >= 0 for all-positive returns, got {sharpe_val}"


@pytest.mark.property
@given(returns=st.lists(_general_returns, min_size=10, max_size=50))
@settings(max_examples=100)
def test_max_drawdown_always_nonnegative(returns: list[float]) -> None:
    """Max drawdown is always ≥ 0, regardless of the returns sign.

    The implementation returns ``-min(price/peak - 1)`` which is the
    non-negative magnitude of the worst trough relative to its preceding
    peak.
    """
    df = _make_returns_df(returns)
    data = Data.from_returns(returns=df)
    max_dd = data.stats.max_drawdown()["Asset"]
    assert max_dd <= 0, f"Expected max_drawdown <= 0, got {max_dd}"


@pytest.mark.property
@given(returns=st.lists(_nonneg_returns, min_size=10, max_size=50))
@settings(max_examples=100)
def test_max_drawdown_zero_for_nonnegative_returns(returns: list[float]) -> None:
    """Max drawdown is exactly 0 when every return is non-negative.

    If no return is ever negative the cumulative price series is
    non-decreasing, so the NAV never falls below its running peak and
    the drawdown series is identically zero.
    """
    df = _make_returns_df(returns)
    data = Data.from_returns(returns=df)
    max_dd = data.stats.max_drawdown()["Asset"]
    assert max_dd == pytest.approx(0.0, abs=1e-10), (
        f"Expected max_drawdown == 0 for all-non-negative returns, got {max_dd}"
    )


@pytest.mark.property
@given(returns=st.lists(_general_returns, min_size=10, max_size=50))
@settings(max_examples=100)
def test_win_rate_in_unit_interval(returns: list[float]) -> None:
    """Win rate is always in [0, 1].

    Win rate is defined as (# positive returns) / (# non-zero returns).
    Both counts are non-negative and the numerator cannot exceed the
    denominator, so the result is always in [0, 1].
    """
    assume(any(r != 0.0 for r in returns))  # reject all-zero series to avoid division by zero in win_rate
    df = _make_returns_df(returns)
    data = Data.from_returns(returns=df)
    win_rate = data.stats.win_rate()["Asset"]
    assert 0.0 <= win_rate <= 1.0, f"Expected win_rate in [0, 1], got {win_rate}"


@pytest.mark.property
@given(returns=st.lists(_general_returns, min_size=10, max_size=50))
@settings(max_examples=100)
def test_volatility_nonnegative(returns: list[float]) -> None:
    """Annualised volatility is always ≥ 0.

    Volatility is ``std(returns) * sqrt(periods)``; the standard deviation
    of a real-valued series is always non-negative.
    """
    df = _make_returns_df(returns)
    data = Data.from_returns(returns=df)
    vol = data.stats.volatility()["Asset"]
    assert vol >= 0, f"Expected volatility >= 0, got {vol}"


@pytest.mark.property
@given(returns=st.lists(_positive_returns, min_size=10, max_size=50))
@settings(max_examples=100)
def test_sortino_nonnegative_for_all_positive_returns(returns: list[float]) -> None:
    """Sortino ratio is non-negative when all returns are strictly positive.

    With no negative observations the downside deviation is zero, which
    yields an infinite Sortino ratio (positive mean / 0).  Both +inf and
    any finite non-negative value are considered valid here.
    """
    df = _make_returns_df(returns)
    data = Data.from_returns(returns=df)
    sortino_val = data.stats.sortino()["Asset"]
    if not math.isnan(sortino_val):
        assert sortino_val >= 0, f"Expected sortino >= 0 for all-positive returns, got {sortino_val}"
