"""Shared fixtures for Portfolio test suite."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from jquantstats import Portfolio


@pytest.fixture
def prices():
    """Three-day, two-asset price frame with exact geometric ratios.

    Asset A: 100 → 110 → 121 (10 % gain each day).
    Asset B: 200 → 180 → 198 (−10 % then +10 %).
    """
    return pl.DataFrame(
        {
            "date": pl.date_range(start=date(2020, 1, 1), end=date(2020, 1, 3), interval="1d", eager=True).cast(
                pl.Date
            ),
            "A": pl.Series([100.0, 110.0, 121.0], dtype=pl.Float64),
            "B": pl.Series([200.0, 180.0, 198.0], dtype=pl.Float64),
        }
    )


@pytest.fixture
def positions():
    """Three-day cash-position frame aligned with the prices fixture.

    Asset A: 1 000 units held throughout.
    Asset B: 0 on day 1, 500 from day 2 onward.
    """
    return pl.DataFrame(
        {
            "date": pl.date_range(start=date(2020, 1, 1), end=date(2020, 1, 3), interval="1d", eager=True).cast(
                pl.Date
            ),
            "A": pl.Series([1000.0, 1000.0, 1000.0], dtype=pl.Float64),
            "B": pl.Series([0.0, 500.0, 500.0], dtype=pl.Float64),
        }
    )


@pytest.fixture
def portfolio(prices, positions):
    """Create Portfolio instance for testing (3-day, exact numeric values)."""
    return Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e5)


@pytest.fixture
def monthly_portfolio():
    """Build a small deterministic Portfolio for monthly aggregation tests."""
    start = date(2020, 1, 10)
    days = 80
    end = start + timedelta(days=days - 1)
    dates = pl.date_range(start=start, end=end, interval="1d", eager=True).cast(pl.Date)

    rng = np.random.default_rng(42)
    returns = rng.normal(0.0, 0.01, size=days)
    prices_arr = (1.0 + returns).cumprod()

    return Portfolio(
        prices=pl.DataFrame({"date": dates, "A": prices_arr}),
        cashposition=pl.DataFrame({"date": dates, "A": pl.Series([1000.0] * days, dtype=pl.Float64)}),
        aum=10000,
    )


@pytest.fixture
def truncate_portfolio():
    """Small 6-day portfolio so truncation height assertions are exact."""
    n = 6
    start = date(2020, 1, 1)
    end = start + timedelta(days=n - 1)
    dates = pl.date_range(start=start, end=end, interval="1d", eager=True).cast(pl.Date)
    return Portfolio.from_cash_position(
        prices=pl.DataFrame(
            {
                "date": dates,
                "A": pl.Series([100.0 + 10.0 * i for i in range(n)], dtype=pl.Float64),
                "B": pl.Series([200.0 - 5.0 * i for i in range(n)], dtype=pl.Float64),
            }
        ),
        cash_position=pl.DataFrame(
            {
                "date": dates,
                "A": pl.Series([1000.0] * n, dtype=pl.Float64),
                "B": pl.Series([500.0] * n, dtype=pl.Float64),
            }
        ),
        aum=1e6,
    )


@pytest.fixture
def int_portfolio():
    """A small Portfolio with no 'date' column (integer-indexed rows)."""
    n = 6
    return Portfolio.from_cash_position(
        prices=pl.DataFrame(
            {
                "A": pl.Series([100.0 + 10.0 * i for i in range(n)], dtype=pl.Float64),
                "B": pl.Series([200.0 - 5.0 * i for i in range(n)], dtype=pl.Float64),
            }
        ),
        cash_position=pl.DataFrame(
            {
                "A": pl.Series([1000.0] * n, dtype=pl.Float64),
                "B": pl.Series([500.0] * n, dtype=pl.Float64),
            }
        ),
        aum=1e6,
    )


@pytest.fixture
def turnover_portfolio():
    """A 10-day, two-asset portfolio with linearly changing positions.

    Asset A increases by 100 each day (position_t = 100 * t).
    Asset B decreases by 50 each day (position_t = 500 - 50 * t).
    AUM = 10_000.
    Daily turnover = (|ΔA| + |ΔB|) / AUM = (100 + 50) / 10_000 = 0.015 per day
    (except row 0 which is 0.0).
    """
    n = 10
    start = date(2020, 1, 1)
    end = start + timedelta(days=n - 1)
    dates = pl.date_range(start=start, end=end, interval="1d", eager=True).cast(pl.Date)
    return Portfolio.from_cash_position(
        prices=pl.DataFrame({"date": dates, "A": pl.Series([100.0] * n), "B": pl.Series([200.0] * n)}),
        cash_position=pl.DataFrame(
            {
                "date": dates,
                "A": pl.Series([100.0 * i for i in range(n)], dtype=pl.Float64),
                "B": pl.Series([500.0 - 50.0 * i for i in range(n)], dtype=pl.Float64),
            }
        ),
        aum=10_000.0,
    )
