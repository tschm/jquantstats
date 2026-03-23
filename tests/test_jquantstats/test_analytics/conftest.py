"""Fixtures for analytics tests.

Security note: Test files use assert statements (S101) which are safe here as
they are only executed by pytest, never in production code.
"""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from jquantstats import Portfolio


@pytest.fixture
def portfolio() -> Portfolio:
    """Shared Portfolio fixture: 60-day, 2-asset deterministic portfolio."""
    n = 60
    start = date(2020, 1, 1)
    end = start + timedelta(days=n - 1)
    dates = pl.date_range(start=start, end=end, interval="1d", eager=True).cast(pl.Date)

    a = pl.Series([100.0 * (1.01**i) for i in range(n)], dtype=pl.Float64)
    b = pl.Series([200.0 + 5.0 * np.sin(0.15 * i) for i in range(n)], dtype=pl.Float64)
    prices = pl.DataFrame({"date": dates, "A": a, "B": b})

    pos_a = pl.Series([1000.0 + 2.0 * i for i in range(n)], dtype=pl.Float64)
    pos_b = pl.Series([500.0 + (i % 3) for i in range(n)], dtype=pl.Float64)
    positions = pl.DataFrame({"date": dates, "A": pos_a, "B": pos_b})

    return Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e6)
