"""Tests for Portfolio.from_risk_position factory method.

Covers: per-asset vola dict, vola validation, vol_cap, and cost parameter
forwarding.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from jquantstats import CostModel, Portfolio


def _make_prices_risk(n_days: int = 60) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Helper: build a simple price and risk-position frame of length n_days."""
    dates = pl.date_range(
        start=date(2020, 1, 1),
        end=date(2020, 1, 1) + __import__("datetime").timedelta(days=n_days - 1),
        interval="1d",
        eager=True,
    ).cast(pl.Date)
    prices = pl.DataFrame({"date": dates, "A": pl.Series(np.linspace(100, 120, n_days), dtype=pl.Float64)})
    risk = pl.DataFrame({"date": dates, "A": pl.Series([1.0] * n_days, dtype=pl.Float64)})
    return prices, risk


# ─── per-asset vola dict ──────────────────────────────────────────────────────


def test_from_risk_position_per_asset_vola():
    """Different vola values per asset produce different cash positions."""
    dates = pl.date_range(start=date(2020, 1, 1), end=date(2020, 3, 1), interval="1d", eager=True).cast(pl.Date)
    prices = pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series(np.linspace(100, 120, len(dates)), dtype=pl.Float64),
            "B": pl.Series(np.linspace(50, 60, len(dates)), dtype=pl.Float64),
        }
    )
    riskposition = pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series([1.0] * len(dates), dtype=pl.Float64),
            "B": pl.Series([1.0] * len(dates), dtype=pl.Float64),
        }
    )
    pf_uniform = Portfolio.from_risk_position(prices, riskposition, vola=8, aum=1e8)
    pf_per_asset = Portfolio.from_risk_position(prices, riskposition, vola={"A": 8, "B": 32}, aum=1e8)
    # Same vola for A → identical A columns
    assert pf_uniform.cashposition["A"].to_list() == pytest.approx(
        pf_per_asset.cashposition["A"].to_list(), nan_ok=True
    )
    # Different vola for B → different B columns
    b_uniform = pf_uniform.cashposition["B"].drop_nulls().to_list()
    b_per_asset = pf_per_asset.cashposition["B"].drop_nulls().to_list()
    assert b_uniform != pytest.approx(b_per_asset)


def test_from_risk_position_dict_vola_missing_key_falls_back_to_32():
    """Assets absent from the vola dict use span=32 as the default."""
    dates = pl.date_range(start=date(2020, 1, 1), end=date(2020, 3, 1), interval="1d", eager=True).cast(pl.Date)
    prices = pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series(np.linspace(100, 120, len(dates)), dtype=pl.Float64),
        }
    )
    riskposition = pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series([1.0] * len(dates), dtype=pl.Float64),
        }
    )
    pf_dict = Portfolio.from_risk_position(prices, riskposition, vola={}, aum=1e8)
    pf_int = Portfolio.from_risk_position(prices, riskposition, vola=32, aum=1e8)
    a_dict = pf_dict.cashposition["A"].drop_nulls().to_list()
    a_int = pf_int.cashposition["A"].drop_nulls().to_list()
    assert a_dict == pytest.approx(a_int, nan_ok=True)


# ─── vola validation ──────────────────────────────────────────────────────────


def test_from_risk_position_zero_span_raises():
    """A span of zero must raise ValueError."""
    prices, risk = _make_prices_risk()
    with pytest.raises(ValueError, match="positive integer"):
        Portfolio.from_risk_position(prices, risk, vola=0, aum=1e8)


def test_from_risk_position_negative_span_raises():
    """A negative span must raise ValueError."""
    prices, risk = _make_prices_risk()
    with pytest.raises(ValueError, match="positive integer"):
        Portfolio.from_risk_position(prices, risk, vola=-5, aum=1e8)


def test_from_risk_position_dict_zero_span_raises():
    """A zero span value inside a vola dict must raise ValueError."""
    prices, risk = _make_prices_risk()
    with pytest.raises(ValueError, match="positive integer"):
        Portfolio.from_risk_position(prices, risk, vola={"A": 0}, aum=1e8)


def test_from_risk_position_dict_unknown_key_raises():
    """A vola dict key that does not match any column must raise ValueError."""
    prices, risk = _make_prices_risk()
    with pytest.raises(ValueError, match="UNKNOWN"):
        Portfolio.from_risk_position(prices, risk, vola={"UNKNOWN": 8}, aum=1e8)


# ─── vol_cap ──────────────────────────────────────────────────────────────────


def test_from_risk_position_vol_cap_clips_vol():
    """vol_cap clips the EWMA vol from below, reducing position sizes in calm regimes."""
    dates = pl.date_range(start=date(2020, 1, 1), end=date(2020, 3, 1), interval="1d", eager=True).cast(pl.Date)
    prices = pl.DataFrame({"date": dates, "A": pl.Series(np.ones(len(dates)) * 100.0, dtype=pl.Float64)})
    risk = pl.DataFrame({"date": dates, "A": pl.Series([1.0] * len(dates), dtype=pl.Float64)})
    pf_uncapped = Portfolio.from_risk_position(prices, risk, vola=2, aum=1e8)
    pf_capped = Portfolio.from_risk_position(prices, risk, vola=2, aum=1e8, vol_cap=1.0)
    uncapped_vals = pf_uncapped.cashposition["A"].drop_nulls().abs()
    capped_vals = pf_capped.cashposition["A"].drop_nulls().abs()
    assert (capped_vals <= uncapped_vals + 1e-9).all()


def test_from_risk_position_vol_cap_zero_raises():
    """A vol_cap of zero must raise ValueError."""
    prices, risk = _make_prices_risk()
    with pytest.raises(ValueError, match="positive"):
        Portfolio.from_risk_position(prices, risk, vola=8, aum=1e8, vol_cap=0.0)


def test_from_risk_position_vol_cap_negative_raises():
    """A negative vol_cap must raise ValueError."""
    prices, risk = _make_prices_risk()
    with pytest.raises(ValueError, match="positive"):
        Portfolio.from_risk_position(prices, risk, vola=8, aum=1e8, vol_cap=-0.05)


# ─── cost parameter forwarding ────────────────────────────────────────────────


def test_from_risk_position_forwards_cost_per_unit():
    """from_risk_position must preserve the cost_per_unit parameter."""
    prices, risk = _make_prices_risk()
    pf = Portfolio.from_risk_position(prices, risk, aum=1e8, cost_per_unit=0.05)
    assert pf.cost_per_unit == pytest.approx(0.05)


def test_from_risk_position_forwards_cost_bps():
    """from_risk_position must preserve the cost_bps parameter."""
    prices, risk = _make_prices_risk()
    pf = Portfolio.from_risk_position(prices, risk, aum=1e8, cost_bps=5.0)
    assert pf.cost_bps == pytest.approx(5.0)


def test_from_risk_position_with_cost_model():
    """from_risk_position passes cost_model values to the Portfolio."""
    dates = pl.date_range(start=date(2020, 1, 1), end=date(2020, 2, 10), interval="1d", eager=True).cast(pl.Date)
    prices = pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series(np.linspace(100, 120, len(dates)), dtype=pl.Float64),
        }
    )
    risk_position = pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series(np.sin(np.linspace(0, 3.14, len(dates))), dtype=pl.Float64),
        }
    )
    cm = CostModel.per_unit(0.05)
    pf = Portfolio.from_risk_position(prices, risk_position, vola=8, aum=1e8, cost_model=cm)
    assert isinstance(pf, Portfolio)
    assert pf.cost_per_unit == pytest.approx(0.05)
    assert pf.cost_bps == pytest.approx(0.0)
