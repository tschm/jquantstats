"""Tests for DataUtils and PortfolioUtils.

Security note: Test code uses pytest assertions (S101), which are intentional
and safe in the test context. No subprocess calls (S603/S607) are used here.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl
import pytest

from jquantstats import Data, Portfolio
from jquantstats._utils import DataUtils, PortfolioUtils
from jquantstats.exceptions import MissingDateColumnError

# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def simple_data() -> Data:
    """5-day single-asset Data with known returns (0.01 per day)."""
    n = 5
    dates = pl.date_range(
        start=date(2020, 1, 1),
        end=date(2020, 1, 1) + timedelta(days=n - 1),
        interval="1d",
        eager=True,
    )
    returns = pl.DataFrame({"Date": dates, "A": pl.Series([0.01] * n, dtype=pl.Float64)})
    return Data.from_returns(returns=returns)


@pytest.fixture
def multi_asset_data() -> Data:
    """5-day two-asset Data."""
    n = 5
    dates = pl.date_range(
        start=date(2020, 1, 1),
        end=date(2020, 1, 1) + timedelta(days=n - 1),
        interval="1d",
        eager=True,
    )
    returns = pl.DataFrame(
        {
            "Date": dates,
            "A": pl.Series([0.01, -0.02, 0.03, 0.00, 0.01], dtype=pl.Float64),
            "B": pl.Series([0.005, 0.010, -0.005, 0.020, 0.015], dtype=pl.Float64),
        }
    )
    return Data.from_returns(returns=returns)


@pytest.fixture
def int_data() -> Data:
    """Integer-indexed (no date column) Data."""
    n = 5
    returns = pl.DataFrame({"A": pl.Series([0.01, 0.02, -0.01, 0.00, 0.03], dtype=pl.Float64)})
    index = pl.DataFrame({"index": list(range(n))})
    return Data(returns=returns, index=index)


@pytest.fixture
def portfolio_pf() -> Portfolio:
    """20-day single-asset Portfolio for PortfolioUtils tests."""
    n = 20
    start = date(2020, 1, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True).cast(pl.Date)
    return Portfolio.from_cash_position(
        prices=pl.DataFrame({"date": dates, "A": pl.Series([100.0 * (1.005**i) for i in range(n)])}),
        cash_position=pl.DataFrame({"date": dates, "A": pl.Series([1000.0] * n, dtype=pl.Float64)}),
        aum=1e5,
    )


# ─── DataUtils: access via data.utils ────────────────────────────────────────


def test_data_utils_property_returns_data_utils(simple_data):
    """data.utils must return a DataUtils instance."""
    assert isinstance(simple_data.utils, DataUtils)


def test_data_utils_repr(simple_data):
    """DataUtils.__repr__ should include asset names."""
    r = repr(simple_data.utils)
    assert r.startswith("DataUtils(assets=")
    assert "A" in r


# ─── to_prices ────────────────────────────────────────────────────────────────


def test_to_prices_columns(simple_data):
    """to_prices must return date + asset columns."""
    prices = simple_data.utils.to_prices()
    assert "A" in prices.columns


def test_to_prices_shape(simple_data):
    """to_prices output must have same row count as input returns."""
    prices = simple_data.utils.to_prices()
    assert prices.height == simple_data.returns.height


def test_to_prices_positive_values(simple_data):
    """All price values must be positive for positive returns."""
    prices = simple_data.utils.to_prices()
    assert (prices["A"] > 0).all()


def test_to_prices_base_parameter(simple_data):
    """The first price value must equal base * (1 + r[0])."""
    base = 500.0
    prices = simple_data.utils.to_prices(base=base)
    # cumprod starts at (1 + r[0]) * base
    expected_first = base * (1.0 + 0.01)
    assert prices["A"][0] == pytest.approx(expected_first, rel=1e-9)


def test_to_prices_default_base_is_1e5(simple_data):
    """Default base should be 1e5."""
    prices = simple_data.utils.to_prices()
    assert prices["A"][0] == pytest.approx(1e5 * 1.01, rel=1e-9)


def test_to_prices_multi_asset(multi_asset_data):
    """to_prices must work for multi-asset data."""
    prices = multi_asset_data.utils.to_prices()
    assert "A" in prices.columns
    assert "B" in prices.columns


# ─── to_log_returns ───────────────────────────────────────────────────────────


def test_to_log_returns_columns(simple_data):
    """to_log_returns must return the same columns as input."""
    log_rets = simple_data.utils.to_log_returns()
    assert "A" in log_rets.columns


def test_to_log_returns_values(simple_data):
    """Log return must equal ln(1 + r) for each observation."""
    log_rets = simple_data.utils.to_log_returns()
    for val in log_rets["A"].to_list():
        assert val == pytest.approx(math.log(1.01), rel=1e-9)


def test_log_returns_is_alias(simple_data):
    """log_returns() must return the same result as to_log_returns()."""
    a = simple_data.utils.to_log_returns()
    b = simple_data.utils.log_returns()
    assert a["A"].to_list() == pytest.approx(b["A"].to_list())


def test_to_log_returns_less_than_simple(simple_data):
    """Log returns must be less than simple returns for positive r."""
    simple_rets = simple_data.returns["A"].to_list()
    log_rets = simple_data.utils.to_log_returns()["A"].to_list()
    for s, log_r in zip(simple_rets, log_rets, strict=False):
        assert log_r < s


# ─── rebase ───────────────────────────────────────────────────────────────────


def test_rebase_starts_at_base(simple_data):
    """Rebased series must start at exactly base."""
    rebased = simple_data.utils.rebase(base=100.0)
    assert float(rebased["A"][0]) == pytest.approx(100.0, rel=1e-9)


def test_rebase_custom_base(simple_data):
    """Rebased series must start at the specified base."""
    rebased = simple_data.utils.rebase(base=1000.0)
    assert float(rebased["A"][0]) == pytest.approx(1000.0, rel=1e-9)


def test_rebase_preserves_shape(simple_data):
    """Rebase must preserve the row count."""
    rebased = simple_data.utils.rebase()
    assert rebased.height == simple_data.returns.height


def test_rebase_monotone_for_constant_positive_returns(simple_data):
    """Rebased prices are monotonically increasing for constant positive returns."""
    rebased = simple_data.utils.rebase()
    vals = rebased["A"].to_list()
    assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))


# ─── group_returns / aggregate_returns ───────────────────────────────────────


def test_group_returns_requires_temporal_index(int_data):
    """group_returns must raise MissingDateColumnError for integer-indexed data."""
    with pytest.raises(MissingDateColumnError):
        int_data.utils.group_returns()


def test_group_returns_monthly_row_count():
    """Monthly grouping of 90-day data should yield roughly 3 rows."""
    n = 90
    start = date(2020, 1, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True)
    data = Data.from_returns(returns=pl.DataFrame({"Date": dates, "A": pl.Series([0.001] * n, dtype=pl.Float64)}))
    grouped = data.utils.group_returns(period="1mo")
    assert grouped.height >= 2


def test_group_returns_compounded_greater_than_sum():
    """Compounded group return must exceed simple sum for positive returns."""
    n = 30
    start = date(2020, 1, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True)
    data = Data.from_returns(returns=pl.DataFrame({"Date": dates, "A": pl.Series([0.005] * n, dtype=pl.Float64)}))
    comp = float(data.utils.group_returns(period="1mo", compounded=True)["A"].sum())
    simple = float(data.utils.group_returns(period="1mo", compounded=False)["A"].sum())
    assert comp > simple


def test_aggregate_returns_alias_matches_group_returns():
    """aggregate_returns must return the same result as group_returns."""
    n = 30
    start = date(2020, 1, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True)
    data = Data.from_returns(returns=pl.DataFrame({"Date": dates, "A": pl.Series([0.002] * n, dtype=pl.Float64)}))
    a = data.utils.group_returns(period="1mo")
    b = data.utils.aggregate_returns(period="1mo")
    assert a["A"].to_list() == pytest.approx(b["A"].to_list())


def test_group_returns_human_readable_period():
    """Human-readable period alias 'monthly' should work like '1mo'."""
    n = 30
    start = date(2020, 1, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True)
    data = Data.from_returns(returns=pl.DataFrame({"Date": dates, "A": pl.Series([0.001] * n, dtype=pl.Float64)}))
    a = data.utils.group_returns(period="1mo")
    b = data.utils.group_returns(period="monthly")
    assert a.height == b.height
    assert a["A"].to_list() == pytest.approx(b["A"].to_list())


# ─── to_volatility_adjusted_returns ───────────────────────────────────────────


@pytest.fixture
def long_data() -> Data:
    """100-day single-asset Data for rolling-window tests."""
    n = 100
    dates = pl.date_range(
        start=date(2020, 1, 1),
        end=date(2020, 1, 1) + timedelta(days=n - 1),
        interval="1d",
        eager=True,
    )
    returns = pl.DataFrame({"Date": dates, "A": pl.Series([0.01 * ((-1) ** i) for i in range(n)], dtype=pl.Float64)})
    return Data.from_returns(returns=returns)


@pytest.fixture
def long_multi_asset_data() -> Data:
    """100-day two-asset Data for rolling-window tests."""
    n = 100
    dates = pl.date_range(
        start=date(2020, 1, 1),
        end=date(2020, 1, 1) + timedelta(days=n - 1),
        interval="1d",
        eager=True,
    )
    returns = pl.DataFrame(
        {
            "Date": dates,
            "A": pl.Series([0.01 * ((-1) ** i) for i in range(n)], dtype=pl.Float64),
            "B": pl.Series([0.005 * ((-1) ** i) for i in range(n)], dtype=pl.Float64),
        }
    )
    return Data.from_returns(returns=returns)


def test_volatility_adjusted_returns_columns(long_data):
    """to_volatility_adjusted_returns must preserve asset columns."""
    result = long_data.utils.to_volatility_adjusted_returns(window=10)
    assert "A" in result.columns


def test_volatility_adjusted_returns_row_count(long_data):
    """to_volatility_adjusted_returns must preserve the row count."""
    result = long_data.utils.to_volatility_adjusted_returns(window=10)
    assert result.height == long_data.returns.height


def test_volatility_adjusted_returns_early_nulls(long_data):
    """First *window* rows must be null (shift avoids look-ahead bias)."""
    window = 10
    result = long_data.utils.to_volatility_adjusted_returns(window=window)
    early = result["A"][:window].to_list()
    assert all(v is None for v in early)
    # Row at index `window` should have a value
    assert result["A"][window] is not None


def test_volatility_adjusted_returns_multi_asset(long_multi_asset_data):
    """to_volatility_adjusted_returns must work for multiple asset columns."""
    result = long_multi_asset_data.utils.to_volatility_adjusted_returns(window=10)
    assert "A" in result.columns
    assert "B" in result.columns
    # Both columns should have valid (non-null) values after the window
    assert result["A"][15] is not None
    assert result["B"][15] is not None


def test_volatility_adjusted_returns_custom_estimator(long_data):
    """A custom vol_estimator callable must be used instead of rolling_std."""

    # Use rolling_mean of absolute returns as an alternative vol proxy
    def custom(expr):
        return expr.abs().rolling_mean(5)

    result = long_data.utils.to_volatility_adjusted_returns(vol_estimator=custom)
    # Should differ from the default rolling_std result
    default = long_data.utils.to_volatility_adjusted_returns(window=5)
    # Compare a non-null row — they should not be equal
    assert result["A"][10] != pytest.approx(default["A"][10])


# ─── to_excess_returns ────────────────────────────────────────────────────────


def test_to_excess_returns_zero_rf_unchanged(simple_data):
    """to_excess_returns(rf=0) must return the original returns."""
    excess = simple_data.utils.to_excess_returns(rf=0.0)
    original = simple_data.returns["A"].to_list()
    assert excess["A"].to_list() == pytest.approx(original)


def test_to_excess_returns_reduces_returns(simple_data):
    """Positive rf must reduce all return values."""
    excess = simple_data.utils.to_excess_returns(rf=0.05)
    original = simple_data.returns["A"].to_list()
    for e, o in zip(excess["A"].to_list(), original, strict=False):
        assert e < o


def test_to_excess_returns_with_nperiods(simple_data):
    """When nperiods is supplied rf is converted to per-period rate."""
    excess_annual = simple_data.utils.to_excess_returns(rf=0.05)
    excess_daily = simple_data.utils.to_excess_returns(rf=0.05, nperiods=252)
    # Per-period rate is much smaller than annual → smaller deduction
    assert float(excess_daily["A"].sum()) > float(excess_annual["A"].sum())


def test_to_excess_returns_preserves_shape(simple_data):
    """to_excess_returns must preserve the row count."""
    excess = simple_data.utils.to_excess_returns(rf=0.01)
    assert excess.height == simple_data.returns.height


# ─── exponential_stdev ────────────────────────────────────────────────────────


def test_exponential_stdev_columns(simple_data):
    """exponential_stdev must return the same columns as input."""
    ewm = simple_data.utils.exponential_stdev()
    assert "A" in ewm.columns


def test_exponential_stdev_nonnegative(multi_asset_data):
    """All EWMA std values must be non-negative."""
    ewm = multi_asset_data.utils.exponential_stdev(window=3)
    for col in ["A", "B"]:
        assert (ewm[col].drop_nulls() >= 0).all()


def test_exponential_stdev_halflife_mode(multi_asset_data):
    """is_halflife=True must produce different (smaller span) results than default."""
    span_result = multi_asset_data.utils.exponential_stdev(window=5, is_halflife=False)
    hl_result = multi_asset_data.utils.exponential_stdev(window=5, is_halflife=True)
    # They should differ (different decay parametrisations)
    assert span_result["A"].to_list() != hl_result["A"].to_list()


def test_exponential_stdev_preserves_row_count(simple_data):
    """exponential_stdev must preserve the row count."""
    ewm = simple_data.utils.exponential_stdev(window=3)
    assert ewm.height == simple_data.returns.height


# ─── PortfolioUtils ───────────────────────────────────────────────────────────


def test_portfolio_utils_property_returns_portfolio_utils(portfolio_pf):
    """portfolio.utils must return a PortfolioUtils instance."""
    assert isinstance(portfolio_pf.utils, PortfolioUtils)


def test_portfolio_utils_cached(portfolio_pf):
    """portfolio.utils must return the same object on repeated calls."""
    assert portfolio_pf.utils is portfolio_pf.utils


def test_portfolio_utils_repr(portfolio_pf):
    """PortfolioUtils.__repr__ should include asset names."""
    r = repr(portfolio_pf.utils)
    assert r.startswith("PortfolioUtils(assets=")
    assert "A" in r


def test_portfolio_utils_to_prices(portfolio_pf):
    """portfolio.utils.to_prices must return a DataFrame with a price column."""
    prices = portfolio_pf.utils.to_prices()
    # The portfolio data bridge exposes a single 'returns' column.
    assert "returns" in prices.columns
    assert (prices["returns"] > 0).all()


def test_portfolio_utils_to_log_returns(portfolio_pf):
    """portfolio.utils.to_log_returns must return log-transformed returns."""
    log_rets = portfolio_pf.utils.to_log_returns()
    simple_rets = portfolio_pf.utils._du().data.returns["returns"].to_list()
    log_list = log_rets["returns"].to_list()
    for s, log_r in zip(simple_rets, log_list, strict=False):
        assert log_r == pytest.approx(math.log(1.0 + s), rel=1e-9)


def test_portfolio_utils_rebase(portfolio_pf):
    """portfolio.utils.rebase must start at exactly base."""
    rebased = portfolio_pf.utils.rebase(base=100.0)
    assert float(rebased["returns"][0]) == pytest.approx(100.0, rel=1e-9)


def test_portfolio_utils_group_returns(portfolio_pf):
    """portfolio.utils.group_returns must aggregate over the full date range."""
    grouped = portfolio_pf.utils.group_returns(period="1w")
    assert grouped.height >= 1
    assert "returns" in grouped.columns


def test_portfolio_utils_to_excess_returns(portfolio_pf):
    """portfolio.utils.to_excess_returns must reduce returns relative to base."""
    base_sum = float(portfolio_pf.utils._du().data.returns["returns"].sum())
    excess_sum = float(portfolio_pf.utils.to_excess_returns(rf=0.05)["returns"].sum())
    assert excess_sum < base_sum


def test_portfolio_utils_exponential_stdev(portfolio_pf):
    """portfolio.utils.exponential_stdev must return non-negative values."""
    ewm = portfolio_pf.utils.exponential_stdev(window=5)
    assert (ewm["returns"].drop_nulls() >= 0).all()


def test_portfolio_utils_log_returns_alias(portfolio_pf):
    """portfolio.utils.log_returns must match to_log_returns."""
    a = portfolio_pf.utils.to_log_returns()["returns"].to_list()
    b = portfolio_pf.utils.log_returns()["returns"].to_list()
    assert a == pytest.approx(b)


def test_portfolio_utils_aggregate_returns_alias(portfolio_pf):
    """portfolio.utils.aggregate_returns must match group_returns."""
    a = portfolio_pf.utils.group_returns(period="1w")
    b = portfolio_pf.utils.aggregate_returns(period="1w")
    assert a["returns"].to_list() == pytest.approx(b["returns"].to_list())


def test_portfolio_utils_to_volatility_adjusted_returns(portfolio_pf):
    """portfolio.utils.to_volatility_adjusted_returns must preserve row count and columns."""
    result = portfolio_pf.utils.to_volatility_adjusted_returns(window=5)
    assert "returns" in result.columns
    assert result.height == portfolio_pf.returns.height


# ─── Winsorise ────────────────────────────────────────────────────────────────


def test_winsorise_preserves_columns(long_data):
    """Winsorise must preserve asset columns."""
    result = long_data.utils.winsorise(window=7, n_sigma=3.0)
    assert "A" in result.columns


def test_winsorise_preserves_row_count(long_data):
    """Winsorise must preserve the row count."""
    result = long_data.utils.winsorise(window=7, n_sigma=3.0)
    assert result.height == long_data.returns.height


def test_winsorise_no_change_for_normal_data(long_data):
    """Alternating ±0.01 data should be unchanged with 3-sigma clipping."""
    result = long_data.utils.winsorise(window=7, n_sigma=3.0)
    original = long_data.returns["A"].to_list()
    clipped = result["A"].to_list()
    # After the warm-up period, values should be identical
    for i in range(10, len(original)):
        if clipped[i] is not None:
            assert clipped[i] == pytest.approx(original[i])


def test_winsorise_clips_outlier():
    """A large outlier should be clipped by tight n_sigma bounds."""
    n = 30
    dates = pl.date_range(
        start=date(2020, 1, 1),
        end=date(2020, 1, 1) + timedelta(days=n - 1),
        interval="1d",
        eager=True,
    )
    vals = [0.01] * n
    vals[20] = 0.50  # huge outlier
    data = Data.from_returns(returns=pl.DataFrame({"Date": dates, "A": pl.Series(vals, dtype=pl.Float64)}))
    result = data.utils.winsorise(window=7, n_sigma=1.0)
    # The outlier at index 20 should have been clipped down
    assert result["A"][20] < 0.50


def test_winsorise_multi_asset(long_multi_asset_data):
    """Winsorise must work for multiple asset columns."""
    result = long_multi_asset_data.utils.winsorise(window=7, n_sigma=3.0)
    assert "A" in result.columns
    assert "B" in result.columns


def test_winsorise_early_nulls_passthrough(long_data):
    """First rows (before window + shift fill) should pass through unchanged."""
    window = 7
    result = long_data.utils.winsorise(window=window, n_sigma=3.0)
    original = long_data.returns["A"].to_list()
    clipped = result["A"].to_list()
    # The first `window` rows have null rolling stats due to shift, so clip
    # becomes clip(null, null) which passes through unchanged.
    for i in range(window):
        assert clipped[i] == pytest.approx(original[i])


def test_winsorise_default_parameters(long_data):
    """Calling winsorise with no args should work without error."""
    result = long_data.utils.winsorise()
    assert result.height == long_data.returns.height


def test_winsorise_shift_prevents_self_normalisation():
    """With shift(1), an outlier should be clipped more aggressively.

    Without the shift, the outlier inflates rolling_std and partly
    normalises itself, resulting in a less aggressive clip.
    """
    n = 30
    dates = pl.date_range(
        start=date(2020, 1, 1),
        end=date(2020, 1, 1) + timedelta(days=n - 1),
        interval="1d",
        eager=True,
    )
    vals = [0.01] * n
    vals[20] = 1.0  # extreme outlier
    df = pl.DataFrame({"Date": dates, "A": pl.Series(vals, dtype=pl.Float64)})

    # With shift (current implementation) — bounds at t=20 use data up to t=19
    data = Data.from_returns(returns=df)
    shifted = data.utils.winsorise(window=7, n_sigma=2.0)
    clipped_shifted = shifted["A"][20]

    # Without shift — manually compute what clip would do
    col_vals = df["A"]
    r_mean_no_shift = col_vals.rolling_mean(7)
    r_std_no_shift = col_vals.rolling_std(7)
    lower_no_shift = r_mean_no_shift[20] - 2.0 * r_std_no_shift[20]
    upper_no_shift = r_mean_no_shift[20] + 2.0 * r_std_no_shift[20]
    clipped_no_shift = max(lower_no_shift, min(1.0, upper_no_shift))

    # The shifted version clips more aggressively (lower value) because the
    # outlier didn't inflate the std used for bounds
    assert clipped_shifted < clipped_no_shift
