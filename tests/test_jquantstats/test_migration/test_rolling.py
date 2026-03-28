"""Parametrized migration tests comparing rolling stats against quantstats."""

import numpy as np
import pytest
import quantstats as qs


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("rolling_sharpe", {"rolling_period": 126, "periods_per_year": 252}),
        ("rolling_volatility", {"rolling_period": 126, "periods_per_year": 252}),
        ("rolling_sortino", {"rolling_period": 126, "periods_per_year": 252}),
    ],
)
def test_rolling(stats, method, kwargs):
    """Verify each rolling method produces the same time-series as quantstats."""
    aapl = stats.all.to_pandas().set_index("Date")["AAPL"]
    jqs_df = getattr(stats, method)(**kwargs)
    qs_series = getattr(qs.stats, method)(aapl, **kwargs)

    jqs_pd = jqs_df.to_pandas().set_index("Date")["AAPL"].dropna()
    qs_clean = qs_series.dropna()
    common = jqs_pd.index.intersection(qs_clean.index)

    assert len(common) > 0
    np.testing.assert_allclose(jqs_pd[common].values, qs_clean[common].values, atol=1e-6)
