# jquantstats

**jquantstats** is a quantitative financial analytics library built on [Polars](https://pola.rs/) for fast, expressive portfolio analysis.

## Quick Start

**Entry point 1 — prices + positions (recommended for active portfolios):**

```python
from jquantstats import Portfolio
import polars as pl

pf = Portfolio.from_cash_position(
    prices=prices_df,
    cash_position=positions_df,
    aum=1_000_000,
)
pf.stats.sharpe()
pf.plots.snapshot()
```

**Entry point 2 — returns series (for arbitrary return streams):**

```python
from jquantstats import build_data
import polars as pl

data = build_data(returns=returns_df, benchmark=bench_df)
data.stats.sharpe()
data.plots.plot_snapshot(title="Performance")
```

The two APIs are layered: `portfolio.data` returns a [`Data`][jquantstats._data.Data] object so you can always drop into the returns-series API from a Portfolio.

## Reference

| Class | Description |
|---|---|
| [`Data`](reference/data.md) | Container for financial returns data; entry point for stats, plots, and reports |
| [`Stats`](reference/stats.md) | Statistical analysis tools — Sharpe, Sortino, drawdown, VaR, and more |
