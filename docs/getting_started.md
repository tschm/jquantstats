# Getting Started

jQuantStats provides two entry points depending on what data you have available:

| You have… | Use… |
|-----------|------|
| Prices **and** positions | `Portfolio` |
| Returns (or prices only) | `Data` |

Both give access to the same stats, plots, and report API.

---

## Installation

```bash
pip install jquantstats
```

Python 3.11+ is required. Optional extras:

```bash
pip install jquantstats[plot]   # static chart export (kaleido)
pip install jquantstats[web]    # FastAPI web server
```

---

## Route A: Portfolio

Use `Portfolio` when you have **prices** and **positions**. This unlocks
position-level analytics — turnover, cost modelling, execution-delay analysis —
that are impossible from returns alone.

### Build a Portfolio

```python
from datetime import date
import polars as pl
from jquantstats import Portfolio

prices = pl.DataFrame({
    "date": [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)],
    "AAPL": [75.09, 74.36, 75.80],
    "MSFT": [160.62, 158.96, 159.03],
})

# Cash positions: dollar amount held per asset each day
positions = pl.DataFrame({
    "date": [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)],
    "AAPL": [500_000.0, 500_000.0, 600_000.0],
    "MSFT": [300_000.0, 300_000.0, 300_000.0],
})

pf = Portfolio.from_cash_position(
    prices=prices,
    cash_position=positions,
    aum=1_000_000,
)
```

If you track **share counts** rather than dollar amounts, use
`Portfolio.from_position` instead — it multiplies units by prices internally.

### Core series

```python
pf.returns          # daily portfolio returns (pl.Series)
pf.nav_compounded   # compounded NAV curve (pl.Series)
pf.drawdown         # drawdown from high-water mark (pl.Series)
```

### Stats

```python
pf.stats.sharpe()        # {'AAPL': ..., 'MSFT': ..., 'portfolio': ...}
pf.stats.max_drawdown()
pf.stats.volatility()
pf.stats.summary()       # full metrics table as pl.DataFrame
```

### Plots

```python
fig = pf.plots.snapshot()           # NAV + drawdown dashboard
fig = pf.plots.rolling_sharpe(window=60)
fig.show()                          # opens in browser / notebook
```

### Report

```python
html = pf.report.to_html()

with open("report.html", "w") as f:
    f.write(html)
```

---

## Route B: Data

Use `Data` when you already have a **return series** (or just prices without
positions). This is the lighter-weight path and accepts pandas DataFrames too.

```python
from datetime import date
import polars as pl
from jquantstats import Data

returns = pl.DataFrame({
    "Date": [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)],
    "Strategy": [0.012, -0.009, 0.005],
    "Benchmark": [0.004, -0.003, 0.002],
})

data = Data.from_returns(
    returns=returns,
    benchmark="Benchmark",  # column name to use as benchmark
    rf=0.0,                 # risk-free rate (float or time-varying DataFrame)
)
```

To start from **prices** instead:

```python
data = Data.from_prices(prices=prices_df)
```

### Stats

```python
data.stats.sharpe()
data.stats.sortino()
data.stats.cagr()
data.stats.annual_breakdown()   # pl.DataFrame: year | return | sharpe | ...
```

### Plots

```python
fig = data.plots.snapshot()
fig = data.plots.monthly_heatmap()
fig = data.plots.returns_distribution()
fig.show()
```

### Report

```python
html = data.reports.to_html()
```

---

## Execution-Delay Analysis

Simulate what happens if signals are executed one or more days late:

```python
pf_t0 = pf           # ideal T+0
pf_t1 = pf.lag(1)    # T+1 fill (signal available today, fills tomorrow)
pf_t2 = pf.lag(2)    # T+2 fill

print(pf_t0.stats.sharpe())
print(pf_t1.stats.sharpe())
print(pf_t2.stats.sharpe())

# Or view the full sweep as a chart
fig = pf.plots.lead_lag_ir_plot(start=-5, end=10)
fig.show()
```

---

## Transaction Costs

See [Cost Models](cost_models.md) for full details. Quick example:

```python
from jquantstats import CostModel, Portfolio

# 5 bps one-way cost on AUM turnover
pf = Portfolio.from_cash_position(
    prices=prices,
    cash_position=positions,
    aum=1_000_000,
    cost_model=CostModel.turnover_bps(5.0),
)

# Sweep Sharpe across 0–20 bps
impact = pf.trading_cost_impact(max_bps=20)
```

---

## Next Steps

- [Cost Models](cost_models.md) — model transaction costs
- [API Reference](api.md) — full method signatures
- [Marimo Notebooks](notebooks.md) — interactive worked examples
