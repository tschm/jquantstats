---
icon: lucide/chart-line
hide:
  - toc
---

# jquantstats

**Portfolio analytics for quants** — built on Polars, powered by Plotly.

[![PyPI version](https://badge.fury.io/py/jquantstats.svg)](https://badge.fury.io/py/jquantstats)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://pypi.org/project/jquantstats/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/jebel-quant/jquantstats/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/personalized-badge/jquantstats?period=month&units=international_system&left_color=black&right_color=orange&left_text=downloads%2Fmonth)](https://pepy.tech/project/jquantstats)

---

jQuantStats is a Python library for portfolio analytics that helps quants and portfolio managers
understand their strategy performance in depth. It provides two complementary entry points:
a **Portfolio route** that works directly from price and position data, and a **Data route**
for arbitrary return streams. All analytics, visualisations, and HTML reports are available
from either entry point.

## Why jquantstats?

<div class="grid cards" markdown>

-   :bar_chart: **Position-level analytics**

    ---

    Work from raw prices and positions instead of a return series. Unlocks execution-delay
    analysis, tilt/timing decomposition, and turnover analytics that are impossible from
    returns alone.

-   :zap: **Polars-native**

    ---

    Zero pandas runtime dependency. Built on [Polars](https://pola.rs/) for fast,
    type-safe data manipulation. Accepts pandas input via automatic conversion.

-   :chart_with_upwards_trend: **Interactive charts**

    ---

    Every plot returns an interactive [Plotly](https://plotly.com/python/) figure —
    zoom, pan, export. No static matplotlib images.

-   :white_check_mark: **Fully type-annotated**

    ---

    Complete PEP 484 type annotations, `py.typed` marker, and thorough test coverage.
    Works great with mypy and Pyright.

</div>

---

## Install

```bash
pip install jquantstats
```

Python **3.11+** is required. Add optional extras as needed:

```bash
pip install jquantstats[plot]   # static chart export (kaleido)
pip install jquantstats[web]    # FastAPI web server
```

---

## Two entry points

=== "Portfolio route"

    Use `Portfolio` when you have **prices** *and* **positions**.
    This is the recommended entry point — it exposes the full analytics suite.

    ```python
    import polars as pl
    from jquantstats import Portfolio

    pf = Portfolio.from_cash_position(
        prices=prices_df,      # pl.DataFrame: date + asset columns
        cash_position=pos_df,  # pl.DataFrame: date + asset columns (£ amounts)
        aum=1_000_000,
    )

    pf.stats.sharpe()          # {"AAPL": 1.34, "MSFT": 0.91, "portfolio": 1.21}
    pf.plots.snapshot()        # interactive NAV + drawdown dashboard
    pf.report.to_html()        # full HTML tearsheet
    ```

    Only `Portfolio` gives you:

    - `pf.lag(n)` — execution-delay analysis
    - `pf.tilt` / `pf.timing` — allocation vs. timing skill decomposition
    - `pf.turnover` / `pf.turnover_summary()` — turnover analytics
    - `pf.trading_cost_impact(max_bps=20)` — cost sensitivity sweep

=== "Data route"

    Use `Data` when you only have a **return series** (or prices without positions).
    This is the lighter-weight path and the easiest migration from QuantStats.

    ```python
    import polars as pl
    from jquantstats import Data

    data = Data.from_returns(
        returns=returns_df,     # pl.DataFrame: date + return columns
        benchmark="Benchmark",  # optional column name
        rf=0.0,                 # risk-free rate
    )

    data.stats.sharpe()
    data.stats.sortino()
    data.stats.annual_breakdown()   # year-by-year table
    data.plots.monthly_heatmap()
    data.reports.to_html()
    ```

---

## Execution-delay analysis

Simulate what happens when signals fill one or more days late — a real-world concern
that a return series hides completely.

```python
pf_t0 = pf           # ideal T+0 execution
pf_t1 = pf.lag(1)    # signal fires today, fills tomorrow
pf_t2 = pf.lag(2)    # fills the day after

print(pf_t0.stats.sharpe())  # {"portfolio": 1.34}
print(pf_t1.stats.sharpe())  # {"portfolio": 1.28}
print(pf_t2.stats.sharpe())  # {"portfolio": 1.19}

# Or visualise the full sweep as a chart
fig = pf.plots.lead_lag_ir_plot(start=-5, end=10)
fig.show()
```

---

## Transaction costs

Model bid/ask spreads and per-unit commissions, then sweep cost sensitivity in one call.

```python
from jquantstats import CostModel, Portfolio

pf = Portfolio.from_cash_position(
    prices=prices_df,
    cash_position=pos_df,
    aum=1_000_000,
    cost_model=CostModel.turnover_bps(5.0),  # 5 bps one-way
)

# Sharpe across 0 → 20 bps — how robust is the strategy?
impact = pf.trading_cost_impact(max_bps=20)
```

See [Cost Models](cost_models.md) for the full reference.

---

## Comparison with QuantStats

| Feature | QuantStats | jquantstats |
|---|---|---|
| DataFrame engine | pandas | Polars |
| Charts | matplotlib (static) | Plotly (interactive) |
| Multi-asset in one call | no | yes |
| Portfolio route | no | yes |
| Execution-delay analysis | no | `pf.lag(n)` |
| Tilt / timing decomp | no | yes |
| Turnover analytics | no | yes |
| Cost modelling | no | yes |
| Type annotations | partial | full |

See the [Migration Guide](MIGRATION.md) for a complete API mapping.

---

## Next steps

<div class="grid cards" markdown>

-   :rocket: [**Getting Started**](getting_started.md)

    Step-by-step guide for both entry points.

-   :money_with_wings: [**Cost Models**](cost_models.md)

    Model transaction costs and sweep sensitivity.

-   :arrows_counterclockwise: [**Migration from QuantStats**](MIGRATION.md)

    Complete API mapping and behavioural differences.

-   :books: [**API Reference**](api.md)

    Full method signatures and docstrings.

</div>
