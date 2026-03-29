# Migrating from QuantStats to jquantstats

This guide covers everything you need to move from
[QuantStats](https://github.com/ranaroussi/quantstats) to **jquantstats**.

---

## Why migrate

| | QuantStats | jquantstats |
|---|---|---|
| DataFrame engine | pandas | Polars (zero pandas at runtime) |
| Charts | static matplotlib/seaborn | interactive Plotly |
| Type annotations | partial | full (PEP 484, `py.typed`) |
| Multi-asset | single `pd.Series` per call | multi-column DataFrame, one call |
| API style | functions (`qs.stats.sharpe(r)`) | object methods (`data.stats.sharpe()`) |
| Portfolio analytics | not supported | prices + positions route via `Portfolio` |
| Execution-delay analysis | not supported | `pf.lag(n)` + lead-lag IR plot |
| Tilt / timing decomposition | not supported | `pf.tilt`, `pf.timing`, `pf.tilt_timing_decomp` |
| Turnover analytics | not supported | `pf.turnover`, `pf.turnover_weekly()` |
| Cost modelling | not supported | `CostModel.per_unit()` / `CostModel.turnover_bps()` |
| Python version | 3.7+ | 3.11+ |

---

## Installation

```bash
pip install jquantstats
pip install jquantstats[plot]   # adds kaleido for static image export
```

Uninstall QuantStats once you're done:

```bash
pip uninstall quantstats
```

---

## Data format

QuantStats expects a **`pd.Series`** with a `DatetimeIndex`.
jquantstats expects a **`pl.DataFrame`** with a dedicated date column.

```python
import pandas as pd
import polars as pl

# QuantStats — pd.Series with DatetimeIndex
returns_pd: pd.Series = ...

# jquantstats — convert from pandas
returns_pl = pl.from_pandas(returns_pd.rename("MyStrategy").reset_index())
# → DataFrame columns: ["Date", "MyStrategy"]

# jquantstats — build from scratch
from datetime import date
returns_pl = pl.DataFrame({
    "Date":       [date(2020, 1, 2), date(2020, 1, 3), ...],
    "MyStrategy": [0.01, -0.005, ...],
})
```

### Benchmarks

```python
# QuantStats — benchmark passed to every function call
import quantstats as qs
qs.stats.information_ratio(returns_pd, benchmark=benchmark_pd)

# jquantstats — benchmark provided once at construction time
import jquantstats as jqs
data = jqs.Data.from_returns(
    returns=returns_pl,
    benchmark=benchmark_pl,
    date_col="Date",
)
data.stats.information_ratio()   # benchmark used automatically
```

---

## Entry points

jquantstats has two entry points depending on what data you have.

### `Data` — returns series (drop-in QuantStats replacement)

```python
import jquantstats as jqs

data = jqs.Data.from_returns(
    returns=returns_pl,     # pl.DataFrame with date + return columns
    benchmark=benchmark_pl, # optional
    date_col="Date",
)

data.stats.sharpe()
data.plots.plot_snapshot()
data.reports.summary()
```

You can also construct from prices:

```python
data = jqs.Data.from_prices(prices=prices_pl, date_col="Date")
```

### `Portfolio` — prices + positions (no QuantStats equivalent)

If you have asset prices and position sizes, use `Portfolio` to get
execution-delay analysis, tilt/timing decomposition, turnover analytics,
and cost modelling — none of which are available in QuantStats.

```python
pf = jqs.Portfolio.from_cash_position(
    prices=prices_df,        # pl.DataFrame: date + asset columns
    cash_position=pos_df,    # pl.DataFrame: date + asset columns (£ amounts)
    aum=1_000_000,
)

# All stats, plots, and reports are available
pf.stats.sharpe()
pf.plots.snapshot()
pf.report.to_html()

# Drop down to the returns-series API at any time
pf.data.stats.calmar()
```

Other factory methods:

```python
# From share counts
pf = jqs.Portfolio.from_position(prices, position, aum)

# From risk-scaled positions
pf = jqs.Portfolio.from_risk_position(prices, risk_position, aum, vola=vola_df)
```

---

## API mapping

All jquantstats analytics live on `.stats`, `.plots`, or `.reports` of a
`Data` (or `Portfolio`) instance.  Methods return a **`dict` keyed by
column name** rather than a scalar.

```python
# one-time setup
data = jqs.Data.from_returns(returns=returns_pl, benchmark=benchmark_pl)
```

### Statistics

| QuantStats | jquantstats |
|---|---|
| `qs.stats.sharpe(r, periods=252)` | `data.stats.sharpe(periods=252)` |
| `qs.stats.sortino(r, periods=252)` | `data.stats.sortino(periods=252)` |
| `qs.stats.calmar(r)` | `data.stats.calmar()` |
| `qs.stats.omega(r)` | `data.stats.omega()` |
| `qs.stats.volatility(r, periods=252)` | `data.stats.volatility(periods=252)` |
| `qs.stats.skew(r)` | `data.stats.skew()` |
| `qs.stats.kurtosis(r)` | `data.stats.kurtosis()` |
| `qs.stats.max_drawdown(r)` | `data.stats.max_drawdown()` |
| `qs.stats.avg_drawdown(r)` | `data.stats.avg_drawdown()` |
| `qs.stats.value_at_risk(r)` | `data.stats.value_at_risk(alpha=0.05)` |
| `qs.stats.conditional_value_at_risk(r, confidence=0.95)` | `data.stats.conditional_value_at_risk(alpha=0.05)` |
| `qs.stats.win_rate(r)` | `data.stats.win_rate()` |
| `qs.stats.avg_return(r)` | `data.stats.avg_return()` |
| `qs.stats.avg_win(r)` | `data.stats.avg_win()` |
| `qs.stats.avg_loss(r)` | `data.stats.avg_loss()` |
| `qs.stats.best(r)` | `data.stats.best()` |
| `qs.stats.worst(r)` | `data.stats.worst()` |
| `qs.stats.profit_factor(r)` | `data.stats.profit_factor()` |
| `qs.stats.profit_ratio(r)` | `data.stats.profit_ratio()` |
| `qs.stats.payoff_ratio(r)` | `data.stats.payoff_ratio()` |
| `qs.stats.win_loss_ratio(r)` | `data.stats.win_loss_ratio()` |
| `qs.stats.gain_to_pain_ratio(r)` | `data.stats.gain_to_pain_ratio()` |
| `qs.stats.kelly_criterion(r)` | `data.stats.kelly_criterion()` |
| `qs.stats.exposure(r)` | `data.stats.exposure()` |
| `qs.stats.cagr(r)` | `data.stats.cagr()` |
| `qs.stats.rar(r)` | `data.stats.rar()` |
| `qs.stats.information_ratio(r, benchmark=b)` | `data.stats.information_ratio()` |
| `qs.stats.r_squared(r, benchmark=b)` | `data.stats.r_squared()` |
| `qs.stats.greeks(r, benchmark=b)` | `data.stats.greeks()` |
| `qs.stats.rolling_sharpe(r)` | `data.stats.rolling_sharpe()` |
| `qs.stats.rolling_sortino(r)` | `data.stats.rolling_sortino()` |
| `qs.stats.rolling_volatility(r)` | `data.stats.rolling_volatility()` |

### Plots

| QuantStats | jquantstats |
|---|---|
| `qs.plots.snapshot(r)` | `data.plots.plot_snapshot()` |
| `qs.plots.drawdown(r)` | `data.plots.plot_drawdown()` |
| `qs.plots.returns(r)` | `data.plots.plot_returns()` |
| `qs.plots.monthly_heatmap(r)` | `data.plots.plot_monthly_heatmap()` |
| `qs.plots.distribution(r)` | `data.plots.plot_distribution()` |
| `qs.plots.rolling_sharpe(r)` | `data.plots.plot_rolling_sharpe()` |
| `qs.plots.rolling_volatility(r)` | `data.plots.plot_rolling_volatility()` |

All `data.plots.*` methods return an interactive **Plotly figure** instead
of a static matplotlib figure.

### Reports

```python
# QuantStats
qs.reports.full(returns_pd, benchmark=benchmark_pd)
qs.reports.metrics(returns_pd)

# jquantstats
data.reports.summary()
data.reports.metrics()
data.reports.to_html()          # full HTML report
```

---

## Accessing results

QuantStats functions return scalars. jquantstats methods return a `dict`
keyed by column name.

```python
# QuantStats — scalar
sharpe = qs.stats.sharpe(returns_pd)   # 1.23

# jquantstats — dict
result = data.stats.sharpe()            # {"MyStrategy": 1.23}
sharpe = result["MyStrategy"]           # 1.23
```

Multi-asset results come back in the same call:

```python
returns_multi = pl.DataFrame({
    "Date": dates,
    "AAPL": aapl_rets,
    "MSFT": msft_rets,
})
data = jqs.Data.from_returns(returns=returns_multi)
data.stats.sharpe()
# → {"AAPL": 1.34, "MSFT": 0.91}
```

---

## Behavioural differences

### `information_ratio` — annualisation

QuantStats returns a **non-annualised** IR. jquantstats **annualises** by
default (`× √periods_per_year`).

```python
import numpy as np
qs_ir  = qs.stats.information_ratio(returns_pd, benchmark=benchmark_pd)
jqs_ir = data.stats.information_ratio(periods_per_year=252)["MyStrategy"]
# qs_ir * np.sqrt(252) ≈ jqs_ir
```

To obtain the raw, non-annualised ratio (matching QuantStats), pass `annualise=False`:

```python
jqs_ir_raw = data.stats.information_ratio(periods_per_year=252, annualise=False)["MyStrategy"]
# jqs_ir_raw ≈ qs_ir
```

### NaN / null handling

pandas (and QuantStats) silently drop `NaN` values in most calculations.
Polars propagates `null` by default — if any value in a column is `null`,
most statistics will return `null` instead of a numeric result.

> **Note:** in Polars, `null` (a missing entry, equivalent to pandas `NaN`)
> and `NaN` (IEEE-754 "Not a Number") are distinct. jquantstats treats `null`
> as missing; `NaN` is a numeric value that propagates through calculations.

You can clean data manually before constructing `Data`:

```python
returns_pl = returns_pl.drop_nulls()
# or fill forward
returns_pl = returns_pl.with_columns(pl.all().forward_fill())
```

Or use the `null_strategy` parameter on `Data.from_returns` / `Data.from_prices`:

```python
# Mirrors pandas / QuantStats behaviour: silently drop rows with nulls
data = jqs.Data.from_returns(returns=returns_pl, null_strategy="drop")

# Forward-fill nulls before computing statistics
data = jqs.Data.from_returns(returns=returns_pl, null_strategy="forward_fill")

# Raise an informative error if any null is found (useful during development)
data = jqs.Data.from_returns(returns=returns_pl, null_strategy="raise")
```

The default (`null_strategy=None`) passes nulls through unchanged.

### No top-level functions

QuantStats exposes bare module-level functions. jquantstats has none.

```python
# ❌ does not exist
jqs.stats.sharpe(returns)

# ✅ correct
data = jqs.Data.from_returns(returns=returns_pl)
data.stats.sharpe()
```

---

## Portfolio-only features

These capabilities have no QuantStats equivalent and are only available
through the `Portfolio` entry point.

```python
pf = jqs.Portfolio.from_cash_position(prices, positions, aum=1_000_000)

# Execution-delay analysis
pf_lagged = pf.lag(1)                    # shift positions forward by 1 day
pf.plots.lead_lag_ir_plot(max_lag=5)     # information ratio across lags

# Tilt / timing decomposition
pf.tilt                                  # constant-weight (allocation skill)
pf.timing                                # weight deviations (timing skill)
pf.tilt_timing_decomp                    # side-by-side NAV comparison

# Turnover analytics
pf.turnover                              # daily one-way turnover (fraction of AUM)
pf.turnover_weekly()                     # weekly aggregate
pf.turnover_summary()                    # {"mean_daily": ..., "mean_weekly": ..., "std": ...}

# Cost modelling
from jquantstats import CostModel

pf_a = jqs.Portfolio.from_cash_position(
    prices, positions, aum,
    cost_model=CostModel.per_unit(0.01),    # £0.01 per share traded
)
pf_b = jqs.Portfolio.from_cash_position(
    prices, positions, aum,
    cost_model=CostModel.turnover_bps(5),   # 5 bps per unit of AUM turnover
)

pf.trading_cost_impact(max_bps=20)          # sweep cost sensitivity 0 → 20 bps
```

---

## Minimal migration checklist

1. **Install** `jquantstats` (`pip install jquantstats`).
2. **Convert** your `pd.Series` to a `pl.DataFrame` with a date column.
3. **Construct** a `Data` object once with `Data.from_returns(...)`.
4. **Replace** every `qs.stats.foo(r)` call with `data.stats.foo()["col"]`.
5. **Scale** any stored `information_ratio` values by `1 / √252` if you need
   to match old QuantStats numbers; or pass `annualise=False` to get the raw IR.
7. **Drop** any `NaN`-filled rows before passing data in, or use `null_strategy`.

---

## Further reading

- [API Reference](docs/api.md) — complete method signatures
- [Architecture](docs/ARCHITECTURE.md) — how the library is structured
- [Changelog](CHANGELOG.md) — version history and new features
- [API Stability](docs/stability.md) — versioning and deprecation policy
- [Quick Reference](docs/QUICK_REFERENCE.md) — one-page lookup table
