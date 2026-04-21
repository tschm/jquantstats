---
icon: material/swap-horizontal
---

# Migrating from QuantStats to jquantstats

**A heartfelt thank you to [Ran Aroussi](https://github.com/ranaroussi) for creating
[QuantStats](https://github.com/ranaroussi/quantstats) — a brilliant and widely loved
library that has helped countless quants and portfolio managers understand their
strategies better. This project would simply not exist without that foundation.
We have enormous respect for the original work and encourage everyone to check it out.**

`jquantstats` is a modern variation on the theme set by QuantStats — not a copy, not a
drop-in replacement, but a different take on the same problem space. Where QuantStats
offers a collection of standalone functions each operating on a return series,
jquantstats is built around a **portfolio-centric entry point**: you start from a
`Portfolio` object constructed from prices and positions, and analytics flow naturally
from there. This makes multi-asset analysis, cost modelling, and execution-lag studies
first-class citizens rather than afterthoughts.

This guide explains the **key conceptual and API differences** between the two libraries
so you know what to expect, and provides **concrete code translations** to help you move
your existing workflows over.


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
data.plots.snapshot()
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

| QuantStats | jquantstats | Notes |
|---|---|---|
| `qs.stats.sharpe(r, periods=252)` | `data.stats.sharpe(periods=252)` | |
| `qs.stats.smart_sharpe(r, periods=252)` | `data.stats.smart_sharpe(periods=252)` | |
| `qs.stats.sortino(r, periods=252)` | `data.stats.sortino(periods=252)` | |
| `qs.stats.smart_sortino(r)` | `data.stats.smart_sortino()` | |
| `qs.stats.adjusted_sortino(r)` | `data.stats.adjusted_sortino()` | |
| `qs.stats.calmar(r)` | `data.stats.calmar()` | |
| `qs.stats.omega(r)` | `data.stats.omega()` | |
| `qs.stats.treynor_ratio(r, benchmark=b)` | `data.stats.treynor_ratio()` | |
| `qs.stats.volatility(r, periods=252)` | `data.stats.volatility(periods=252)` | |
| `qs.stats.implied_volatility(r)` | `data.stats.implied_volatility()` | |
| `qs.stats.skew(r)` | `data.stats.skew()` | |
| `qs.stats.kurtosis(r)` | `data.stats.kurtosis()` | |
| `qs.stats.max_drawdown(r)` | `data.stats.max_drawdown()` | |
| `qs.stats.value_at_risk(r)` | `data.stats.value_at_risk(alpha=0.05)` | |
| `qs.stats.conditional_value_at_risk(r, confidence=0.95)` | `data.stats.conditional_value_at_risk(alpha=0.05)` | `alpha = 1 − confidence`; see note below |
| `qs.stats.win_rate(r)` | `data.stats.win_rate()` | |
| `qs.stats.avg_return(r)` | `data.stats.avg_return()` | |
| `qs.stats.avg_win(r)` | `data.stats.avg_win()` | |
| `qs.stats.avg_loss(r)` | `data.stats.avg_loss()` | |
| `qs.stats.best(r)` | `data.stats.best()` | |
| `qs.stats.worst(r)` | `data.stats.worst()` | |
| `qs.stats.consecutive_wins(r)` | `data.stats.consecutive_wins()` | |
| `qs.stats.consecutive_losses(r)` | `data.stats.consecutive_losses()` | |
| `qs.stats.profit_factor(r)` | `data.stats.profit_factor()` | |
| `qs.stats.profit_ratio(r)` | `data.stats.profit_ratio()` | |
| `qs.stats.payoff_ratio(r)` | `data.stats.payoff_ratio()` | |
| `qs.stats.win_loss_ratio(r)` | `data.stats.win_loss_ratio()` | |
| `qs.stats.gain_to_pain_ratio(r)` | `data.stats.gain_to_pain_ratio()` | |
| `qs.stats.risk_return_ratio(r)` | `data.stats.risk_return_ratio()` | |
| `qs.stats.cpc_index(r)` | `data.stats.cpc_index()` | |
| `qs.stats.common_sense_ratio(r)` | `data.stats.common_sense_ratio()` | |
| `qs.stats.tail_ratio(r)` | `data.stats.tail_ratio()` | |
| `qs.stats.outlier_win_ratio(r)` | `data.stats.outlier_win_ratio()` | |
| `qs.stats.outlier_loss_ratio(r)` | `data.stats.outlier_loss_ratio()` | |
| `qs.stats.kelly_criterion(r)` | `data.stats.kelly_criterion()` | |
| `qs.stats.exposure(r)` | `data.stats.exposure()` | |
| `qs.stats.cagr(r)` | `data.stats.cagr()` | |
| `qs.stats.rar(r)` | `data.stats.rar()` | |
| `qs.stats.recovery_factor(r)` | `data.stats.recovery_factor()` | |
| `qs.stats.risk_of_ruin(r)` | `data.stats.risk_of_ruin()` | |
| `qs.stats.ulcer_index(r)` | `data.stats.ulcer_index()` | |
| `qs.stats.ulcer_performance_index(r)` | `data.stats.ulcer_performance_index()` | |
| `qs.stats.serenity_index(r)` | `data.stats.serenity_index()` | |
| `qs.stats.information_ratio(r, benchmark=b)` | `data.stats.information_ratio()` | jquantstats annualises; see note below |
| `qs.stats.r_squared(r, benchmark=b)` | `data.stats.r_squared()` | |
| `qs.stats.greeks(r, benchmark=b)` | `data.stats.greeks()` | |
| `qs.stats.probabilistic_sharpe_ratio(r)` | `data.stats.probabilistic_sharpe_ratio()` | |
| `qs.stats.probabilistic_sortino_ratio(r)` | `data.stats.probabilistic_sortino_ratio()` | |
| `qs.stats.probabilistic_adjusted_sortino_ratio(r)` | `data.stats.probabilistic_adjusted_sortino_ratio()` | |
| `qs.stats.geometric_mean(r)` | `data.stats.geometric_mean()` | |
| `qs.stats.ghpr(r)` | `data.stats.ghpr()` | |
| `qs.stats.expected_return(r)` | `data.stats.expected_return()` | |
| `qs.stats.outliers(r)` | `data.stats.outliers()` | |
| `qs.stats.remove_outliers(r)` | `data.stats.remove_outliers()` | |
| `qs.stats.drawdown_details(r)` | `data.stats.drawdown_details()` | |
| `qs.stats.monthly_returns(r)` | `data.stats.monthly_returns()` | |
| `qs.stats.compare(r, benchmark=b)` | `data.stats.compare()` | |
| `qs.stats.rolling_sharpe(r)` | `data.stats.rolling_sharpe()` | |
| `qs.stats.rolling_sortino(r)` | `data.stats.rolling_sortino()` | |
| `qs.stats.rolling_volatility(r)` | `data.stats.rolling_volatility()` | |
| `qs.stats.rolling_greeks(r, benchmark=b)` | `data.stats.rolling_greeks()` | |
| `qs.stats.autocorr_penalty(r)` | `data.stats.autocorr_penalty()` | |
| `qs.stats.pct_rank(r)` | `data.stats.pct_rank()` | |

#### QuantStats functions with no jquantstats equivalent

| QuantStats | Notes |
|---|---|
| `qs.stats.montecarlo(r)` | Monte Carlo simulations not currently implemented |
| `qs.stats.montecarlo_sharpe(r)` | |
| `qs.stats.montecarlo_drawdown(r)` | |
| `qs.stats.montecarlo_cagr(r)` | |

#### jquantstats-only stats methods

These methods have **no QuantStats equivalent** and are unique to jquantstats.

| jquantstats | Description |
|---|---|
| `data.stats.acf(nlags=20)` | Full autocorrelation function series |
| `data.stats.annual_breakdown()` | Year-by-year performance table |
| `data.stats.autocorr(lag=1)` | Autocorrelation at a given lag (qs only has `autocorr_penalty`) |
| `data.stats.avg_drawdown()` | Average drawdown across all drawdown episodes |
| `data.stats.down_capture()` | Downside capture ratio vs benchmark |
| `data.stats.hhi_positive()` | Herfindahl-Hirschman concentration index for positive returns |
| `data.stats.hhi_negative()` | Herfindahl-Hirschman concentration index for negative returns |
| `data.stats.max_drawdown_duration()` | Duration of the maximum drawdown in periods |
| `data.stats.monthly_win_rate()` | Win rate computed on monthly aggregated returns |
| `data.stats.periods_per_year` | Inferred annualisation factor (property) |
| `data.stats.sharpe_variance()` | Variance-penalised Sharpe variant |
| `data.stats.summary()` | Composite stats summary as a `pl.DataFrame` |
| `data.stats.up_capture()` | Upside capture ratio vs benchmark |
| `data.stats.worst_n_periods(n=5)` | Worst N individual return periods |

### Plots

| QuantStats | jquantstats |
|---|---|
| `qs.plots.snapshot(r)` | `data.plots.snapshot()` |
| `qs.plots.drawdown(r)` | `data.plots.drawdown()` |
| `qs.plots.returns(r)` | `data.plots.returns()` |
| `qs.plots.monthly_heatmap(r)` | `data.plots.monthly_heatmap()` |
| `qs.plots.distribution(r)` | `data.plots.distribution()` |
| `qs.plots.rolling_sharpe(r)` | `data.plots.rolling_sharpe()` |
| `qs.plots.rolling_volatility(r)` | `data.plots.rolling_volatility()` |

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
5. **Drop** any `NaN`-filled rows before passing data in, or use `null_strategy`.

---

## Further reading

- [API Reference](docs/api.md) — complete method signatures
- [Architecture](docs/ARCHITECTURE.md) — how the library is structured
- [Changelog](CHANGELOG.md) — version history and new features
- [API Stability](docs/stability.md) — versioning and deprecation policy
- [Quick Reference](docs/QUICK_REFERENCE.md) — one-page lookup table
