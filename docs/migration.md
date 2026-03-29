# Migrating from QuantStats to jquantstats

This guide helps users of [QuantStats](https://github.com/ranaroussi/quantstats) move to
**jquantstats**, a modern replacement built on [Polars](https://pola.rs/).

---

## Why migrate

| | QuantStats | jquantstats |
|---|---|---|
| Dataframe engine | pandas | Polars (zero pandas at runtime) |
| Charts | static matplotlib | interactive Plotly |
| Type annotations | partial | full (PEP 484) |
| Multi-asset | single Series | multi-column DataFrame |
| Input style | pass Series to every call | create a `Data` object once |

Key benefits at a glance:

- **Polars-native** — faster data processing, no pandas dependency at runtime.
- **Interactive charts** — Plotly charts you can zoom, hover, and export.
- **Full type annotations** — works seamlessly with IDEs and type checkers.
- **Multi-asset in one call** — pass a multi-column DataFrame and get back
  results for every asset at once.

---

## Data format conversion

QuantStats expects a **pandas `Series`** with a `DatetimeIndex`.
jquantstats expects a **Polars `DataFrame`** with a dedicated date column.

```python
import pandas as pd
import polars as pl

# --- QuantStats style ---
# returns is a pd.Series with a DatetimeIndex
returns_pd: pd.Series = ...

# --- jquantstats style ---
# convert a pandas Series to a polars DataFrame
returns_pl = pl.from_pandas(returns_pd.rename("MyStrategy").reset_index())
# → DataFrame with columns ["Date", "MyStrategy"]

# or build directly from polars
returns_pl = pl.DataFrame({
    "Date": dates,          # list[datetime.date] or pl.Series of pl.Date
    "MyStrategy": values,   # list[float]
})
```

### Using a benchmark

```python
# QuantStats — pass benchmark separately to each function
import quantstats as qs
qs.stats.sharpe(returns_pd, benchmark=benchmark_pd)

# jquantstats — pass benchmark once at construction time
import jquantstats as jqs
data = jqs.Data.from_returns(
    returns=returns_pl,
    benchmark=benchmark_pl,   # same shape as returns_pl
    date_col="Date",
)
# benchmark is used automatically by stats that need it (e.g. information_ratio)
```

---

## Entry point mapping

### Returns-series workflow

```python
# QuantStats
import quantstats as qs
qs.reports.full(returns_pd, benchmark=benchmark_pd)

# jquantstats
import jquantstats as jqs
data = jqs.Data.from_returns(returns=returns_pl, benchmark=benchmark_pl)
data.reports.summary()   # tabular summary
```

### Prices + positions workflow (active portfolio)

```python
# jquantstats — Portfolio entry point (no QuantStats equivalent)
import jquantstats as jqs
pf = jqs.Portfolio.from_cash_position(
    prices=prices_df,
    cash_position=positions_df,
    aum=1_000_000,
)
pf.stats.sharpe()
pf.plots.snapshot()
pf.report.to_html()

# Drop down to the returns-series API at any time
data = pf.data
data.stats.sharpe()
```

---

## API mapping table

The table below maps the most common QuantStats calls to their jquantstats
equivalents.  All jquantstats methods are called on the `stats` attribute of
a `Data` (or `Portfolio`) object and return a **dict-like result keyed by
column name**, so you can request results for multiple assets in one call.

```python
# one-time setup (replaces passing the Series to every call)
data = jqs.Data.from_returns(returns=returns_pl, benchmark=benchmark_pl)
```

| QuantStats | jquantstats | Notes |
|---|---|---|
| `qs.stats.sharpe(r, periods=252)` | `data.stats.sharpe(periods=252)` | Same formula; `periods` defaults to the inferred frequency |
| `qs.stats.sortino(r)` | `data.stats.sortino(periods=252)` | Pass `periods` explicitly when you want a specific annualisation |
| `qs.stats.volatility(r, periods=252)` | `data.stats.volatility(periods=252)` | Same formula |
| `qs.stats.max_drawdown(r)` | `data.stats.max_drawdown()` | Identical result; both return a negative fraction |
| `qs.stats.avg_drawdown(r)` | `data.stats.avg_drawdown()` | Identical result; both return a negative fraction |
| `qs.stats.calmar(r)` | `data.stats.calmar()` | Identical result |
| `qs.stats.value_at_risk(r)` | `data.stats.value_at_risk(alpha=0.05)` | See note on parameter naming below |
| `qs.stats.conditional_value_at_risk(r, confidence=0.95)` | `data.stats.conditional_value_at_risk(alpha=0.05)` | `alpha = 1 - confidence` |
| `qs.stats.information_ratio(r, benchmark=b)` | `data.stats.information_ratio(periods_per_year=252)` | jquantstats **annualises** the IR; QuantStats does not — see note below |
| `qs.stats.r_squared(r, benchmark=b)` | `data.stats.r_squared()` | Benchmark is set at construction time |
| `qs.stats.greeks(r, benchmark=b)` | `data.stats.greeks()` | Returns `{"alpha": ..., "beta": ...}` per column |
| `qs.stats.win_rate(r)` | `data.stats.win_rate()` | Identical result |
| `qs.stats.avg_return(r)` | `data.stats.avg_return()` | Identical result |
| `qs.stats.avg_win(r)` | `data.stats.avg_win()` | Identical result |
| `qs.stats.avg_loss(r)` | `data.stats.avg_loss()` | Identical result |
| `qs.stats.best(r)` | `data.stats.best()` | Identical result |
| `qs.stats.worst(r)` | `data.stats.worst()` | Identical result |
| `qs.stats.profit_factor(r)` | `data.stats.profit_factor()` | Identical result |
| `qs.stats.profit_ratio(r)` | `data.stats.profit_ratio()` | Identical result |
| `qs.stats.payoff_ratio(r)` | `data.stats.payoff_ratio()` | Identical result |
| `qs.stats.win_loss_ratio(r)` | `data.stats.win_loss_ratio()` | Identical result |
| `qs.stats.gain_to_pain_ratio(r)` | `data.stats.gain_to_pain_ratio()` | Identical result |
| `qs.stats.risk_return_ratio(r)` | `data.stats.risk_return_ratio()` | Identical result |
| `qs.stats.kelly_criterion(r)` | `data.stats.kelly_criterion()` | Identical result |
| `qs.stats.exposure(r)` | `data.stats.exposure()` | Identical result |
| `qs.stats.skew(r)` | `data.stats.skew()` | Identical result |
| `qs.stats.kurtosis(r)` | `data.stats.kurtosis()` | Identical result |
| `qs.plots.snapshot(r)` | `data.plots.plot_snapshot()` | Returns an interactive Plotly figure |
| `qs.plots.drawdown(r)` | `data.plots.plot_drawdown()` | Returns an interactive Plotly figure |
| `qs.plots.returns(r)` | `data.plots.plot_returns()` | Returns an interactive Plotly figure |
| `qs.reports.full(r)` | `data.reports.summary()` | See entry-point mapping above |

### Accessing a single-asset result

```python
# QuantStats returns a scalar
sharpe_value = qs.stats.sharpe(returns_pd, periods=252)   # float

# jquantstats returns a dict keyed by column name
result = data.stats.sharpe(periods=252)
sharpe_value = result["MyStrategy"]                        # float
```

---

## What's different

### Drawdown sign convention

Both `max_drawdown` and `avg_drawdown` follow the **QuantStats sign
convention**: drawdowns are returned as **negative fractions** (≤ 0).

```python
# QuantStats
qs.stats.max_drawdown(returns_pd)   # e.g. -0.35
qs.stats.avg_drawdown(returns_pd)   # e.g. -0.12

# jquantstats — same sign convention
data.stats.max_drawdown()["MyStrategy"]   # e.g. -0.35
data.stats.avg_drawdown()["MyStrategy"]   # e.g. -0.12
```

> **Tip:** If you compare against a threshold, use negative values as you
> would with QuantStats:
>
> ```python
> if data.stats.max_drawdown()["MyStrategy"] < -0.20:
>     print("Drawdown exceeds 20 %")
> ```

### `information_ratio` — annualisation

QuantStats returns a **non-annualised** information ratio.
jquantstats **annualises** by default (multiplies by `sqrt(periods_per_year)`).

```python
import numpy as np

# equivalent expressions
qs_ir = qs.stats.information_ratio(returns_pd, benchmark=benchmark_pd)
jqs_ir = data.stats.information_ratio(periods_per_year=252)["MyStrategy"]

# qs_ir * sqrt(252) ≈ jqs_ir
```

To obtain the raw, non-annualised ratio (matching QuantStats), pass `annualise=False`:

```python
# non-annualised — matches qs.stats.information_ratio directly
jqs_ir_raw = data.stats.information_ratio(periods_per_year=252, annualise=False)["MyStrategy"]
# jqs_ir_raw ≈ qs_ir
```

### `conditional_value_at_risk` — parameter naming

QuantStats uses `confidence` (e.g. `0.95`) for the confidence level.
jquantstats uses `alpha` (e.g. `0.05`) for the tail probability (`alpha = 1 − confidence`).

```python
# QuantStats
cvar = qs.stats.conditional_value_at_risk(returns_pd, confidence=0.95)

# jquantstats (alpha = 1 - confidence)
cvar = data.stats.conditional_value_at_risk(alpha=0.05)["MyStrategy"]
```

For migration convenience, passing `confidence` is accepted but emits a
`DeprecationWarning` and will be removed in a future release:

```python
# Accepted but deprecated — emits DeprecationWarning
cvar = data.stats.conditional_value_at_risk(confidence=0.95)["MyStrategy"]
```

### Multi-asset results

jquantstats is designed to work on **multiple assets simultaneously**.
Pass a DataFrame with several return columns and every method returns a
result for all of them at once.

```python
returns_multi = pl.DataFrame({
    "Date": dates,
    "AAPL": aapl_returns,
    "MSFT": msft_returns,
    "SPY":  spy_returns,
})

data = jqs.Data.from_returns(returns=returns_multi, date_col="Date")
data.stats.sharpe()
# → {"AAPL": 1.23, "MSFT": 0.98, "SPY": 0.76}
```

### No top-level module functions

QuantStats exposes bare functions in `qs.stats.*` that each accept a
`pd.Series`.  jquantstats has **no top-level functions**; all analytics
live on the `Stats` object that is accessed via `.stats` on a `Data` or
`Portfolio` instance.

```python
# ❌ does not exist in jquantstats
jqs.stats.sharpe(returns)

# ✅ correct jquantstats style
data = jqs.Data.from_returns(returns=returns_pl)
data.stats.sharpe()
```

### `null` / missing-value handling

pandas (and QuantStats) **silently drop** `NaN` values in aggregate
calculations.  Polars **propagates** `null` values — if any value in a
column is `null`, most statistics will return `null` instead of a
numeric result.

> **Important:** in Polars, `null` (a missing entry, equivalent to
> pandas `NaN`) and `NaN` (IEEE-754 "Not a Number", a valid float value)
> are different things.  jquantstats uses the Polars convention throughout:
> `null` means "missing" and is subject to the `null_strategy` below;
> `NaN` is a numeric value that propagates through calculations.

#### Recommended pre-processing

Clean your data **before** calling `Data.from_returns` or
`Data.from_prices`:

```python
import polars as pl

# Option 1 — drop rows that contain any null
returns_pl = returns_pl.drop_nulls()

# Option 2 — forward-fill missing values
returns_pl = returns_pl.with_columns(pl.all().forward_fill())
```

#### Using `null_strategy` for automatic handling

Both `Data.from_returns` and `Data.from_prices` accept a `null_strategy`
parameter that applies the chosen strategy at construction time:

```python
import jquantstats as jqs

# Mirrors pandas / QuantStats behaviour: silently drop rows with nulls
data = jqs.Data.from_returns(returns=returns_pl, null_strategy="drop")

# Forward-fill nulls before computing excess returns
data = jqs.Data.from_returns(returns=returns_pl, null_strategy="forward_fill")

# Raise an informative error if any null is found (useful during development)
data = jqs.Data.from_returns(returns=returns_pl, null_strategy="raise")
```

The default (`null_strategy=None`) preserves the current behaviour:
nulls are passed through unchanged and will propagate into statistics.

#### Multi-asset portfolios with different start dates

A common case is a multi-asset DataFrame where some assets have a shorter
history than others.  The earlier rows for newer assets will be `null`:

```python
# AAPL has data from 1980; META only from 2012 → META column is mostly null
returns_multi = pl.DataFrame({
    "Date": dates,          # starts 1980
    "AAPL": aapl_returns,
    "META": meta_returns,   # null before 2012
})

# Drop rows where any column is null (keeps only dates where ALL assets
# have data — shortest common history)
data = jqs.Data.from_returns(returns=returns_multi, null_strategy="drop")

# Alternatively, keep all dates and fill nulls for META before 2012
data = jqs.Data.from_returns(returns=returns_multi, null_strategy="forward_fill")
```

---

## Further reading

- [Quick Start](index.md) — minimal working example
- [API Reference](api.md) — complete method list with signatures
- [API Stability](stability.md) — versioning and deprecation policy
