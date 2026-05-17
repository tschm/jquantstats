# jquantstats vs quantstats — Benchmark Comparison

> Assessed: 2026-05-17  
> quantstats: `.venv/lib/python3.12/site-packages/quantstats/`  
> jquantstats: `src/jquantstats/` (main, post-PR #749)  
> Ratings are 1–10 where 10 = jquantstats is clearly superior, 5 = parity, 1 = quantstats is clearly superior.  
> A score below 5 means jquantstats has a gap to close.

---

## Scorecard

| Dimension | Score | Winner |
|---|---|---|
| Architecture & structure | 8 | jquantstats |
| Data model | 9 | jquantstats |
| API design & consistency | 8 | jquantstats |
| Stats metric coverage | 9 | jquantstats |
| Plot coverage | 8 | jquantstats |
| Reports | 7 | jquantstats |
| Error handling | 9 | jquantstats |
| Performance | 9 | jquantstats |
| Type safety | 9 | jquantstats |
| Test quality | 9 | jquantstats |
| **Overall** | **8.5** | **jquantstats** |

---

## 1. Architecture & Structure — 8/10

**quantstats:** Four flat modules (`stats.py` 3 307 LOC, `reports.py` 2 515 LOC, `_plotting/core.py` 2 137 LOC, `utils.py` 1 002 LOC). Entirely procedural — every metric is a free function imported and called directly:

```python
qs.stats.sharpe(returns)
qs.plots.snapshot(returns)
```

No concept of a data container or session object. Benchmark is passed as an argument to every benchmark-aware function independently.

**jquantstats:** Object-oriented with a clear layered architecture. Data lives in a `Data` or `Portfolio` object; analytics are exposed as lazy facades (`.stats`, `.plots`, `.reports`, `.utils`). The mixin architecture keeps file sizes manageable (808–875 LOC per stats mixin) and responsibilities clearly separated.

**Gap:** quantstats' flat structure is easy to get started with for one-off calculations, but it scales poorly — there is no shared state, so calling ten metrics on the same dataset requires passing the same series ten times. jquantstats' object model eliminates that friction. The one cost is a higher upfront learning curve.

---

## 2. Data Model — 9/10

**quantstats:** Accepts any pandas `Series` or `DataFrame`. Index must be `DatetimeIndex` or convertible. No null strategy — NaNs propagate silently. Benchmark is a loose argument.

**jquantstats:** Accepts any narwhals-compatible frame (pandas, Polars, PyArrow) and converts immediately to Polars internally. Strict null handling with three explicit strategies (`raise`, `drop`, `forward_fill`). Benchmark is stored once on the `Data` object and applied to all benchmark-aware metrics automatically. `Portfolio` additionally accepts raw prices, cash positions, and AUM and derives NAV and returns internally.

**Gap:** The main gap is in the other direction — quantstats' pandas-native model has zero friction for users already in a pandas workflow, and some users will object to the forced Polars conversion. jquantstats scores higher on correctness and safety, but loses a point for the potential pandas interop friction.

---

## 3. API Design & Consistency — 8/10

**quantstats:** Free-function API. Return types are inconsistent: most metrics return `float`, some return `pd.Series`, others return `pd.DataFrame`. No enforced contract.

```python
qs.stats.sharpe(returns)            # → float
qs.stats.monthly_returns(returns)   # → pd.DataFrame
qs.stats.rolling_sharpe(returns)    # → pd.Series
```

**jquantstats:** The `@columnwise_stat` decorator enforces `dict[str, float]` for all scalar metrics; `@to_frame` enforces `pl.DataFrame` with a date column for all time-series metrics. Every asset produces the same key in the dict. There are no surprises.

```python
data.stats.sharpe()           # → {"AAPL": 1.52, "META": 0.89}
data.stats.rolling_sharpe()   # → pl.DataFrame (Date | AAPL | META)
data.stats.monthly_returns()  # → pl.DataFrame
```

**Gap:** The alias methods (`ghpr`, `r2`, `win_loss_ratio`) slightly pollute the surface. ~~The asymmetry in `rolling_sortino` (uses `@to_frame` + `pl.Expr`) vs the other three rolling methods (operate on `self.all` directly) is a minor inconsistency.~~ The `rolling_sortino` asymmetry has been **fixed** — merged [PR #723](https://github.com/Jebel-Quant/jquantstats/pull/723) ✅. The decorator-contract enforcement gap has also been **closed** — merged [PR #735](https://github.com/Jebel-Quant/jquantstats/pull/735) ✅

---

## 4. Stats Metric Coverage — 9/10

**quantstats:** 79 public functions. Monte Carlo simulation is a distinctive feature (4 functions: `montecarlo`, `montecarlo_sharpe`, `montecarlo_drawdown`, `montecarlo_cagr`).

**jquantstats:** 100+ methods across five mixins (including `_MonteCarloStatsMixin` added via [PR #751](https://github.com/Jebel-Quant/jquantstats/pull/751) ✅). Several metrics are absent from quantstats:

| Only in jquantstats | Description |
|---|---|
| `hhi_positive` / `hhi_negative` | Herfindahl–Hirschman Index for return concentration |
| `sharpe_variance` | Asymptotic variance of the Sharpe ratio |
| `autocorr` / `acf` | Autocorrelation and autocorrelation function |
| `up_capture` / `down_capture` | Capture ratios in up/down benchmark markets |
| `annual_breakdown` | Year-by-year return and drawdown table |
| `summary` | Single-call comprehensive metrics DataFrame |

All four quantstats Monte Carlo methods are now present in jquantstats:

| Method | Status |
|---|---|
| `montecarlo` | ✅ added via PR #751 |
| `montecarlo_sharpe` | ✅ added via PR #751 |
| `montecarlo_drawdown` | ✅ added via PR #751 |
| `montecarlo_cagr` | ✅ added via PR #751 |

**Implementation differences for shared metrics:**

- **`implied_volatility`:** quantstats returns a scalar `float`; jquantstats returns a rolling `pl.DataFrame` when `annualize=True` or a `dict[str, float]` when `annualize=False`. The jquantstats version is more expressive.
- **`downside_deviation` (used in Sortino):** quantstats divides by the count of negative returns; jquantstats divides by the total observation count (matching the Red Rock Capital paper). The two formulas agree only when all returns are negative.

**Gap:** The Monte Carlo gap is now closed. Two points remain deducted for minor coverage differences in edge-case metrics and the alias methods (`ghpr`, `r2`, `win_loss_ratio`) that add surface without value.

---

## 5. Plot Coverage — 8/10

**quantstats:** ~42 plot functions via matplotlib. All output static images. Coverage includes Monte Carlo plots (`montecarlo`, `montecarlo_distribution`) and `compare` plots.

**jquantstats:** ~24 plot methods across `DataPlots` and `PortfolioPlots` via Plotly. Interactive (zoom, pan, hover, date range selectors). Portfolio-specific plots have no quantstats equivalent: `lead_lag_ir_plot`, `lagged_performance`, `smoothed_holdings`.

**Gap:** jquantstats has fewer plots by raw count (~24 vs ~42), but each plot is meaningfully higher quality (interactive Plotly vs static matplotlib). Monte Carlo plots (`montecarlo`, `montecarlo_distribution`) added via [PR #749](https://github.com/Jebel-Quant/jquantstats/pull/749) ✅; `compare` and rolling_beta figsize parity added via [PR #750](https://github.com/Jebel-Quant/jquantstats/pull/750) ✅.

| Category | quantstats | jquantstats |
|---|---|---|
| Basic performance | ✓ snapshot, earnings, daily/yearly/monthly returns | ✓ same |
| Heatmap | ✓ monthly_heatmap | ✓ monthly_heatmap |
| Drawdown | ✓ drawdown, drawdowns_periods | ✓ same |
| Rolling | ✓ rolling_volatility, rolling_sharpe, rolling_sortino, rolling_beta | ✓ same |
| Distribution | ✓ distribution, histogram | ✓ same |
| Compare | ✓ compare | ✓ compare (added PR #750) |
| Monte Carlo | ✓ montecarlo, montecarlo_distribution | ✓ same (added PR #749) |
| Portfolio-specific | ✗ | ✓ lead_lag_ir, lagged_performance, smoothed_holdings |
| Interactivity | ✗ (static) | ✓ (Plotly) |
| Date range selector | ✗ | ✓ |

---

## 6. Reports — 7/10

**quantstats:** `qs.reports.html()` generates a full HTML tearsheet embedding base64-encoded matplotlib figures. Three modes: `html()`, `full()` (verbose DataFrame), `basic()` (condensed). Opens automatically in the browser. Well-known and widely used in the quant community.

**jquantstats:** `.reports.html()` generates an HTML tearsheet with embedded Plotly divs (fully interactive). Portfolio reports include attribution analysis (tilt vs timing decomposition) not present in quantstats. Multi-asset reports work out of the box.

**Gap:** quantstats' reports are the de facto standard for tearsheets in the Python quant community — they are instantly recognisable and widely cited. jquantstats' Plotly-based reports are better in isolation (interactive, multi-asset), but users migrating from quantstats may find the output style unfamiliar. The portfolio attribution section is a clear advantage.

---

## 7. Error Handling — 9/10

**quantstats:** 5 exception types (`QuantStatsError`, `DataValidationError`, `CalculationError`, `PlottingError`, `BenchmarkError`). Basic validation on input type and emptiness. NaN propagation is largely silent.

**jquantstats:** 8+ domain-specific exceptions that map to exact failure modes (`MissingDateColumnError`, `NullsInReturnsError`, `RowCountMismatchError`, `NonPositiveAumError`, etc.). Three explicit null strategies force the caller to declare intent. Alignment validation between assets and benchmark is enforced at construction time, not at computation time.

**Gap:** jquantstats is substantially stronger here. The only point deducted is that the null-handling inconsistency within the stats mixins means the validation guarantees at construction time are not fully carried through to computation.

---

## 8. Performance — 9/10

**quantstats:** pandas + NumPy + scipy. Performance is adequate for typical series lengths (daily returns, 5–20 years ≈ 1 250–5 000 rows). Large DataFrames with many assets can be slow due to pandas copy-on-write overhead and row-wise Python loops in some metrics.

**jquantstats:** Polars (Apache Arrow columnar format) + scipy. Polars outperforms pandas by 5–50× for typical analytical queries, especially on multi-column DataFrames. The `columnwise_stat` decorator applies metrics per-column without Python-level loops.

~~One point deducted for `rolling_sortino` using `map_elements` (a Python-level UDF) — a known Polars performance antipattern and likely slower than an equivalent native Polars expression.~~ **Fixed** — `rolling_sortino` has been rewritten using native Polars expressions via [PR #740](https://github.com/Jebel-Quant/jquantstats/pull/740) ✅. Score raised from 8 → 9.

**Gap:** Benchmarks have not been run in this repo, so the 9/10 is inferred from the underlying library performance characteristics. No known Polars antipatterns remain in the codebase.

---

## 9. Type Safety — 9/10

**quantstats:** Minimal type hints. A `Returns = pd.Series | pd.DataFrame` alias is defined, and a handful of functions have `-> float` return annotations, but most parameters are unannotated. No static type checker configuration found.

**jquantstats:** Comprehensive type annotations throughout. `ty` is configured and runs clean (`All checks passed!`). Frozen dataclasses with `slots=True` for data containers. Protocol classes for structural typing. `TYPE_CHECKING` guards prevent circular imports while preserving IDE intelligence.

**Gap:** jquantstats is clearly stronger. One point deducted for the `cast(float, ...)` workaround pattern needed because Polars `.mean()` returns `float | None` — a minor friction with the type system rather than a design flaw.

---

## 10. Test Quality — 9/10

**quantstats:** No test suite is distributed with the package. The upstream repository may have tests but they are not installed and not observable here.

**jquantstats:** ~870 tests across 40+ test files. 100% code coverage. Includes:

- **Migration tests** (`test_migration/`) — numeric comparison against quantstats itself with `atol=1e-6`
- **Edge case tests** — empty series, all-NaN, single-observation, zero-variance inputs
- **Property-based tests** — `hypothesis`-driven invariant checking
- **Snapshot tests** — 13 plot snapshots pinned with `syrupy`
- **API contract tests** — assert public interface stability

**Gap:** The migration tests are a particularly strong design choice — they use quantstats as a ground-truth oracle, which both validates correctness and documents intentional divergences (e.g. the downside-deviation convention difference).

---

## Strategic Summary

jquantstats is a material improvement over quantstats on almost every engineering dimension: type safety, data model, error handling, API consistency, and test quality. The analytical capability gap is now largely closed:

1. ~~**Monte Carlo simulation** — 4 functions with no jquantstats equivalent.~~ **Closed** — `_MonteCarloStatsMixin` adds all four methods via PR #751 ✅
2. ~~**Monte Carlo plots** — `montecarlo`, `montecarlo_distribution` missing.~~ **Closed** — added via PR #749 ✅
3. **Plot count** — ~42 vs ~24 functions. The jquantstats plots are higher quality and interactive, but the breadth remains narrower (non-Monte-Carlo tearsheet plots not yet ported).
4. **Community recognition** — quantstats' HTML tearsheet is the de facto standard. jquantstats' reports are technically superior but unfamiliar.

The performance and correctness advantages of Polars, combined with the Portfolio data model, attribution analytics, and now full Monte Carlo parity, position jquantstats clearly ahead of quantstats for production use.
