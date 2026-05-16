# jquantstats — Plan to 10.0

> Based on `report.md` (internal quality, current overall **7.4**) and `benchmark.md`
> (vs quantstats, current overall **8.0**).
>
> Tasks are grouped by theme and ordered within each theme by effort × impact.
> Each task carries an effort estimate, the source score it closes, and the target
> score achievable once the whole theme is done.

---

## Theme 1 — Code Duplication (7 → 10)

### T1.1 — Extract shared report formatting helpers ✅

> **Done** — merged [PR #727](https://github.com/Jebel-Quant/jquantstats/pull/727) · closes [#715](https://github.com/Jebel-Quant/jquantstats/issues/715)

**Files:** `_reports/_data.py:17`, `_reports/_portfolio.py:34`

Create `src/jquantstats/_reports/_formatting.py` containing:

```python
def _is_finite(v: Any) -> TypeGuard[int | float]: ...
def _fmt(value: Any, fmt: str = ".4f", suffix: str = "") -> str: ...
```

Replace the two inline definitions with imports. Unify the `bool` vs
`TypeGuard[int | float]` return annotation — `TypeGuard` is strictly more
informative and should be the canonical version.

**Effort:** 30 min · **Removes:** ~30 lines

---

### T1.2 — Consolidate positive/negative series filtering ✅

> **Done** — merged [PR #725](https://github.com/Jebel-Quant/jquantstats/pull/725) · closes [#717](https://github.com/Jebel-Quant/jquantstats/issues/717)

**File:** `src/jquantstats/_stats/_basic.py`

Add two static helpers alongside the existing `_mean_positive_expr` /
`_mean_negative_expr`:

```python
@staticmethod
def _positive(series: pl.Series) -> pl.Series:
    return series.filter(series > 0)

@staticmethod
def _negative(series: pl.Series) -> pl.Series:
    return series.filter(series < 0)
```

Replace the ~12 inline `series.filter(series > 0)` chains in `payoff_ratio`,
`profit_ratio`, `profit_factor`, `win_rate`, `risk_of_ruin`, and related methods
with calls to these helpers.

**Effort:** 1 hr · **Removes:** ~30 lines of duplication

---

### T1.3 — Remove the `prices()` / `_nav_series` duplication ✅

> **Done** — `prices()` now delegates directly to `_nav_series`; no separate PR.

**Files:** `_internals.py`, `_performance.py`

`_PerformanceStatsMixin.prices` recomputes `(1 + series).cumprod()` inline.
Change it to call `_nav_series` from `_internals.py`. Add a note in `_internals.py`
that `prices()` is the public entry point for consumers.

**Effort:** 15 min · **Removes:** 1 duplicate computation path

---

## Theme 2 — API Surface & Naming (7 → 10)

### T2.1 — Remove alias methods

**File:** `src/jquantstats/_stats/_performance.py`, `_basic.py`

Delete `ghpr()` (`_performance.py:315`), `r2()` (`_performance.py:724`), and
`win_loss_ratio()` (`_basic.py:206`). Remove their entries from `StatsLike`.
Add a one-line migration note to `CHANGELOG.md` under a `Deprecated` heading.

If quantstats name-compatibility is later required, re-introduce as
`@deprecated` shims rather than silent aliases.

**Effort:** 30 min · **Removes:** ~20 lines + 3 protocol stubs

---

### T2.2 — Complete the `PortfolioUtils` facade ✅

> **Done** — merged [PR #726](https://github.com/Jebel-Quant/jquantstats/pull/726) · closes [#716](https://github.com/Jebel-Quant/jquantstats/issues/716)

**File:** `src/jquantstats/_utils/_portfolio.py`

Add delegation methods for the two `DataUtils` methods not yet exposed:

```python
def winsorise(self, window: int = 7, n_sigma: float = 3.0) -> pl.DataFrame:
    """..."""
    return self._du().winsorise(window=window, n_sigma=n_sigma)

def exponential_cov(
    self, window: int = 30, is_halflife: bool = False, warmup: int = 0
) -> dict[str, dict[str, float]]:
    """..."""
    return self._du().exponential_cov(
        window=window, is_halflife=is_halflife, warmup=warmup
    )
```

Add tests for both methods via the `portfolio_pf` fixture.

**Effort:** 30 min · **Closes:** uniform-delegation design principle

---

### T2.3 — Use the public `periods_per_year` property in rolling methods ✅

> **Done** — `1e989a5` (direct commit, no separate PR)

**File:** `src/jquantstats/_stats/_rolling.py`

All four occurrences of `self._data._periods_per_year` in `rolling_sortino`,
`rolling_sharpe`, `rolling_greeks`, and `rolling_volatility` replaced with
`self.periods_per_year`.

**Effort:** 15 min · **Closes:** private-attribute leakage

---

## Theme 3 — Abstraction & Indirection (7 → 10)

### T3.1 — Make decorator coupling to `self._data` explicit ✅

> **Done** — `da3fd15` (direct commit, no separate PR)

**File:** `src/jquantstats/_stats/_core.py`

Both `columnwise_stat` and `to_frame` docstrings now state the `self._data` /
`self.all` preconditions explicitly. Both documentation and enforcement at
decoration time are now complete — [PR #735](https://github.com/Jebel-Quant/jquantstats/pull/735) ✅

**Effort:** 30 min · **Closes:** silent runtime failure for out-of-context use

---

### T3.2 — Rename `_PerformanceStatsMixin` for clarity ✅

> **Done** — merged [PR #732](https://github.com/Jebel-Quant/jquantstats/pull/732) · closes [#731](https://github.com/Jebel-Quant/jquantstats/issues/731)

**File:** `src/jquantstats/_stats/_performance.py` and all imports

The mixin's content spans Sharpe/Sortino ratios, drawdown metrics, Greeks, HHI,
R-squared, and Kelly criterion. A name like `_RiskAdjustedStatsMixin` more
accurately describes its scope. Coordinate with `_stats.py`, protocol files, and
any `TYPE_CHECKING` imports. This is a pure rename — no logic changes.

**Effort:** 45 min · **Improves:** mixin boundary legibility

---

### T3.3 — Standardise rolling method implementation shape ✅

> **Done** — merged [PR #723](https://github.com/Jebel-Quant/jquantstats/pull/723) · closes [#721](https://github.com/Jebel-Quant/jquantstats/issues/721)

**File:** `src/jquantstats/_stats/_rolling.py`

`rolling_sortino` uses `@to_frame` + `pl.Expr` while the other three operate on
`self.all` directly. Rewrite `rolling_sortino` to match the pattern used by
`rolling_sharpe` / `rolling_volatility`: operate on `self.all`, build the output
column as a `pl.Expr`, and return a `pl.DataFrame` directly. Remove the `@to_frame`
decorator.

**Effort:** 1 hr · **Closes:** asymmetry in the rolling API

---

## Theme 4 — Null / Error-Handling Consistency (6 → 10)

### T4.1 — Document the null-return convention ✅

> **Done** — merged [PR #724](https://github.com/Jebel-Quant/jquantstats/pull/724) · closes [#720](https://github.com/Jebel-Quant/jquantstats/issues/720)

**File:** `src/jquantstats/_stats/_core.py`

Convention documented in `_core.py`: scalar metrics return `float("nan")` when the
series has no non-null observations; ratio metrics return `float("nan")` when the
denominator is zero or indeterminate.

**Effort:** 15 min

---

### T4.2 — Normalise callsites to the declared convention ✅

> **Done** — merged [PR #724](https://github.com/Jebel-Quant/jquantstats/pull/724) · closes [#720](https://github.com/Jebel-Quant/jquantstats/issues/720)

**Files:** `_basic.py`, `_performance.py`, `_reporting.py`, `_rolling.py`

`cast(float, series.mean())` callsites replaced with `_mean(series)` throughout.
The new `_mean` helper (added to `_core.py`) returns `float("nan")` when the
series is empty or all-null — consistent with the documented convention.

**Effort:** 2 hr · **Affected sites:** ~35

---

## Theme 5 — Mixin Architecture & Coupling (7 → 10)

### T5.1 — Document cross-mixin dependencies ✅

> **Done** — merged [PR #752](https://github.com/Jebel-Quant/jquantstats/pull/752)

**Files:** `_reporting.py`, `_performance.py`

For each method that calls into a sibling mixin (e.g. `rar()` calling
`self.exposure()`), add a one-line comment identifying the source mixin:

```python
# exposure() is provided by _BasicStatsMixin
exp = self.exposure()
```

Alternatively, add a `# Cross-mixin dependencies:` section to the class docstring
listing every external method used. This makes the coupling visible without
changing behaviour.

**Effort:** 30 min

---

### T5.2 — Add a cross-mixin integration test

**File:** `tests/test_jquantstats/test__stats/test_mixin_isolation.py` (new)

Add a test that instantiates `_ReportingStatsMixin` in isolation (without the other
mixins) and asserts that calling `rar()` raises `AttributeError` with a useful
message, confirming the dependency is known and expected rather than an accident.

**Effort:** 30 min · **Closes:** silent cross-mixin failure gap

---

## Theme 6 — Protocol Design (6 → 10)

### T6.1 — Trim `StatsLike` to its actual consumers

**File:** `src/jquantstats/_reports/_protocol.py`

Grep for every `StatsLike` method called inside `_reports/`. Replace the 66-stub
protocol with a minimal version covering only those methods. Expected reduction:
66 stubs → ~12 stubs, ~150 lines removed.

Remove `@runtime_checkable` from `StatsLike` unless an `isinstance` check against
it exists somewhere in non-test code — `@runtime_checkable` is only useful at
runtime and adds maintenance overhead for no static benefit.

**Effort:** 1 hr · **Removes:** ~150 lines

---

### T6.2 — Consolidate `DataLike` into one authoritative definition ✅

> **Done** — merged [PR #736](https://github.com/Jebel-Quant/jquantstats/pull/736) · closes [#734](https://github.com/Jebel-Quant/jquantstats/issues/734)

**Files:** `_plots/_protocol.py`, `_reports/_protocol.py`, `_utils/_protocol.py`

All three define a `DataLike` protocol with overlapping attributes. Create a single
`src/jquantstats/_protocol.py` at the package root and import from there. Each
sub-package may define a narrower `DataLike` alias if it genuinely needs fewer
attributes, but all must extend or import the root definition rather than redefine
it independently.

**Effort:** 1 hr · **Closes:** three-way divergence risk

---

## Theme 7 — Dead Code (8 → 10)

### T7.1 — Clarify `hhi_positive` / `hhi_negative` intent ✅

> **Done** — merged [PR #730](https://github.com/Jebel-Quant/jquantstats/pull/730) · closes [#722](https://github.com/Jebel-Quant/jquantstats/issues/722)

**File:** `src/jquantstats/_stats/_performance.py:154`

These methods are tested but never appear in `summary()`, any report template, or
any public export. Both are clarified as intentionally public optional metrics.

**Effort:** 15 min

---

## Theme 8 — Test Quality (9 → 10)

### T8.1 — Refactor large test functions in `test_stats.py`

**File:** `tests/test_jquantstats/test__stats/test_stats.py`

`test_stats.py` is 1 775 lines. Identify the 5–10 longest test functions and
extract their inline setup into named sub-fixtures in `conftest.py` or into a
`@pytest.fixture` scoped to the test class. Target: no test function exceeds 20
lines of setup code.

> **Partial** — [PR #711](https://github.com/Jebel-Quant/jquantstats/pull/711) extracted the `integer_indexed_data` sub-fixture ✅ (partial); further refactoring open in [PR #747](https://github.com/Jebel-Quant/jquantstats/pull/747)

**Effort:** 1 hr · **Improves:** readability, future maintainability

---

## Theme 9 — Stats Coverage (7 → 10 vs quantstats)

### T9.1 — Implement Monte Carlo simulation suite

**File:** `src/jquantstats/_stats/_montecarlo.py` (new), exposed via `Stats`

Implement the four Monte Carlo methods present in quantstats:

| Method | Description |
|---|---|
| `montecarlo(n, period)` | Simulate `n` return paths over `period` observations |
| `montecarlo_sharpe(n, period, periods_per_year)` | Distribution of simulated Sharpe ratios |
| `montecarlo_drawdown(n, period)` | Distribution of simulated max drawdowns |
| `montecarlo_cagr(n, period, periods_per_year)` | Distribution of simulated CAGRs |

Each method should return a `pl.DataFrame` (one column per asset, one row per
simulation). The simulation baseline is block bootstrap with replacement, matching
quantstats' approach, so migration tests can verify numeric equivalence.

Add a `_MonteCarloStatsMixin` and include it in `Stats`.

**Effort:** 3–4 hr · **Closes:** biggest functional gap vs quantstats

---

## Theme 10 — Plot Coverage (6 → 10 vs quantstats)

### T10.1 — Add Monte Carlo plots

**File:** `src/jquantstats/_plots/_data.py`

Add two plot methods to `DataPlots`:

- `montecarlo(n, period)` — fan chart of simulated return paths
- `montecarlo_distribution(n, period, metric)` — histogram of simulated metric
  (Sharpe, drawdown, CAGR) distribution with observed value marked

Both should return `go.Figure` (Plotly), consistent with the existing plot API.

**Effort:** 2 hr · **Depends on:** T9.1

---

### T10.2 — Close remaining plot gaps vs quantstats

Audit `quantstats/_plotting/wrappers.py` against `DataPlots` / `PortfolioPlots`
and implement the missing plots. Priority order based on usage frequency in
quantstats tearsheets:

| Plot | Notes |
|---|---|
| `log_returns` (cumulative, log scale) | May already exist — verify |
| `compare(benchmark)` | Overlay of asset vs benchmark cumulative returns |
| `rolling_beta` | Rolling beta vs benchmark |

Each plot should follow the existing Plotly pattern: return `go.Figure`, accept
`title` and `figsize` kwargs, include a date range selector.

**Effort:** 2–3 hr per plot

---

## Theme 11 — Performance (8 → 10)

### T11.1 — Rewrite `rolling_sortino` using native Polars expressions ✅

> **Done** — merged [PR #740](https://github.com/Jebel-Quant/jquantstats/pull/740)

**File:** `src/jquantstats/_stats/_rolling.py:123`

The current implementation uses `map_elements` (a Python-level UDF per element),
which bypasses Polars' query optimiser and is significantly slower than native
expressions. Replace with:

```python
mean_ret = pl.col(col).rolling_mean(window_size=rolling_period)
downside = (
    pl.when(pl.col(col) < 0)
    .then(pl.col(col) ** 2)
    .otherwise(0.0)
    .rolling_mean(window_size=rolling_period)
)
sortino = (mean_ret / downside.sqrt()) * scale
```

Add a benchmark test (`pytest-benchmark`) comparing the old and new
implementations on a 10-year daily series to confirm the speedup.

**Effort:** 1 hr · **Closes:** known Polars UDF antipattern

---

## Theme 12 — Error Handling (9 → 10)

### T12.1 — Carry null-handling guarantees through to computation

**Files:** `_basic.py`, `_performance.py`

After `Data.__post_init__` enforces the null strategy, individual metrics should
not need defensive null checks for values that the strategy already guarantees are
absent. Audit for guards of the form `if series.is_empty() or series.null_count() > 0`
that duplicate construction-time validation and remove them, replacing with
`assert`s that document the invariant for maintainers.

**Effort:** 1 hr

---

## Theme 13 — Type Safety (9 → 10)

### T13.1 — Eliminate `cast(float, ...)` noise via a typed helper ✅

> **Done** — merged [PR #724](https://github.com/Jebel-Quant/jquantstats/pull/724)

**Files:** across stats mixins

`_mean` added to `_core.py`. Returns `float("nan")` (not `0.0`) when the series
is empty or all-null — consistent with the scalar-metric convention established in
T4.1. All `cast(float, series.mean())` callsites replaced.

**Effort:** 1 hr · **Eliminates:** ~20 type-cast suppressions

---

## Consolidated Effort Estimate

| Theme | Tasks | Effort |
|---|---|---|
| 1. Code duplication | ~~T1.1~~, ~~T1.2~~, ~~T1.3~~ | ✅ done |
| 2. API surface | ~~T2.2~~, ~~T2.3~~; T2.1 open | ~0.5 hr |
| 3. Abstraction | ~~T3.1~~, ~~T3.2~~, ~~T3.3~~ | ✅ done |
| 4. Null handling | ~~T4.1~~, ~~T4.2~~ | ✅ done |
| 5. Mixin coupling | ~~T5.1~~; T5.2 open | ~0.5 hr |
| 6. Protocol design | ~~T6.2~~; T6.1 open | 1 hr |
| 7. Dead code | ~~T7.1~~ | ✅ done |
| 8. Test quality | T8.1 (partial) | ~0.75 hr |
| 9. Monte Carlo stats | T9.1 | 3.5 hr |
| 10. Plot coverage | T10.1–T10.2 | 6 hr |
| 11. Performance | ~~T11.1~~ | ✅ done |
| 12. Error handling | T12.1 | 1 hr |
| 13. Type safety | ~~T13.1~~ | ✅ done |
| **Total remaining** | **7 tasks** | **~13.25 hr** |

---

## Recommended Sequence

**Sprint 1 — Quick wins (~4 hr, no API changes)** ✅ complete
~~T1.1~~, ~~T1.2~~, ~~T1.3~~, ~~T2.3~~, ~~T3.3~~, ~~T4.1~~, ~~T7.1~~, ~~T13.1~~

**Sprint 2 — API clean-up (~4 hr, minor breaking changes)**
T2.1, ~~T2.2~~, ~~T3.1~~, ~~T3.2~~, ~~T4.2~~, ~~T5.1~~, T5.2, T8.1

**Sprint 3 — Architecture (~4 hr, protocol restructure)**
T6.1, ~~T6.2~~, T12.1

**Sprint 4 — Feature parity (~8 hr, Monte Carlo)**
T9.1, T10.1, ~~T11.1~~

**Sprint 5 — Plot parity (~4 hr)**
T10.2

After Sprint 3 the internal quality score reaches **10.0**.
After Sprint 5 the benchmark score reaches **10.0**.
