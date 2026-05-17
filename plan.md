# jquantstats — Plan to 10.0

> Based on `report.md` (internal quality, current overall **9.9**) and `benchmark.md`
> (vs quantstats, current overall **9.1**).
>
> Tasks are grouped by theme and ordered within each theme by effort × impact.
> Each task carries an effort estimate, the source score it closes, and the target
> score achievable once the whole theme is done.

---

## Theme 1 — Code Duplication (7 → 9 → 10)

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

### T1.4 — Eliminate residual code duplication ✅

> **Done** — merged [PR #760](https://github.com/Jebel-Quant/jquantstats/pull/760) · closes [#758](https://github.com/Jebel-Quant/jquantstats/issues/758)

Audit the current source tree for any duplication introduced since T1.1–T1.3. Check `_stats/` for patterns not yet covered by `_positive`/`_negative`/`_mean`, and `_reports/` and `_plots/` for repeated figure-construction or rendering logic. Extract shared helpers where a block of 3+ lines appears in more than one location.

**Effort:** 1 hr · **Raises:** Code duplication 9 → 10

---

## Theme 2 — API Surface & Naming (7 → 9 → 10)

### T2.1 — Remove alias methods ✅

> **Done** — merged [PR #756](https://github.com/Jebel-Quant/jquantstats/pull/756) · closes [#718](https://github.com/Jebel-Quant/jquantstats/issues/718)

**File:** `src/jquantstats/_stats/_performance.py`, `_basic.py`

`ghpr()`, `r2()`, and `win_loss_ratio()` converted to `@deprecated` shims that emit
`DeprecationWarning` and delegate to their canonical counterparts. Migration note added
to `CHANGELOG.md`.

**Effort:** 30 min · **Removes:** silent alias surface

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

### T2.4 — Delete deprecated alias methods

**Issue:** [#757](https://github.com/Jebel-Quant/jquantstats/issues/757)

Remove `ghpr()`, `r2()`, and `win_loss_ratio()` from the stats mixins entirely. Remove their stubs from `StatsLike`. Update any tests that reference the deprecated names to use the canonical methods. Add a `Removed` entry to `CHANGELOG.md`.

**Effort:** 30 min · **Raises:** API surface & naming 9 → 10

---

## Theme 3 — Abstraction & Indirection (7 → 10) ✅

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

## Theme 4 — Null / Error-Handling Consistency (6 → 9 → 10)

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

### T4.3 — Resolve remaining `is_empty()` post-filter guards ✅

> **Done** — merged [PR #761](https://github.com/Jebel-Quant/jquantstats/pull/761) · closes [#759](https://github.com/Jebel-Quant/jquantstats/issues/759)

For each remaining guard in `_basic.py` and `_performance.py`, determine whether it is a legitimate post-filter check or a duplicate of the construction-time invariant. Legitimate guards get a one-line comment and a `float("nan")` return; redundant guards get replaced with an `assert`.

**Known locations:** `_basic.py` ~241 (`wins/losses`), `_basic.py` ~762 (`paired`), `_performance.py` ~443 (`dd_frame`).

**Effort:** 1 hr · **Raises:** Null / error-handling consistency 9 → 10

---

## Theme 5 — Mixin Architecture & Coupling (7 → 10) ✅

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

### T5.2 — Add a cross-mixin integration test ✅

> **Done** — merged [PR #748](https://github.com/Jebel-Quant/jquantstats/pull/748)

**File:** `tests/test_jquantstats/test__stats/test_mixin_isolation.py` (new)

Add a test that instantiates `_ReportingStatsMixin` in isolation (without the other
mixins) and asserts that calling `rar()` raises `AttributeError` with a useful
message, confirming the dependency is known and expected rather than an accident.

**Effort:** 30 min · **Closes:** silent cross-mixin failure gap

---

## Theme 6 — Protocol Design (6 → 10) ✅

### T6.1 — Trim `StatsLike` to its actual consumers ✅

> **Done** — merged [PR #755](https://github.com/Jebel-Quant/jquantstats/pull/755) · closes [#719](https://github.com/Jebel-Quant/jquantstats/issues/719)

**File:** `src/jquantstats/_reports/_protocol.py`

`StatsLike` trimmed from 66 stubs (~210 lines) to the ~12 methods `Reports` actually
calls. `@runtime_checkable` retained only where an `isinstance` check exists. Protocol
surface now matches the true consumer boundary.

**Effort:** 1 hr · **Removed:** ~150 lines

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

## Theme 7 — Dead Code (8 → 10) ✅

### T7.1 — Clarify `hhi_positive` / `hhi_negative` intent ✅

> **Done** — merged [PR #730](https://github.com/Jebel-Quant/jquantstats/pull/730) · closes [#722](https://github.com/Jebel-Quant/jquantstats/issues/722)

**File:** `src/jquantstats/_stats/_performance.py:154`

These methods are tested but never appear in `summary()`, any report template, or
any public export. Both are clarified as intentionally public optional metrics.

**Effort:** 15 min

---

## Theme 8 — Test Quality (9 → 10) ✅

### T8.1 — Refactor large test functions in `test_stats.py` ✅

> **Done** — [PR #711](https://github.com/Jebel-Quant/jquantstats/pull/711) extracted `integer_indexed_data` sub-fixture; [PR #747](https://github.com/Jebel-Quant/jquantstats/pull/747) completed the broader sub-fixture extraction ✅

**File:** `tests/test_jquantstats/test__stats/test_stats.py`

`test_stats.py` is 1 775 lines. Identify the 5–10 longest test functions and
extract their inline setup into named sub-fixtures in `conftest.py` or into a
`@pytest.fixture` scoped to the test class. Target: no test function exceeds 20
lines of setup code.

**Effort:** 1 hr · **Improves:** readability, future maintainability

---

## Theme 9 — Stats Coverage (7 → 9 → 10 vs quantstats)

### T9.1 — Implement Monte Carlo simulation suite ✅

> **Done** — merged [PR #751](https://github.com/Jebel-Quant/jquantstats/pull/751)

**File:** `src/jquantstats/_stats/_montecarlo.py` (new), exposed via `Stats`

All four Monte Carlo methods implemented as `_MonteCarloStatsMixin`:

| Method | Description |
|---|---|
| `montecarlo(n, period)` | Simulate `n` return paths over `period` observations |
| `montecarlo_sharpe(n, period, periods_per_year)` | Distribution of simulated Sharpe ratios |
| `montecarlo_drawdown(n, period)` | Distribution of simulated max drawdowns |
| `montecarlo_cagr(n, period, periods_per_year)` | Distribution of simulated CAGRs |

Block bootstrap with replacement. Returns `pl.DataFrame` (one column per asset,
one row per simulation). Mixin included in `Stats`.

**Effort:** 3–4 hr · **Closes:** biggest functional gap vs quantstats

---

### T9.2 — Close edge-case metric coverage gaps vs quantstats ✅

> **Done** — merged [PR #766](https://github.com/Jebel-Quant/jquantstats/pull/766) · closes [#763](https://github.com/Jebel-Quant/jquantstats/issues/763)

Audit quantstats `stats.py` against `jquantstats` to identify any remaining metrics present in quantstats but absent or subtly different in jquantstats. For each gap: implement the missing method or document the intentional divergence with a note in `benchmark.md`. Migration tests for new metrics added via [PR #770](https://github.com/Jebel-Quant/jquantstats/pull/770) ✅.

**Effort:** 2 hr · **Raises:** Stats coverage 9 → 10 vs quantstats

---

## Theme 10 — Plot Coverage (6 → 8 → 10 vs quantstats)

### T10.1 — Add Monte Carlo plots ✅

> **Done** — merged [PR #749](https://github.com/Jebel-Quant/jquantstats/pull/749)

**File:** `src/jquantstats/_plots/_data.py`

Two plot methods added to `DataPlots`:

- `montecarlo(n, period)` — fan chart of simulated return paths
- `montecarlo_distribution(n, period, metric)` — histogram of simulated metric
  (Sharpe, drawdown, CAGR) distribution with observed value marked

Both return `go.Figure` (Plotly), consistent with the existing plot API.

**Effort:** 2 hr · **Depends on:** T9.1

---

### T10.2 — Close remaining plot gaps vs quantstats ✅

> **Done** — merged [PR #750](https://github.com/Jebel-Quant/jquantstats/pull/750)

Audit `quantstats/_plotting/wrappers.py` against `DataPlots` / `PortfolioPlots`
and implement the missing plots. Priority order based on usage frequency in
quantstats tearsheets:

| Plot | Notes |
|---|---|
| `log_returns` (cumulative, log scale) | Already existed — `figsize` parity added |
| `compare(benchmark)` | Added — overlay of asset vs benchmark cumulative returns |
| `rolling_beta` | Already existed — `figsize` parity added |

Each plot follows the existing Plotly pattern: returns `go.Figure`, accepts
`title` and `figsize` kwargs, includes a date range selector.

**Effort:** 2–3 hr per plot

---

### T10.3 — Port remaining quantstats tearsheet plots ✅

> **Done** — merged [PR #765](https://github.com/Jebel-Quant/jquantstats/pull/765) · closes [#764](https://github.com/Jebel-Quant/jquantstats/issues/764)

Audit `quantstats/_plotting/wrappers.py` for the ~18 plots present in quantstats but not yet in `DataPlots`. Implement each following the existing Plotly pattern (`go.Figure`, `title`/`figsize` kwargs, date range selector). Priority order: plots that appear in the default `html()` tearsheet first. All plots pinned with snapshot tests.

**Effort:** 4–6 hr · **Raises:** Plot coverage 8 → 10 vs quantstats

---

## Theme 11 — Performance (8 → 9 → 10)

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

### T11.2 — Run actual performance benchmarks and identify optimisation targets ✅

> **Done** — merged [PR #768](https://github.com/Jebel-Quant/jquantstats/pull/768) · closes [#767](https://github.com/Jebel-Quant/jquantstats/issues/767)

The 9/10 performance score is inferred from Polars vs pandas library characteristics; no in-repo benchmarks verify it end-to-end. Add `pytest-benchmark` timing tests for the most computationally intensive methods (e.g. `rolling_sortino`, `sharpe` across a 50-asset frame, `summary`). If any method shows unexpected slowness, rewrite using native Polars expressions. Remaining `map_elements` / Python-loop antipatterns also eliminated.

**Effort:** 2 hr · **Raises:** Performance 9 → 10

---

## Theme 12 — Error Handling (9 → 10) ✅

### T12.1 — Carry null-handling guarantees through to computation ✅

> **Done** — merged [PR #739](https://github.com/Jebel-Quant/jquantstats/pull/739), refined in [PR #754](https://github.com/Jebel-Quant/jquantstats/pull/754)

**Files:** `_basic.py`, `_performance.py`

Redundant null guards that duplicated `Data.__post_init__` construction-time validation
removed across the stats mixins. Benchmark-aware null handling in `information_ratio`
additionally refined. Remaining `is_empty()` checks are post-filter guards on derived
frames (legitimate, not duplicates of the construction invariant).

**Effort:** 1 hr

---

## Theme 13 — Type Safety (9 → 10) ✅

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
| 1. Code duplication | ~~T1.1~~, ~~T1.2~~, ~~T1.3~~, ~~T1.4~~ | ✅ done |
| 2. API surface | ~~T2.1~~, ~~T2.2~~, ~~T2.3~~; T2.4 open | ~0.5 hr |
| 3. Abstraction | ~~T3.1~~, ~~T3.2~~, ~~T3.3~~ | ✅ done |
| 4. Null handling | ~~T4.1~~, ~~T4.2~~, ~~T4.3~~ | ✅ done |
| 5. Mixin coupling | ~~T5.1~~, ~~T5.2~~ | ✅ done |
| 6. Protocol design | ~~T6.1~~, ~~T6.2~~ | ✅ done |
| 7. Dead code | ~~T7.1~~ | ✅ done |
| 8. Test quality | ~~T8.1~~ | ✅ done |
| 9. Monte Carlo stats | ~~T9.1~~, ~~T9.2~~ | ✅ done |
| 10. Plot coverage | ~~T10.1~~, ~~T10.2~~, ~~T10.3~~ | ✅ done |
| 11. Performance | ~~T11.1~~, ~~T11.2~~ | ✅ done |
| 12. Error handling | ~~T12.1~~ | ✅ done |
| 13. Type safety | ~~T13.1~~ | ✅ done |
| **Total remaining** | **1 task** | **~0.5 hr** |

---

## Recommended Sequence

**Sprint 1 — Quick wins (~4 hr, no API changes)** ✅ complete
~~T1.1~~, ~~T1.2~~, ~~T1.3~~, ~~T2.3~~, ~~T3.3~~, ~~T4.1~~, ~~T7.1~~, ~~T13.1~~

**Sprint 2 — API clean-up (~4 hr, minor breaking changes)** ✅ complete
~~T2.1~~, ~~T2.2~~, ~~T3.1~~, ~~T3.2~~, ~~T4.2~~, ~~T5.1~~, ~~T5.2~~, ~~T8.1~~

**Sprint 3 — Architecture (~4 hr, protocol restructure)** ✅ complete
~~T6.1~~, ~~T6.2~~, ~~T12.1~~

**Sprint 4 — Feature parity (~8 hr, Monte Carlo)** ✅ complete
~~T9.1~~, ~~T10.1~~, ~~T11.1~~

**Sprint 5 — Plot parity (~4 hr)** ✅ complete
~~T10.2~~

**Sprint 6 — Path to 10.0 (~9–11 hr)** ✅ nearly complete
~~T1.4~~, T2.4, ~~T4.3~~, ~~T9.2~~, ~~T10.3~~, ~~T11.2~~

After T2.4 completes, the internal quality score reaches **10.0**. The benchmark score is already **9.1** (only the community-recognition gap remains, which is not addressable by code changes).
