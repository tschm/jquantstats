# jquantstats — Code Quality Report

> Assessed: 2026-05-17 · `main` v0.9.0 post-PR #770 · ~10 500 source lines · ~900 tests

Scores are 1–10. **10 = no actionable improvements. 1 = immediate attention required.**

---

## Scorecard

| Category | Score |
|---|:---:|
| Code duplication | 10 |
| API surface & naming | 9 |
| Abstraction & indirection | 10 |
| Null / error-handling consistency | 10 |
| Mixin architecture & coupling | 10 |
| Protocol design | 10 |
| Test quality | 10 |
| Documentation coverage | 10 |
| Dead code | 10 |
| **Overall** | **9.9** |

---

## 1. Code Duplication — 10/10

**Strengths.** `_internals.py` centralises the four core computational helpers
(`_comp_return`, `_nav_series`, `_annualization_factor`, `_downside_deviation`)
that would otherwise be copy-pasted across all four stats mixins. The
`@columnwise_stat` and `@to_frame` decorators in `_core.py` eliminate ~120 lines
of per-column iteration boilerplate.

~~**`_is_finite` / `_fmt` defined twice.**~~ **Fixed** — merged [PR #727](https://github.com/Jebel-Quant/jquantstats/pull/727) ✅

~~**Positive/negative filter chains repeated inline.**~~ **Fixed** — merged [PR #725](https://github.com/Jebel-Quant/jquantstats/pull/725) ✅

~~**Residual duplication in reports, stats, and plot helpers.**~~ **Fixed** — merged [PR #760](https://github.com/Jebel-Quant/jquantstats/pull/760) ✅

Remaining patterns audited and consolidated. No duplicated block of 3+ lines remains without a shared helper.

---

## 2. API Surface & Naming — 9/10

**Strengths.** Method names match quantstats conventions, easing migration.
`@columnwise_stat` enforces `dict[str, float]` uniformly. The public proxy
properties on `Stats` (`assets`, `returns`, `benchmark`, `date_col`, `index`)
cleanly hide the internal `_data` object.

~~**Three alias methods add no value.**~~ **Fixed** — merged [PR #756](https://github.com/Jebel-Quant/jquantstats/pull/756) ✅

`ghpr()`, `r2()`, and `win_loss_ratio()` converted to `DeprecationWarning` shims that
delegate to their canonical counterparts (`geometric_mean`, `r_squared`, `payoff_ratio`).
Migration note added to `CHANGELOG.md`. The surface is clean for new callers; existing
callers receive clear guidance at runtime.

~~**`PortfolioUtils` is missing two methods.**~~ **Fixed** — merged [PR #726](https://github.com/Jebel-Quant/jquantstats/pull/726) ✅

~~**`_periods_per_year` accessed as a private attribute across class boundaries.**
Four rolling methods each write `periods_per_year or self._data._periods_per_year`
rather than using the public `periods_per_year` property already exposed by
`_ReportingStatsMixin` (`_reporting.py:89`).~~ **Fixed** — `1e989a5` ✅

---

## 3. Abstraction & Indirection — 10/10

**Strengths.** Facade classes (`DataUtils`, `PortfolioUtils`, `DataPlots`,
`PortfolioPlots`, `Reports`) each have one responsibility: wrap a data or portfolio
object and expose domain methods. `@columnwise_stat` and `@to_frame` are
well-chosen abstractions that pay for themselves across 100+ methods.

~~**Decorator internals are implicitly coupled to `self._data`.**
`columnwise_stat` (`_core.py:116`) and `to_frame` (`_core.py:136`) reach directly
into `self._data` and `self._data.items()`. The coupling is invisible at the
decorator call site and only discovered at runtime.~~ **Documented** — both
decorators now state the `self._data` / `self.all` preconditions explicitly in
their docstrings (`da3fd15`) ✅. Both documentation and enforcement at decoration
time are now complete — merged [PR #735](https://github.com/Jebel-Quant/jquantstats/pull/735), closes [#733](https://github.com/Jebel-Quant/jquantstats/issues/733) ✅

~~**`_nav_series` is effectively reimplemented in `prices()`.**~~ **Fixed** — `prices()` now delegates to `_nav_series` directly ✅

---

## 4. Null / Error-Handling Consistency — 10/10

**Strengths.** `Data.__post_init__` enforces a declared null strategy (`raise`,
`drop`, `forward_fill`) at construction time. Domain-specific exceptions
(`NullsInReturnsError`, `RowCountMismatchError`, etc.) pinpoint the exact failure
mode.

~~**Five different null-return patterns coexist across the stats mixins.**~~ **Fixed** — merged [PR #724](https://github.com/Jebel-Quant/jquantstats/pull/724) ✅

The convention is now documented in `_core.py` and enforced via a new `_mean`
helper: scalar metrics return `float("nan")` when the series has no non-null
observations; ratio metrics return `float("nan")` when the denominator is zero or
indeterminate. `cast(float, series.mean())` calls replaced throughout.

~~**Redundant null guards duplicating construction-time validation.**~~ **Fixed** — merged [PR #739](https://github.com/Jebel-Quant/jquantstats/pull/739), refined in [PR #754](https://github.com/Jebel-Quant/jquantstats/pull/754) ✅

Defensive `is_empty()` / `null_count()` checks that duplicated `Data.__post_init__`
guarantees removed. Benchmark-aware null handling in `information_ratio` additionally
refined. Remaining `is_empty()` checks documented as legitimate post-filter guards — merged [PR #761](https://github.com/Jebel-Quant/jquantstats/pull/761) ✅

Every guard is now either removed (redundant) or annotated with a one-line comment explaining why it survives construction-time validation.

---

## 5. Mixin Architecture & Coupling — 10/10

**Strengths.** Splitting ~2 500 lines of stats logic into four focused mixins
(`_basic`, `_performance`, `_reporting`, `_rolling`) keeps each file manageable.
`Stats` itself is only 116 lines. Cross-mixin dependencies on `_ReportingStatsMixin`
are declared centrally via `TYPE_CHECKING` stubs, each annotated with their source
mixin.

~~**The "performance" mixin boundary is not intuitive.**
`_PerformanceStatsMixin` spans Sharpe/Sortino ratios, drawdown metrics, Greeks,
HHI, R-squared, and Kelly criterion — three distinct conceptual domains. The split
between this mixin and `_ReportingStatsMixin` (CAGR, Calmar, recovery factor,
`summary`) is not self-evident.~~ **Fixed** — renamed to `_RiskStatsMixin` via
[PR #732](https://github.com/Jebel-Quant/jquantstats/pull/732) ✅

Cross-mixin method dependencies are additionally documented inline via
[PR #752](https://github.com/Jebel-Quant/jquantstats/pull/752) ✅

A cross-mixin isolation test confirming `_ReportingStatsMixin.rar()` raises `AttributeError` when called without `_BasicStatsMixin` is in place via [PR #748](https://github.com/Jebel-Quant/jquantstats/pull/748) ✅

~~**`rolling_sortino` is inconsistent with the other rolling methods.**~~ **Fixed** — merged [PR #723](https://github.com/Jebel-Quant/jquantstats/pull/723) ✅

---

## 6. Protocol Design — 10/10

**Strengths.** Structural protocols (`StatsLike`, `DataLike`, `PlotsLike`,
`PortfolioLike`) let consumers type-annotate without importing concrete classes,
keeping circular imports at bay. `@runtime_checkable` enables `isinstance` checks
in tests.

~~**`StatsLike` is 66 method stubs (~210 lines).**~~ **Fixed** — merged [PR #755](https://github.com/Jebel-Quant/jquantstats/pull/755) ✅

`StatsLike` trimmed from 66 stubs (~210 lines) to the ~12 methods `Reports` actually
calls. Every new metric no longer requires a manual protocol update. `@runtime_checkable`
retained only where an `isinstance` check genuinely exists.

~~**`DataLike` is defined independently in three sub-packages.**
`_plots/_protocol.py`, `_reports/_protocol.py`, and `_utils/_protocol.py` each
define a `DataLike` protocol with overlapping but slightly different attribute sets.
A class implementing all three must manually verify compliance against each variant.~~ **Fixed** — `DataLike` centralised at the package root, sub-package redefinitions removed via [PR #736](https://github.com/Jebel-Quant/jquantstats/pull/736) ✅

---

## 7. Test Quality — 10/10

~900 tests, 100% code and branch coverage. The suite is well-structured across
concerns:

- **Migration tests** — parametrized numeric comparison against quantstats with
  `atol=1e-6`, making quantstats the ground-truth oracle.
- **Edge-case tests** — empty series, all-NaN, single observation, zero variance.
- **Property-based tests** — `hypothesis`-driven invariant checking.
- **Snapshot tests** — 13 plot snapshots pinned with `syrupy`.
- **API contract tests** — assert public interface stability across versions.
- **Benchmark tests** — `pytest-benchmark` timing comparison for `rolling_sortino` native vs legacy.

~~Minor: `test_stats.py` is 1 775 lines with some 30–40-line test functions that inline their own fixture setup.~~ **Fixed** — reusable sub-fixtures extracted across two PRs ([#711](https://github.com/Jebel-Quant/jquantstats/pull/711), [#747](https://github.com/Jebel-Quant/jquantstats/pull/747)) ✅

---

## 8. Documentation Coverage — 10/10

`interrogate` reports 447/447 items covered against a 100% minimum threshold. All
public classes, methods, and module docstrings are present. `_internals.py`
includes worked examples that function as doctests.

---

## 9. Dead Code — 10/10

~~**`hhi_positive` / `hhi_negative` are tested but unused downstream.**
`_performance.py:154` and `_performance.py:186` implement the Herfindahl–Hirschman
Index for return concentration. Both have tests and docstrings, but neither appears
in `summary()`, any report template, or any downstream method, and neither is
exported via `__all__`. If they are exploratory, a comment saying so prevents
confusion; otherwise they are removal candidates.~~ **Clarified** — both methods
are intentionally public optional metrics, as documented via
[PR #730](https://github.com/Jebel-Quant/jquantstats/pull/730) ✅

No stale imports or unused variables were found anywhere in the source tree.

---

## Recommendations

| # | Finding | Effort | Impact |
|---|---|---|---|
| ~~1~~ | ~~Extract `_is_finite` / `_fmt` to `_reports/_formatting.py`~~ | ~~30 min~~ | ✅ done |
| ~~2~~ | ~~Add `winsorise` + `exponential_cov` to `PortfolioUtils`~~ | ~~30 min~~ | ✅ done |
| ~~3~~ | ~~Use existing filter helpers consistently in `_basic.py`~~ | ~~1 hr~~ | ✅ done |
| ~~4~~ | ~~Standardise rolling methods to one implementation shape~~ | ~~1 hr~~ | ✅ done |
| ~~5~~ | ~~Document + normalise null-return convention in `_core.py`~~ | ~~2 hr~~ | ✅ done |
| ~~6~~ | ~~Document decorator contract (`self._data` requirement) in `_core.py`~~ | ~~30 min~~ | ✅ done |
| ~~7~~ | ~~Remove `ghpr`, `r2`, `win_loss_ratio` aliases ([#718](https://github.com/Jebel-Quant/jquantstats/issues/718))~~ | ~~30 min~~ | ✅ merged [PR #756](https://github.com/Jebel-Quant/jquantstats/pull/756) |
| ~~8~~ | ~~Replace `self._data._periods_per_year` with public property in rolling methods~~ | ~~30 min~~ | ✅ done |
| ~~9~~ | ~~Rename `_PerformanceStatsMixin` to clarify scope ([#731](https://github.com/Jebel-Quant/jquantstats/issues/731))~~ | ~~45 min~~ | ✅ merged [PR #732](https://github.com/Jebel-Quant/jquantstats/pull/732) |
| ~~10~~ | ~~Enforce decorator contract at decoration time ([#733](https://github.com/Jebel-Quant/jquantstats/issues/733))~~ | ~~30 min~~ | ✅ merged [PR #735](https://github.com/Jebel-Quant/jquantstats/pull/735) |
| ~~11~~ | ~~Trim `StatsLike` to the ~12 methods `Reports` calls ([#719](https://github.com/Jebel-Quant/jquantstats/issues/719))~~ | ~~1 hr~~ | ✅ merged [PR #755](https://github.com/Jebel-Quant/jquantstats/pull/755) |
| ~~12~~ | ~~Unify the three `DataLike` protocol definitions ([#734](https://github.com/Jebel-Quant/jquantstats/issues/734))~~ | ~~1 hr~~ | ✅ merged [PR #736](https://github.com/Jebel-Quant/jquantstats/pull/736) |
| ~~13~~ | ~~Clarify or remove `hhi_positive` / `hhi_negative` ([#722](https://github.com/Jebel-Quant/jquantstats/issues/722))~~ | ~~15 min~~ | ✅ merged [PR #730](https://github.com/Jebel-Quant/jquantstats/pull/730) |
| 14 | Remove deprecated alias methods `ghpr`, `r2`, `win_loss_ratio` ([#757](https://github.com/Jebel-Quant/jquantstats/issues/757)) | 30 min | API surface 9 → 10 |
| ~~15~~ | ~~Audit and eliminate residual code duplication ([#758](https://github.com/Jebel-Quant/jquantstats/issues/758))~~ | ~~1 hr~~ | ✅ merged [PR #760](https://github.com/Jebel-Quant/jquantstats/pull/760) |
| ~~16~~ | ~~Resolve remaining `is_empty()` post-filter guards ([#759](https://github.com/Jebel-Quant/jquantstats/issues/759))~~ | ~~1 hr~~ | ✅ merged [PR #761](https://github.com/Jebel-Quant/jquantstats/pull/761) |

Item 14 remains open (Sprint 6). Completion raises the internal quality score from **9.9 → 10.0**.
