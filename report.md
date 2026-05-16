# jquantstats — Code Quality Report

> Assessed: 2026-05-16 · `main` post-PR #727 · ~8 700 source lines · 780 tests

Scores are 1–10. **10 = no actionable improvements. 1 = immediate attention required.**

---

## Scorecard

| Category | Score |
|---|:---:|
| Code duplication | 9 |
| API surface & naming | 8 |
| Abstraction & indirection | 8 |
| Null / error-handling consistency | 8 |
| Mixin architecture & coupling | 8 |
| Protocol design | 6 |
| Test quality | 9 |
| Documentation coverage | 10 |
| Dead code | 8 |
| **Overall** | **8.2** |

---

## 1. Code Duplication — 9/10

**Strengths.** `_internals.py` centralises the four core computational helpers
(`_comp_return`, `_nav_series`, `_annualization_factor`, `_downside_deviation`)
that would otherwise be copy-pasted across all four stats mixins. The
`@columnwise_stat` and `@to_frame` decorators in `_core.py` eliminate ~120 lines
of per-column iteration boilerplate.

~~**`_is_finite` / `_fmt` defined twice.**
`_reports/_data.py:17` and `_reports/_portfolio.py:34` each define the same two
private formatting helpers. The bodies are identical; only the return annotation
differs (`bool` vs `TypeGuard[int | float]`). Extracting both to
`_reports/_formatting.py` removes ~30 lines.~~ **Fixed** — merged [PR #727](https://github.com/Jebel-Quant/jquantstats/pull/727) ✅

~~**Positive/negative filter chains repeated inline.**
`_basic.py` already has `_mean_positive_expr` / `_mean_negative_expr` helpers
(lines 41–47), but `payoff_ratio` (line 202), `profit_ratio` (line 215), and
several others re-inline `series.filter(series > 0).mean()` rather than calling
them. The pattern appears ~12 times in `_basic.py` alone. Introducing
`_positive_values` / `_negative_values` series-filter companions and using them
consistently would collapse ~30 lines of duplication.~~ **Fixed** — merged [PR #725](https://github.com/Jebel-Quant/jquantstats/pull/725) ✅

---

## 2. API Surface & Naming — 8/10

**Strengths.** Method names match quantstats conventions, easing migration.
`@columnwise_stat` enforces `dict[str, float]` uniformly. The public proxy
properties on `Stats` (`assets`, `returns`, `benchmark`, `date_col`, `index`)
cleanly hide the internal `_data` object.

**Three alias methods add no value.**

| Alias | Canonical | Location |
|---|---|---|
| `ghpr()` | `geometric_mean()` | `_performance.py:314` |
| `r2()` | `r_squared()` | `_performance.py:721` |
| `win_loss_ratio()` | `payoff_ratio()` | `_basic.py:216` |

Each costs ~6 lines plus one entry in the `StatsLike` protocol. Keep them only if
quantstats compatibility is explicitly required; otherwise remove.

~~**`PortfolioUtils` is missing two methods.**~~ **Fixed** — merged [PR #726](https://github.com/Jebel-Quant/jquantstats/pull/726) ✅

**`_periods_per_year` accessed as a private attribute across class boundaries.**
Four rolling methods each write `periods_per_year or self._data._periods_per_year`
rather than using the public `periods_per_year` property already exposed by
`_ReportingStatsMixin` (`_reporting.py:89`).

---

## 3. Abstraction & Indirection — 8/10

**Strengths.** Facade classes (`DataUtils`, `PortfolioUtils`, `DataPlots`,
`PortfolioPlots`, `Reports`) each have one responsibility: wrap a data or portfolio
object and expose domain methods. `@columnwise_stat` and `@to_frame` are
well-chosen abstractions that pay for themselves across 100+ methods.

**Decorator internals are implicitly coupled to `self._data`.**
`columnwise_stat` (`_core.py:116`) and `to_frame` (`_core.py:136`) reach directly
into `self._data` and `self._data.items()`. The coupling is invisible at the
decorator call site and only discovered at runtime. Documenting the required
interface — or accepting a `DataLike` argument — would make the contract explicit.

~~**`_nav_series` is effectively reimplemented in `prices()`.**~~ **Fixed** — `prices()` now delegates to `_nav_series` directly ✅

---

## 4. Null / Error-Handling Consistency — 8/10

**Strengths.** `Data.__post_init__` enforces a declared null strategy (`raise`,
`drop`, `forward_fill`) at construction time. Domain-specific exceptions
(`NullsInReturnsError`, `RowCountMismatchError`, etc.) pinpoint the exact failure
mode.

~~**Five different null-return patterns coexist across the stats mixins.**~~ **Fixed** — merged [PR #724](https://github.com/Jebel-Quant/jquantstats/pull/724) ✅

The convention is now documented in `_core.py` and enforced via a new `_mean`
helper: scalar metrics return `float("nan")` when the series has no non-null
observations; ratio metrics return `float("nan")` when the denominator is zero or
indeterminate. `cast(float, series.mean())` calls replaced throughout.

---

## 5. Mixin Architecture & Coupling — 8/10

**Strengths.** Splitting ~2 500 lines of stats logic into four focused mixins
(`_basic`, `_performance`, `_reporting`, `_rolling`) keeps each file manageable.
`Stats` itself is only 116 lines.

**Cross-mixin calls are invisible at the call site.**
`_ReportingStatsMixin.rar()` calls `self.cagr()` (own mixin) and `self.exposure()`
(from `_BasicStatsMixin`). This works only because `Stats` happens to inherit both,
but nothing at the call site makes the cross-mixin dependency visible. A consumer
attempting to use `_ReportingStatsMixin` in isolation would get a runtime error.

**The "performance" mixin boundary is not intuitive.**
`_PerformanceStatsMixin` spans Sharpe/Sortino ratios, drawdown metrics, Greeks,
HHI, R-squared, and Kelly criterion — three distinct conceptual domains. The split
between this mixin and `_ReportingStatsMixin` (CAGR, Calmar, recovery factor,
`summary`) is not self-evident.

~~**`rolling_sortino` is inconsistent with the other rolling methods.**~~ **Fixed** — merged [PR #723](https://github.com/Jebel-Quant/jquantstats/pull/723) ✅

---

## 6. Protocol Design — 6/10

**Strengths.** Structural protocols (`StatsLike`, `DataLike`, `PlotsLike`,
`PortfolioLike`) let consumers type-annotate without importing concrete classes,
keeping circular imports at bay. `@runtime_checkable` enables `isinstance` checks
in tests.

**`StatsLike` is 66 method stubs (~210 lines).**
`_reports/_protocol.py` lists the entire `Stats` surface. It is referenced only in
two attributes of other protocols (`DataLike.stats`, `PortfolioLike.stats`), which
in turn are used only inside `_reports`. The protocol is so broad that:

- Every new metric added to `Stats` requires a manual protocol update.
- A mock implementing the 12 methods `Reports` actually calls cannot satisfy it.
- `@runtime_checkable` checks only name existence, so the 66 stubs add no runtime
  safety beyond `hasattr`.

Replacing it with a minimal protocol scoped to what `Reports` genuinely needs would
remove ~150 lines and make the true dependency explicit.

**`DataLike` is defined independently in three sub-packages.**
`_plots/_protocol.py`, `_reports/_protocol.py`, and `_utils/_protocol.py` each
define a `DataLike` protocol with overlapping but slightly different attribute sets.
A class implementing all three must manually verify compliance against each variant.

---

## 7. Test Quality — 9/10

780 tests, 100% code and branch coverage. The suite is well-structured across
concerns:

- **Migration tests** — parametrized numeric comparison against quantstats with
  `atol=1e-6`, making quantstats the ground-truth oracle.
- **Edge-case tests** — empty series, all-NaN, single observation, zero variance.
- **Property-based tests** — `hypothesis`-driven invariant checking.
- **Snapshot tests** — 13 plot snapshots pinned with `syrupy`.
- **API contract tests** — assert public interface stability across versions.

Minor: `test_stats.py` is 1 775 lines with some 30–40-line test functions that
inline their own fixture setup. Extracting reusable sub-fixtures would reduce
duplication without adding abstraction.

---

## 8. Documentation Coverage — 10/10

`interrogate` reports 447/447 items covered against a 100% minimum threshold. All
public classes, methods, and module docstrings are present. `_internals.py`
includes worked examples that function as doctests.

---

## 9. Dead Code — 8/10

**`hhi_positive` / `hhi_negative` are tested but unused downstream.**
`_performance.py:154` and `_performance.py:186` implement the Herfindahl–Hirschman
Index for return concentration. Both have tests and docstrings, but neither appears
in `summary()`, any report template, or any downstream method, and neither is
exported via `__all__`. If they are exploratory, a comment saying so prevents
confusion; otherwise they are removal candidates.

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
| 6 | Remove `ghpr`, `r2`, `win_loss_ratio` aliases | 30 min | removes 20 lines |
| 7 | Replace `self._data._periods_per_year` with public property in rolling methods | 30 min | removes private boundary crossing |
| 8 | Document decorator contract (`self._data` requirement) in `_core.py` | 30 min | readability |
| 9 | Trim `StatsLike` to the ~12 methods `Reports` calls | 1 hr | removes 150 lines |
| 10 | Unify the three `DataLike` protocol definitions | 1 hr | removes attribute-set divergence |
| 11 | Clarify or remove `hhi_positive` / `hhi_negative` | 15 min | removes 60 lines |

Items 6–11 together take roughly 4 hours and remove or consolidate ~230 lines.
