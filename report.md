# jquantstats — Code Quality Report

> Assessed: 2026-05-16 · `main` post-PR #709 · ~8 700 source lines · 780 tests

Scores are 1–10. **10 = no actionable improvements. 1 = immediate attention required.**

---

## Scorecard

| Category | Score |
|---|:---:|
| Code duplication | 7 |
| API surface & naming | 7 |
| Abstraction & indirection | 7 |
| Null / error-handling consistency | 6 |
| Mixin architecture & coupling | 7 |
| Protocol design | 6 |
| Test quality | 9 |
| Documentation coverage | 10 |
| Dead code | 8 |
| **Overall** | **7.4** |

---

## 1. Code Duplication — 7/10

**Strengths.** `_internals.py` centralises the four core computational helpers
(`_comp_return`, `_nav_series`, `_annualization_factor`, `_downside_deviation`)
that would otherwise be copy-pasted across all four stats mixins. The
`@columnwise_stat` and `@to_frame` decorators in `_core.py` eliminate ~120 lines
of per-column iteration boilerplate.

**`_is_finite` / `_fmt` defined twice.**
`_reports/_data.py:17` and `_reports/_portfolio.py:34` each define the same two
private formatting helpers. The bodies are identical; only the return annotation
differs (`bool` vs `TypeGuard[int | float]`). Extracting both to
`_reports/_formatting.py` removes ~30 lines.

**Positive/negative filter chains repeated inline.**
`_basic.py` already has `_mean_positive_expr` / `_mean_negative_expr` helpers
(lines 41–47), but `payoff_ratio` (line 202), `profit_ratio` (line 215), and
several others re-inline `series.filter(series > 0).mean()` rather than calling
them. The pattern appears ~12 times in `_basic.py` alone. Introducing
`_positive_values` / `_negative_values` series-filter companions and using them
consistently would collapse ~30 lines of duplication.

---

## 2. API Surface & Naming — 7/10

**Strengths.** Method names match quantstats conventions, easing migration.
`@columnwise_stat` enforces `dict[str, float]` uniformly. The public proxy
properties on `Stats` (`assets`, `returns`, `benchmark`, `date_col`, `index`)
cleanly hide the internal `_data` object.

**Three alias methods add no value.**

| Alias | Canonical | Location |
|---|---|---|
| `ghpr()` | `geometric_mean()` | `_performance.py:315` |
| `r2()` | `r_squared()` | `_performance.py:724` |
| `win_loss_ratio()` | `payoff_ratio()` | `_basic.py:206` |

Each costs ~6 lines plus one entry in the `StatsLike` protocol. Keep them only if
quantstats compatibility is explicitly required; otherwise remove.

**`PortfolioUtils` is missing two methods.** ~~`DataUtils` exposes 12 methods; `PortfolioUtils` delegates only 10. The two
omissions — `winsorise` (`_data.py:168`) and `exponential_cov` (`_data.py:296`)
— mean a `Portfolio` user must reach into `.data.utils` directly, breaking the
uniform-delegation design.~~ **Fixed** — merged [PR #726](https://github.com/Jebel-Quant/jquantstats/pull/726) ✅

**`_periods_per_year` accessed as a private attribute across class boundaries.**
Four rolling methods each write `periods_per_year or self._data._periods_per_year`
rather than using the public `periods_per_year` property already exposed by
`_ReportingStatsMixin` (`_reporting.py:89`).

---

## 3. Abstraction & Indirection — 7/10

**Strengths.** Facade classes (`DataUtils`, `PortfolioUtils`, `DataPlots`,
`PortfolioPlots`, `Reports`) each have one responsibility: wrap a data or portfolio
object and expose domain methods. `@columnwise_stat` and `@to_frame` are
well-chosen abstractions that pay for themselves across 100+ methods.

**Decorator internals are implicitly coupled to `self._data`.**
`columnwise_stat` (`_core.py:73`) and `to_frame` (`_core.py:92`) reach directly
into `self._data` and `self._data.items()`. The coupling is invisible at the
decorator call site and only discovered at runtime. Documenting the required
interface — or accepting a `DataLike` argument — would make the contract explicit.

**`_nav_series` is effectively reimplemented in `prices()`.**
`_internals.py` exports `_nav_series` (`(1 + r).cumprod()`), but
`_PerformanceStatsMixin.prices` is a `@staticmethod` that does the same
computation inline without calling the helper. Both paths are live.

---

## 4. Null / Error-Handling Consistency — 6/10

**Strengths.** `Data.__post_init__` enforces a declared null strategy (`raise`,
`drop`, `forward_fill`) at construction time. Domain-specific exceptions
(`NullsInReturnsError`, `RowCountMismatchError`, etc.) pinpoint the exact failure
mode. The shared `_to_float` helper in `_core.py` handles `None → 0.0` cleanly.

**Five different null-return patterns coexist across the stats mixins.**

| Pattern | Approx. occurrences | Example |
|---|:---:|---|
| `cast(float, series.mean())` | ~20 | `_basic.py:42` |
| `float(np.nan)` | ~10 | `_basic.py:152` |
| `_to_float(...)` | ~8 | `_performance.py` |
| `fill_nan(0).fill_null(0)` inline | 2 | `_rolling.py:148` |
| `if x is None: return 0.0` | 3 | `_performance.py` |

The inconsistency means different metrics silently disagree on what to return when
the series contains no data — some return `0.0`, others `nan`, others `None`.
Standardising on `_to_float` for the scalar case and documenting the chosen
convention in `_core.py` would close this.

---

## 5. Mixin Architecture & Coupling — 7/10

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

**`rolling_sortino` is inconsistent with the other rolling methods.** ~~`rolling_sortino` (`_rolling.py:123`) is decorated with `@to_frame` and receives a
`pl.Expr`; `rolling_sharpe`, `rolling_volatility`, and `rolling_greeks` operate on
`self.all` directly and return a `pl.DataFrame`. A reader of the module sees four
methods with what appears to be the same purpose but two different implementation
shapes.~~ **Fixed** — merged [PR #723](https://github.com/Jebel-Quant/jquantstats/pull/723) ✅

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
| 1 | Extract `_is_finite` / `_fmt` to `_reports/_formatting.py` | 30 min | removes 30 lines |
| 2 | Add `winsorise` + `exponential_cov` to `PortfolioUtils` | 30 min | closes API gap |
| 3 | Use existing filter helpers consistently in `_basic.py` | 1 hr | removes 30 lines |
| 4 | Remove `ghpr`, `r2`, `win_loss_ratio` aliases | 30 min | removes 20 lines |
| 5 | Trim `StatsLike` to the ~12 methods `Reports` calls | 1 hr | removes 150 lines |
| 6 | Document + normalise null-return convention in `_core.py` | 2 hr | correctness |
| 7 | Standardise rolling methods to one implementation shape | 1 hr | readability |
| 8 | Clarify or remove `hhi_positive` / `hhi_negative` | 15 min | removes 60 lines |

Items 1–5 together take roughly 3 hours and remove or consolidate ~230 lines.
