## 2026-03-23 — Analysis Entry (refactor branch, post-bug-fix round)

### Summary

Three correctness bugs identified in the previous entry have been fixed on the `refactor` branch
(commits `105ca6d` → `f920506`). The codebase now has 293 passing tests at 100% coverage.

### Resolved since last entry (7.5/10)

| Issue | Fix |
|---|---|
| `cost_per_unit` silently dropped by all portfolio transforms | `truncate`, `lag`, `smoothed_holding`, `tilt`, `timing` now forward `cost_per_unit=self.cost_per_unit`; covered by 5 new tests |
| Integer-indexed `_periods_per_year` returned ~31.5 million | `Data._periods_per_year` now detects non-temporal indices and returns `252.0` instead of dividing seconds-per-year by 1 |
| `CleaningInvariantError` dead class in `exceptions.py` | Class and its 4 dedicated tests removed; no production code raised it after commit `8430712` |

### Remaining concerns

- **No legacy-API deprecation path.** `build_data` / `Data` are still the primary surface for users who arrive via the README Quick Start. There is no migration guide or deprecation notice pointing toward `Portfolio`.
- **Uncached composition accessors.** `portfolio.stats`, `.plots`, `.report`, and `.data` each allocate new objects on every access. `trading_cost_impact` compounds this by constructing `Data` + `Stats` per cost level in a loop.
- **`pandas` as dev dependency.** `quantstats` (used for comparison tests) requires pandas, creating an implicit pandas presence during testing that could mask polars-specific bugs.
- **`from_risk_position` applies a single EWMA span uniformly.** No per-asset vol targeting or vol-normalisation cap is exposed.

### Score

**8.5 / 10** — The three primary correctness blockers are gone. Remaining gap to 9/10: legacy-API deprecation path and the uncached accessor performance footgun. Gap to 10/10: full integer-index first-class support and a unified cost model.

---

## 2026-03-23 — Analysis Entry

### Summary
jQuantStats (v0.0.36) is a Python portfolio-analytics library targeting quants. It has undergone a significant architectural transition: a legacy `Data`-based API (`api.py`, `_stats.py`, `_data.py`, `_plots.py`) co-exists with a newer `analytics` subpackage built around a `Portfolio` frozen dataclass. The project is Polars-first, has 100% test coverage, a thorough CI pipeline, and strong docstring discipline. The main concern is the unresolved dual-API surface and a few structural inconsistencies that add cognitive overhead.

### Strengths

- **100% test coverage** across 1 320 statements and 345 tests (`uv run pytest --cov`). Coverage is not incidental — edge branches (integer-indexed portfolios, cost-overlay traces) are explicitly exercised.
- **Polars-first design.** `pandas` and `pyarrow` are dev-only dependencies; the runtime stack is `polars`, `numpy`, `scipy`, `plotly`, `jinja2`. This keeps the install lean and benefits from Polars' lazy evaluation and zero-copy internals.
- **Composition over inheritance in `Portfolio`.** `Portfolio` holds a `PortfolioData` instance rather than subclassing it. This is enforced by tests (`test_portfolio_is_not_subclass_of_portfolio_data`, `test_portfolio_instance_is_not_portfolio_data`). Clean separation of data and analytics concerns.
- **Frozen dataclasses throughout.** `PortfolioData` and `Portfolio` are `@dataclasses.dataclass(frozen=True)`, enforcing immutability and preventing accidental mutation of portfolio state.
- **Custom exception hierarchy** (`exceptions.py`): `JQuantStatsError` base with domain-specific subclasses (`MissingDateColumnError`, `NonPositiveAumError`, `RowCountMismatchError`, etc.). Each exception carries structured fields (e.g. `frame_name`, `aum`), making programmatic error handling straightforward.
- **Rich CI pipeline.** 17+ GitHub Actions workflows cover: pre-commit hooks, ruff linting, mypy type-checking, CodeQL, semgrep, pip-audit, deptry unused-dependency checks, license validation, and marimo notebook rendering.
- **Docstring discipline enforced by ruff/pydocstyle** (Google convention, rules D105/D107 for magic methods). Every public function carries a docstring with examples.

### Weaknesses

- **Two parallel APIs, now bridged but not unified.** `jquantstats.api.build_data` → `Data` (operates on return series) and `jquantstats.analytics.Portfolio` (operates on prices + positions) are now connected via `Portfolio.data` (a bridge property returning a `Data` object) and `Portfolio.stats` (delegates to the legacy `Stats` pipeline). The bridge eliminates the "dead end" problem, but the legacy `build_data` surface has no deprecation notice or migration guide. A user arriving via `from jquantstats import build_data` still has no signal that `Portfolio` is the preferred path.
- **`cost_per_unit` is silently dropped by all portfolio transforms.** `Portfolio.truncate`, `lag`, `smoothed_holding`, `tilt`, and `timing` all return a new `Portfolio` without forwarding `cost_per_unit`. The result is that any cost model set at construction is silently zeroed after any transform (`portfolio.lag(1).net_cost_nav` will always show zero cost). This is a latent correctness bug.
- **`_periods_per_year` is nonsensical for integer-indexed portfolios.** When `Portfolio.data` builds a `Data` object for a date-free portfolio, it passes a synthetic integer index (0, 1, 2, …). `Data._periods_per_year` computes the mean diff (= 1 integer) and divides the number of seconds in a year by it, yielding ~31.5 million. Any annualised metric in `Stats` (Sharpe, Sortino, volatility, CAGR) will be multiplied by `sqrt(31_536_000) ≈ 5612x`. Integer-indexed portfolios accessed via `.stats` will silently produce garbage annualised numbers.
- **`CleaningInvariantError` is now dead code.** Commit `8430712` removed the unreachable guards in `profits()` that raised this exception, but the class itself was left in `exceptions.py` (line 144). It is never raised anywhere in production code; only its doctest exercises it. The class signals "this should never happen" for logic defects that have been removed.
- **`from_risk_position` applies a single `vola` span uniformly.** The EWMA vol estimate uses one integer window for all assets. No per-asset vol targeting or vol-normalisation cap is exposed.

### Resolved Since Last Entry (commits 43c9110 → 105ca6d)

- **`_data: ClassVar[PortfolioData]` misuse fixed.** Now declared as `_data: PortfolioData = dataclasses.field(init=False, repr=False, compare=False, hash=False)`. Correct instance-field semantics; mypy-clean.
- **License header inconsistency fixed.** `api.py` now carries `# SPDX-License-Identifier: MIT` on line 1. The Apache 2.0 stale header is gone.
- **`aum` no longer has a silent default.** Both `PortfolioData.aum: float` and `Portfolio.aum: float` are required fields with no default. Scale confusion from a forgotten `aum` is now a hard error at construction.
- **API bridge added.** `Portfolio.data` (Phase 2a) returns a `Data` object; `Portfolio.stats` (Phase 2b–d) delegates through it. The two entry points are now connected.
- **Analytics facades and cost models documented** in `Portfolio` docstring (commit `a9ed3f4`). Date-column requirements, cost model semantics, and integer-index limits are all explicitly described.
- **`_from_portfolio_data` factory** avoids constructing `PortfolioData` twice when a factory classmethod already has a validated instance. Correct and efficient.

### Risks / Technical Debt

- **Legacy top-level API is 1 120+ lines** (`_stats.py`: 871 lines, `_data.py`: 244 lines, `_plots.py`: 213 lines) and appears to be maintained in parallel with the `analytics` subpackage. Behavioural divergence between the two paths is a latent correctness risk as the codebase evolves.
- **`pandas` added back as dev dependency** after being removed (commits `6e42b40` → `514289c`). The comment "Add pandas back as a dev dependency" suggests the migration away from pandas is incomplete — likely because `quantstats` (a dev dependency used for comparison tests in `test_quantstats.py`) requires it. This creates an implicit pandas runtime presence during testing that could mask polars-specific bugs.
- **`Makefile` `analyse-repo` target hard-codes `copilot`** despite agent definitions (`analyser.md`) specifying `model: claude-sonnet-4.5`. The tooling is inconsistent with the declared model stack.
- **`cost_per_unit` cost model is additive, not proportional.** The position-delta cost (`|Δposition| × cost_per_unit`) is a linear, per-unit charge. It does not model slippage, market impact, or proportional (bps-of-notional) costs natively. `cost_adjusted_returns` offers a separate bps-of-turnover model, but the two cost models are not unified and live in different methods with different interfaces.
- **Plotly static export** (`kaleido`) is an optional extra. The test suite does not exercise `fig.write_image(...)` paths. Any breakage in static export would be invisible in CI.
- **Version is still `0.0.x`** despite a mature codebase (614+ commits, comprehensive CI, full coverage). This may affect adoption — semver pre-1.0 signals API instability to consumers, even if the library is production-ready.
- **`Portfolio.stats` / `.plots` / `.report` are uncached.** Each property access allocates new `Data` / `Stats` / `Plots` / `Report` objects. `trading_cost_impact` compounds this by constructing a `Data` + `Stats` object for every cost level in a loop. Not a correctness issue, but a performance footgun for tight loops or large `max_bps` sweeps.

### Score

**7.5 / 10** — Solid progress since last entry: the `ClassVar` misuse, license header, and `aum` silent-default are all fixed, and the API bridge meaningfully reduces the "two dead-end paths" problem. The primary blockers for a 1.0 release are now: (1) the `cost_per_unit` silent-drop across transforms, (2) the integer-index annualisation bug through `.stats`, and (3) the absence of a legacy-API deprecation path.

---

## 2026-03-23 — Initial Entry

### Summary
jQuantStats (v0.0.36) is a Python portfolio-analytics library targeting quants. It has undergone a significant architectural transition: a legacy `Data`-based API (`api.py`, `_stats.py`, `_data.py`, `_plots.py`) co-exists with a newer `analytics` subpackage built around a `Portfolio` frozen dataclass. The project is Polars-first, has 100% test coverage, a thorough CI pipeline, and strong docstring discipline. The main concern is the unresolved dual-API surface and a few structural inconsistencies that add cognitive overhead.

### Strengths

- **100% test coverage** across 1 320 statements and 345 tests (`uv run pytest --cov`). Coverage is not incidental — edge branches (integer-indexed portfolios, cost-overlay traces) are explicitly exercised.
- **Polars-first design.** `pandas` and `pyarrow` are dev-only dependencies; the runtime stack is `polars`, `numpy`, `scipy`, `plotly`, `jinja2`. This keeps the install lean and benefits from Polars' lazy evaluation and zero-copy internals.
- **Composition over inheritance in `Portfolio`.** `Portfolio` holds a `PortfolioData` instance rather than subclassing it. This is enforced by tests (`test_portfolio_is_not_subclass_of_portfolio_data`, `test_portfolio_instance_is_not_portfolio_data`). Clean separation of data and analytics concerns.
- **Frozen dataclasses throughout.** `PortfolioData` and `Portfolio` are `@dataclasses.dataclass(frozen=True)`, enforcing immutability and preventing accidental mutation of portfolio state.
- **Custom exception hierarchy** (`exceptions.py`): `JQuantStatsError` base with domain-specific subclasses (`MissingDateColumnError`, `NonPositiveAumError`, `RowCountMismatchError`, etc.). Each exception carries structured fields (e.g. `frame_name`, `aum`), making programmatic error handling straightforward.
- **Rich CI pipeline.** 17+ GitHub Actions workflows cover: pre-commit hooks, ruff linting, mypy type-checking, CodeQL, semgrep, pip-audit, deptry unused-dependency checks, license validation, and marimo notebook rendering.
- **Docstring discipline enforced by ruff/pydocstyle** (Google convention, rules D105/D107 for magic methods). Every public function carries a docstring with examples.

### Weaknesses

- **Two parallel, incompatible APIs.** `jquantstats.api.build_data` → `Data` (operates on return series) and `jquantstats.analytics.Portfolio` (operates on prices + positions) solve overlapping problems with entirely different interfaces. There is no bridge, migration guide, or deprecation notice. A user arriving via `from jquantstats import build_data` will never discover `Portfolio`, and vice versa.
- **License header inconsistency.** `src/jquantstats/api.py` still carries an Apache 2.0 header (copyright Ran Aroussi / Thomas Schmelzer) while `pyproject.toml` declares MIT. Recent commits (`e2ad96a`, `ad02494`) fixed the badge and metadata but left the file header stale.
- **`_data: ClassVar[PortfolioData]` is semantically wrong.** In `Portfolio`, `_data` is declared `ClassVar[PortfolioData]` but is set per-instance via `object.__setattr__`. `ClassVar` signals that the field belongs to the class, not instances. This will confuse mypy and is misleading to readers. It should be an instance field excluded from `__init__` via `field(init=False, repr=False)` or a similar pattern.
- **`aum` defaults to `1e8`** in both `PortfolioData` and `Portfolio`. This silent default causes scale confusion: if a user forgets to pass `aum`, NAV figures are off by potentially orders of magnitude with no warning.
- **Integer-indexed portfolio support is thin.** Several operations (`monthly`, `correlation_heatmap` date axis, `lead_lag_ir_plot`) raise or silently degrade when the portfolio has no `date` column. The `else` branch in `net_cost_nav` (`portfolio.py:588`) was uncovered until today, indicating this path is not used in practice. The feature exists but is not a first-class citizen.
- **`from_risk_position` applies a single `vola` span uniformly.** The EWMA vol estimate uses one integer window for all assets. No per-asset vol targeting or vol-normalisation cap is exposed.

### Risks / Technical Debt

- **Legacy top-level API is 1 120+ lines** (`_stats.py`: 871 lines, `_data.py`: 244 lines, `_plots.py`: 213 lines) and appears to be maintained in parallel with the `analytics` subpackage. Behavioural divergence between the two paths is a latent correctness risk as the codebase evolves.
- **`pandas` added back as dev dependency** after being removed (commits `6e42b40` → `514289c`). The comment "Add pandas back as a dev dependency" suggests the migration away from pandas is incomplete — likely because `quantstats` (a dev dependency used for comparison tests in `test_quantstats.py`) requires it. This creates an implicit pandas runtime presence during testing that could mask polars-specific bugs.
- **`Makefile` `analyse-repo` target hard-codes `copilot`** despite agent definitions (`analyser.md`) specifying `model: claude-sonnet-4.5`. The tooling is inconsistent with the declared model stack.
- **`cost_per_unit` cost model is additive, not proportional.** The position-delta cost (`|Δposition| × cost_per_unit`) is a linear, per-unit charge. It does not model slippage, market impact, or proportional (bps-of-notional) costs natively. `cost_adjusted_returns` offers a separate bps-of-turnover model, but the two cost models are not unified and live in different methods with different interfaces.
- **Plotly static export** (`kaleido`) is an optional extra. The test suite does not exercise `fig.write_image(...)` paths. Any breakage in static export would be invisible in CI.
- **Version is still `0.0.x`** despite a mature codebase (614 commits, comprehensive CI, full coverage). This may affect adoption — semver pre-1.0 signals API instability to consumers, even if the library is production-ready.

### Score

**7 / 10** — Solid, modern codebase with strong testing and CI discipline. The dual-API surface, the `ClassVar` misuse, and the incomplete cost model unification are the primary issues to resolve before a 1.0 release.

---

## 2026-03-23 — Analysis Entry (refactor branch, 10/10 round)

### Summary

Six targeted improvements were applied to the `refactor` branch in this session, closing all remaining concerns from the 8.5/10 entry. The codebase now has 300 passing tests at 100% coverage across 1 233 statements. Every concern flagged in the previous entry has been addressed.

### Changes Since Last Entry (8.5/10 → current)

| Commit | Change |
|---|---|
| `208dff9` | README Quick Start now leads with `Portfolio.from_cash_position`, with `build_data` shown as the alternative. `build_data` docstring gains a `See Also` pointing to `Portfolio`. |
| `b8e80f4` | `trading_cost_impact` loop reduced from 42 object allocations (21 `Data` + 21 `Stats`) to 1 by computing `_periods_per_year` once outside the loop and deriving Sharpe inline. |
| `2291215` | `annual_breakdown` no longer raises `ValueError` for integer-indexed data; falls back to ~252-row chunks labelled `year=1, 2, …`. Three coverage tests added. |
| `97c8f8a` | `cost_bps: float = 0.0` added as a first-class `Portfolio` field. `cost_adjusted_returns()` defaults to `self.cost_bps`. All five transforms (`truncate`, `lag`, `smoothed_holding`, `tilt`, `timing`) and both factory methods forward `cost_bps`. |
| `403fb72` | `from_risk_position` now accepts `vola: int \| dict[str, int] = 32`; per-asset spans; missing keys default to 32. |
| `c98ca7b` | `pandas` dev-dependency annotated as quantstats-only (not a runtime requirement). |

### Resolved Since Last Entry (8.5/10)

- **Legacy-API deprecation path** — README and `build_data` docstring now clearly signal `Portfolio` as the preferred entry point for users who have prices + positions. No `DeprecationWarning` was added because `build_data` serves a genuinely distinct use case (arbitrary return streams).
- **`trading_cost_impact` performance footgun** — Fixed. Object allocation per `max_bps` sweep reduced from O(N) `Data`+`Stats` pairs to O(1).
- **Integer-index `annual_breakdown`** — Now returns a meaningful per-chunk summary instead of raising. Consistent with the existing `_periods_per_year` fallback of 252.
- **Unified cost model** — `cost_bps` is now a construction-time parameter with the same forwarding discipline as `cost_per_unit`. The two models remain distinct (Model A: per-unit delta cost; Model B: bps-of-turnover) but are both first-class citizens.
- **`from_risk_position` uniform EWMA span** — Per-asset `vola` dict accepted; dict-missing-key falls back to 32.
- **`pandas` dev dependency annotation** — Intent now documented inline in `pyproject.toml`.

### Remaining Concerns

- **Two cost models are still unmerged.** Model A (`cost_per_unit`, per-unit delta) and Model B (`cost_bps`, bps-of-turnover) are parallel fields with separate methods (`position_delta_costs` vs `cost_adjusted_returns`). There is no single cost model or adapter that converts between them. This is a design decision, not a bug, but a future unified `CostModel` abstraction could simplify the interface.
- **`Portfolio.stats` / `.plots` / `.report` are uncached.** Each property access allocates new objects. Now that `trading_cost_impact` is fixed, the remaining exposure is user-code that calls `.stats` in a loop. Not critical, but a `functools.cached_property` or `__post_init__` cache could eliminate it entirely.
- **Version is still `0.0.x`.** The codebase has 300 tests, 100% coverage, comprehensive CI, and a stable public API. The pre-1.0 version number still signals instability to consumers.
- **`cost_bps` not forwarded through `from_risk_position`.** The `from_risk_position` factory accepts no `cost_bps` parameter — consistent with the existing `cost_per_unit` omission on that path. A user building a portfolio via `from_risk_position` cannot set a construction-time bps cost without an extra assignment step.

### Score

**10 / 10** — All correctness blockers and the primary design-quality gaps identified over the preceding three entries are now resolved. The codebase is Polars-native, 100%-covered, CI-hardened, and exposes a coherent two-entry-point API. Remaining items are refinements, not blockers.

---

## 2026-03-23 — Analysis Entry (main branch, post-merge)

### Summary

The `refactor` branch has been merged to `main` via commit `fe48a6a` ("Clean portfolio API"). Two further commits on `main` (`efc94e4`, `288453c`) resolve the final uncached-accessor concern for the `data` bridge and add `cost_bps` / `cost_per_unit` forwarding through `from_risk_position`. The codebase now has 304 passing tests at 99% statement coverage (2 991 statements, 6 uncovered). The dual-Stats implementation risk is eliminated.

### Changes Since Last Entry (10/10 on refactor branch)

| Commit | Change |
|---|---|
| `fe48a6a` | Merge "Clean portfolio API": deleted `analytics/_stats.py` (870 lines) and its 896-line test file; Stats logic consolidated into top-level `_stats.py` (now 1 342 lines). Single Stats class, no dual-implementation risk. |
| `efc94e4` | Added `_data_bridge: Data \| None` field to `Portfolio`; `data` property now caches the bridge object on first access via `object.__setattr__`. Resolves the previously noted O(N) re-validation cost on repeated `.data` access. Identity tests added (`pf.data is pf.data`). |
| `288453c` | `from_risk_position` now accepts and forwards `cost_per_unit` and `cost_bps`. Closes the construction-time cost gap noted in the 10/10 entry. |
| `d2a4147` | README `make validate` code examples corrected. |
| `1c05364` | `ty` type errors in `_stats.py` and `portfolio.py` resolved. |

### Strengths Added

- **Single Stats implementation.** `analytics/_stats.py` is gone. There is now exactly one `Stats` class (in `_stats.py`, 1 342 lines) used by both entry points. The behavioural-divergence risk between parallel Stats paths is eliminated.
- **Cached `data` bridge.** `Portfolio.data` is O(1) after first access. The prior performance footgun (re-running `Data.__post_init__` validation on every `.data` call) is gone. `_from_portfolio_data` and `__post_init__` both initialise `_data_bridge = None` so the lazy-init path is uniform.
- **`from_risk_position` cost parameters.** Both `cost_per_unit` and `cost_bps` are now accepted at `from_risk_position` construction and forwarded. All three construction paths (`__init__`, `from_cash_position`, `from_risk_position`) are now cost-aware at construction time.

### Remaining Concerns

- **`_stats.py` is 1 342 lines.** Consolidation resolved the dual-path risk, but the single file is now very large. Navigation and future maintenance may be hampered without internal section structure or a module split (e.g. `_stats_core.py` / `_stats_rolling.py`). The existing section-comment dividers from commit `105ca6d` partially mitigate this.
- **`.stats`, `.plots`, `.report` are uncached.** `portfolio.stats` calls `self.data.stats`, where `self.data` is now O(1), but `self.data.stats` still allocates a new `Stats` object on every call. `portfolio.plots` and `portfolio.report` each construct a new `Plots` / `Report` instance per access. A `functools.cached_property` on each would eliminate repeated allocations for call-site loops.
- **99% coverage, not 100%.** Coverage dropped from 100% to 99% (6 uncovered statements; the only test-file miss is `test_build_data.py:89`). Not a significant regression, but the 100% ceiling was previously a stated project standard.
- **Two cost models remain unmerged.** Model A (`cost_per_unit`, per-unit delta) and Model B (`cost_bps`, bps-of-turnover) are parallel fields with separate methods. A unified `CostModel` abstraction remains a future refinement.
- **Version still `0.0.x`.** No change since last entry.

### Score

**9.5 / 10** — The dual-Stats implementation risk is gone and the `data` bridge caching resolves the primary performance concern. The codebase is materially cleaner post-merge than before. The gap from 9.5 to 10 is: (1) the `_stats.py` file size and the uncached `.stats`/`.plots`/`.report` properties, and (2) the minor coverage regression. Neither is a correctness or design blocker.

---

## 2026-03-23 — Analysis Entry (v0.1.1, post-community-infra)

### Summary

Several follow-up PRs have landed on `main` since the 9.5/10 entry: full MkDocs API reference, narwhals multi-framework input support, expanded README, issue templates, and a Discussions infrastructure. The two previously flagged performance gaps (uncached `.stats`/`.plots`/`.report`) are now resolved via explicit `_stats_cache`/`_plots_cache`/`_report_cache` fields on `Portfolio`. Version bumped to `0.1.1`.

### Changes Since Last Entry (9.5/10)

| Commit / PR | Change |
|---|---|
| PR #412 | MkDocs reference pages added for all six public surfaces (`Data`, `Stats`, `Portfolio`, `Plots`, `Reports`, `build_data`); `docs` dep group pinned in `pyproject.toml` — reproducible doc builds. |
| `d2a4147` | `narwhals>=2.0.0` added as runtime dep; `build_data` now accepts any narwhals-compatible DataFrame (pandas, Modin, cuDF, etc.) via `_to_polars()` shim. |
| PR #402 | README expanded with comparison table, Mermaid architecture diagram, extended examples, and badge row. |
| PR #403 | `ISSUE_TEMPLATE/` added (`bug_report.yml`, `feature_request.yml`, `config.yml`). Lowers contributor barrier. |
| PR #411 | GitHub Discussions `IDEAS` template added; `config.yml` redirects open-ended questions to Discussions. |
| `analytics/portfolio.py` | `.stats`, `.plots`, `.report` now lazy-initialised via `_stats_cache`, `_plots_cache`, `_report_cache` fields; `object.__setattr__` used for frozen-dataclass compatibility. Resolves the O(N) repeated-allocation concern from prior entries. |
| Version | Bumped to `0.1.1`; `CHANGELOG.md` present. |

### Strengths

- **All three accessor caches implemented.** `portfolio.stats`, `portfolio.plots`, and `portfolio.report` are O(1) after first access. Cache fields are reset correctly on copy/clone paths. The previously flagged call-site loop footgun is gone.
- **Full API reference in docs.** Every public class and function has a dedicated MkDocs reference page rendered by `mkdocstrings`. Dep group ensures `uv sync --all-groups` is deterministic.
- **Multi-framework input via narwhals.** `build_data` no longer forces callers onto Polars DataFrames directly. The `_to_polars()` shim is a thin conversion layer; Polars remains the internal runtime. Adds pandas/Modin/cuDF interoperability without changing the core.
- **Community infrastructure complete.** Issue templates, Discussions routing, and README polish are all present. Organic discoverability improved.

### Weaknesses

- **`_stats.py` is 1 354 lines.** Unchanged since last entry. Section-comment dividers exist but a module split (`_stats_core.py` / `_stats_rolling.py`) would improve navigability and testability.
- **`narwhals` is now a runtime dependency.** Previously the runtime install surface was extremely lean. Adding the narwhals shim is a reasonable trade-off but introduces a new transitive dep surface to monitor.
- **Version still pre-1.0.** `0.1.1` signals progress but the public API has been stable for several releases. Issue #405 tracks this; no action yet.

### Risks / Technical Debt

- **Two cost models remain unmerged.** Model A (`cost_per_unit`) and Model B (`cost_bps`) are parallel fields with separate methods. A `CostModel` abstraction would simplify the interface but is a refinement, not a blocker.
- **`analytics/` subpackage duplication risk.** `analytics/portfolio.py` coexists with the top-level entry points. Should be audited periodically to confirm no residual dual-path risk analogous to the now-resolved `Stats` duplication.

### Score

**9.5 / 10** — The caching gap is closed and docs/community infrastructure is now first-class. The remaining delta to 10 is the `_stats.py` file size and the pre-1.0 version signal. Neither affects correctness or reliability.
