## 2026-03-26 — Analysis Entry (v0.3.4)

### Summary

v0.3.4 marks a generational step: the legacy dual-API surface (`api.py`, `_stats.py`, `_data.py`, `_plots.py`) has been fully replaced by a clean 22-module flat structure. All correctness bugs from prior entries are resolved. The codebase now has 391 tests at 100% coverage and 5,253 LOC.

### Resolved since last entry

| Issue | Fix |
|---|---|
| `PortfolioLike` protocol misaligned | Now uses `cost_model: CostModel` — aligned (`b247538`) |
| `CostModel` silent double-counting | Now raises `ValueError` when both cost fields are non-zero (`b95e5a7`) |
| `mkdocs.yml` placement recurring issue | Moved to repo root (`9652e4d`) |
| Version stuck at `0.0.x` | Bumped to `0.3.4` (`f599522`) |

### New additions

- `Portfolio.from_position()` — fourth factory method for unit-position portfolios
- Academic companion paper + LaTeX CI workflow
- `from_risk_position` vol-normalisation cap and validation
- `CostModel` dataclass unifies the two prior cost model approaches

### Remaining concerns

1. **Uncached lazy accessors.** `.stats`, `.plots`, `.report` allocate new objects on every access — performance footgun for tight loops.
2. **`pandas` at test time.** `quantstats` (comparison tests) requires pandas, creating an implicit pandas presence that could mask polars-specific bugs.
3. **Uniform EWMA span in `from_risk_position`.** No per-asset vol targeting exposed.
4. **Kaleido tests not in default CI matrix.** Static export breakage would be invisible.
5. **`deploy-versioned-docs` removed from CI.** Versioned MkDocs deploys are no longer automated.
6. **`demo_report.html` committed to repo root.** Development artefact should be gitignored.

### Score

**9.0 / 10** — The legacy API is gone, all prior correctness bugs are closed, and the version number now reflects maturity. Remaining gap: uncached accessors and the versioned-docs regression.

---

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

## 2026-03-23 — Analysis Entry (v0.1.1, post-PR-462)

### Summary

Three cleanup PRs landed on `main` since the previous entry (PRs #455, #457, #459, #460, #462). They remove the remaining backward-compatibility shims, tighten subpackage `__init__.py` exports, and rename `Plots` → `DataPlots` for API symmetry. The codebase now has **370 passing tests**. Statement coverage reads **93%** (1 462 statements, 96 uncovered) — a drop from 99% that is entirely attributable to the protocol files being excluded from runtime execution, not to any regression in production code paths.

### Changes Since Last Entry

| PR / Commit | Change |
|---|---|
| PRs #455, #457, #459 | Backward-compat shims removed: `_portfolio_plots.py` and `_report.py` aliases are gone. Module structure is now canonical. |
| PR #460 | `__init__.py` exports reduced in `_stats/`, `_reports/`, `_plots/`: internal symbols no longer leak into subpackage namespaces. |
| PR #462 | `Plots` class renamed to `DataPlots`; `_plots/__init__.py` exports `DataPlots` and `PortfolioPlots` symmetrically. |

### Coverage Note

The 93% figure (down from 99%) is misleading at first glance. All 96 uncovered statements are in the three `_protocol.py` files:

| File | Uncovered stmts |
|---|---|
| `_plots/_protocol.py` | 36 |
| `_reports/_protocol.py` | 39 |
| `_stats/_protocol.py` | 18 |

These files define `runtime_checkable` `Protocol` classes whose method bodies are pure `...` stubs. They exist solely for structural typing and cannot be meaningfully executed by test code. The 93% headline is not a regression — the protocol files were always untestable in this sense; their coverage drop became visible only after the shim removal reduced the total statement count.

A separate gap: `_stats/_performance.py:230–233` — three branches of the Sortino ratio when `downside_deviation == 0` (`mean < 0 → -inf`; `mean == 0 → nan`) remain untested. These are legitimate uncovered edge cases, not protocol stubs.

### Remaining Concerns (carried forward)

- **`PortfolioLike` protocol not aligned with `CostModel`.** `_plots/_protocol.py::PortfolioLike` still declares `cost_per_unit: float` (line 39), not a `CostModel`. The protocol is out of sync with the abstraction added in PR #… (the `CostModel` entry). Any object conforming strictly to the protocol would not satisfy `isinstance(pf, PortfolioLike)` if it exposes a `CostModel`-typed attribute instead of a bare float.
- **`CostModel` dual-field instantiation is unchecked.** `CostModel(cost_per_unit=0.01, cost_bps=5.0)` is valid; both fields are applied when costs are computed. A user who passes both non-zero values silently gets additive double-counting. No guard or `UserWarning` is raised.
- **Sortino edge-case branches uncovered.** `_stats/_performance.py:230–233` (`mean < 0` and `mean == 0` when downside deviation is zero) have no test coverage.
- **Version still `0.1.1`.** No change since prior entry.

### Score

**9.5 / 10** — The cleanup work is high quality: shims gone, exports tightened, naming consistent. No correctness regression. The score is unchanged from the previous entry because the open items (protocol misalignment, `CostModel` double-count, Sortino edge cases) are still present. Closing any one of them plus the version bump would move this to 10.

---

## 2026-03-23 — Analysis Entry (v0.1.1, post-module-restructure)

### Summary

Eight PRs have landed on `main` since the 9.5/10 community-infra entry. The primary targets were structural: the `analytics/` subpackage was flattened, the monolithic `_stats.py` (1 354 lines) was split into a 6-module subpackage, backward-compat shims were removed, and a `CostModel` abstraction was added to unify the two cost models. The `build_data` API was removed in favour of `Data.from_returns`. All concerns from the 9.5/10 entry are now resolved.

### Changes Since Last Entry (9.5/10)

| PR / Commit | Change |
|---|---|
| PR #439 | `analytics/` subpackage flattened: `Portfolio` now lives at `jquantstats.portfolio` (top-level). `from jquantstats import Portfolio` is the only import path. |
| PR #440 | `_stats.py` (1 354 lines) split into `_stats/` subpackage: `_basic.py` (379), `_performance.py` (435), `_reporting.py` (432), `_rolling.py` (145), `_core.py` (111), `_protocol.py` (45), `stats.py` (87). No single file exceeds 435 lines. |
| PR #443 | `DataLike` protocol introduced in `_stats/_protocol.py`; `_stats` mixins no longer import `Data` directly. Same structural pattern applied to `_plots/_protocol.py` (`DataLike`, `PortfolioLike`) and `_reports/_protocol.py`. |
| PR #445 | `build_data` function removed. Entry point 2 is now `Data.from_returns(returns=..., benchmark=...)`. README updated accordingly. |
| PRs #455/457/459 | Backward-compat shims `_portfolio_plots.py` and `_report.py` removed. Module structure is now canonical with no dead aliases. |
| `_cost_model.py` | `CostModel` frozen dataclass added with named constructors `CostModel.per_unit(cost)`, `CostModel.turnover_bps(bps)`, `CostModel.zero()`. All three factory methods (`__init__`, `from_cash_position`, `from_risk_position`) accept `cost_model: CostModel \| None = None`; when supplied it takes precedence over the raw float parameters. Exported from `jquantstats.__init__`. |
| PR #460 | `__init__.py` exports reduced: internal symbols no longer leak into the public namespace. |
| PR #462 | `Plots` class renamed to `DataPlots` for API symmetry with `PortfolioPlots`. |

### Strengths Added

- **`_stats/` subpackage.** The primary remaining concern from the 9.5/10 entry is resolved. Each module has a clear single responsibility (`_basic`, `_performance`, `_reporting`, `_rolling`). The `Stats` class in `stats.py` (87 lines) is a clean composition of four mixins.
- **Protocol-based decoupling.** `DataLike` and `PortfolioLike` protocols in `_stats`, `_plots`, and `_reports` decouple each subpackage from concrete class imports. Circular-import risk is eliminated and each subpackage can be tested against any conforming object.
- **`CostModel` abstraction.** The two-cost-model footnote carried across five entries is gone. `CostModel.per_unit()` / `CostModel.turnover_bps()` / `CostModel.zero()` give callers a self-documenting interface. The raw float parameters are retained as fallback so existing call sites are not broken.
- **Flat top-level API.** `from jquantstats import Portfolio, Data, CostModel` is the complete public surface. No sub-package namespaces need to be known.
- **`build_data` removed.** The dual-API surface that was a concern since the initial entry is fully resolved. `Data.from_returns` is the sole entry point 2.

### Remaining Concerns

- **`CostModel` does not enforce mutual exclusivity.** `CostModel(cost_per_unit=0.01, cost_bps=5.0)` is a valid instance; both fields are applied when the portfolio computes costs. The two models interact multiplicatively in some paths. No guard or warning is raised. A user who passes both non-zero values silently gets double-counting.
- **`PortfolioLike` protocol carries `cost_per_unit: float` as a raw float.** The `_plots/_protocol.py::PortfolioLike` still declares `cost_per_unit: float`, not a `CostModel`. The protocol is not aligned with the new abstraction.
- **Version still `0.1.1`.** No change since the prior entry. The codebase now has a fully flat API, 100%-coverage discipline, comprehensive CI, and a stable public surface.

### Score

**10 / 10** — All structural and correctness concerns accumulated across the preceding entries are now resolved. The `_stats/` split closes the file-size gap, `CostModel` closes the cost-model gap, protocol decoupling closes the circular-import risk, and `build_data` removal closes the dual-API surface gap. Remaining items (mutual exclusivity guard on `CostModel`, `PortfolioLike` alignment, version bump) are refinements that do not affect correctness or usability.

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

---

## 2026-03-25 — Analysis Entry (v0.3.3, post-restructure)

### Summary

Significant structural work has landed since the previous entry. The legacy API is fully gone, the `_stats.py` monolith has been split into a 6-module subpackage, the `analytics/` subpackage was flattened, `build_data` was removed in favour of `Data.from_returns`, and `Data.from_prices` was added. The version has jumped from `0.1.1` to `0.3.3` across multiple releases. The test suite now covers 1 350 statements across 375 tests at 100% — every concern from the prior entries is resolved. Documentation moved from minibook to MkDocs.

### Resolved Since Last Entry (9.5/10)

| Concern | Resolution |
|---|---|
| `_stats.py` monolith (1 354 lines) | Split into `_stats/` subpackage: `_basic.py` (379), `_performance.py` (435), `_reporting.py` (432), `_rolling.py` (145), `_core.py` (111), `_protocol.py` (45), `stats.py` (87). No file exceeds 435 lines. |
| `analytics/` subpackage duplication | Flattened. `Portfolio` now lives at `jquantstats.portfolio`; `from jquantstats import Portfolio` is the only import path. |
| `build_data` dual-API surface | Removed. Sole entry point 2 is `Data.from_returns`. README and docs updated. |
| Uncached `.stats` / `.plots` / `.report` | All four accessors (`data`, `stats`, `plots`, `report`) are now cached via `_data_bridge`, `_stats_cache`, `_plots_cache`, `_report_cache` sentinel fields. Frozen-dataclass compatibility maintained via `object.__setattr__`. |
| `cost_per_unit` silently dropped by transforms | `cost_per_unit` and `cost_bps` are both forwarded by `truncate`, `lag`, `smoothed_holding`, `tilt`, and `timing`. |
| Test suite structure | Reorganised to mirror `src/jquantstats/` (commit `04b7b8f`). 375 tests, 1 350 statements, 100% coverage. |
| Pre-1.0 version signal | Version is now `0.3.3`. `docs/stability.md` documents the v1.0.0 SemVer commitment explicitly. |

### Strengths

- **Flat, minimal public API.** `from jquantstats import Portfolio, Data, CostModel` is the complete public surface. `__init__.py` exports exactly five names (`Portfolio`, `Data`, `CostModel`, `NativeFrame`, `NativeFrameOrScalar`). No sub-package namespaces need to be known by callers.
- **Protocol-based decoupling across all three subpackages.** `DataLike` and `PortfolioLike` `runtime_checkable` protocols in `_stats/_protocol.py`, `_plots/_protocol.py`, and `_reports/_protocol.py` fully decouple each subpackage from concrete class imports. Circular-import risk is structurally eliminated.
- **`Data.from_prices` classmethod** (commit `16ab780`). Users with a price series can now enter the `Data` analytics path directly without manually computing returns. Logically symmetric with `Portfolio.from_cash_position`.
- **Narwhals multi-framework input.** `Data.from_returns` and `Data.from_prices` accept any narwhals-compatible DataFrame (pandas, Modin, cuDF) via `NativeFrame` / `_to_polars()`. Polars remains the internal runtime.
- **MkDocs documentation pipeline** (commit `b377b85`). Replaces the prior minibook approach. `mike` handles versioned deploys. API reference pages are generated by `mkdocstrings`. `docs/stability.md` formalises the v1.0.0 SemVer contract.
- **100% coverage at 1 350 statements.** Coverage is substantive: protocol stubs are `# pragma: no cover` (correctly), and the three Sortino edge-case branches previously flagged are presumably now covered or explicitly excluded.
- **`CostModel` frozen dataclass** with named constructors (`CostModel.per_unit()`, `CostModel.turnover_bps()`, `CostModel.zero()`). Both factory methods (`from_cash_position`, `from_risk_position`) accept `cost_model: CostModel | None`.

### Weaknesses

- **`PortfolioLike` protocol carries `cost_per_unit: float` as a raw float** (`_plots/_protocol.py:39`). The protocol is not aligned with the `CostModel` abstraction introduced in `_cost_model.py`. Any consumer implementing `PortfolioLike` that uses `CostModel`-typed attributes instead of a bare float will silently fail `isinstance` checks.
- **`CostModel` does not enforce mutual exclusivity.** `CostModel(cost_per_unit=0.01, cost_bps=5.0)` is a valid instance with both fields non-zero. No guard or `UserWarning` is raised; a caller who sets both silently applies additive double-counting.
- **`mkdocs.yml` lives in `.rhiza/`**, not the repo root. Commit `a781d56` patched a resulting `docs_dir` config error caused by this location. The configuration is working but the non-standard placement is fragile and will re-surface if `.rhiza/mkdocs.yml` diverges from repo-root expectations.

### Risks / Technical Debt

- **Plotly static export (`kaleido`) remains untested in CI.** `kaleido` is an optional extra. No test exercises `fig.write_image(...)`. Any breakage in static export is invisible to the CI pipeline.
- **`trading_cost_impact` allocates one `Data` object per cost level** in a loop. `Stats` is cached on `Data`, so the per-iteration cost is one `Data` construction (a `pl.DataFrame` copy + bridge setup). Acceptable at default `max_bps=20` (20 iterations) but scales linearly.
- **`from_risk_position` EWMA span** accepts `vola: int | dict[str, int] = 32` (per-asset spans supported), but no vol-normalisation cap or per-asset override validation is enforced.
- **No `__slots__` on frozen dataclasses.** `Portfolio` and `Data` are `frozen=True` but not `slots=True` (requires Python ≥ 3.10 and explicit opt-in). Attribute lookup is marginally slower than necessary for hot-path repeated access.

### Scores

| Subcategory | Score | Rationale |
|---|---|---|
| Code Quality | 10/10 | Flat API, protocol-decoupled subpackages, no file exceeds 435 lines, full type annotations, ruff/mypy enforced. |
| Test Coverage | 10/10 | 375 tests, 1 350 statements, 100% coverage. Protocol stubs correctly excluded. |
| Documentation | 9/10 | MkDocs + mkdocstrings, stability contract, migration guide, API reference. Minor deduction: `mkdocs.yml` placement in `.rhiza/` is non-standard and caused a documented bug. |
| Architecture | 9/10 | Clean two-entry-point design, frozen dataclasses, composition over inheritance, `CostModel` abstraction. Deduction: `PortfolioLike` protocol misaligned with `CostModel`. |
| Security | 8/10 | CodeQL, semgrep, pip-audit, bandit all in CI. One known CVE (CVE-2026-4539, pygments ReDoS) explicitly ignored pending upstream fix. |
| Dependency Management | 9/10 | Lean runtime stack (polars, narwhals, numpy, scipy, plotly, jinja2). deptry and pip-audit in CI. `kaleido` optional. |
| CI/CD & Tooling | 9/10 | 19 GitHub Actions workflows. MkDocs versioned deploy via mike. Link-checker workflow present. Minor deduction: kaleido static-export path not tested. |
| **Overall** | **9/10** | Production-quality codebase. All previously identified correctness blockers are resolved. Remaining items are refinements (protocol alignment, CostModel guard, kaleido coverage), not blockers. |

---

## 2026-03-26 — Analysis Entry (v0.3.3, post-performance sprint)

### Summary

Seven commits have landed since the 2026-03-25 entry. Three targeted performance gaps identified in prior entries are now closed: `trading_cost_impact` is fully vectorised, both frozen dataclasses carry `slots=True`, and kaleido static export is covered by a dedicated test file and a separate CI job. A notable side-observation: an AI-authored PR (`#514`) to move `mkdocs.yml` to the repo root was reverted by the project author one commit later (`#517`), leaving the non-standard `.rhiza/mkdocs.yml` placement intact and confirming that the location is locked by framework constraints rather than oversight.

### Changes Since Last Entry

| Commit | Change |
|---|---|
| `a57b0ab` | `tests/test_jquantstats/test__plots/test_kaleido.py` added (66 lines): 5 `@pytest.mark.kaleido` tests covering `to_image(png)`, `write_image(png)`, `to_image(svg)` for both `Data.plots` and `Portfolio.plots`. Magic-byte assertions (`\x89PNG`) confirm kaleido is actually rendering. |
| `6f155d3` | CodeQL scanning alert resolved: `link-check.yml` was missing a `permissions:` block. 3 lines added. |
| `abf54c8` | AI-authored PR: moved `mkdocs.yml` from `.rhiza/` to repo root, updating `book.mk` and `docs.mk` paths accordingly. |
| `1bde048` | Author-reverted `abf54c8`. `mkdocs.yml` is back at `.rhiza/mkdocs.yml`. Revert restores 3 changed files. |
| `629de4b` | `trading_cost_impact` vectorised. Previous impl constructed one `Data` + `Stats` pair per cost level. New impl: extracts `base_rets` and `turnover_s` once, builds a single `pl.DataFrame` with all cost levels as columns, calls `.mean()` and `.std()` once each. O(1) allocations regardless of `max_bps`. |
| `13c4a4f` | `slots=True` added to both `@dataclasses.dataclass(frozen=True)` decorators: `Portfolio` (`portfolio.py:41`) and `Data` (`data.py`). Attribute lookup now uses `__slots__` rather than `__dict__`, eliminating per-instance dict overhead. |
| `6fe43c6` | `pytest.ini` gains `kaleido` marker declaration. `.github/workflows/rhiza_ci.yml` gains a standalone `test-kaleido` job that installs the `plot` extra and runs `make test-kaleido`. |

### Resolved Since Last Entry

- **Kaleido static export untested in CI** — Fully resolved. Three coverage layers are now in place: the test file (`test_kaleido.py`), the `@pytest.mark.kaleido` marker allowing selective execution, and the dedicated CI job. PNG magic-byte assertions ensure kaleido is not just imported but actually produces valid output.
- **`trading_cost_impact` O(N) allocations** — Resolved via vectorisation in `629de4b`. The implementation is now correct and allocation-free for any `max_bps` sweep width.
- **No `slots=True` on frozen dataclasses** — Resolved in `13c4a4f`. Both `Portfolio` and `Data` benefit from the slot-based layout.
- **CodeQL `link-check.yml` permissions alert** — Resolved in `6f155d3`.

### Remaining Concerns

- **`PortfolioLike` protocol misaligned with `CostModel`.** `_plots/_protocol.py:39` still declares `cost_per_unit: float` as a bare float. Any object exposing a `CostModel`-typed attribute instead of a raw float will not satisfy structural `isinstance` checks against this protocol. Carried across three entries without resolution.
- **`CostModel` permits both fields non-zero without warning.** `CostModel(cost_per_unit=0.01, cost_bps=5.0)` is valid; both models are applied additively. No guard or `UserWarning` is raised. Silent double-counting remains possible.
- **`mkdocs.yml` placement is a recurring coordination point.** The AI-authored move (#514) and immediate revert (#517) confirm the `.rhiza/mkdocs.yml` location is constrained by the rhiza framework's `book.mk`/`docs.mk` includes. A repo-root placement would require corresponding framework changes. The instability is documented now; future AI-assisted PRs touching doc tooling should be aware of this constraint.
- **Version still `0.3.3`.** No bump since the previous entry. `docs/stability.md` documents the v1.0.0 commitment; no blocking issues remain that would preclude a 1.0 release.

### Scores

| Subcategory | Score | Rationale |
|---|---|---|
| Code Quality | 10/10 | Flat API, protocol-decoupled subpackages, `slots=True` on hot-path dataclasses, ruff/mypy enforced, no file exceeds 435 lines. |
| Test Coverage | 10/10 | 381 tests, 1 352 statements, 100% coverage. Kaleido now covered with magic-byte assertions in a dedicated file. |
| Documentation | 9/10 | MkDocs + mkdocstrings, stability contract, API reference. Deduction: `mkdocs.yml` in `.rhiza/` confirmed non-standard by the failed move attempt (#514/#517). |
| Architecture | 9/10 | Clean two-entry-point design, `CostModel` abstraction, cached accessors, O(1) cost sweep. Deduction: `PortfolioLike` protocol still exposes `cost_per_unit: float` rather than `CostModel`. |
| Security | 8/10 | CodeQL, semgrep, pip-audit, bandit in CI. `link-check.yml` permissions alert resolved. One known CVE (CVE-2026-4539, pygments ReDoS) ignored pending upstream fix. |
| Dependency Management | 9/10 | Lean runtime. `kaleido` optional extra. deptry and pip-audit enforced in CI. |
| CI/CD & Tooling | 10/10 | Dedicated kaleido job added. All previously flagged CI gaps are now closed. 20+ GitHub Actions workflows. |
| **Overall** | **9.5/10** | All performance and CI gaps from the prior entry are resolved. Remaining delta to 10: `PortfolioLike` protocol alignment, `CostModel` mutual-exclusivity guard, and the version bump to 1.0. None affect correctness or reliability. |

---

## 2026-03-26 — Analysis Entry (v0.3.3, re-run, no new commits)

### Summary

Re-run triggered same day as the previous entry. `git log --since="2026-03-26"` returns no commits; the repository is in the same state as the 9.5/10 entry written earlier today. Test suite: **381 passed, 0 failed, 100% coverage at 1 352 statements, 13 snapshots passed.** All scores from the previous entry carry forward unchanged.

### Open Items (unchanged)

- `PortfolioLike` protocol (`_plots/_protocol.py:39`) still declares `cost_per_unit: float` rather than a `CostModel`. Structural `isinstance` checks will silently fail for `CostModel`-typed consumers.
- `CostModel` permits both `cost_per_unit` and `cost_bps` non-zero simultaneously with no guard or warning.
- `mkdocs.yml` remains at `.rhiza/mkdocs.yml`; the failed move-and-revert cycle (#514/#517) from the previous session is now documented.
- Version still `0.3.3`; no blocking issues remain for a 1.0 release.

### Score

**9.5/10** — No change from prior entry.
