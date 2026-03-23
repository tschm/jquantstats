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
