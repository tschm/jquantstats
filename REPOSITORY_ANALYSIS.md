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
