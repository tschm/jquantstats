# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- git-cliff: generate entries below this line -->

## [Unreleased]

## [0.5.0] - 2026-03-29

### Added

- `drawdown_details`, `expected_return`, `rolling_greeks`, and related stats (#583)
- `comp`, `compsum`, `ghpr` stats (#582)
- `outliers`, `remove_outliers`, `outlier_win_ratio`, `outlier_loss_ratio` stats
- `geometric_mean` and risk stats (value-at-risk, conditional-VaR, etc.) (#581)
- `autocorrelation()` and `acf()` in Stats (#544)
- `omega_ratio` in Stats (#554)
- Quantstats parity benchmarks (#579)

### Fixed

- GitHub org rename: updated all remaining `tschm` → `jebel-quant` references

### Changed

- `Reports.metrics()` decomposed into focused helper methods (#586)
- CI workflows consolidated: `rhiza_validate`, `rhiza_quality` now cover security, semgrep, typecheck, pip-audit, deptry, pre-commit, and link-check in unified jobs
- Copilot agent hooks and setup workflow removed (sync out #587)
- `demo_report.html` and `REPOSITORY_ANALYSIS.md` removed from repository

## [0.4.0] - 2026-03-26

### Added

- `Portfolio.from_position()` — fourth factory method for unit-position portfolios
- 1/n equal-weight portfolio SVG charts from real AAPL/META data

### Fixed

- `generate_svgs` path scope and coverage badge URL
- `CostModel` now raises `ValueError` when constructed with both cost fields non-zero

### Changed

- Figure 1 in companion paper: facades column reordered to `.stats` > `.report` > `.plots`

### Removed

- `generate_svgs.py` removed from repository
- Autopilot CI workflow removed

## [0.3.4] - 2026-03-26

### Added

- Vol-normalisation cap and input validation in `from_risk_position` (#521)
- Kaleido marker and dedicated CI job for static image export (#520)
- Kaleido static image export tests (#504)
- Per-subcategory 1–10 scores in analyser agent
- MkDocs-based book pipeline replacing minibook (#502)
- Companion paper and LaTeX CI workflow

### Fixed

- `PortfolioLike` protocol aligned with `CostModel` abstraction (#522)
- API Reference TOC depth limited to 3 to reduce clutter
- `mkdocs.yml` moved to repo root (#514, #526)
- Workflow missing permissions (code scanning alert #1) (#503)

### Performance

- `slots=True` added to `Portfolio` and `Data` frozen dataclasses (#515)
- `trading_cost_impact` vectorised — O(1) allocations for the full cost sweep (#516)

## [0.3.3] - 2026-03-24

### Added

- `Data.from_prices` classmethod (#499)

### Fixed

- `mkdocs` `site_dir` nested inside `docs_dir`

### Changed

- Tests reorganised to mirror `src/jquantstats/` structure (#500)

## [0.3.2] - 2026-03-24

### Added

- Link-checking GitHub Actions workflow for README (#497)

### Fixed

- `mike` deploy failing due to `mkdocs.yml` not found at repo root

## [0.3.1] - 2026-03-24

### Changed

- README: absolute URLs for references, static Python version badge, removed stale badges
- Dependency updates (python-dependencies group, 3 packages)

## [0.3.0] - 2026-03-24

### Added

- `__repr__` with date range on `Data` and `Portfolio` (#489)
- Versioned documentation with `mike` (#480)
- macOS and Windows added to CI test matrix (#482)
- Dashboard screenshot, quick-start output, and Marimo badge in README (#481)
- PyPI classifiers and explicit `__all__` (#491)

### Changed

- `PortfolioData` collapsed into `Portfolio` (#473)
- Test coverage raised to 100%

## [0.2.0] - 2026-03-23

### Added

- `Data.describe()` method and tests
- `Data.from_returns` classmethod; `build_data` retained as alias
- `DataLike` protocol; upward `Data` imports removed from `_stats` mixins
- `NativeFrame` type alias replacing `Any` for narwhals inputs
- `interrogate` enforcing 100% docstring coverage
- `ROADMAP.md` and `CITATIONS.bib` / Citing section in README
- Narwhals support to accept pandas/polars/modin inputs in `build_data`
- `Data.truncate` method with tests
- Property-based tests with `hypothesis`; `ZeroDivisionError` in `sortino` fixed
- conda-forge recipe and README installation docs
- `CHANGELOG.md`, `cliff.toml`, and `make changelog` target
- Release notes automated with `git-cliff` in release workflow
- API contract test (`test_api_contract.py`) to guard public API surface
- QuantStats migration guide (`docs/migration.md`)
- `docs/stability.md` and updated mkdocs nav
- GitHub Discussions templates; updated contributing guide
- `ISSUE_TEMPLATE` config, `PULL_REQUEST_TEMPLATE.md`
- Full API reference pages in mkdocs nav
- `py.typed` marker; `.stats`, `.plots`, `.report` cached on `Portfolio`
- ANN2 return-type annotation enforcement in ruff

### Changed

- `Plots` renamed to `DataPlots` for symmetry with `PortfolioPlots` (#462)
- `_stats*.py` files grouped into `_stats/` subpackage
- `_report.py` and `_portfolio_plots.py` backward-compat shims removed
- `__init__.py` exports reduced in `_stats`, `_reports`, `_plots` subpackages
- Analytics subpackage flattened; notebooks updated for marimo 0.20.4
- Mypy configuration removed from `pyproject.toml`

### Fixed

- README code examples corrected to pass `make validate`
- `book/` excluded from `interrogate` pre-commit hook

## [0.1.1] - 2026-03-23

### Added

- **Core library** — polars-native portfolio analytics with zero pandas runtime dependency.
- **`Portfolio` class** — high-level entry point for price + position workflows; compiles NAV curves and exposes `.stats`, `.plots`, and `.report` attributes.
- **`build_data` function** — lower-level entry point for working directly with return series and optional benchmarks.
- **`Stats` API** — performance metrics including Sharpe ratio, Sortino ratio, Calmar ratio, volatility, CAGR, max drawdown, Value at Risk (VaR), Conditional VaR, win rate, and more.
- **`Plots` API** — interactive Plotly visualisations: performance dashboard (snapshot), drawdown chart, return distribution, and monthly return heatmap.
- **`Report` API** — Jinja2-powered HTML report generation combining metrics and charts.
- **Benchmark comparison** — align portfolio returns against a benchmark series and compute relative metrics.
- **Risk-free rate support** — adjustable daily risk-free rate parameter for excess-return calculations.
- **PEP 561 compliance** — `py.typed` marker included for downstream type checker support.
- **GitHub Release Notes** — `.github/release.yml` configured to categorise auto-generated release notes by label (breaking changes, features, bug fixes, documentation, maintenance).
- **Full CI/CD pipeline** via Rhiza templates: tests with coverage, type checking, pre-commit hooks, dependency auditing, CodeQL and Semgrep security scanning, PyPI publishing, and documentation building.
- **Interactive notebooks** — Marimo notebooks for exploratory analysis.
- **Documentation book** — rendered documentation available at <https://jebel-quant.github.io/jquantstats/book>.

[Unreleased]: https://github.com/jebel-quant/jquantstats/compare/v0.3.4...HEAD
[0.3.4]: https://github.com/jebel-quant/jquantstats/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/jebel-quant/jquantstats/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/jebel-quant/jquantstats/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/jebel-quant/jquantstats/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/jebel-quant/jquantstats/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/jebel-quant/jquantstats/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/jebel-quant/jquantstats/releases/tag/v0.1.1
