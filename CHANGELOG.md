# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- git-cliff: generate entries below this line -->

## [Unreleased]

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
- **Documentation book** — rendered documentation available at <https://tschm.github.io/jquantstats/book>.

[Unreleased]: https://github.com/tschm/jquantstats/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/tschm/jquantstats/releases/tag/v0.1.1
