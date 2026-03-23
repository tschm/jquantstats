# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- git-cliff: generate entries below this line -->

## [Unreleased]

## [1.0.0] - 2026-03-23

### Stable API

This release establishes the **stable public API** for jQuantStats.
The following surfaces are covered by semantic versioning guarantees — breaking changes
will not be introduced without a major version bump.

#### Entry points (top-level imports)

| Symbol | Type | Description |
|--------|------|-------------|
| `jquantstats.Portfolio` | class | High-level entry point for price + position workflows |
| `jquantstats.build_data` | function | Lower-level entry point for pre-computed return series |
| `jquantstats.__version__` | str | Package version string |

#### `Portfolio` class

| Member | Kind | Description |
|--------|------|-------------|
| `Portfolio.from_cash_position(prices, cash_position, aum)` | class method | Construct from absolute position sizes |
| `Portfolio.from_risk_position(prices, risk_position, aum, vola)` | class method | Construct from risk-scaled positions |
| `.stats` | property | Returns a `Stats` object (see Stats API below) |
| `.plots` | property | Returns a portfolio-specific `Plots` object |
| `.report` | property | Returns a `Report` object |
| `.data` | property | Returns the underlying `Data` object (Entry point 2) |

#### `build_data` function

```python
build_data(returns, benchmark=None, rf=0.0, date_col="Date") -> Data
```

#### `Stats` API (available via `.stats` on both `Portfolio` and `Data`)

| Method | Description |
|--------|-------------|
| `sharpe()` | Sharpe ratio |
| `sortino()` | Sortino ratio |
| `calmar()` | Calmar ratio |
| `volatility()` | Annualised volatility |
| `cagr()` | Compound annual growth rate |
| `max_drawdown()` | Maximum drawdown |
| `avg_drawdown()` | Average drawdown |
| `max_drawdown_duration()` | Longest drawdown period |
| `value_at_risk()` | Value at Risk (VaR) |
| `conditional_value_at_risk()` | Conditional VaR (CVaR / Expected Shortfall) |
| `win_rate()` | Fraction of positive-return periods |
| `avg_win()` | Average winning-period return |
| `avg_loss()` | Average losing-period return |
| `payoff_ratio()` | Average win / average loss |
| `profit_factor()` | Gross profit / gross loss |
| `profit_ratio()` | Net profit ratio |
| `kelly_criterion()` | Kelly fraction |
| `skew()` | Return skewness |
| `kurtosis()` | Return kurtosis (excess) |
| `avg_return()` | Mean period return |
| `recovery_factor()` | Net profit / max drawdown |
| `risk_return_ratio()` | CAGR / volatility |
| `gain_to_pain_ratio()` | Sum of returns / sum of losses |
| `information_ratio()` | Active return / tracking error (requires benchmark) |
| `r_squared()` / `r2()` | R² vs benchmark (requires benchmark) |
| `greeks()` | Alpha and beta vs benchmark (requires benchmark) |
| `up_capture()` | Capture ratio in up markets (requires benchmark) |
| `down_capture()` | Capture ratio in down markets (requires benchmark) |
| `adjusted_sortino()` | Adjusted Sortino ratio |
| `rolling_sharpe()` | Rolling Sharpe series |
| `rolling_sortino()` | Rolling Sortino series |
| `rolling_volatility()` | Rolling volatility series |
| `drawdown()` | Full drawdown series |
| `prices()` | Cumulative price series from returns |
| `summary()` | Key-metric summary table |
| `annual_breakdown()` | Year-by-year performance table |
| `monthly_win_rate()` | Win rate broken down by calendar month |
| `worst_n_periods()` | Worst N return periods |

#### `Plots` API (available via `.plots` on both `Portfolio` and `Data`)

| Method | Description |
|--------|-------------|
| `plot_snapshot(title, log_scale)` | Full performance dashboard (Plotly figure) |

#### `Report` API (available via `.report` / `.reports`)

| Method | Description |
|--------|-------------|
| `metrics(periods)` | DataFrame of key metrics across time periods |

### Added

- **Stable-API declaration** — the surfaces listed above are now covered by
  semantic versioning guarantees; breaking changes require a major version bump.

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

[Unreleased]: https://github.com/tschm/jquantstats/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/tschm/jquantstats/compare/v0.1.1...v1.0.0
[0.1.1]: https://github.com/tschm/jquantstats/releases/tag/v0.1.1
