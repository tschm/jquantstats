# Repository Analysis: jQuantStats

_Last updated: 2026-03-28_

## 1. Project Purpose

**jQuantStats** is a modern Python library for portfolio analytics designed for quantitative finance professionals and portfolio managers. It provides comprehensive performance metrics, risk analysis, and interactive visualizations for portfolio evaluation.

Key differentiators from the original QuantStats library:
- **Polars-native design**: Zero pandas runtime dependency; pure Polars for all data processing
- **Modern visualizations**: Interactive Plotly charts instead of static matplotlib/seaborn
- **Full type annotations**: Complete type hints with `py.typed` marker
- **Comprehensive test coverage**: pytest-based test suite with extensive validation
- **Clean, layered API**: Two entry points for different use cases (Portfolio vs Data)

Current version: **0.4.0** (Production/Stable)

---

## 2. Directory Structure

```
jquantstats/
├── src/jquantstats/              # Main package source code
│   ├── __init__.py               # Public API exports
│   ├── data.py                   # Data container & manipulation
│   ├── portfolio.py              # Portfolio analytics
│   ├── _cost_model.py            # Cost modeling for transactions
│   ├── _types.py                 # Type aliases (narwhals compatibility)
│   ├── exceptions.py             # Domain-specific exceptions
│   ├── _stats/                   # Statistical analysis subpackage
│   │   ├── stats.py              # Main Stats class (4 mixins)
│   │   ├── _basic.py             # Basic stats (volatility, skew, VaR, Kelly)
│   │   ├── _performance.py       # Performance metrics (Sharpe, Sortino, alpha, beta)
│   │   ├── _reporting.py         # Temporal reporting (annual breakdown, Calmar)
│   │   ├── _rolling.py           # Rolling window metrics
│   │   ├── _core.py              # Helper functions & decorators
│   │   └── _protocol.py          # Protocol definitions
│   ├── _plots/                   # Visualization subpackage
│   │   ├── _data.py              # DataPlots - snapshot, heatmap, distribution
│   │   ├── _portfolio.py         # PortfolioPlots - NAV overlays, correlation heatmaps
│   │   └── _protocol.py          # Plot protocols
│   ├── _reports/                 # Report generation subpackage
│   │   ├── _data.py              # Reports for Data objects
│   │   ├── _portfolio.py         # Reports for Portfolio objects
│   │   └── _protocol.py          # Report protocols
│   ├── templates/                # Jinja2 HTML templates for reports
│   └── py.typed                  # PEP 561 marker for type checking
│
├── tests/test_jquantstats/       # Comprehensive pytest suite
│   ├── conftest.py               # Pytest fixtures
│   ├── test_data.py
│   ├── test_data_from_returns.py
│   ├── test_data_from_prices.py
│   ├── test_portfolio.py
│   ├── test_api_contract.py
│   ├── test_edge_cases.py
│   ├── test_properties.py
│   ├── test_version.py
│   ├── test__stats/
│   ├── test__plots/
│   ├── test__reports/
│   ├── test_migration/           # Comparison tests against QuantStats
│   └── resources/                # Test data (meta.csv, portfolio.csv, benchmark.csv)
│
├── docs/                         # MkDocs documentation
├── book/marimo/                  # Interactive Marimo notebooks
├── .github/workflows/            # GitHub Actions CI/CD (Rhiza-managed)
├── .rhiza/                       # Rhiza template management system
├── pyproject.toml
├── pytest.ini
├── ruff.toml
├── Makefile
├── README.md
├── CHANGELOG.md
└── ROADMAP.md
```

---

## 3. Dependencies

### Runtime
| Package | Version | Purpose |
|---------|---------|---------|
| `polars` | >=1.18.0 | Primary dataframe engine |
| `narwhals` | >=2.0.0 | Multi-dataframe abstraction layer |
| `numpy` | >=2.0.0 | Numerical computing |
| `plotly` | >=6.0.0 | Interactive visualizations |
| `scipy` | >=1.14.1 | Statistics and optimization |
| `jinja2` | >=3.1.0 | HTML template rendering |

### Optional
| Package | Version | Purpose |
|---------|---------|---------|
| `kaleido` | ==1.2.0 | Static image export (PNG/SVG) |

### Development
| Package | Purpose |
|---------|---------|
| `pandas` | Comparison testing only (NOT runtime) |
| `quantstats` | Reference implementation for metric validation |
| `marimo` | Interactive notebooks/dashboards |
| `yfinance` | Test data fixtures |
| `pyarrow` | Parquet support, Arrow interop |

**Python**: 3.11, 3.12, 3.13 supported. Primary dev: 3.13.

---

## 4. Public API

### Top-Level Exports (`jquantstats`)
```python
from jquantstats import Data, Portfolio, CostModel, NativeFrame
```

### Data API
```python
# Factory methods
Data.from_returns(returns, rf=0.0, benchmark=None, date_col="Date") → Data
Data.from_prices(prices, rf=0.0, benchmark=None, date_col="Date") → Data

# Key properties
.returns: pl.DataFrame
.benchmark: pl.DataFrame | None
.stats: Stats
.plots: DataPlots
.reports: Reports
.assets: list[str]

# Methods
.resample(every="1mo") → Data
.truncate(start=None, end=None) → Data
.describe() → pl.DataFrame
.items() → Iterator[tuple[str, pl.Series]]
```

### Portfolio API
```python
# Factory method
Portfolio.from_cash_position(prices, cashposition, aum, ...) → Portfolio

# Derived series (lazy computed)
.profits: pl.DataFrame          # Per-asset daily P&L
.nav_accumulated: pl.DataFrame  # Cumulative additive NAV
.nav_compounded: pl.DataFrame   # Compounded NAV
.returns: pl.DataFrame          # Daily returns
.drawdown: pl.DataFrame         # Drawdown from HWM
.highwater: pl.DataFrame        # Running high-water mark

# Analytics facades
.stats: Stats
.plots: PortfolioPlots
.report: Report
.data: Data                     # Bridge to Data API

# Attribution & costs
.tilt, .timing, .tilt_timing_decomp
.turnover, .turnover_weekly, .turnover_summary()
.cost_adjusted_returns(cost_bps) → pl.DataFrame
```

### Stats API (50+ metrics)
```python
# Basic
.mean(), .std(), .skew(), .kurtosis(), .volatility()

# Risk
.max_drawdown(), .value_at_risk(), .conditional_value_at_risk()
.win_rate(), .profit_factor(), .payoff_ratio(), .kelly_criterion()

# Performance ratios
.sharpe(), .sortino(), .calmar(), .information_ratio()
.recovery_factor(), .max_drawdown_duration()

# Factor analysis (requires benchmark)
.greeks()         # → {asset: {alpha, beta, r2}}
.capture_ratio()

# Temporal
.annual_breakdown(), .monthly_breakdown(), .summary()

# Rolling
.rolling_sharpe(window=60), .rolling_sortino(), .rolling_volatility()
```

### CostModel API
```python
CostModel.per_unit(cost: float) → CostModel      # Per-share cost
CostModel.turnover_bps(bps: float) → CostModel   # BPS of AUM turnover
CostModel.zero() → CostModel                     # No transaction costs
```

---

## 5. Architecture Decisions

### Layered Entry Points
Two APIs for different use cases:
- **Portfolio API**: price + position data → compiles NAV → full suite
- **Data API**: returns series from data vendors → optional benchmark → lighter-weight
- Bridge: `portfolio.data` drops down to Data API

### Frozen Dataclasses + Lazy Caching
All main classes are frozen (`Data`, `Portfolio`, `Stats`, `CostModel`, etc.) for immutability and thread safety. `.stats`, `.plots`, `.report` properties lazily instantiate and cache via `object.__setattr__()`.

### Mixin-Based Stats
`Stats` composes 4 mixins: `_BasicStatsMixin`, `_PerformanceStatsMixin`, `_ReportingStatsMixin`, `_RollingStatsMixin`. A `@columnwise_stat` decorator applies each metric to all assets, returning a `dict[str, float]`.

### Narwhals Compatibility
Type alias `NativeFrame = nw_typing.IntoDataFrame` allows input from pandas, dask, cudf. All inputs are normalized to Polars internally via `nw.from_native().to_polars()`.

### Dual Cost Models
- **Model A**: `cost_per_unit` — per-share cost (equity portfolios)
- **Model B**: `cost_bps` — basis points of AUM turnover (macro/fund-of-funds)
- Validation enforces only one model active at a time.

### Exception Hierarchy
All exceptions inherit `JQuantStatsError`. Domain-specific: `MissingDateColumnError`, `InvalidCashPositionTypeError`, `NonPositiveAumError`, `RowCountMismatchError`, etc.

---

## 6. Testing

### Structure
| Category | Files | Notes |
|----------|-------|-------|
| Core functionality | `test_data.py`, `test_portfolio.py`, `test_api_contract.py`, `test_edge_cases.py` | Factory methods, validation, API stability |
| Statistics | `test__stats/test_stats.py` | ~50+ metric tests |
| Visualization | `test__plots/` | Plot generation, snapshot regression |
| Reports | `test__reports/test_reports.py` | HTML report generation |
| Migration | `test_migration/` | Direct comparison against QuantStats reference |
| Property-based | `test_properties.py` | Hypothesis-based tests |

### Fixtures (`conftest.py`)
- `returns` — Meta stock returns (from `resources/meta.csv`)
- `benchmark_frame` — SPY benchmark
- `portfolio` — Multi-asset portfolio (AAPL, META)
- `data` — Composed `Data` object with benchmark

### Markers
- `@pytest.mark.stress` — performance/stress tests
- `@pytest.mark.property` — property-based tests
- `@pytest.mark.kaleido` — requires kaleido optional dep

---

## 7. CI/CD

All workflows are Rhiza-managed (synced from `jebel-quant/rhiza` template).

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `rhiza_ci.yml` | push/PR | Multi-OS (ubuntu/macos/windows) × Python (3.11–3.13) test matrix |
| `rhiza_release.yml` | tag `v*` | Build + publish to PyPI, draft GitHub Release |
| `rhiza_validate.yml` | push/PR | Type checking, formatting, linting, 100% docstring coverage |
| `rhiza_security.yml` | push, scheduled | pip-audit, bandit, Semgrep, license compliance |
| `rhiza_codeql.yml` | push, scheduled | GitHub CodeQL semantic analysis |
| `rhiza_book.yml` | push | MkDocs site build + GitHub Pages deploy |

### Local Development
```bash
make install       # Install via uv
make test          # Run full test suite
make validate      # format + lint + typecheck
make security      # Security scanning
make docs          # Generate documentation
```

Dependency management: `uv` with `uv.lock`.

---

## 8. Key Files Reference

| File | Purpose |
|------|---------|
| `src/jquantstats/__init__.py` | Public API, version metadata |
| `src/jquantstats/data.py` | `Data` dataclass — returns container |
| `src/jquantstats/portfolio.py` | `Portfolio` dataclass — full analytics |
| `src/jquantstats/_cost_model.py` | `CostModel` — transaction costs |
| `src/jquantstats/_types.py` | `NativeFrame` type alias |
| `src/jquantstats/exceptions.py` | Exception hierarchy |
| `src/jquantstats/_stats/stats.py` | `Stats` class (4 mixins) |
| `src/jquantstats/_stats/_basic.py` | Volatility, VaR, Kelly, skew |
| `src/jquantstats/_stats/_performance.py` | Sharpe, Sortino, alpha/beta |
| `src/jquantstats/_stats/_reporting.py` | Annual/monthly breakdown |
| `src/jquantstats/_stats/_rolling.py` | Rolling window metrics |
| `src/jquantstats/_plots/_data.py` | `DataPlots` — snapshot, heatmap, drawdown |
| `src/jquantstats/_plots/_portfolio.py` | `PortfolioPlots` — NAV, correlation |
| `src/jquantstats/_reports/_portfolio.py` | HTML report generation |
| `tests/test_jquantstats/conftest.py` | Shared pytest fixtures |
| `pyproject.toml` | Build config, dependencies, tool settings |
| `pytest.ini` | Test configuration |
| `ruff.toml` | Linter/formatter configuration |
