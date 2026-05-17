---
icon: material/chart-bar
---

# Plot coverage benchmark

This page tracks parity between the **jquantstats** `DataPlots` API and the
reference implementation in
[quantstats `_plotting/wrappers.py`](https://github.com/ranaroussi/quantstats).

**Plot coverage score: 10 / 10** ✅

---

## DataPlots method coverage

All plots that appear in the quantstats default HTML tearsheet (`qs.reports.html()`)
are implemented in `DataPlots`.

| quantstats | jquantstats | In default tearsheet |
|---|---|:---:|
| `qs.plots.snapshot(r)` | `data.plots.snapshot()` | ✅ |
| `qs.plots.returns(r)` | `data.plots.returns()` | ✅ |
| `qs.plots.log_returns(r)` | `data.plots.log_returns()` | ✅ |
| `qs.plots.daily_returns(r)` | `data.plots.daily_returns()` | ✅ |
| `qs.plots.yearly_returns(r)` | `data.plots.yearly_returns()` | ✅ |
| `qs.plots.monthly_returns(r)` | `data.plots.monthly_returns()` | ✅ |
| `qs.plots.monthly_heatmap(r)` | `data.plots.monthly_heatmap()` | ✅ |
| `qs.plots.histogram(r)` | `data.plots.histogram()` | ✅ |
| `qs.plots.distribution(r)` | `data.plots.distribution()` | ✅ |
| `qs.plots.drawdown(r)` | `data.plots.drawdown()` | ✅ |
| `qs.plots.drawdowns_periods(r)` | `data.plots.drawdowns_periods()` | ✅ |
| `qs.plots.earnings(r)` | `data.plots.earnings()` | — |
| `qs.plots.rolling_sharpe(r)` | `data.plots.rolling_sharpe()` | ✅ |
| `qs.plots.rolling_sortino(r)` | `data.plots.rolling_sortino()` | ✅ |
| `qs.plots.rolling_volatility(r)` | `data.plots.rolling_volatility()` | ✅ |
| `qs.plots.rolling_beta(r, benchmark=b)` | `data.plots.rolling_beta()` | ✅ |
| `qs.plots.montecarlo(r)` | `data.plots.montecarlo()` | — |
| `qs.plots.montecarlo_distribution(r)` | `data.plots.montecarlo_distribution()` | — |
| — | `data.plots.compare()` | — |

## Improvements over quantstats

- All plots return interactive **Plotly figures** instead of static matplotlib figures.
- Every method accepts `title` and (where applicable) `figsize` keyword arguments.
- Charts include a **date range selector** for quick zoom to 6m / 1y / 3y / YTD / All.
- `data.plots.compare()` — cumulative return comparison vs benchmark — has no
  direct quantstats counterpart.

## Snapshot test coverage

Every `DataPlots` method listed above is guarded by at least one
[syrupy](https://github.com/syrupy-project/syrupy) snapshot test in
`tests/test_jquantstats/test__plots/test_plot_snapshots.py`.  The snapshot
captures the structural fingerprint of the figure (trace types, names, and key
layout properties) without storing raw data arrays, so tests remain stable
across data changes while still catching structural regressions.
