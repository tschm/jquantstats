# Why Polars? — Performance Benchmarks

jQuantStats is built on [Polars](https://pola.rs/) instead of pandas — the same
foundation used by the original [QuantStats](https://github.com/ranaroussi/quantstats).
This page quantifies what that means in practice.

---

## Methodology

| Parameter | Value |
|---|---|
| Dataset | 2520 synthetic daily returns (≈ 10 trading years) |
| Seed | `numpy.random.default_rng(42)` — fully reproducible |
| Repeats | 10 calls per operation (mean reported) |
| Timing | `time.perf_counter` wall-clock |
| Memory | `tracemalloc` peak traced allocation |

Three representative operations are compared:

| # | Operation | jQuantStats API | QuantStats API |
|---|---|---|---|
| 1 | Sharpe ratio | `data.stats.sharpe()` | `qs.stats.sharpe(series)` |
| 2 | Max drawdown | `data.stats.max_drawdown()` | `qs.stats.max_drawdown(series)` |
| 3 | Metrics report | `data.reports.metrics()` | `qs.reports.html(series, output=…)` |

> **Note on report comparison**
> `data.reports.metrics()` returns a Polars DataFrame of all key metrics.
> `qs.reports.html()` generates a full HTML file with matplotlib / seaborn charts.
> The operations are not identical, but both represent the primary "full summary"
> output for each library's returns-only workflow.

---

## Results

Numbers below were produced by running `uv run python benchmarks/run.py`
against a fresh clone (Data object built once before timing loop).

| Operation | jqs (ms) | qs (ms) | × faster | jqs (KiB) | qs (KiB) | × less memory |
|---|---:|---:|---:|---:|---:|---:|
| Sharpe ratio | **0.5** | 1.0 | **2.2 ×** | **11** | 87 | **8 ×** |
| Max drawdown | **0.3** | 3.3 | **9.7 ×** | **5** | 182 | **36 ×** |
| Metrics report | **2.6** | 7204 | **2782 ×** | **18** | 20179 | **1121 ×** |

*ms = mean wall-clock time per call · KiB = peak traced memory per call*

---

## Interpretation

### Sharpe ratio

jQuantStats is **2.2× faster** on raw computation and uses **8× less memory**
than the pandas-backed QuantStats path.  Both libraries complete the calculation
in under 1 ms once the input data is loaded.

### Max drawdown

Polars' efficient `cum_max` / `shift` operations on the cumulative-return
series give jQuantStats a **9.7× speed advantage** while allocating
**36× less memory** — relevant when analysing many assets or running
rolling windows.

### Metrics report

This is where the gap is most dramatic.  QuantStats' `reports.html()` renders
a full matplotlib / seaborn figure suite, serialises them to a temporary PNG
buffer, and embeds them in HTML — all backed by pandas.  jQuantStats'
`reports.metrics()` returns a lightweight Polars DataFrame of the same
numerical metrics in **2.6 ms vs 7204 ms** — nearly **2800× faster** — and
uses **1000× less memory**.

---

## Reproduce Locally

```bash
# 1. Clone and install dev dependencies
git clone https://github.com/tschm/jquantstats
cd jquantstats
make install

# 2. Run the standalone comparison script
uv run python benchmarks/run.py

# 3. (Optional) Run the full pytest-benchmark suite
make benchmark
```

The benchmark scripts live in two places:

| Path | Purpose |
|---|---|
| `benchmarks/run.py` | Standalone script — human-readable table |
| `tests/benchmarks/test_vs_quantstats.py` | pytest-benchmark — used by `make benchmark` and CI |

---

## CI

Benchmarks run automatically on every push to `main` via the
[`rhiza_benchmark` workflow](https://github.com/tschm/jquantstats/actions/workflows/rhiza_benchmark.yml).
Results are stored as workflow artifacts so you can track trends over time.
