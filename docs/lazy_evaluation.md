# Lazy Evaluation — Verdict

## TL;DR

**Lazy evaluation (`LazyData`) is consistently slower than eager evaluation (`Data`)
for the workloads typical in `jquantstats`.**

For in-memory operations on a 20-year daily × 100-asset portfolio, lazy adds
9–30 % overhead.  The predicate-pushdown advantage of `scan_parquet` is
negligible when the dataset fits in RAM.

Use `LazyData` only when you cannot load the full dataset into memory, and even
then treat it as a streaming convenience rather than a speed optimisation.

---

## Benchmark

Scenario: **20 years of daily (Mon–Fri) returns, 100 assets ≈ 5 200 rows × 100
columns**.

All benchmarks were produced by running
`make benchmark` → `tests/benchmarks/test_lazy_vs_eager.py`
with `pytest-benchmark 5.2.3` on the CI runner (Azure, Linux, Python 3.12).

### Results

| Scenario | Eager | Lazy | Δ |
|---|---|---|---|
| `resample("1mo")` on full history | **8.83 ms** | 9.63 ms | +9 % |
| `truncate(5 yr) → resample("1mo")` | **4.82 ms** | 6.28 ms | +30 % |
| `scan_parquet → truncate → resample` | **14.05 ms** | 15.31 ms | +9 % |

*Lower is better.*

### Interpretation

| Finding | Why |
|---|---|
| Lazy is **slower** in every scenario | Building a `LazyFrame` query plan, materialising schema metadata (`collect_schema()`), and calling `.collect()` all add overhead that Polars' query optimiser cannot recover for datasets that already fit in memory. |
| `truncate → resample` gap is largest (+30 %) | The eager path slices a contiguous block of rows and then aggregates; the lazy path re-attaches the date column, builds a combined lazy frame, and then optimises — more bookkeeping for no gain. |
| `scan_parquet` lazy is not faster | For a ~42 MB, single-file Parquet the file I/O dominates and predicate pushdown provides no meaningful row skipping (Polars reads row groups; the filter only saves the final selection). |

---

## When `LazyData` *would* be worth it

Polars lazy evaluation does pay off in scenarios that `jquantstats` does not
currently target:

| Scenario | Why lazy helps |
|---|---|
| Dataset **does not fit in RAM** | Streaming mode (`LazyFrame.collect(streaming=True)`) processes data in batches. |
| Very **wide joins** across many tables | Polars can push projections down and avoid materialising intermediate wide frames. |
| **Chained SQL-style queries** on raw data | Multiple filter + group-by + join steps benefit from full query fusion. |

For a typical quant analytics workflow — where the data is already in memory as
a clean returns matrix — eager Polars is the right default, and `jquantstats`
should stay eager-first.

---

## Conclusion

The `LazyData` class remains in the library as a **convenience API** for users
who want to:

* scan large Parquet / CSV files without loading them completely into memory
  (`LazyData.scan_parquet`, `LazyData.scan_csv`), then chain a single
  `.collect()` to get an eager `Data` object.
* defer computation until a final `.collect()` call as a style preference.

It is **not** recommended as a performance optimisation for in-memory
workloads.  The public API is:

```python
from jquantstats import LazyData

# Good use-case: file too large to load fully
result = (
    LazyData.scan_parquet("decades_of_ticks.parquet")
    .truncate(start=date(2020, 1, 1))
    .collect()          # ← single I/O pass
)

# Unnecessary overhead: data already in memory
result = data.lazy().resample("1mo").collect()   # ← no benefit vs eager
# Just do this instead:
result = data.resample("1mo")
```
