# Roadmap

## v1.0.0 (stable API)
- [ ] Stable public API surface documented in `docs/stability.md`
- [ ] Coverage threshold enforced at 90%
- [ ] Release notes automated via git-cliff

## Future
- Autocorrelation / ACF (#19)

## Completed explorations
- **LazyData / Polars lazy path**: benchmarked; lazy adds 9–30 % overhead for
  in-memory workloads. `LazyData` kept as a file-scanning convenience
  (`scan_parquet`, `scan_csv`) but not recommended as a speed optimisation.
  See `docs/lazy_evaluation.md` for the full verdict.
