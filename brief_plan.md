# Brief Plan

| # | Issue | Effort | Impact | Status |
|---|---|---|---|---|
| 1 | [Extract `_is_finite` / `_fmt` to `_reports/_formatting.py`](https://github.com/Jebel-Quant/jquantstats/issues/715) | 30 min | removes 30 lines | ✅ merged [#727](https://github.com/Jebel-Quant/jquantstats/pull/727) |
| 2 | [Add `winsorise` + `exponential_cov` to `PortfolioUtils`](https://github.com/Jebel-Quant/jquantstats/issues/716) | 30 min | closes API gap | ✅ merged [#726](https://github.com/Jebel-Quant/jquantstats/pull/726) |
| 3 | [Use existing filter helpers consistently in `_basic.py`](https://github.com/Jebel-Quant/jquantstats/issues/717) | 1 hr | removes 30 lines | ✅ merged [#725](https://github.com/Jebel-Quant/jquantstats/pull/725) |
| 4 | [Standardise rolling methods to one implementation shape](https://github.com/Jebel-Quant/jquantstats/issues/721) | 1 hr | readability | ✅ merged [#723](https://github.com/Jebel-Quant/jquantstats/pull/723) |
| 5 | [Document + normalise null-return convention in `_core.py`](https://github.com/Jebel-Quant/jquantstats/issues/720) | 2 hr | correctness | ✅ merged [#724](https://github.com/Jebel-Quant/jquantstats/pull/724) |
| 6 | Document decorator contract (`self._data` / `self.all` requirement) | 30 min | readability | ✅ done `da3fd15` |
| 7 | [Remove `ghpr`, `r2`, `win_loss_ratio` aliases](https://github.com/Jebel-Quant/jquantstats/issues/718) | 30 min | removes 20 lines | ✅ merged [#756](https://github.com/Jebel-Quant/jquantstats/pull/756) |
| 8 | Replace `self._data._periods_per_year` with public property | 30 min | removes private boundary crossing | ✅ done `1e989a5` |
| 9 | [Rename `_PerformanceStatsMixin` to clarify scope](https://github.com/Jebel-Quant/jquantstats/issues/731) | 45 min | readability | ✅ merged [#732](https://github.com/Jebel-Quant/jquantstats/pull/732) |
| 10 | [Enforce decorator contract at decoration time](https://github.com/Jebel-Quant/jquantstats/issues/733) | 30 min | fail-fast on misuse | ✅ merged [#735](https://github.com/Jebel-Quant/jquantstats/pull/735), closes [#733](https://github.com/Jebel-Quant/jquantstats/issues/733) |
| 11 | [Trim `StatsLike` to the ~12 methods `Reports` calls](https://github.com/Jebel-Quant/jquantstats/issues/719) | 1 hr | removes 150 lines | ✅ merged [#755](https://github.com/Jebel-Quant/jquantstats/pull/755) |
| 12 | [Consolidate three `DataLike` protocol definitions](https://github.com/Jebel-Quant/jquantstats/issues/734) | 1 hr | removes attribute-set divergence | ✅ merged [#736](https://github.com/Jebel-Quant/jquantstats/pull/736) |
| 13 | [Clarify or remove `hhi_positive` / `hhi_negative`](https://github.com/Jebel-Quant/jquantstats/issues/722) | 15 min | removes 60 lines | ✅ merged [#730](https://github.com/Jebel-Quant/jquantstats/pull/730) |
| 14 | [Remove deprecated alias methods `ghpr`, `r2`, `win_loss_ratio`](https://github.com/Jebel-Quant/jquantstats/issues/757) | 30 min | API surface 9 → 10 | not started |
| 15 | [Audit and eliminate residual code duplication](https://github.com/Jebel-Quant/jquantstats/issues/758) | 1 hr | code duplication 9 → 10 | ✅ merged [#760](https://github.com/Jebel-Quant/jquantstats/pull/760) |
| 16 | [Resolve remaining `is_empty()` post-filter guards in stats mixins](https://github.com/Jebel-Quant/jquantstats/issues/759) | 1 hr | null handling 9 → 10 | ✅ merged [#761](https://github.com/Jebel-Quant/jquantstats/pull/761) |
| 17 | [Close edge-case metric coverage gaps vs quantstats](https://github.com/Jebel-Quant/jquantstats/issues/763) | 2 hr | stats coverage 9 → 10 | ✅ merged [#766](https://github.com/Jebel-Quant/jquantstats/pull/766) |
| 18 | [Port remaining quantstats tearsheet plots to DataPlots](https://github.com/Jebel-Quant/jquantstats/issues/764) | 4–6 hr | plot coverage 8 → 10 | ✅ merged [#765](https://github.com/Jebel-Quant/jquantstats/pull/765) |
| 19 | [Add end-to-end performance benchmarks and close optimisation gaps](https://github.com/Jebel-Quant/jquantstats/issues/767) | 2 hr | performance 9 → 10 | ✅ merged [#768](https://github.com/Jebel-Quant/jquantstats/pull/768) |
