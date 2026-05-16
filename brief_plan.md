# Brief Plan

| # | Issue | Effort | Impact | Status |
|---|---|---|---|---|
| 1 | [Extract `_is_finite` / `_fmt` to `_reports/_formatting.py`](https://github.com/Jebel-Quant/jquantstats/issues/715) | 30 min | removes 30 lines | ✅ merged [#727](https://github.com/Jebel-Quant/jquantstats/pull/727) |
| 2 | [Add `winsorise` + `exponential_cov` to `PortfolioUtils`](https://github.com/Jebel-Quant/jquantstats/issues/716) | 30 min | closes API gap | ✅ merged [#726](https://github.com/Jebel-Quant/jquantstats/pull/726) |
| 3 | [Use existing filter helpers consistently in `_basic.py`](https://github.com/Jebel-Quant/jquantstats/issues/717) | 1 hr | removes 30 lines | ✅ merged [#725](https://github.com/Jebel-Quant/jquantstats/pull/725) |
| 4 | [Standardise rolling methods to one implementation shape](https://github.com/Jebel-Quant/jquantstats/issues/721) | 1 hr | readability | ✅ merged [#723](https://github.com/Jebel-Quant/jquantstats/pull/723) |
| 5 | [Document + normalise null-return convention in `_core.py`](https://github.com/Jebel-Quant/jquantstats/issues/720) | 2 hr | correctness | ✅ merged [#724](https://github.com/Jebel-Quant/jquantstats/pull/724) |
| 6 | Document decorator contract (`self._data` / `self.all` requirement) | 30 min | readability | ✅ done `da3fd15` |
| 7 | [Remove `ghpr`, `r2`, `win_loss_ratio` aliases](https://github.com/Jebel-Quant/jquantstats/issues/718) | 30 min | removes 20 lines | not started |
| 8 | Replace `self._data._periods_per_year` with public property | 30 min | removes private boundary crossing | ✅ done `1e989a5` |
| 9 | [Rename `_PerformanceStatsMixin` to clarify scope](https://github.com/Jebel-Quant/jquantstats/issues/731) | 45 min | readability | not started |
| 10 | [Enforce decorator contract at decoration time](https://github.com/Jebel-Quant/jquantstats/issues/733) | 30 min | fail-fast on misuse | not started |
| 11 | [Trim `StatsLike` to the ~12 methods `Reports` calls](https://github.com/Jebel-Quant/jquantstats/issues/719) | 1 hr | removes 150 lines | not started |
| 12 | [Consolidate three `DataLike` protocol definitions](https://github.com/Jebel-Quant/jquantstats/issues/734) | 1 hr | removes attribute-set divergence | not started |
| 13 | [Clarify or remove `hhi_positive` / `hhi_negative`](https://github.com/Jebel-Quant/jquantstats/issues/722) | 15 min | removes 60 lines | not started |
