---
icon: lucide/shield-check
---

# API Stability

This document defines the public API surface of **jquantstats** and the
stability guarantees that apply from **v1.0.0** onwards.

## Stable public exports

The following names are exported from the top-level `jquantstats` package and
are covered by the stability guarantee described below.

| Name | Kind | Imported from |
|------|------|---------------|
| `Portfolio` | class | `jquantstats.portfolio` |
| `Data` | class | `jquantstats.data` |
| `Stats` | class | `jquantstats._stats` |
| `Plots` | class | `jquantstats._plots` |
| `NativeFrame` | type alias | `jquantstats._types` |
| `NativeFrameOrScalar` | type alias | `jquantstats._types` |

All of the above are importable directly from `jquantstats`:

```python
from jquantstats import Portfolio, Data
```

`Data`, `Stats`, and `Plots` instances are returned by the public API
(e.g. `Data.from_returns()` returns a `Data`; `data.stats` is a `Stats`;
`data.plots` is a `Plots`).  Their public methods and attributes are
stable even though the classes live in private modules.

## What "stable" means

From **v1.0.0** onwards jquantstats follows [Semantic Versioning](https://semver.org/):

- **Patch** (`1.x.y → 1.x.z`): bug fixes only; no API changes.
- **Minor** (`1.x → 1.y`): backwards-compatible additions (new methods,
  new keyword arguments with defaults, new exports).  Existing code
  continues to work unchanged.
- **Major** (`1.x → 2.0`): breaking changes are permitted.  A breaking
  change is any removal, rename, or signature change of a stable export
  or its public methods/attributes.

## What is *not* stable

Anything that is **not** in the table above is considered internal and
may change or be removed in any release:

- Private modules: `_stats.py`, `_plots.py`, `_reports.py`,
  `_types.py`, `_portfolio_data.py`.
- Private classes, functions, or attributes whose names begin with an
  underscore (e.g. `Data._raw_returns`, `Stats._df`).
- Sub-module paths such as `jquantstats.portfolio`
  — import from the top-level package instead.

```python
# ✅ stable
from jquantstats import Portfolio, Data

# ❌ not stable — internal path, may change
from jquantstats.portfolio import Portfolio
```

## Deprecation policy

When a stable export needs to be changed or removed:

1. The old name / signature is kept for **one minor version** with a
   `DeprecationWarning` that names the replacement.
2. The breaking change is then made in the **next minor release** (or in
   the next major release if the migration is larger).

Example timeline:

| Release | Action |
|---------|--------|
| `1.3.0` | `old_name` deprecated; `DeprecationWarning` raised on use; `new_name` available |
| `1.4.0` | `old_name` removed |

## Pre-release versions

Releases tagged `0.x.y` carry **no stability guarantee**.  The API may
change in any release.  Once `v1.0.0` is tagged the guarantees above
apply.
