"""jQuantStats: Portfolio analytics for quants.

Two entry points
----------------
**Entry point 1 — prices + positions (recommended for active portfolios):**

Use :class:`~jquantstats.analytics.Portfolio` when you have price series and
position sizes.  Portfolio compiles the NAV curve from raw inputs and exposes
the full analytics suite via ``.stats``, ``.plots``, and ``.report``.

```python
from jquantstats import Portfolio
import polars as pl

pf = Portfolio.from_cash_position(
    prices=prices_df,
    cash_position=positions_df,
    aum=1_000_000,
)
pf.stats.sharpe()
pf.plots.snapshot()
```

**Entry point 2 — returns series (for arbitrary return streams):**

Use :func:`~jquantstats.api.build_data` when you already have a returns series
(e.g. downloaded from a data vendor) and want benchmark comparison or
factor analytics.

```python
from jquantstats import build_data
import polars as pl

data = build_data(returns=returns_df, benchmark=bench_df)
data.stats.sharpe()
data.plots.plot_snapshot(title="Performance")
```

The two APIs are layered: ``portfolio.data`` returns a :class:`~jquantstats._data.Data`
object so you can always drop into the returns-series API from a Portfolio.

For more information, visit the `jQuantStats Documentation <https://tschm.github.io/jquantstats/book>`_.
"""

import importlib.metadata

from ._types import NativeFrame as NativeFrame
from ._types import NativeFrameOrScalar as NativeFrameOrScalar
from .analytics import CostModel as CostModel
from .analytics import Portfolio as Portfolio
from .api import build_data  # noqa: F401

__version__ = importlib.metadata.version("jquantstats")
