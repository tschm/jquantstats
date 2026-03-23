"""jQuantStats: Portfolio analytics for quants.

Two entry points
----------------
**Entry point 1 — prices + positions (recommended for active portfolios):**

Use :class:`~jquantstats.portfolio.Portfolio` when you have price series and
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

Use :class:`~jquantstats._data.Data` when you already have a returns series
(e.g. downloaded from a data vendor) and want benchmark comparison or
factor analytics.

```python
from jquantstats import Data
import polars as pl

data = Data.from_returns(returns=returns_df, benchmark=bench_df)
data.stats.sharpe()
data.plots.plot_snapshot(title="Performance")
```

:func:`~jquantstats.api.build_data` is kept as a convenience alias for
``Data.from_returns()`` for backward compatibility.

The two APIs are layered: ``portfolio.data`` returns a :class:`~jquantstats._data.Data`
object so you can always drop into the returns-series API from a Portfolio.

For more information, visit the `jQuantStats Documentation <https://tschm.github.io/jquantstats/book>`_.
"""

import importlib.metadata

from ._cost_model import CostModel as CostModel
from ._data import Data as Data
from ._types import NativeFrame as NativeFrame
from ._types import NativeFrameOrScalar as NativeFrameOrScalar
from .api import build_data  # noqa: F401
from .portfolio import Portfolio as Portfolio

__version__ = importlib.metadata.version("jquantstats")
