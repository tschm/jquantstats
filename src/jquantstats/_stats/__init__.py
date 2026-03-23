"""Statistical analysis subpackage.

Re-exports :class:`Stats` and the module-level helpers from
:mod:`jquantstats._stats._core` so that existing imports such as
``from jquantstats._stats import Stats`` continue to work.
"""

from ._core import _drawdown_series as _drawdown_series
from ._core import _to_float as _to_float
from ._core import columnwise_stat as columnwise_stat
from ._core import to_frame as to_frame
from .stats import Stats as Stats

__all__ = [
    "Stats",
    "_drawdown_series",
    "_to_float",
    "columnwise_stat",
    "to_frame",
]
