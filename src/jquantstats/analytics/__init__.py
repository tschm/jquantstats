"""Analytics subpackage for jquantstats.

Provides the Portfolio data model and related analytics helpers.

Public API:
- jquantstats.analytics.portfolio — home of jquantstats.analytics.portfolio.Portfolio.

Private modules (subject to change):
- jquantstats.analytics._portfolio_data
- jquantstats.analytics._stats
- jquantstats.analytics._plots

Usage:
    >>> from jquantstats.analytics import Portfolio  # doctest: +SKIP
    >>> from jquantstats.analytics import Portfolio
    >>> issubclass(Portfolio, object)
    True

Notes:
    Direct imports from private modules are discouraged as they may change
    without notice.
"""

# Public re-exports (explicit aliases so linters recognize intent)
from .portfolio import Portfolio as Portfolio
