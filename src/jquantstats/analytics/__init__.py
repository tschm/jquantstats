"""Analytics subpackage for jquantstats.

Provides the :class:`Portfolio` data model, the :class:`CostModel` abstraction,
and related analytics helpers.

Subpackage boundary
-------------------
This subpackage is the **canonical** implementation of portfolio-level
analytics.  It coexists with the top-level entry points
(:func:`~jquantstats.api.build_data` and the :class:`~jquantstats._stats.Stats`
class) which operate on bare return streams.

The separation is intentional:

- ``jquantstats.analytics`` — has access to raw prices, positions, and AUM;
  provides richer portfolio-specific visualisations, cost analysis, and
  attribution decomposition.
- ``jquantstats.api`` / ``jquantstats._stats`` — operates on pre-computed
  return streams; provides point-in-time and rolling financial metrics.

There is **no** dual-implementation risk: the previous ``analytics/_stats.py``
that shadowed the top-level ``Stats`` class was removed.
``analytics/portfolio.py`` delegates statistical computation to the top-level
``Stats`` class via the ``.stats`` and ``.data`` bridge properties.

Public API:
    :class:`Portfolio` — main portfolio analytics facade.
    :class:`CostModel` — unified transaction-cost model (Model A or Model B).

Private modules (subject to change without notice):
    ``_portfolio_data``, ``_plots``, ``_report``, ``exceptions``,
    ``_cost_model``.

Usage:
    >>> from jquantstats.analytics import Portfolio  # doctest: +SKIP
    >>> from jquantstats.analytics import Portfolio
    >>> issubclass(Portfolio, object)
    True
"""

# Public re-exports (explicit aliases so linters recognize intent)
from ._cost_model import CostModel as CostModel
from .portfolio import Portfolio as Portfolio
