"""Shared test utilities for plot-structure snapshot tests.

Provides :func:`figure_structure`, a helper that converts a Plotly
``go.Figure`` into a JSON-serialisable dict capturing its structural
fingerprint (trace types/names and key layout properties) **without**
the raw x/y data arrays.  This keeps snapshot files small and stable
across data changes while still detecting structural regressions.
"""

from __future__ import annotations

import plotly.graph_objects as go


def figure_structure(fig: go.Figure) -> dict:
    """Extract a structural fingerprint of a Plotly figure.

    Returns a dict capturing trace metadata (type and name) and key layout
    properties, deliberately *excluding* the raw ``x``/``y`` data arrays so
    that the snapshot stays stable across different data inputs while still
    catching any structural regressions.

    Args:
        fig: A Plotly Figure object to fingerprint.

    Returns:
        dict: A JSON-serialisable dict with ``num_traces``, ``traces``, and
        ``layout`` keys.

    """
    return {
        "num_traces": len(fig.data),
        "traces": [{"type": type(t).__name__, "name": t.name} for t in fig.data],
        "layout": {
            "title": fig.layout.title.text if fig.layout.title else None,
            "hovermode": fig.layout.hovermode,
            "plot_bgcolor": fig.layout.plot_bgcolor,
            "subplot_titles": [
                ann.text for ann in (fig.layout.annotations or []) if ann.showarrow is False and ann.text
            ],
        },
    }
