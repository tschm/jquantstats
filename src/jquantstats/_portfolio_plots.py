"""Backward-compatibility shim: re-exports from the _plots subpackage.

The class formerly known as ``Plots`` in this module has been moved to
:mod:`jquantstats._plots._portfolio` and renamed :class:`PortfolioPlots`.
This shim keeps existing imports working.
"""

from __future__ import annotations

from ._plots._portfolio import PortfolioPlots as Plots

__all__ = ["Plots"]
