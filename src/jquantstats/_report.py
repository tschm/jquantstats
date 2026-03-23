"""Backward-compatibility shim: re-exports from the _reports subpackage.

The classes and helpers formerly in this module have been moved to
:mod:`jquantstats._reports._portfolio`. This shim keeps existing imports working.
"""

from __future__ import annotations

from ._reports._portfolio import (
    Report,
    _fmt,
    _stats_table_html,
)

__all__ = ["Report", "_fmt", "_stats_table_html"]
