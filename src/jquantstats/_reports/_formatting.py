"""Shared formatting helpers for report modules."""

from __future__ import annotations

import math
from typing import Any, TypeGuard


def _is_finite(v: Any) -> TypeGuard[int | float]:
    """Return True when *v* is a real, finite number."""
    if not isinstance(v, (int, float)):
        return False
    return math.isfinite(float(v))


def _fmt(value: Any, fmt: str = ".4f", suffix: str = "") -> str:
    """Format *value* for display; return ``"N/A"`` for non-finite values."""
    if not _is_finite(value):
        return "N/A"
    return f"{float(value):{fmt}}{suffix}"
