"""Shared protocol definitions used across jquantstats subpackages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import polars as pl

if TYPE_CHECKING:
    from jquantstats._reports._protocol import StatsLike


@runtime_checkable
class DataLike(Protocol):  # pragma: no cover
    """Authoritative structural interface for Data consumers."""

    returns: pl.DataFrame
    index: pl.DataFrame

    @property
    def all(self) -> pl.DataFrame:
        """Combined DataFrame of date index and return columns."""
        ...

    @property
    def assets(self) -> list[str]:
        """Names of the asset return columns."""
        ...

    @property
    def date_col(self) -> list[str]:
        """Column names used as the date/time index."""
        ...

    @property
    def stats(self) -> StatsLike:
        """Statistics facade used by reports."""
        ...
