"""Protocol describing the subset of Data that the _stats mixins require."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class DataLike(Protocol):  # pragma: no cover
    """Structural interface required by the Stats mixin classes.

    Any object satisfying this protocol can be passed as ``data`` to the
    mixin methods without a concrete dependency on :class:`~jquantstats._data.Data`.
    """

    returns: pl.DataFrame
    index: pl.DataFrame
    benchmark: pl.DataFrame | None

    @property
    def lazy(self) -> pl.LazyFrame:
        """LazyFrame view of all data, for use with expr-based stat factories."""
        ...

    @property
    def date_col(self) -> list[str]:
        """Column names used as the date/time index."""
        ...

    @property
    def assets(self) -> list[str]:
        """Names of the asset return columns."""
        ...

    @property
    def all(self) -> pl.DataFrame:
        """Combined DataFrame of returns, index, and benchmark columns."""
        ...

    @property
    def _periods_per_year(self) -> float:
        """Estimated number of return periods per calendar year."""
        ...

    def items(self) -> Iterator[tuple[str, pl.Series]]:
        """Iterate over (asset_name, returns_series) pairs."""
        ...
