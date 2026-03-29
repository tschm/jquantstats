"""Protocols describing the minimal interfaces required by the _utils subpackage."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class DataLike(Protocol):  # pragma: no cover
    """Structural interface required by :class:`~jquantstats._utils._data.DataUtils`.

    Any object satisfying this protocol can be passed as ``data`` without a
    concrete dependency on :class:`~jquantstats.data.Data`.
    """

    returns: pl.DataFrame
    index: pl.DataFrame

    @property
    def date_col(self) -> list[str]:
        """Column names used as the date/time index."""
        ...


@runtime_checkable
class PortfolioLike(Protocol):  # pragma: no cover
    """Structural interface required by :class:`~jquantstats._utils._portfolio.PortfolioUtils`.

    Any object satisfying this protocol can be passed as ``portfolio`` without a
    concrete dependency on :class:`~jquantstats.portfolio.Portfolio`.
    """

    @property
    def data(self) -> DataLike:
        """Bridge to the Data analytics object for this portfolio."""
        ...

    @property
    def assets(self) -> list[str]:
        """Asset column names."""
        ...
