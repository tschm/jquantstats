"""Protocols describing the minimal interfaces required by the _utils subpackage."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from jquantstats._protocol import DataLike


@runtime_checkable
class PortfolioLike(Protocol):  # pragma: no cover
    """Structural interface required by `PortfolioUtils`.

    Any object satisfying this protocol can be passed as ``portfolio`` without a
    concrete dependency on `Portfolio`.
    """

    @property
    def data(self) -> DataLike:
        """Bridge to the Data analytics object for this portfolio."""
        ...

    @property
    def assets(self) -> list[str]:
        """Asset column names."""
        ...
