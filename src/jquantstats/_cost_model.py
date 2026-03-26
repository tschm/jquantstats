"""Unified cost model for Portfolio analytics.

This module provides :class:`CostModel`, a single abstraction that covers both
cost models available in :class:`~jquantstats.portfolio.Portfolio`:

**Model A — position-delta** (``cost_per_unit``)
    One-way cost per unit of traded notional (e.g. £0.01 per share).  Best for
    equity portfolios where cost scales with shares traded.

**Model B — turnover-bps** (``cost_bps``)
    One-way cost in basis points of AUM turnover (e.g. 5 bps).  Best for
    macro / fund-of-funds portfolios where cost scales with notional traded.

Use the class-method constructors to make intent explicit::

    CostModel.per_unit(0.01)       # Model A: £0.01 per share traded
    CostModel.turnover_bps(5.0)    # Model B: 5 bps per unit of AUM turnover
    CostModel.zero()               # No transaction costs

"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class CostModel:
    """Unified representation of a portfolio transaction-cost model.

    Eliminates the implicit "pick one" contract between the two independent
    cost parameters (``cost_per_unit`` and ``cost_bps``) on
    :class:`~jquantstats.portfolio.Portfolio`.  A ``CostModel``
    instance encapsulates one model at a time and can be passed to any
    Portfolio factory method instead of specifying the raw float parameters.

    Attributes:
        cost_per_unit: One-way cost per unit of position change (Model A).
            Defaults to 0.0.
        cost_bps: One-way cost in basis points of AUM turnover (Model B).
            Defaults to 0.0.

    Raises:
        ValueError: If ``cost_per_unit`` or ``cost_bps`` is negative, or if
            both are non-zero (which would silently double-count costs).

    Examples:
        >>> CostModel.per_unit(0.01)
        CostModel(cost_per_unit=0.01, cost_bps=0.0)
        >>> CostModel.turnover_bps(5.0)
        CostModel(cost_per_unit=0.0, cost_bps=5.0)
        >>> CostModel.zero()
        CostModel(cost_per_unit=0.0, cost_bps=0.0)
    """

    cost_per_unit: float = 0.0
    cost_bps: float = 0.0

    def __post_init__(self) -> None:
        if self.cost_per_unit < 0:
            raise ValueError(f"cost_per_unit must be non-negative, got {self.cost_per_unit}")  # noqa: TRY003
        if self.cost_bps < 0:
            raise ValueError(f"cost_bps must be non-negative, got {self.cost_bps}")  # noqa: TRY003
        if self.cost_per_unit > 0 and self.cost_bps > 0:
            raise ValueError(  # noqa: TRY003
                "Only one cost model may be active at a time: "
                f"got cost_per_unit={self.cost_per_unit} and cost_bps={self.cost_bps}. "
                "Use CostModel.per_unit() or CostModel.turnover_bps() to make intent explicit."
            )

    # ── Named constructors ────────────────────────────────────────────────────

    @classmethod
    def per_unit(cls, cost: float) -> CostModel:
        """Create a Model A (position-delta) cost model.

        Args:
            cost: One-way cost per unit of position change.  Must be
                non-negative.

        Returns:
            A :class:`CostModel` with ``cost_per_unit=cost`` and
            ``cost_bps=0.0``.

        Examples:
            >>> CostModel.per_unit(0.01)
            CostModel(cost_per_unit=0.01, cost_bps=0.0)
        """
        return cls(cost_per_unit=cost, cost_bps=0.0)

    @classmethod
    def turnover_bps(cls, bps: float) -> CostModel:
        """Create a Model B (turnover-bps) cost model.

        Args:
            bps: One-way cost in basis points of AUM turnover.  Must be
                non-negative.

        Returns:
            A :class:`CostModel` with ``cost_per_unit=0.0`` and
            ``cost_bps=bps``.

        Examples:
            >>> CostModel.turnover_bps(5.0)
            CostModel(cost_per_unit=0.0, cost_bps=5.0)
        """
        return cls(cost_per_unit=0.0, cost_bps=bps)

    @classmethod
    def zero(cls) -> CostModel:
        """Create a zero-cost model (no transaction costs).

        Returns:
            A :class:`CostModel` with both parameters set to 0.0.

        Examples:
            >>> CostModel.zero()
            CostModel(cost_per_unit=0.0, cost_bps=0.0)
        """
        return cls(cost_per_unit=0.0, cost_bps=0.0)
