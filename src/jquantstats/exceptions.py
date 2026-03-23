"""Domain-specific exception types for the jquantstats package.

This module defines a hierarchy of exceptions that provide meaningful context
when data-validation errors occur within the package.

All exceptions inherit from :class:`JQuantStatsError` so callers can catch the
entire family with a single ``except JQuantStatsError`` clause if they prefer.

Examples:
    >>> raise MissingDateColumnError("prices")  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    jquantstats.exceptions.MissingDateColumnError: ...
"""

from __future__ import annotations


class JQuantStatsError(Exception):
    """Base class for all JQuantStats domain errors."""


class MissingDateColumnError(JQuantStatsError, ValueError):
    """Raised when a required ``'date'`` column is absent from a DataFrame.

    Args:
        frame_name: Descriptive name of the frame missing the column (e.g. ``"prices"``).

    Examples:
        >>> raise MissingDateColumnError("prices")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        jquantstats.exceptions.MissingDateColumnError: ...
    """

    def __init__(self, frame_name: str) -> None:
        """Initialize with the name of the frame that is missing the column."""
        super().__init__(f"DataFrame '{frame_name}' is missing the required 'date' column.")
        self.frame_name = frame_name


class InvalidCashPositionTypeError(JQuantStatsError, TypeError):
    """Raised when ``cashposition`` is not a :class:`polars.DataFrame`.

    Args:
        actual_type: The ``type.__name__`` of the value that was supplied.

    Examples:
        >>> raise InvalidCashPositionTypeError("dict")
        Traceback (most recent call last):
            ...
        jquantstats.exceptions.InvalidCashPositionTypeError: cashposition must be pl.DataFrame, got dict.
    """

    def __init__(self, actual_type: str) -> None:
        """Initialize with the offending type name."""
        super().__init__(f"cashposition must be pl.DataFrame, got {actual_type}.")
        self.actual_type = actual_type


class InvalidPricesTypeError(JQuantStatsError, TypeError):
    """Raised when ``prices`` is not a :class:`polars.DataFrame`.

    Args:
        actual_type: The ``type.__name__`` of the value that was supplied.

    Examples:
        >>> raise InvalidPricesTypeError("list")
        Traceback (most recent call last):
            ...
        jquantstats.exceptions.InvalidPricesTypeError: prices must be pl.DataFrame, got list.
    """

    def __init__(self, actual_type: str) -> None:
        """Initialize with the offending type name."""
        super().__init__(f"prices must be pl.DataFrame, got {actual_type}.")
        self.actual_type = actual_type


class NonPositiveAumError(JQuantStatsError, ValueError):
    """Raised when ``aum`` is not strictly positive.

    Args:
        aum: The non-positive value that was supplied.

    Examples:
        >>> raise NonPositiveAumError(0.0)
        Traceback (most recent call last):
            ...
        jquantstats.exceptions.NonPositiveAumError: aum must be strictly positive, got 0.0.
    """

    def __init__(self, aum: float) -> None:
        """Initialize with the offending aum value."""
        super().__init__(f"aum must be strictly positive, got {aum}.")
        self.aum = aum


class RowCountMismatchError(JQuantStatsError, ValueError):
    """Raised when ``prices`` and ``cashposition`` have different numbers of rows.

    Args:
        prices_rows: Number of rows in the prices DataFrame.
        cashposition_rows: Number of rows in the cashposition DataFrame.

    Examples:
        >>> raise RowCountMismatchError(10, 9)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        jquantstats.exceptions.RowCountMismatchError: ...
    """

    def __init__(self, prices_rows: int, cashposition_rows: int) -> None:
        """Initialize with the row counts of the two mismatched DataFrames."""
        super().__init__(
            f"cashposition and prices must have the same number of rows, "
            f"got cashposition={cashposition_rows} and prices={prices_rows}."
        )
        self.prices_rows = prices_rows
        self.cashposition_rows = cashposition_rows


class IntegerIndexBoundError(JQuantStatsError, TypeError):
    """Raised when a row-index bound is not an integer.

    Args:
        param: Name of the offending parameter (e.g. ``"start"`` or ``"end"``).
        actual_type: The ``type.__name__`` of the value that was supplied.

    Examples:
        >>> raise IntegerIndexBoundError("start", "str")
        Traceback (most recent call last):
            ...
        jquantstats.exceptions.IntegerIndexBoundError: start must be an integer, got str.
    """

    def __init__(self, param: str, actual_type: str) -> None:
        """Initialize with the parameter name and the offending type."""
        super().__init__(f"{param} must be an integer, got {actual_type}.")
        self.param = param
        self.actual_type = actual_type
