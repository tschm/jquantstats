"""Unit tests for .rhiza/utils/version_matrix.py.

Tests the version parsing and specifier matching functions used for
determining supported Python versions in CI workflows.
"""

import sys
from pathlib import Path

import pytest

# Add .rhiza/utils to path so we can import version_matrix
sys.path.insert(0, str(Path(__file__).parent.parent.parent / ".rhiza" / "utils"))

from version_matrix import _check_operator, parse_version, satisfies


class TestParseVersion:
    """Tests for parse_version function."""

    def test_simple_version(self):
        """Parse a simple two-part version."""
        assert parse_version("3.11") == (3, 11)

    def test_three_part_version(self):
        """Parse a three-part version."""
        assert parse_version("3.11.5") == (3, 11, 5)

    def test_single_part_version(self):
        """Parse a single-part version."""
        assert parse_version("3") == (3,)

    def test_version_with_rc_suffix(self):
        """Parse version with release candidate suffix - numeric prefix is extracted."""
        assert parse_version("3.14.0rc1") == (3, 14, 0)

    def test_version_with_alpha_suffix(self):
        """Parse version with alpha suffix."""
        assert parse_version("3.13.0a1") == (3, 13, 0)

    def test_invalid_version_no_leading_digit(self):
        """Raise ValueError when component has no leading digit."""
        with pytest.raises(ValueError, match="Invalid version component"):
            parse_version("3.abc")

    def test_invalid_version_empty_component(self):
        """Raise ValueError for empty component."""
        with pytest.raises(ValueError, match="Invalid version component"):
            parse_version("3..11")


class TestCheckOperator:
    """Tests for _check_operator function."""

    def test_greater_equal_true(self):
        """Version >= specifier returns True when satisfied."""
        assert _check_operator((3, 11), ">=", (3, 11)) is True
        assert _check_operator((3, 12), ">=", (3, 11)) is True

    def test_greater_equal_false(self):
        """Version >= specifier returns False when not satisfied."""
        assert _check_operator((3, 10), ">=", (3, 11)) is False

    def test_less_equal_true(self):
        """Version <= specifier returns True when satisfied."""
        assert _check_operator((3, 11), "<=", (3, 11)) is True
        assert _check_operator((3, 10), "<=", (3, 11)) is True

    def test_less_equal_false(self):
        """Version <= specifier returns False when not satisfied."""
        assert _check_operator((3, 12), "<=", (3, 11)) is False

    def test_greater_than(self):
        """Version > specifier."""
        assert _check_operator((3, 12), ">", (3, 11)) is True
        assert _check_operator((3, 11), ">", (3, 11)) is False

    def test_less_than(self):
        """Version < specifier."""
        assert _check_operator((3, 10), "<", (3, 11)) is True
        assert _check_operator((3, 11), "<", (3, 11)) is False

    def test_equal(self):
        """Version == specifier."""
        assert _check_operator((3, 11), "==", (3, 11)) is True
        assert _check_operator((3, 12), "==", (3, 11)) is False

    def test_not_equal(self):
        """Version != specifier."""
        assert _check_operator((3, 12), "!=", (3, 11)) is True
        assert _check_operator((3, 11), "!=", (3, 11)) is False


class TestSatisfies:
    """Tests for satisfies function."""

    def test_greater_equal(self):
        """Version satisfies >= constraint."""
        assert satisfies("3.11", ">=3.11") is True
        assert satisfies("3.12", ">=3.11") is True
        assert satisfies("3.10", ">=3.11") is False

    def test_less_than(self):
        """Version satisfies < constraint."""
        assert satisfies("3.10", "<3.11") is True
        assert satisfies("3.11", "<3.11") is False

    def test_combined_constraints(self):
        """Version satisfies multiple comma-separated constraints."""
        assert satisfies("3.11", ">=3.11,<3.14") is True
        assert satisfies("3.13", ">=3.11,<3.14") is True
        assert satisfies("3.14", ">=3.11,<3.14") is False
        assert satisfies("3.10", ">=3.11,<3.14") is False

    def test_exact_version_no_operator(self):
        """Version matches exact version without operator."""
        assert satisfies("3.11", "3.11") is True
        assert satisfies("3.12", "3.11") is False

    def test_whitespace_handling(self):
        """Constraints with whitespace are handled correctly."""
        assert satisfies("3.11", ">= 3.10, < 3.14") is True

    def test_not_equal_constraint(self):
        """Version satisfies != constraint."""
        assert satisfies("3.12", "!=3.11") is True
        assert satisfies("3.11", "!=3.11") is False

    def test_invalid_specifier(self):
        """Raise ValueError for invalid specifier."""
        with pytest.raises(ValueError, match="Invalid specifier"):
            satisfies("3.11", "~=3.11")  # ~= not supported

    def test_three_part_versions(self):
        """Handle three-part versions in constraints."""
        assert satisfies("3.11.5", ">=3.11.0") is True
        assert satisfies("3.11.0", ">=3.11.5") is False
