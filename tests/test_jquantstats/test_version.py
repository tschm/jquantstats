"""Tests for the version module."""

import jquantstats


def test_version():
    """Tests that the package has a version.

    Verifies:
        The __version__ attribute of the jquantstats package is not None.
    """
    assert jquantstats.__version__ is not None


def test_public_api_exports():
    """Both Portfolio and Data are importable from the top-level package."""
    from jquantstats import Data, Portfolio

    assert Data is not None
    assert Portfolio is not None


def test_data_has_from_returns_classmethod():
    """Data exposes a callable from_returns classmethod."""
    from jquantstats import Data

    assert hasattr(Data, "from_returns")
    assert callable(Data.from_returns)
