"""Tests for the version module."""
import jquantstats


def test_version():
    """Tests that the package has a version.

    Verifies:
        The __version__ attribute of the jquantstats package is not None.
    """
    assert jquantstats.__version__ is not None
