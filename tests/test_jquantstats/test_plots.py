"""Tests for the plots module."""

import pytest

from jquantstats import Data


@pytest.fixture
def plots(data):
    """Fixture that returns the plots property of the data fixture.

    Args:
        data: The data fixture containing a Data object.

    Returns:
        DataPlots: The plots property of the data fixture.

    """
    return data.plots


def test_plot_snapshot(plots):
    """Tests that the plot_snapshot method works correctly.

    Args:
        plots: The plots fixture.

    Verifies:
        1. The method returns a plotly Figure object.
        2. The method doesn't raise any exceptions.
        3. The method works with different parameters.

    """
    # Test with default parameters
    fig = plots.plot_snapshot()
    assert fig is not None
    assert hasattr(fig, "show")

    # Test with custom parameters
    fig = plots.plot_snapshot(title="Custom Title", log_scale=True)
    assert fig is not None
    assert hasattr(fig, "show")

    # causing sometimes problems
    # fig.show()


def test_plot_snapshot_one_symbol(returns):
    """Tests that the plot_snapshot method works correctly with a single symbol.

    Args:
        returns: The returns fixture containing a DataFrame with a single symbol.

    Verifies:
        1. The method returns a plotly Figure object.
        2. The method doesn't raise any exceptions when working with a single symbol.

    """
    fig = Data.from_returns(returns=returns).plots.plot_snapshot()

    assert fig is not None
    assert hasattr(fig, "show")
    # causing sometimes problems
    # fig.show()


def test_repr(plots):
    """Tests that DataPlots.__repr__ returns an informative string."""
    r = repr(plots)
    assert r.startswith("DataPlots(assets=")
    for asset in plots.data.assets:
        assert asset in r
