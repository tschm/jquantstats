import pytest


@pytest.fixture
def plots(data):
    """
    Fixture that returns the plots property of the data fixture.

    Args:
        data: The data fixture containing a Data object.

    Returns:
        Plots: The plots property of the data fixture.
    """
    return data.plots


def test_plot_returns_bars(plots):
    """
    Tests that the plot_returns_bars method works correctly.

    Args:
        plots: The plots fixture.

    Verifies:
        1. The method returns a plotly Figure object.
        2. The method doesn't raise any exceptions.
    """
    fig = plots.plot_returns_bars()
    assert fig is not None
    assert hasattr(fig, 'show')


def test_plot_snapshot(plots):
    """
    Tests that the plot_snapshot method works correctly.

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
    assert hasattr(fig, 'show')

    # Test with custom parameters
    fig = plots.plot_snapshot(title="Custom Title", compounded=False, log_scale=True)
    assert fig is not None
    assert hasattr(fig, 'show')


def test_monthly_heatmap(plots):
    """
    Tests that the monthly_heatmap method works correctly.

    Args:
        plots: The plots fixture.

    Verifies:
        1. The method returns a plotly Figure object.
        2. The method doesn't raise any exceptions.
        3. The method works with different parameters.
    """
    print(plots.data.all)
    print(plots.data.resample(every="1mo", compounded=False).all)

    # Test with default parameters
    fig = plots.monthly_heatmap(col="AAPL", compounded=False)
    assert fig is not None
    assert hasattr(fig, 'show')

    # Test with custom parameters
    fig = plots.monthly_heatmap(col="AAPL", annot_size=10, cbar=False, returns_label="AAPL",
                               compounded=False, fontname="Courier", ylabel=False)
    assert fig is not None
    assert hasattr(fig, 'show')
