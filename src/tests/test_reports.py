"""Tests for the reports module functionality."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal


@pytest.fixture
def reports(data):
    """Fixture that returns the stats property of the data fixture.

    Args:
        data: The data fixture containing a Data object.

    Returns:
        Stats: The stats property of the data fixture.

    """
    return data.reports

@pytest.fixture
def metrics(resource_dir):
    """Fixture that returns the metrics CSV file as a DataFrame.

    Args:
        resource_dir: The resource_dir fixture containing the path to the resources directory.

    Returns:
        pl.DataFrame: A DataFrame containing the metrics data.

    """
    return pl.read_csv(resource_dir / 'metrics.csv')


def test_metric(reports, metrics):
    """Tests that the metrics method returns the correct metrics.

    Args:
        reports: The reports fixture containing a Reports object.
        metrics: The metrics fixture containing the expected metrics.

    Verifies:
        The metrics method returns a DataFrame that matches the expected metrics.

    """
    #reports.metrics().write_csv("metrics.csv")
    assert_frame_equal(metrics, reports.metrics())
