import polars as pl
import pytest
from polars.testing import assert_frame_equal


@pytest.fixture
def reports(data):
    """
    Fixture that returns the stats property of the data fixture.

    Args:
        data: The data fixture containing a Data object.

    Returns:
        Stats: The stats property of the data fixture.
    """
    return data.reports

@pytest.fixture
def metrics(resource_dir):
    return pl.read_csv(resource_dir / 'metrics.csv')


def test_metric(reports, metrics):
    assert_frame_equal(metrics, reports.metrics())
