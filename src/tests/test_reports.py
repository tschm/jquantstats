import pytest


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

def test_reports(reports):
    reports.basic()
