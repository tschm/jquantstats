import pandas as pd
import pytest

from jquantstats.api import build_data


@pytest.fixture
def data_no_benchmark(returns):
    """
    Fixture that returns a Data object with no benchmark.

    Args:
        returns: The returns fixture containing asset returns.

    Returns:
        _Data: A Data object with returns but no benchmark.
    """
    return build_data(returns=returns)


def test_copy_no_benchmark(data_no_benchmark):
    """
    Tests that the copy() method works correctly when benchmark is None.

    Args:
        data_no_benchmark: Fixture that returns a Data object with no benchmark.

    Verifies:
        1. The return value is a Data object.
        2. The copied object has the same returns as the original.
        3. The benchmark is None in both the original and the copy.
        4. Modifying the copied object does not affect the original.
    """
    # Create a copy of the data object
    data_copy = data_no_benchmark.copy()

    # Verify the copy has the same returns as the original
    pd.testing.assert_frame_equal(data_copy.returns, data_no_benchmark.returns)

    # Verify the benchmark is None in both the original and the copy
    assert data_no_benchmark.benchmark is None
    assert data_copy.benchmark is None

    # Verify that modifying the copy doesn't affect the original
    assert data_copy is not data_no_benchmark
    assert data_copy.returns is not data_no_benchmark.returns


def test_r_squared_no_benchmark(data_no_benchmark):
    """
    Tests that the r_squared() method raises an AttributeError when benchmark is None.

    Args:
        data_no_benchmark: Fixture that returns a Data object with no benchmark.

    Verifies:
        1. Calling r_squared() raises an AttributeError with the expected message.
    """
    # Verify that calling r_squared() raises an AttributeError
    with pytest.raises(AttributeError, match="No benchmark data available"):
        data_no_benchmark.stats.r_squared()
