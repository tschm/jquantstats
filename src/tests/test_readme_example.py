"""Tests for the code example in the README.md file using doctest."""
import doctest
from pathlib import Path

import numpy as np
import plotly.graph_objs
import polars as pl

from jquantstats.api import build_data


def test_readme_examples():
    """Test the examples from the README using doctest."""
    # Get the path to the README.md file
    readme_path = Path(__file__).parent.parent.parent / "README.md"

    # Set up the globals for doctest
    globs = {
        'pl': pl,
        'build_data': build_data,
        'plotly': plotly,
        'np': np
    }

    # Run doctest on the README.md file
    # Set optionflags to ignore whitespace differences and normalize whitespace
    result = doctest.testfile(
        str(readme_path),
        module_relative=False,
        globs=globs,
        optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
    )

    # Check if there were any failures
    assert result.failed == 0, f"Doctest failed with {result.failed} failures"

    # Verify that tests were actually run
    assert result.attempted > 0, "No doctests were found in the README.md file"
