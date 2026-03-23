"""Guard the public API surface against accidental removals."""

import jquantstats

PUBLIC_API = [
    "CostModel",
    "Data",
    "NativeFrame",
    "NativeFrameOrScalar",
    "Portfolio",
]


def test_public_api_complete() -> None:
    """Assert every name in PUBLIC_API is present in the jquantstats namespace."""
    for name in PUBLIC_API:
        assert hasattr(jquantstats, name), f"{name!r} missing from public API"


def test_version_exposed() -> None:
    """Assert that __version__ is exposed at the top-level package."""
    assert hasattr(jquantstats, "__version__")
