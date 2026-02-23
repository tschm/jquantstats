"""Tests for the Rhiza badge in README.md."""

RHIZA_BADGE_URL = (
    "https://img.shields.io/badge/dynamic/yaml"
    "?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftschm%2Fjquantstats%2Fmain%2F.rhiza%2Ftemplate.yml"
    "&query=%24.ref"
    "&label=rhiza"
)


def test_rhiza_badge_present(readme_path):
    """README.md should contain a dynamic Rhiza badge.

    The badge must reference .rhiza/template.yml via shields.io dynamic endpoint,
    not hardcode the version.
    """
    content = readme_path.read_text(encoding="utf-8")
    assert RHIZA_BADGE_URL in content, (
        "README.md should contain the dynamic Rhiza badge URL that reads from .rhiza/template.yml"
    )
