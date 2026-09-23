"""The package version is defined in two places; keep them in sync."""

# standard imports
import re
from pathlib import Path

# custom imports
import online_alignment


def test_version_matches_pyproject():
    pyproject = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()
    version = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE).group(1)
    assert online_alignment.__version__ == version
