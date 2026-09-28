"""Every source file must compile with no warnings.

Added after a fresh Docker build showed "SyntaxWarning: invalid escape sequence '\\-'" from a
docstring diagram in src/shadow.py. Locally it never appeared: Python only warns while compiling,
and cached .pyc files skipped that. Future Python versions make it an error (the app wouldn't
start), so compile everything from source here with warnings as errors.
"""

import warnings
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"


@pytest.mark.parametrize("path", sorted(SRC.rglob("*.py")), ids=lambda p: str(p.relative_to(SRC)))
def test_compiles_without_warnings(path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
