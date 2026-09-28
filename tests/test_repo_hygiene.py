"""The public repo must never track local state: databases, secrets, the RAG index.

Added after SQLite companion files (acme-agent.db-wal / -shm) slipped into a commit: .gitignore
covered *.db but not the -wal/-shm files SQLite writes next to it. This test fails in CI if any
such file is tracked again.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN = re.compile(
    r"(\.db|\.db-wal|\.db-shm|\.db-journal|\.sqlite3?|\.sqlite-\w+)$"   # databases + companions
    r"|(^|/)\.env$"                                                        # secrets
    r"|(^|/)data/chroma/"                                                  # the built index
    r"|\.gguf$")                                                           # model weights


@pytest.mark.skipif(not shutil.which("git") or not (ROOT / ".git").exists(), reason="needs git")
def test_no_local_state_is_tracked():
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout.splitlines()
    offenders = [f for f in tracked if FORBIDDEN.search(f)]
    assert offenders == [], f"local state tracked in git: {offenders}"
