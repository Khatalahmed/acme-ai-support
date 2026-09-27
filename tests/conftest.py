"""Shared test setup. Tests are hermetic: no Azure, Jev, Ollama or RAG index, whatever .env says."""

import copy
import os
import sys
from datetime import timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Set BEFORE any src module loads .env: load_dotenv never overrides an existing variable,
# so empty keys stay empty and no test can reach a paid API.
os.environ["LLM_BACKEND"] = "ollama"
os.environ["ROUTER_BACKEND"] = "llm"
for key in ("TYPESAFE_API_KEY", "AI_GATEWAY_API_KEY", "JEV_MODEL", "JEV_MIN_CONFIDENCE"):
    os.environ[key] = ""

for p in (ROOT, ROOT / "src", ROOT / "src" / "tools"):
    sys.path.insert(0, str(p))

import actions  # noqa: E402
import backend  # noqa: E402  (same module object actions and the API use)


@pytest.fixture(autouse=True)
def fresh_state(tmp_path, monkeypatch):
    """Every test gets its own database file and a pristine copy of the mock bookings."""
    monkeypatch.setenv("ACME_DB_PATH", str(tmp_path / "acme.db"))
    saved = copy.deepcopy(backend.FLIGHTS)
    yield
    backend.FLIGHTS.clear()
    backend.FLIGHTS.update(saved)


@pytest.fixture
def clock(monkeypatch):
    """Controllable time: clock.advance(minutes=6) jumps forward instantly."""
    class Clock:
        def __init__(self):
            self.t = actions.now()

        def advance(self, **delta):
            self.t += timedelta(**delta)

    c = Clock()
    monkeypatch.setattr(actions, "now", lambda: c.t)
    return c


def events(**filters):
    """Audit event names, oldest first."""
    return [row["event"] for row in actions.audit_trail(**filters)]
