"""'My activity' and 'My requests': the audit log and hand-off tickets, readable and private."""

import pytest
from fastapi.testclient import TestClient

from src.api import main

ROUTES = {
    "Cancel ACX456": {"tool": "cancel_ticket", "arguments": {"pnr": "ACX456"}},
    "Cancel ACX789": {"tool": "cancel_ticket", "arguments": {"pnr": "ACX789"}},
    "ACX123 options": {"tool": "disruption_help", "arguments": {"pnr": "ACX123"}},
    "ACX789 cancelled": {"tool": "disruption_help", "arguments": {"pnr": "ACX789"}},
    "my passport is Z1234567, get me a manager": {"tool": "human_agent"},
}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "route_message", lambda m: {
        **ROUTES.get(m, {"tool": None}), "router": "fake", "confidence": None,
        "risk": {"angry": 0.0, "demands_exception": 0.0}})
    monkeypatch.setattr(main, "llm", lambda prompt, system=None, **kw: "stub reply")
    return TestClient(main.app)


def say(client, message, user="asha"):
    return client.post("/v1/chat", json={"message": message, "session_id": "s1"},
                       headers={"Authorization": f"Bearer demo-{user}"}).json()


def mine(client, user="asha"):
    return client.get("/v1/activity", headers={"Authorization": f"Bearer demo-{user}"}).json()


def texts(client, user="asha"):
    return [(e["kind"], e["text"]) for e in mine(client, user)["activity"]]


def test_cancel_then_yes_newest_first(client):
    say(client, "Cancel ACX456")
    say(client, "yes")
    assert texts(client) == [
        ("done", "Cancelled ACX456 - Non-refundable fare: a travel credit voucher will be issued instead"),
        ("pending", "You asked to cancel ACX456 - waiting for your yes"),
    ]


def test_saying_no_is_recorded_as_nothing_changed(client):
    say(client, "Cancel ACX456")
    say(client, "no")
    assert texts(client)[0] == ("stopped", "You said no - nothing changed on ACX456")


def test_disruption_choice_and_refund(client):
    say(client, "ACX789 cancelled", user="ravi")
    say(client, "3", user="ravi")
    say(client, "yes", user="ravi")
    assert texts(client, "ravi")[:2] == [
        ("done", "Refunded Rs 3,200 for ACX789"),
        ("pending", "You chose a full refund for ACX789 - waiting for your yes"),
    ]


def test_voucher(client):
    say(client, "ACX123 options")
    say(client, "1")
    say(client, "yes")
    assert texts(client)[0] == ("done", "Rs 3,000 travel voucher added to ACX123")


def test_someone_elses_booking_shows_as_refused(client):
    say(client, "Cancel ACX789")                     # Ravi's booking
    assert texts(client) == [("blocked", "A request about ACX789 was refused - it isn't on your account")]


def test_only_your_own_activity(client):
    say(client, "Cancel ACX456")
    assert mine(client, "ravi") == {"activity": [], "requests": []}


def test_requests_show_reason_not_your_words(client):
    say(client, "my passport is Z1234567, get me a manager")
    got = mine(client)
    req = got["requests"][0]
    assert req["ref"].startswith("ESC-") and req["status"] == "open"
    assert req["reason"] == "You asked to speak to a person"
    assert got["activity"][0] == {**got["activity"][0], "kind": "handoff",
                                  "text": f"Passed to our team ({req['ref']})"}
    assert "Z1234567" not in str(got)                # the ticket's summary holds the raw message


def test_needs_a_token(client):
    assert client.get("/v1/activity").status_code == 401
