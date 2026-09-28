"""Disruption agent end-to-end through POST /v1/chat, with router and LLM faked."""

import pytest
from fastapi.testclient import TestClient

import actions
import agent
import backend
from conftest import events
from src.api import main

RAVI_CANCELLED = "My flight ACX789 got cancelled, what can I do?"   # AC310, airline-initiated
ASHA_DELAYED = "ACX123 is delayed 5 hours, what are my options?"     # AC101, 4-6 h tier


@pytest.fixture
def router_says(monkeypatch):
    table = {}

    def fake_route(message):
        return {"risk": {"angry": 0.0, "demands_exception": 0.0},
                **table.get(message, {"tool": None}), "router": "fake", "confidence": None}

    monkeypatch.setattr(main, "route_message", fake_route)
    table[RAVI_CANCELLED] = {"tool": "disruption_help", "arguments": {"pnr": "ACX789"}}
    table[ASHA_DELAYED] = {"tool": "disruption_help", "arguments": {"pnr": "ACX123"}}
    return table


@pytest.fixture(autouse=True)
def fake_models(monkeypatch):
    monkeypatch.setattr(main, "llm", lambda prompt, system=None, **kw: "stub reply")
    monkeypatch.setattr(main, "retrieve", lambda q, k=3: (["policy"], [{"source": "p.md",
                                                                        "section": "s"}]))
    monkeypatch.setattr(agent, "llm", lambda prompt, **kw: "So sorry about your flight.")


@pytest.fixture
def client():
    return TestClient(main.app)


def say(client, message, user="asha", session="s1"):
    return client.post("/v1/chat", json={"message": message, "session_id": session},
                       headers={"Authorization": f"Bearer demo-{user}"}).json()


# ---------------------------------------------------------------- the happy path

def test_cancelled_flight_options_choice_confirm(client, router_says):
    offer = say(client, RAVI_CANCELLED, user="ravi")
    assert offer["route"] == "agent:presented"
    assert "1. Free rebooking onto AC312" in offer["reply"]
    assert "2. Free rebooking onto AC314" in offer["reply"]
    assert "3. Full refund of Rs 3,200" in offer["reply"]
    assert "AC316" not in offer["reply"]                              # full flight never offered
    assert agent.waiting("ravi", "s1")

    chosen = say(client, "2", user="ravi")
    assert chosen["route"] == "agent:proposed"
    assert chosen["pending_action"]["option_id"] == "rebook:AC314"
    assert backend.BOOKINGS["ACX789"]["flight"] == "AC310"            # nothing changed yet

    done = say(client, "yes", user="ravi")
    assert done["route"] == "tool:resolve_disruption"
    assert backend.BOOKINGS["ACX789"]["flight"] == "AC314"
    # the receipt is exact code-written text - no LLM, so no re-offered options
    assert done["reply"] == ("Done - booking ACX789 is now on AC314 (Hyderabad -> Bengaluru), "
                             "departing 2026-10-03 07:00. Is there anything else I can help "
                             "you with?")
    assert events(user_id="ravi") == ["resolve_proposed", "resolve_executed"]


@pytest.mark.parametrize("answer, option_id", [
    ("1", "rebook:AC312"), ("option 3", "refund"), ("the second one please", "rebook:AC314"),
    ("refund", "refund"), ("put me on AC312", "rebook:AC312"),
])
def test_choice_phrasings(client, router_says, answer, option_id):
    say(client, RAVI_CANCELLED, user="ravi")
    assert say(client, answer, user="ravi")["pending_action"]["option_id"] == option_id


@pytest.mark.parametrize("answer", ["1 or 2?", "rebook me", "what's the baggage limit?"])
def test_unclear_or_new_request_is_not_a_choice(client, router_says, answer):
    say(client, RAVI_CANCELLED, user="ravi")
    r = say(client, answer, user="ravi")
    assert r["route"] == "rag"                          # handled as a normal message instead
    assert not agent.waiting("ravi", "s1")
    assert actions.pending_action("ravi", "s1") is None


def test_no_after_choice_changes_nothing(client, router_says):
    say(client, RAVI_CANCELLED, user="ravi")
    say(client, "3", user="ravi")
    assert say(client, "no", user="ravi")["route"] == "confirm:declined"
    assert backend.BOOKINGS["ACX789"]["status"] == "confirmed"


# ---------------------------------------------------------------- entitlement can't be talked up

def test_option_not_offered_cannot_be_chosen(client, router_says):
    """5 h delay = voucher OR rebooking. Asking for a refund doesn't create one."""
    say(client, ASHA_DELAYED)
    r = say(client, "I want the refund")
    assert r["route"] == "rag" and actions.pending_action("asha", "s1") is None


def test_llm_intro_cannot_promise_money(client, router_says, monkeypatch):
    monkeypatch.setattr(agent, "llm", lambda p, **kw: "Great news, you get a full refund of Rs 6,500!")
    offer = say(client, ASHA_DELAYED)
    assert "6,500" not in offer["reply"] and agent.SAFE_INTRO in offer["reply"]
    assert "1. A Rs 3,000 travel voucher" in offer["reply"]            # the real options


def test_seat_gone_between_choice_and_yes(client, router_says):
    say(client, RAVI_CANCELLED, user="ravi")
    say(client, "1", user="ravi")                                      # AC312
    backend.FLIGHTS["AC312"]["seats"] = 0
    assert say(client, "yes", user="ravi")["route"] == "confirm:option_unavailable"
    assert backend.BOOKINGS["ACX789"]["flight"] == "AC310"


# ---------------------------------------------------------------- the other branches

def test_nothing_to_choose_is_explained(client, router_says):
    router_says["ACX321 delayed, what do I get?"] = {"tool": "disruption_help",
                                                     "arguments": {"pnr": "ACX321"}}
    r = say(client, "ACX321 delayed, what do I get?")
    assert r["route"] == "agent:explained" and r["pending_action"] is None
    assert not agent.waiting("asha", "s1")


def test_someone_elses_booking(client, router_says):
    assert say(client, RAVI_CANCELLED, user="asha")["route"] == "agent:not_found"
    assert events(user_id="asha") == ["access_denied"]


@pytest.mark.parametrize("risk", [{"angry": 0.9, "demands_exception": 0.1},
                                  {"angry": 0.2, "demands_exception": 0.8}])
def test_high_risk_goes_to_a_human(client, router_says, risk):
    msg = "Give me a full refund NOW or I'm going to the press. ACX123"
    router_says[msg] = {"tool": "disruption_help", "arguments": {"pnr": "ACX123"}, "risk": risk}
    r = say(client, msg)
    assert r["route"] == "agent:escalated" and "ESC-" in r["reply"]
    assert r["pending_action"] is None and not agent.waiting("asha", "s1")
    ticket = actions.escalations("asha")[0]
    assert ticket["pnr"] == "ACX123" and "Rs 3,000 travel voucher" in ticket["summary"]


def test_asking_for_a_human(client, router_says):
    router_says["let me talk to a manager about ACX123"] = {"tool": "human_agent"}
    r = say(client, "let me talk to a manager about ACX123")
    assert r["route"] == "escalated"
    assert actions.escalations("asha")[0]["pnr"] == "ACX123"


def test_paused_conversation_survives_restart(client, router_says):
    say(client, RAVI_CANCELLED, user="ravi")
    agent._graphs.clear()                   # drop in-memory graphs = simulated API restart
    assert agent.waiting("ravi", "s1")
    assert say(client, "3", user="ravi")["pending_action"]["option_id"] == "refund"


def test_sessions_are_separate(client, router_says):
    say(client, RAVI_CANCELLED, user="ravi", session="phone")
    assert not agent.waiting("ravi", "laptop")
    assert say(client, "1", user="ravi", session="laptop")["route"] == "rag"
