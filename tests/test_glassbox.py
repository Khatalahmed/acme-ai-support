"""The "How I decided" trace on every reply, and the numbers-only /insights metrics."""

import types

import pytest
from fastapi.testclient import TestClient

import decision_trace
import llm_backend
from src.api import main

SECRET_MESSAGE = "my passport number is Z1234567, what is the baggage allowance?"


@pytest.fixture
def client(monkeypatch):
    """Real API path down to the model: the model itself is faked with measurable usage, so
    the trace's model-call step is exercised, not stubbed away."""
    table = {"Cancel ACX456": {"tool": "cancel_ticket", "arguments": {"pnr": "ACX456"}},
             "Cancel ACX789": {"tool": "cancel_ticket", "arguments": {"pnr": "ACX789"}},
             "Is ACX456 on time?": {"tool": "get_flight_status", "arguments": {"pnr": "ACX456"}}}
    monkeypatch.setattr(main, "route_message", lambda m: {
        **table.get(m, {"tool": None}), "router": "jev", "confidence": 0.91,
        "risk": {"angry": 0.1, "demands_exception": 0.0}})
    usage = types.SimpleNamespace(prompt_tokens=900, completion_tokens=120)
    monkeypatch.setattr(llm_backend, "_ollama_chat", lambda messages, name: ("An answer.", usage))
    return TestClient(main.app)


def say(client, message, session="s1"):
    return client.post("/v1/chat", json={"message": message, "session_id": session},
                       headers={"Authorization": "Bearer demo-asha"}).json()


def test_policy_answer_trace(client):
    t = say(client, "What is the baggage allowance?")["trace"]
    assert t["router"] == "jev" and t["confidence"] == 0.91 and t["model_calls"] == 1
    assert [s["step"] for s in t["steps"]] == ["Understand", "Act", "Write"]
    assert t["steps"][0]["detail"] == "Jev classifier read this as a policy question (91% confident)"
    assert t["steps"][0]["ms"] is not None
    assert t["steps"][2]["detail"] == "Model call rag.answer: 1,020 tokens"
    assert t["risk"]["flagged"] == []


def test_cancel_trace_shows_the_guardrail(client):
    t = say(client, "Cancel ACX456")["trace"]
    assert t["steps"][1]["detail"] == "Proposed cancel ticket on ACX456 - waiting for YES"
    assert "Nothing changes until you reply YES" in t["safety"]
    assert t["steps"][2]["detail"] == "Written by code - no model call" and t["model_calls"] == 0

    done = say(client, "yes")["trace"]
    assert done["router"] == "confirmation"
    assert done["steps"][0]["detail"] == "Your message answered a pending confirmation"
    assert "Recorded in the audit log" in done["safety"]


def test_status_trace_names_the_backend_call(client):
    t = say(client, "Is ACX456 on time?")["trace"]
    assert t["steps"][1]["detail"] == "Called get_flight_status(ACX456)"
    assert t["steps"][2]["detail"].startswith("Model call tool.reply:")


def test_fallback_and_risk_are_explained():
    decision = {"router": "jev->llm", "tool": "cancel_ticket", "confidence": None,
                "unsure_jev": {"confidence": 0.42},
                "risk": {"angry": 0.9, "demands_exception": 0.1}, "router_ms": 3100}
    t = decision_trace.build("escalated", decision, [], None, [])
    assert t["steps"][0]["detail"] == ("Jev was unsure (42%), so the LLM router decided: "
                                       "a cancel request")
    assert t["risk"]["flagged"] == ["angry"]
    assert "Nothing on the booking was changed" in t["safety"]


def test_reasoning_tokens_are_shown():
    calls = [{"name": "tool.reply", "ms": 2100, "input_tokens": 800, "output_tokens": 700,
              "reasoning_tokens": 512}]
    t = decision_trace.build("tool:get_flight_status", {"router": "jev"}, [], None, calls)
    assert t["steps"][2] == {"step": "Write", "ms": 2100,
                             "detail": "Model call tool.reply: 1,500 tokens (512 spent reasoning)"}


# ---------------------------------------------------------------- /insights

def test_insights_counts_real_traffic(client):
    say(client, "What is the baggage allowance?")
    say(client, "Cancel ACX456")
    say(client, "no")
    say(client, "Cancel ACX789")                     # Ravi's booking: blocked
    live = client.get("/v1/insights").json()["live"]
    assert live["requests"] == 4
    assert live["actions"] == {"proposed": 1, "executed": 0, "declined": 1, "expired": 0,
                               "failed": 0}
    assert live["blocked_access"] == 1
    assert live["tokens"] == 1020 and live["no_model_replies"] == 3
    families = {r["family"]: r["count"] for r in live["routes"]}
    assert families == {"Policy answer": 1, "Confirmations": 2, "Clarifying question": 1}


def test_insights_never_stores_or_returns_messages(client):
    say(client, SECRET_MESSAGE)
    body = client.get("/v1/insights").text
    assert "Z1234567" not in body and "passport" not in body and "asha" not in body
    import actions
    with actions._db() as conn:
        columns = {r[1] for r in conn.execute("PRAGMA table_info(request_log)")}
    assert not columns & {"message", "user_id", "session_id"}


def test_insights_is_public_and_includes_measured_results(client):
    r = client.get("/v1/insights")                   # no token needed: aggregates only
    assert r.status_code == 200 and r.json()["measured"] is not None
