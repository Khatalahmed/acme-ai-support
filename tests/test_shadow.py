"""Shadow routing: the second router is recorded, never used, and never slows the reply."""

import time

import pytest
from fastapi.testclient import TestClient

import actions
import backend
import shadow
from src.api import main

POLICY_Q = "What is the checked baggage allowance?"
CANCEL_Q = "cancel ACX456"
RAG = {"tool": None}
CANCEL = {"tool": "cancel_ticket", "arguments": {"pnr": "ACX456"}}


@pytest.fixture
def served(monkeypatch):
    """The primary router's answers (router name 'jev', as when serving with Jev)."""
    table = {POLICY_Q: RAG, CANCEL_Q: CANCEL}
    monkeypatch.setattr(main, "route_message",
                        lambda m: {**table.get(m, RAG), "router": "jev", "confidence": 0.9,
                                   "risk": {"angry": 0.0, "demands_exception": 0.0}})
    monkeypatch.setattr(main, "llm", lambda *a, **kw: "stub reply")
    monkeypatch.setattr(main, "retrieve", lambda q, k=3: (["p"], [{"source": "p.md", "section": "s"}]))
    return table


@pytest.fixture
def shadow_says(monkeypatch):
    """The shadow (LLM) router's answers; turns shadow mode on."""
    table = {}

    def fake_llm_router(message):
        answer = table.get(message, RAG)
        if isinstance(answer, Exception):
            raise answer
        if callable(answer):
            return answer()
        return {**answer, "router": "llm"}

    monkeypatch.setitem(shadow.SHADOW_ROUTERS, "llm", fake_llm_router)
    monkeypatch.setenv("ROUTER_SHADOW", "llm")
    return table


def say(message, session="s1"):
    return TestClient(main.app).post("/v1/chat", json={"message": message, "session_id": session},
                                     headers={"Authorization": "Bearer demo-asha"}).json()


def logged():
    shadow.drain()
    return shadow.summary()


def test_off_by_default(served):
    say(POLICY_Q)
    assert logged()["compared"] == 0


def test_agreement_is_recorded(served, shadow_says):
    shadow_says[POLICY_Q] = RAG
    say(POLICY_Q)
    s = logged()
    assert (s["compared"], s["agree"]) == (1, 1)
    row = s["rows"][0]
    assert (row["primary_router"], row["shadow_router"], row["primary_outcome"]) == ("jev", "llm", "rag")


def test_disagreement_is_recorded_with_both_answers(served, shadow_says):
    shadow_says[POLICY_Q] = CANCEL
    say(POLICY_Q)
    [d] = logged()["disagreements"]
    assert (d["primary_outcome"], d["shadow_outcome"]) == ("rag", "tool:cancel_ticket ACX456")
    assert d["message"] == POLICY_Q and d["user_id"] == "asha"


def test_shadow_never_slows_the_reply(served, shadow_says):
    def slow():
        time.sleep(1.5)
        return {**RAG, "router": "llm"}
    shadow_says[POLICY_Q] = slow
    t0 = time.perf_counter()
    reply = say(POLICY_Q)
    assert time.perf_counter() - t0 < 1.0              # did not wait 1.5 s for the shadow
    assert reply["route"] == "rag"
    assert logged()["rows"][0]["shadow_ms"] >= 1500     # but the comparison still completed


def test_shadow_never_changes_behaviour(served, shadow_says):
    """Shadow wants to cancel; the customer only ever sees the primary router's decision."""
    shadow_says[POLICY_Q] = CANCEL
    reply = say(POLICY_Q)
    assert reply["route"] == "rag" and reply["pending_action"] is None
    assert actions.pending_action("asha", "s1") is None
    assert backend.BOOKINGS["ACX456"]["status"] == "confirmed"


def test_shadow_errors_are_contained(served, shadow_says):
    shadow_says[POLICY_Q] = RuntimeError("shadow router down")
    assert say(POLICY_Q)["route"] == "rag"
    s = logged()
    assert s["errors"] == 1 and s["compared"] == 0
    assert "shadow router down" in s["rows"][0]["error"]


@pytest.mark.parametrize("served_by", ["llm", "jev->llm"])
def test_skipped_when_that_router_already_decided(served, shadow_says, monkeypatch, served_by):
    monkeypatch.setattr(main, "route_message", lambda m: {**RAG, "router": served_by,
                                                          "confidence": None})
    say(POLICY_Q)
    assert logged()["rows"] == []


def test_unsure_jev_is_compared_for_free(served, shadow_says, monkeypatch):
    """Jev unsure -> LLM decided: compare Jev's own answer, without calling any router again."""
    def llm_decided(m):
        return {**RAG, "router": "jev->llm", "confidence": None,
                "unsure_jev": {**CANCEL, "router": "jev", "confidence": 0.34}}
    monkeypatch.setattr(main, "route_message", llm_decided)
    shadow_says[POLICY_Q] = RuntimeError("must not be called - both opinions already exist")
    monkeypatch.setenv("ROUTER_SHADOW_RATE", "0")        # never sampled away either
    assert say(POLICY_Q)["route"] == "rag"
    [d] = logged()["disagreements"]
    assert (d["primary_router"], d["shadow_router"]) == ("jev->llm", "jev (unsure)")
    assert (d["primary_outcome"], d["shadow_outcome"]) == ("rag", "tool:cancel_ticket ACX456")


def test_router_keeps_jevs_unsure_answer(monkeypatch):
    import router
    monkeypatch.setenv("ROUTER_BACKEND", "jev")
    monkeypatch.setattr(router, "route_jev", lambda m: {**CANCEL, "router": "jev",
                                                        "confidence": 0.3})
    monkeypatch.setattr(router, "route_llm", lambda m: {**RAG, "router": "llm",
                                                        "confidence": None})
    d = router.route("anything")
    assert d["router"] == "jev->llm" and d["tool"] is None
    assert d["unsure_jev"]["tool"] == "cancel_ticket"


def test_sampling_rate(served, shadow_says, monkeypatch):
    monkeypatch.setenv("ROUTER_SHADOW_RATE", "0")
    say(POLICY_Q)
    assert logged()["rows"] == []


def test_only_new_requests_are_shadowed(served, shadow_says):
    """Answering 'are you sure?' isn't routed, so there's nothing to compare."""
    shadow_says[CANCEL_Q] = CANCEL
    say(CANCEL_Q)
    say("yes")
    assert len(logged()["rows"]) == 1


def test_no_langfuse_calls_when_tracing_is_off(served, shadow_says, monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("must not call Langfuse with tracing off")
    monkeypatch.setattr(shadow, "_report", boom)
    shadow_says[POLICY_Q] = CANCEL
    say(POLICY_Q)
    assert len(logged()["disagreements"]) == 1
