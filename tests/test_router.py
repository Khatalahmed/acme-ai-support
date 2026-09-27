"""Router logic with fake model responses: Jev parsing, fallbacks, PNR handling."""

import httpx
import pytest

import router
from evals.router_compare import resolve


def fake_jev(choice, confidence=0.95, status=200, seen=None):
    """Stand-in for httpx.post returning a Jev-shaped response."""
    def post(url, headers, json, timeout):
        if seen is not None:
            seen.update(url=url, auth=headers["Authorization"], body=json)
        body = {"model": "jev-1.13.0",
                "answers": {"intent": {"type": "choice", "choice": choice,
                                       "probabilities": {choice: confidence},
                                       "confidence": confidence}},
                "usage": {"input_tokens": 120, "output_tokens": 0}}
        return httpx.Response(status, json=body, request=httpx.Request("POST", url))
    return post


@pytest.fixture
def jev_on(monkeypatch):
    monkeypatch.setenv("ROUTER_BACKEND", "jev")
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-test")
    # the LLM fallback must never reach a real model in tests
    monkeypatch.setattr(router, "route_llm",
                        lambda msg: {"tool": None, "router": "llm", "confidence": None})


@pytest.mark.parametrize("message, choice, confidence, status, expected, expected_router", [
    ("cancel acx789", "cancel_ticket", 0.95, 200, ("tool:cancel_ticket", "ACX789"), "jev"),
    ("pnr is acx 789, status?", "get_flight_status", 0.95, 200,
     ("tool:get_flight_status", "ACX789"), "jev"),
    ("Is my flight on time?", "get_flight_status", 0.9, 200, ("clarify", None), "jev"),
    ("Please cancel booking AC1234.", "cancel_ticket", 0.9, 200, ("clarify", None), "jev"),
    ("What is the baggage allowance?", "policy", 0.97, 200, ("rag", None), "jev"),
    ("Status of ACX123?", "get_flight_status", 0.4, 200, ("rag", None), "jev->llm"),  # unsure
    ("Status of ACX123?", "get_flight_status", 0.9, 503, ("rag", None), "jev->llm"),  # Jev down
])
def test_jev_router(jev_on, monkeypatch, message, choice, confidence, status,
                    expected, expected_router):
    monkeypatch.setattr(router.httpx, "post", fake_jev(choice, confidence, status))
    decision = router.route(message)
    assert resolve(decision) == expected
    assert decision["router"] == expected_router


def test_jev_request_shape(jev_on, monkeypatch):
    seen = {}
    monkeypatch.setattr(router.httpx, "post", fake_jev("policy", seen=seen))
    router.route("hello")
    assert seen["url"] == "https://api.typesafe.ai/v1/systemone"
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["model"] == "jev-latest"
    assert seen["body"]["questions"]["intent"]["type"] == "choice"


def test_jev_via_vercel_gateway(jev_on, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "")
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "vck-test")
    seen = {}
    monkeypatch.setattr(router.httpx, "post", fake_jev("cancel_ticket", seen=seen))
    assert resolve(router.route("cancel ACX456")) == ("tool:cancel_ticket", "ACX456")
    assert seen["url"] == "https://ai-gateway.vercel.sh/typesafe/v1/systemone"
    assert seen["body"]["model"] == "typesafe-ai/jev"


def test_jev_without_key_falls_back(jev_on, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "")
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "")
    assert router.route("cancel ACX456")["router"] == "jev->llm"


def test_llm_router_normalises_pnr(monkeypatch):
    monkeypatch.setattr(router, "chat", lambda messages:
                        'sure! {"tool": "get_flight_status", "arguments": {"pnr": "acx 789"}}')
    assert resolve(router.route_llm("where is my flight acx 789")) == \
        ("tool:get_flight_status", "ACX789")


@pytest.mark.parametrize("model_output", ["no json here", "{broken json", ""])
def test_llm_router_garbage_means_no_tool(monkeypatch, model_output):
    monkeypatch.setattr(router, "chat", lambda messages: model_output)
    assert resolve(router.route_llm("anything")) == ("rag", None)
