"""Abuse controls for the public demo: per-visitor limit, daily cap, spoof-resistant IP, demo reset."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import backend
import ratelimit
from src.api import main


def test_per_visitor_limit(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_IP", "3")
    assert [ratelimit.check("1.2.3.4", now=100 + i) for i in range(3)] == [None] * 3
    reason, retry = ratelimit.check("1.2.3.4", now=103)
    assert reason == "too many messages" and 0 < retry <= 600
    assert ratelimit.check("5.6.7.8", now=103) is None          # other visitors unaffected


def test_window_slides(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_IP", "1")
    assert ratelimit.check("v", now=0) is None
    assert ratelimit.check("v", now=599) is not None
    assert ratelimit.check("v", now=601) is None                 # 10 minutes later: allowed again


def test_daily_cap_across_visitors(monkeypatch):
    monkeypatch.setenv("DAILY_REQUEST_CAP", "2")
    t = 86400 * 100 + 10
    assert ratelimit.check("a", now=t) is None and ratelimit.check("b", now=t) is None
    assert ratelimit.check("c", now=t)[0] == "daily demo limit reached"
    assert ratelimit.check("c", now=t + 86400) is None           # next day


def test_client_ip_trusts_only_the_proxy_appended_entry():
    """A client can PREPEND fake addresses; Azure's proxy APPENDS the one it saw."""
    spoofed = SimpleNamespace(headers={"x-forwarded-for": "6.6.6.6, 203.0.113.9"},
                              client=SimpleNamespace(host="10.0.0.1"))
    assert ratelimit.client_ip(spoofed) == "203.0.113.9"
    direct = SimpleNamespace(headers={}, client=SimpleNamespace(host="127.0.0.1"))
    assert ratelimit.client_ip(direct) == "127.0.0.1"


def test_api_returns_429_with_retry_after(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_IP", "1")
    monkeypatch.setattr(main, "route_message", lambda m: {"tool": None, "router": "fake",
                                                          "confidence": None})
    monkeypatch.setattr(main, "rag_answer", lambda q, **kw: ("ok", [], []))
    c = TestClient(main.app)
    body = {"message": "hi", "session_id": "s"}
    auth = {"Authorization": "Bearer demo-asha"}
    assert c.post("/v1/chat", json=body, headers=auth).status_code == 200
    r = c.post("/v1/chat", json=body, headers=auth)
    assert r.status_code == 429 and int(r.headers["Retry-After"]) > 0


@pytest.mark.parametrize("flag, status", [("", 404), ("true", 200)])
def test_demo_reset_only_in_demo_mode(monkeypatch, flag, status):
    monkeypatch.setenv("DEMO_MODE", flag)
    backend.BOOKINGS["ACX456"]["status"] = "cancelled"
    r = TestClient(main.app).post("/v1/demo/reset", headers={"Authorization": "Bearer demo-asha"})
    assert r.status_code == status
    expected = "confirmed" if status == 200 else "cancelled"
    assert backend.BOOKINGS["ACX456"]["status"] == expected


def test_demo_reset_needs_sign_in(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    assert TestClient(main.app).post("/v1/demo/reset").status_code == 401
