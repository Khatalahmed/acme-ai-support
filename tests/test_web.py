"""The demo page and its bookings endpoint."""

import re

from fastapi.testclient import TestClient

import backend
from src.api import main

client = TestClient(main.app)


def test_chat_page_is_served():
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert 'role="log"' in r.text and 'id="form"' in r.text


def test_page_warns_messages_are_logged():
    """Messages go to Langfuse on the public demo, so visitors are told before they type."""
    assert "don't enter personal information" in client.get("/").text


def test_page_never_inserts_untrusted_text_as_html():
    """XSS guard: messages and replies go in via textContent, never innerHTML."""
    page = client.get("/").text
    # actual use (an assignment or call), not the word in a comment
    assert not re.search(r"\.(innerHTML|outerHTML)\s*\+?=|insertAdjacentHTML\(|document\.write\(",
                         page)
    assert "textContent" in page


def test_bookings_are_only_your_own():
    asha = client.get("/v1/bookings", headers={"Authorization": "Bearer demo-asha"}).json()
    ravi = client.get("/v1/bookings", headers={"Authorization": "Bearer demo-ravi"}).json()
    owned = lambda user: {p for p, b in backend.BOOKINGS.items() if b["owner"] == user}  # noqa: E731
    assert {b["pnr"] for b in asha} == owned("asha")
    assert {b["pnr"] for b in ravi} == owned("ravi")
    assert not {b["pnr"] for b in asha} & {b["pnr"] for b in ravi}
    assert set(asha[0]) == {"pnr", "flight", "route", "departure", "status"}   # no owner/fare


def test_bookings_need_a_token():
    assert client.get("/v1/bookings").status_code == 401
