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
    assert set(asha[0]) == {"pnr", "flight", "route", "departure", "status",   # no owner/fare
                            "state", "outcome", "origin", "origin_city", "dest", "dest_city"}


def sidebar(user="asha"):
    rows = client.get("/v1/bookings", headers={"Authorization": f"Bearer demo-{user}"}).json()
    return {b["pnr"]: (b["state"], b["outcome"]) for b in rows}


def test_booking_states_for_the_sidebar():
    assert sidebar()["ACX456"] == ("ok", None)
    assert sidebar()["ACX123"] == ("delayed", None)
    assert sidebar("ravi")["ACX789"] == ("cancelled", None)


def test_sidebar_shows_what_was_done():
    """After an action the sidebar must visibly change - the demo video relies on it."""
    backend.refund("ACX789", 3200)
    backend.issue_voucher("ACX123", 3000)
    backend.cancel_ticket("ACX456")
    assert sidebar("ravi")["ACX789"] == ("cancelled", "Refunded Rs 3,200")
    assert sidebar()["ACX123"] == ("delayed", "Rs 3,000 voucher added")
    assert sidebar()["ACX456"] == ("cancelled", "Cancelled by you")


def test_bookings_need_a_token():
    assert client.get("/v1/bookings").status_code == 401


def test_insights_page_is_served_and_linked():
    r = client.get("/insights")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert 'href="/insights"' in client.get("/").text and 'href="/"' in r.text


def test_insights_page_never_inserts_api_text_as_html():
    page = client.get("/insights").text
    assert not re.search(r"\.(innerHTML|outerHTML)\s*\+?=|insertAdjacentHTML\(|document\.write\(",
                         page)
    assert "textContent" in page


def test_trip_cards_get_airport_codes():
    rows = client.get("/v1/bookings", headers={"Authorization": "Bearer demo-asha"}).json()
    acx123 = next(b for b in rows if b["pnr"] == "ACX123")
    assert (acx123["origin"], acx123["origin_city"], acx123["dest"]) == ("PNQ", "Pune", "DEL")


def test_shared_stylesheet_is_served_and_the_github_link_is_gone():
    assert client.get("/static/site.css").status_code == 200
    for page in ("/", "/insights"):
        html = client.get(page).text
        assert '/static/site.css' in html and "github.com" not in html


def test_favicon_served_and_linked():
    for path in ("/favicon.ico", "/static/favicon.svg"):
        r = client.get(path)
        assert r.status_code == 200 and "svg" in r.headers["content-type"]
    for page in ("/", "/insights"):
        assert "/static/favicon.svg" in client.get(page).text


def test_insights_page_has_a_loading_state():
    """Found in a screenshot: before the data arrived the page showed empty cards and a '-'."""
    page = client.get("/insights").text
    assert "Loading the measured results" in page and 'class="card measured"' in page
