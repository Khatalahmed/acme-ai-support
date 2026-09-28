"""Status replies get the policy section the booking's state makes relevant - by lookup, not search."""

import re
from pathlib import Path

import pytest

from src.api import main
from tools.backend import get_flight_status

POLICY_DIR = Path(__file__).resolve().parent.parent / "data" / "policies"


def test_every_mapped_section_exists_in_the_policy_documents():
    """If someone renames a heading, the lookup would silently find nothing - fail here instead."""
    for source, section in main.POLICY_SECTIONS:
        text = (POLICY_DIR / source).read_text(encoding="utf-8")
        headings = re.findall(r"(?m)^## (.+?)\s*$", text)
        assert section in headings, f"{source} has no '## {section}' heading"


@pytest.mark.parametrize("pnr, expected", [
    ("ACX123", main.DELAY_TIERS),        # 5 h technical delay
    ("ACX321", main.DELAY_TIERS),        # 2.5 h technical delay
    ("ACX246", main.DELAY_TIERS),        # 6 h 40 m technical delay
    ("ACX987", main.DELAY_EXCLUSIONS),   # weather delay: not compensated
    ("ACX789", main.AIRLINE_CANCEL),     # technical cancellation
    ("ACX654", main.WEATHER_CANCEL),     # weather cancellation
    ("ACX456", None),                    # on time: no policy applies
])
def test_policy_section_from_booking_state(pnr, expected):
    assert main.policy_section(get_flight_status(pnr)) == expected


def test_on_time_flight_gets_no_policy_and_says_so(monkeypatch):
    prompts = []
    monkeypatch.setattr(main, "llm", lambda prompt, **kw: prompts.append(prompt) or "ok")
    reply, sources = main.phrase_result("get_flight_status", "ACX456",
                                        get_flight_status("ACX456"), "Is ACX456 on time?")
    assert sources == ["backend:get_flight_status(ACX456)"]      # no policy documents
    assert "operating normally, so make no policy statements" in prompts[0]


def test_llm_sees_plain_facts_not_raw_fields(monkeypatch):
    """Seen live: replies recited '(flight_status: scheduled' and 'resolution: null; vouchers: []'."""
    prompts = []
    monkeypatch.setattr(main, "llm", lambda prompt, **kw: prompts.append(prompt) or "ok")
    main.phrase_result("get_flight_status", "ACX456", get_flight_status("ACX456"), "status?")
    for raw in ["flight_status", "delay_min", "cause", "seats_available", "resolution",
                "booking_status", "null", "[]", "{", "refundable\""]:
        assert raw not in prompts[0], raw
    assert "- Status: On time" in prompts[0]
    assert "- Fare type: non-refundable" in prompts[0]


def test_status_reply_closing_is_code(monkeypatch):
    """The LLM isn't asked to offer anything, so it can't offer services that don't exist."""
    monkeypatch.setattr(main, "llm", lambda prompt, **kw: "Your flight AC205 is on time.")
    reply, _ = main.phrase_result("get_flight_status", "ACX456", get_flight_status("ACX456"), "?")
    assert reply == "Your flight AC205 is on time.\n\nIs there anything else I can help you with?"


def test_vouchers_shown_in_plain_words():
    booking = {**get_flight_status("ACX123"), "vouchers": [3000]}
    assert "- Travel vouchers on this booking: Rs 3,000" in main.customer_view(booking)
