"""PNR extraction and yes/no detection - the deterministic guards around the models."""

import pytest

from router import find_pnr
from src.api.main import affirmative, declines, valid_pnr


@pytest.mark.parametrize("message, expected", [
    ("yo is my flight still happening or what, pnr is acx 789", "ACX789"),
    ("PNR: ACX-789", "ACX789"),
    ("booking no. acx 456 please check", "ACX456"),
    ("cancel acx789", "ACX789"),
    ("I paid for 500 rupees extra", None),          # 3-letter word + 3 digits is not a PNR
    ("I paid for 500 rupees, pnr ACX123", "ACX123"),
    ("My flight number is AI202", None),             # flight number, not a PNR
    ("PNR is 123ACX", None),
    ("the gate is B 12 and pnr is xyz 99", None),
])
def test_find_pnr(message, expected):
    assert find_pnr(message) == expected


@pytest.mark.parametrize("message, expected", [
    ("yes", True), ("YES!!", True), ("Yes, cancel it", True), ("haan ji", True),
    ("ok cancel", True),
    ("no", False), ("yes but don't", False), ("not yet", False), ("wait", False),
    ("", False), ("sure, wait actually", False), ("what is the baggage limit?", False),
])
def test_affirmative(message, expected):
    assert affirmative(message) is expected


@pytest.mark.parametrize("message, expected", [
    ("no", True), ("No, keep it", True), ("nahi", True), ("keep my booking", True),
    ("yes", False), ("what is the baggage limit?", False),
])
def test_declines(message, expected):
    assert declines(message) is expected


@pytest.mark.parametrize("pnr, ok", [("ACX123", True), ("acx123", True), ("AC1234", False),
                                     ("ACX12", False), ("", False), (None, False)])
def test_valid_pnr(pnr, ok):
    assert bool(valid_pnr(pnr)) is ok
