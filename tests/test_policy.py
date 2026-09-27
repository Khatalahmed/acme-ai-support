"""policy.py - every tier and every boundary of the disruption policy, with no I/O at all."""

import pytest

import policy

ALTS = [{"flight": "AC900", "departure": "2026-10-02 20:00"}]


def booking(**kw):
    return {"fare": 5000, "status": "confirmed", "resolution": None, **kw}


def flight(status="scheduled", delay_min=0, cause=None):
    return {"status": status, "delay_min": delay_min, "cause": cause}


def kinds(result):
    return [o["kind"] for o in result["options"]]


@pytest.mark.parametrize("delay, expected_options, expected_automatic", [
    (119, [], []),                                   # under 2 h: nothing
    (120, [], ["meal_voucher"]),                     # exactly 2 h: meal voucher
    (239, [], ["meal_voucher"]),
    (240, ["voucher", "rebook"], []),                # exactly 4 h: the 4-6 h tier (our decision)
    (360, ["voucher", "rebook"], []),                # exactly 6 h: still 4-6 h
    (361, ["refund", "rebook"], []),                 # over 6 h
])
def test_delay_tiers_and_boundaries(delay, expected_options, expected_automatic):
    r = policy.entitlements(booking(), flight("delayed", delay, "technical"), ALTS)
    assert kinds(r) == expected_options
    assert [a["kind"] for a in r["automatic"]] == expected_automatic


def test_amounts_come_from_policy_not_input():
    four_to_six = policy.entitlements(booking(), flight("delayed", 300, "technical"), ALTS)
    assert four_to_six["options"][0] == {"id": "voucher", "kind": "voucher", "amount": 3000}
    over_six = policy.entitlements(booking(fare=7200), flight("delayed", 400, "technical"), ALTS)
    assert over_six["options"][0]["amount"] == 7200                   # full fare refund
    assert over_six["options"][1]["voucher"] == 5000                  # rebook + Rs 5,000


@pytest.mark.parametrize("cause", ["weather", "atc"])
def test_weather_and_atc_delays_not_compensated(cause):
    r = policy.entitlements(booking(), flight("delayed", 420, cause), ALTS)
    assert r["options"] == [] and r["automatic"] == []
    assert r["policy"] == "delay-compensation-policy.md §4"


def test_airline_cancellation_rebook_or_refund():
    r = policy.entitlements(booking(), flight("cancelled", cause="technical"), ALTS)
    assert kinds(r) == ["rebook", "refund"]
    assert r["policy"] == "refund-cancellation-policy.md §4"


def test_weather_cancellation_same_choice_different_policy():
    r = policy.entitlements(booking(), flight("cancelled", cause="weather"), ALTS)
    assert kinds(r) == ["rebook", "refund"]
    assert r["policy"] == "refund-cancellation-policy.md §5"


def test_no_seats_anywhere_still_leaves_refund():
    r = policy.entitlements(booking(), flight("cancelled", cause="technical"), [])
    assert kinds(r) == ["refund"]


def test_on_time_flight_owes_nothing():
    r = policy.entitlements(booking(), flight(), ALTS)
    assert r["disruption"] is None and r["options"] == []


@pytest.mark.parametrize("state", [{"resolution": {"kind": "refund"}}, {"status": "cancelled"}])
def test_resolved_or_cancelled_booking_owes_nothing_more(state):
    r = policy.entitlements(booking(**state), flight("cancelled", cause="technical"), ALTS)
    assert r["options"] == []


def test_descriptions_are_deterministic():
    assert policy.describe({"kind": "refund", "amount": 7200}) == \
        "Full refund of Rs 7,200 to your original payment method"
    assert policy.describe({"kind": "rebook", "flight": "AC312", "departure": "2026-10-02 20:30",
                            "voucher": 5000}).endswith("plus a Rs 5,000 travel voucher")
