"""Disruption actions through the tool layer: options, rebook/refund/voucher, escalation."""

import actions
import backend
from conftest import events

ASHA, RAVI = "asha", "ravi"
# Asha: ACX123 (AC101, 5 h delay), ACX321 (2.5 h delay), ACX246 (6 h 40 m delay)
# Ravi: ACX789 (AC310 cancelled, technical), ACX654 (weather cancel), ACX987 (weather delay)


def option_ids(user, pnr):
    return [o["id"] for o in actions.disruption_options(user, "s1", pnr)["options"]]


def test_options_come_from_policy():
    assert option_ids(RAVI, "ACX789") == ["rebook:AC312", "rebook:AC314", "refund"]  # AC316 full
    assert option_ids(ASHA, "ACX123") == ["voucher", "rebook:AC103"]                 # AC105 full


def test_cannot_see_options_for_someone_elses_booking():
    assert actions.disruption_options(ASHA, "s1", "ACX789")["reason"] == "not_found"


def test_option_not_offered_cannot_be_proposed():
    """A 5 h delay gets voucher OR rebooking - 'but my friend got a refund' changes nothing."""
    out = actions.propose_option(ASHA, "s1", "ACX123", "refund")
    assert out == {"ok": False, "reason": "option_unavailable", "pnr": "ACX123"}
    assert actions.pending_action(ASHA, "s1") is None


def test_rebook_takes_a_seat_and_moves_the_booking():
    seats_before = backend.FLIGHTS["AC312"]["seats"]
    actions.propose_option(RAVI, "s1", "ACX789", "rebook:AC312")
    assert backend.BOOKINGS["ACX789"]["flight"] == "AC310"                  # nothing yet
    done = actions.confirm(RAVI, "s1")
    assert done["ok"] and done["result"]["flight"] == "AC312"
    assert backend.BOOKINGS["ACX789"]["flight"] == "AC312"
    assert backend.FLIGHTS["AC312"]["seats"] == seats_before - 1
    assert events() == ["resolve_proposed", "resolve_executed"]


def test_last_seat_taken_between_choice_and_yes():
    actions.propose_option(RAVI, "s1", "ACX789", "rebook:AC312")
    backend.FLIGHTS["AC312"]["seats"] = 0                                   # someone else booked it
    out = actions.confirm(RAVI, "s1")
    assert out["reason"] == "option_unavailable"
    assert backend.BOOKINGS["ACX789"]["flight"] == "AC310"                  # unchanged
    assert events()[-1] == "resolve_failed"


def test_disruption_can_only_be_resolved_once():
    actions.propose_option(RAVI, "s1", "ACX789", "refund")
    assert actions.confirm(RAVI, "s1")["result"]["refund_amount"] == 3200
    # a second option (or a replayed request) is no longer on offer
    assert actions.propose_option(RAVI, "s1", "ACX789", "rebook:AC312")["reason"] in (
        "option_unavailable", "already_cancelled")
    assert backend.BOOKINGS["ACX789"]["status"] == "cancelled"


def test_voucher_keeps_the_flight():
    actions.propose_option(ASHA, "s1", "ACX123", "voucher")
    assert actions.confirm(ASHA, "s1")["result"]["voucher_amount"] == 3000
    assert backend.BOOKINGS["ACX123"]["flight"] == "AC101"
    assert backend.BOOKINGS["ACX123"]["vouchers"] == [3000]


def test_long_delay_rebook_includes_voucher():
    actions.propose_option(ASHA, "s1", "ACX246", "rebook:AC704")
    result = actions.confirm(ASHA, "s1")["result"]
    assert result["flight"] == "AC704" and result["voucher_amount"] == 5000


def test_backend_writes_are_idempotent():
    assert backend.refund("ACX789", 3200)["ok"]
    assert backend.refund("ACX789", 3200)["already_resolved"]
    assert backend.issue_voucher("ACX123", 3000)["ok"]
    assert not backend.issue_voucher("ACX123", 3000)["ok"]


def test_escalation_is_recorded_and_never_attaches_someone_elses_booking():
    mine = actions.escalate(ASHA, "s1", "ACX123", "demands_exception", "Wants refund on 5 h delay")
    theirs = actions.escalate(ASHA, "s1", "ACX789", "angry", "probing")
    assert mine["ref"].startswith("ESC-")
    rows = {r["ref"]: r for r in actions.escalations(ASHA)}
    assert rows[mine["ref"]]["pnr"] == "ACX123"
    assert rows[theirs["ref"]]["pnr"] is None
    assert events(user_id=ASHA) == ["escalated", "escalated"]
