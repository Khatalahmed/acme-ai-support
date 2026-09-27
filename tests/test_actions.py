"""The tool layer (src/actions.py): each test is an attack or accident that must be impossible."""

import threading

import actions
import backend
from conftest import events

ASHA, RAVI = "asha", "ravi"  # Asha owns ACX123 + ACX456; Ravi owns ACX789


# ---------------------------------------------------------------- ownership (BOLA / IDOR)

def test_owner_can_read_own_booking():
    assert actions.flight_status(ASHA, "s1", "acx123")["route"] == "Pune -> Delhi"


def test_other_users_booking_looks_like_it_does_not_exist():
    theirs = actions.flight_status(ASHA, "s1", "ACX789")       # exists, Ravi's
    missing = actions.flight_status(ASHA, "s1", "ACX999")      # doesn't exist
    assert theirs["reason"] == missing["reason"] == "not_found"
    assert "route" not in theirs                               # nothing leaked
    assert events(user_id=ASHA) == ["access_denied", "access_denied"]


def test_cannot_propose_cancelling_someone_elses_booking():
    out = actions.propose_cancel(ASHA, "s1", "ACX789")
    assert out == {"ok": False, "reason": "not_found", "error": "No booking found for PNR ACX789"}
    assert actions.pending_action(ASHA, "s1") is None
    assert backend.FLIGHTS["ACX789"]["status"] == "Cancelled by airline"


# ---------------------------------------------------------------- confirmation

def test_proposing_does_not_cancel():
    out = actions.propose_cancel(ASHA, "s1", "ACX456", router="jev", confidence=0.96)
    assert out["ok"] and out["action_id"].startswith("act_")
    assert backend.FLIGHTS["ACX456"]["status"] == "On time"
    assert events() == ["cancel_proposed"]


def test_confirm_cancels_exactly_once():
    actions.propose_cancel(ASHA, "s1", "ACX123")
    first = actions.confirm(ASHA, "s1")
    replay = actions.confirm(ASHA, "s1")                       # replayed / double-clicked "yes"
    assert first["ok"] and first["result"]["refund_amount"] == 6500
    assert replay == {"ok": False, "reason": "nothing_pending"}
    assert events().count("cancel_executed") == 1


def test_confirm_only_in_same_session_and_user():
    actions.propose_cancel(ASHA, "s1", "ACX456")
    assert actions.confirm(ASHA, "other-session")["reason"] == "nothing_pending"
    assert actions.confirm(RAVI, "s1")["reason"] == "nothing_pending"
    assert backend.FLIGHTS["ACX456"]["status"] == "On time"


def test_decline_closes_without_cancelling():
    actions.propose_cancel(ASHA, "s1", "ACX456")
    assert actions.decline(ASHA, "s1") == {"ok": True, "pnr": "ACX456", "status": "declined"}
    assert actions.confirm(ASHA, "s1")["reason"] == "nothing_pending"
    assert backend.FLIGHTS["ACX456"]["status"] == "On time"
    assert events() == ["cancel_proposed", "cancel_declined"]


def test_newer_request_supersedes_older():
    actions.propose_cancel(ASHA, "s1", "ACX123")
    actions.propose_cancel(ASHA, "s1", "ACX456")               # changed their mind
    assert actions.confirm(ASHA, "s1")["pnr"] == "ACX456"
    assert backend.FLIGHTS["ACX123"]["status"] == "Delayed by 5 hours"


# ---------------------------------------------------------------- expiry

def test_confirmation_expires(clock):
    actions.propose_cancel(ASHA, "s1", "ACX456")
    clock.advance(minutes=5, seconds=1)
    assert actions.confirm(ASHA, "s1") == {"ok": False, "reason": "expired", "pnr": "ACX456"}
    assert backend.FLIGHTS["ACX456"]["status"] == "On time"
    assert actions.confirm(ASHA, "s1")["reason"] == "nothing_pending"  # expired stays expired
    assert events() == ["cancel_proposed", "cancel_expired"]


def test_confirmation_just_inside_window_works(clock):
    actions.propose_cancel(ASHA, "s1", "ACX456")
    clock.advance(minutes=4, seconds=59)
    assert actions.confirm(ASHA, "s1")["ok"]


# ---------------------------------------------------------------- execution-time checks

def test_booking_changed_between_question_and_yes():
    actions.propose_cancel(ASHA, "s1", "ACX123")
    backend.cancel_ticket("ACX123")                            # cancelled via another channel
    assert actions.confirm(ASHA, "s1")["reason"] == "already_cancelled"
    assert events()[-1] == "cancel_failed"


def test_already_cancelled_booking_is_not_proposed():
    backend.cancel_ticket("ACX123")
    assert actions.propose_cancel(ASHA, "s1", "ACX123")["reason"] == "already_cancelled"


def test_backend_cancel_is_idempotent():
    assert backend.cancel_ticket("ACX123")["refund_amount"] == 6500
    again = backend.cancel_ticket("ACX123")
    assert again["already_cancelled"] and again["refund_amount"] == 0


# ---------------------------------------------------------------- concurrency

def test_simultaneous_yes_executes_once():
    """Two 'yes' requests racing (double-click, client retry): exactly one may win."""
    actions.propose_cancel(ASHA, "s1", "ACX123")
    start, results = threading.Barrier(8), []

    def click():
        start.wait()
        results.append(actions.confirm(ASHA, "s1"))

    threads = [threading.Thread(target=click) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(r["ok"] for r in results) == 1
    assert events().count("cancel_executed") == 1
    # Losers must be stopped at the claim, not later by the re-read check: if they reached
    # execution, only the next layer of defence was saving us.
    assert all(r["reason"] == "nothing_pending" for r in results if not r["ok"])
    assert "cancel_failed" not in events()


def test_claim_is_compare_and_set():
    """The pending -> executing step succeeds for exactly one caller."""
    action_id = actions.propose_cancel(ASHA, "s1", "ACX123")["action_id"]
    assert actions._transition(action_id, "pending", "executing") is True
    assert actions._transition(action_id, "pending", "executing") is False
