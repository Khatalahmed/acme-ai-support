"""Tool layer: the safety boundary between every caller (API, agent, MCP) and the airline backend.

Callers never touch backend.py directly. This module enforces, for everyone:
  - ownership:     a user only sees/changes their own bookings; someone else's PNR
                   looks exactly like a PNR that doesn't exist (no enumeration)
  - confirmation:  irreversible actions are PROPOSED, stored server-side as pending, and run
                   only when the same user confirms in the same session
  - expiry:        a pending action dies after CONFIRM_TTL
  - idempotency:   pending -> executing is one conditional UPDATE, so a double-clicked or
                   replayed "yes" executes at most once
  - audit:         every proposal, decision, execution and denied access is logged

Pending-action lifecycle:
    pending -> executing -> executed | failed
    pending -> declined | expired | superseded
"""

import json
import os
import sqlite3
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "tools"))
import backend  # noqa: E402  (same module object the API imports)

CONFIRM_TTL = timedelta(minutes=5)

SCHEMA = """
CREATE TABLE IF NOT EXISTS pending_actions (
    action_id  TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    session_id TEXT NOT NULL,
    action     TEXT NOT NULL,
    pnr        TEXT NOT NULL,
    status     TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    result     TEXT
);
CREATE TABLE IF NOT EXISTS audit_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL,
    event      TEXT NOT NULL,
    user_id    TEXT,
    session_id TEXT,
    action_id  TEXT,
    pnr        TEXT,
    detail     TEXT
);
"""


def now():
    """Current UTC time. Tests replace this to simulate expiry."""
    return datetime.now(timezone.utc)


@contextmanager
def _db():
    path = Path(os.environ.get("ACME_DB_PATH") or ROOT / "data" / "acme.db")
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None)  # autocommit; each statement is atomic
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
    finally:
        conn.close()  # sqlite3's own `with` commits but never closes


def audit(event, user_id, session_id=None, action_id=None, pnr=None, **detail):
    """Append one row to the audit log. Nothing ever updates or deletes these rows."""
    with _db() as conn:
        conn.execute(
            "INSERT INTO audit_log (ts, event, user_id, session_id, action_id, pnr, detail) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (now().isoformat(), event, user_id, session_id, action_id, pnr,
             json.dumps(detail) if detail else None))


def audit_trail(user_id=None, pnr=None):
    """Audit rows, oldest first, optionally filtered."""
    sql, args = "SELECT * FROM audit_log WHERE 1=1", []
    if user_id:
        sql, args = sql + " AND user_id = ?", args + [user_id]
    if pnr:
        sql, args = sql + " AND pnr = ?", args + [pnr]
    with _db() as conn:
        return [dict(r) for r in conn.execute(sql + " ORDER BY id", args)]


def _owned_booking(user_id, pnr):
    """The booking if it exists AND belongs to user_id, else None - same answer either way."""
    booking = backend.FLIGHTS.get(pnr.upper())
    return booking if booking and booking["owner"] == user_id else None


# ---------------------------------------------------------------- read tools

def flight_status(user_id, session_id, pnr):
    pnr = pnr.upper()
    if not _owned_booking(user_id, pnr):
        audit("access_denied", user_id, session_id, pnr=pnr, tool="get_flight_status")
        return {"ok": False, "reason": "not_found", "error": f"No booking found for PNR {pnr}"}
    return backend.get_flight_status(pnr)


# ---------------------------------------------------------------- irreversible: cancel

def propose_cancel(user_id, session_id, pnr, **context):
    """Store a pending cancellation; nothing is cancelled here. context: router, confidence."""
    pnr = pnr.upper()
    booking = _owned_booking(user_id, pnr)
    if not booking:
        audit("access_denied", user_id, session_id, pnr=pnr, tool="cancel_ticket")
        return {"ok": False, "reason": "not_found", "error": f"No booking found for PNR {pnr}"}
    if booking["status"] == "Cancelled by passenger":
        return {"ok": False, "reason": "already_cancelled",
                "error": f"Booking {pnr} is already cancelled"}

    action_id, created = f"act_{uuid.uuid4().hex[:12]}", now()
    expires = created + CONFIRM_TTL
    with _db() as conn:
        # one transaction: supersede the old question and store the new one, or neither
        conn.execute("BEGIN IMMEDIATE")
        # one open question per session: a newer proposal replaces any older one
        conn.execute("UPDATE pending_actions SET status = 'superseded' "
                     "WHERE user_id = ? AND session_id = ? AND status = 'pending'",
                     (user_id, session_id))
        conn.execute("INSERT INTO pending_actions (action_id, user_id, session_id, action, pnr, "
                     "status, created_at, expires_at) VALUES (?, ?, ?, 'cancel_ticket', ?, "
                     "'pending', ?, ?)",
                     (action_id, user_id, session_id, pnr, created.isoformat(), expires.isoformat()))
        conn.execute("COMMIT")
    audit("cancel_proposed", user_id, session_id, action_id, pnr, **context)
    return {"ok": True, "action_id": action_id, "action": "cancel_ticket", "pnr": pnr,
            "expires_at": expires.isoformat(), "booking": backend.get_flight_status(pnr)}


def pending_action(user_id, session_id):
    """The caller's open pending action in this session, or None."""
    with _db() as conn:
        row = conn.execute("SELECT * FROM pending_actions WHERE user_id = ? AND session_id = ? "
                           "AND status = 'pending' ORDER BY created_at DESC LIMIT 1",
                           (user_id, session_id)).fetchone()
    return dict(row) if row else None


def _transition(action_id, from_status, to_status, result=None):
    """Conditional state change; True only for the caller that actually moved it."""
    with _db() as conn:
        cur = conn.execute("UPDATE pending_actions SET status = ?, result = COALESCE(?, result) "
                           "WHERE action_id = ? AND status = ?",
                           (to_status, json.dumps(result) if result else None,
                            action_id, from_status))
        return cur.rowcount == 1


def confirm(user_id, session_id):
    """Run the caller's pending action. Every check happens here, at execution time."""
    action = pending_action(user_id, session_id)
    if not action:
        return {"ok": False, "reason": "nothing_pending"}
    aid, pnr = action["action_id"], action["pnr"]

    if now() >= datetime.fromisoformat(action["expires_at"]):
        if _transition(aid, "pending", "expired"):
            audit("cancel_expired", user_id, session_id, aid, pnr)
        return {"ok": False, "reason": "expired", "pnr": pnr}

    # Claim it. If two "yes" requests race, exactly one gets rowcount == 1.
    if not _transition(aid, "pending", "executing"):
        return {"ok": False, "reason": "nothing_pending"}

    # Re-read: the booking may have changed since we asked (cancelled elsewhere, owner changed).
    booking = _owned_booking(user_id, pnr)
    if not booking or booking["status"] == "Cancelled by passenger":
        reason = "not_found" if not booking else "already_cancelled"
        _transition(aid, "executing", "failed", {"reason": reason})
        audit("cancel_failed", user_id, session_id, aid, pnr, reason=reason)
        return {"ok": False, "reason": reason, "pnr": pnr}

    result = backend.cancel_ticket(pnr)
    _transition(aid, "executing", "executed", result)
    audit("cancel_executed", user_id, session_id, aid, pnr, result=result)
    return {"ok": True, "action_id": aid, "pnr": pnr, "result": result}


def decline(user_id, session_id):
    """Customer said no (or moved on): close the pending action without running it."""
    action = pending_action(user_id, session_id)
    if not action:
        return {"ok": False, "reason": "nothing_pending"}
    aid, pnr = action["action_id"], action["pnr"]
    expired = now() >= datetime.fromisoformat(action["expires_at"])
    status = "expired" if expired else "declined"
    if _transition(aid, "pending", status):
        audit(f"cancel_{status}", user_id, session_id, aid, pnr)
    return {"ok": True, "pnr": pnr, "status": status}
