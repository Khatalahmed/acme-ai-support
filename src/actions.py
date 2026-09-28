"""Tool layer: the safety boundary between every caller (API, agent, MCP) and the airline backend.

Callers never touch backend.py directly. This module enforces, for everyone:
  - ownership:     a user only sees/changes their own bookings; someone else's PNR
                   looks exactly like a PNR that doesn't exist (no enumeration)
  - entitlement:   disruption options come from policy.py (code), never from a model, and are
                   re-computed at execution time (the last seat may have gone)
  - confirmation:  changes are PROPOSED, stored server-side as pending, and run only when the
                   same user confirms in the same session
  - expiry:        a pending action dies after CONFIRM_TTL
  - idempotency:   pending -> executing is one conditional UPDATE, so a double-clicked or
                   replayed "yes" executes at most once
  - audit:         every proposal, decision, execution, escalation and denied access is logged

Actions:  cancel_ticket (voluntary cancel) · resolve_disruption (apply one policy option)
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
import policy  # noqa: E402
from langfuse import observe  # noqa: E402

CONFIRM_TTL = timedelta(minutes=5)
VERB = {"cancel_ticket": "cancel", "resolve_disruption": "resolve"}  # audit event prefixes

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
    result     TEXT,
    params     TEXT
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
CREATE TABLE IF NOT EXISTS escalations (
    ref        TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    session_id TEXT NOT NULL,
    pnr        TEXT,
    reason     TEXT NOT NULL,
    summary    TEXT NOT NULL,
    status     TEXT NOT NULL,
    created_at TEXT NOT NULL
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
        columns = {r["name"] for r in conn.execute("PRAGMA table_info(pending_actions)")}
        if "params" not in columns:  # databases created before disruption actions existed
            conn.execute("ALTER TABLE pending_actions ADD COLUMN params TEXT")
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
    booking = backend.BOOKINGS.get(pnr.upper())
    return booking if booking and booking["owner"] == user_id else None


def _denied(user_id, session_id, pnr, tool):
    audit("access_denied", user_id, session_id, pnr=pnr, tool=tool)
    return {"ok": False, "reason": "not_found", "error": f"No booking found for PNR {pnr}"}


def _entitlements(booking):
    flight_no = booking["flight"]
    return policy.entitlements(booking, backend.FLIGHTS[flight_no],
                               backend.find_alternatives(flight_no))


# ---------------------------------------------------------------- read tools

@observe(name="actions.flight_status", as_type="tool")
def flight_status(user_id, session_id, pnr):
    pnr = pnr.upper()
    if not _owned_booking(user_id, pnr):
        return _denied(user_id, session_id, pnr, "get_flight_status")
    return backend.get_flight_status(pnr)


def my_bookings(user_id):
    """The caller's own bookings (customer-safe view) - never anyone else's."""
    return [backend.get_flight_status(pnr) for pnr, b in backend.BOOKINGS.items()
            if b["owner"] == user_id]


@observe(name="actions.disruption_options", as_type="tool")
def disruption_options(user_id, session_id, pnr):
    """What the passenger is owed for this booking, computed by policy.py."""
    pnr = pnr.upper()
    booking = _owned_booking(user_id, pnr)
    if not booking:
        return _denied(user_id, session_id, pnr, "disruption_options")
    return {"ok": True, "pnr": pnr, "booking": backend.get_flight_status(pnr),
            **_entitlements(booking)}


# ---------------------------------------------------------------- proposing changes

def _validate(action, booking, params):
    """None if the action is allowed right now, else a failure reason. Runs at proposal AND
    again at execution, because the world can change in between."""
    if booking["status"] == "cancelled":
        return "already_cancelled"
    if action == "cancel_ticket":
        return None
    if action == "resolve_disruption":
        offered = {o["id"] for o in _entitlements(booking)["options"]}
        return None if params.get("option_id") in offered else "option_unavailable"
    return "unknown_action"


@observe(name="actions.propose", as_type="tool")
def propose(user_id, session_id, pnr, action, params=None, **context):
    """Store a pending change; nothing is changed here. context: router, confidence."""
    pnr, params = pnr.upper(), params or {}
    booking = _owned_booking(user_id, pnr)
    if not booking:
        return _denied(user_id, session_id, pnr, action)
    reason = _validate(action, booking, params)
    if reason:
        return {"ok": False, "reason": reason, "pnr": pnr}

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
                     "status, created_at, expires_at, params) VALUES (?, ?, ?, ?, ?, 'pending', "
                     "?, ?, ?)",
                     (action_id, user_id, session_id, action, pnr, created.isoformat(),
                      expires.isoformat(), json.dumps(params)))
        conn.execute("COMMIT")
    audit(f"{VERB[action]}_proposed", user_id, session_id, action_id, pnr, **params, **context)
    return {"ok": True, "action_id": action_id, "action": action, "pnr": pnr, "params": params,
            "expires_at": expires.isoformat(), "booking": backend.get_flight_status(pnr)}


def propose_cancel(user_id, session_id, pnr, **context):
    return propose(user_id, session_id, pnr, "cancel_ticket", **context)


def propose_option(user_id, session_id, pnr, option_id, **context):
    return propose(user_id, session_id, pnr, "resolve_disruption", {"option_id": option_id},
                   **context)


def pending_action(user_id, session_id):
    """The caller's open pending action in this session, or None."""
    with _db() as conn:
        row = conn.execute("SELECT * FROM pending_actions WHERE user_id = ? AND session_id = ? "
                           "AND status = 'pending' ORDER BY created_at DESC LIMIT 1",
                           (user_id, session_id)).fetchone()
    if not row:
        return None
    action = dict(row)
    action["params"] = json.loads(action["params"] or "{}")
    return action


def _transition(action_id, from_status, to_status, result=None):
    """Conditional state change; True only for the caller that actually moved it."""
    with _db() as conn:
        cur = conn.execute("UPDATE pending_actions SET status = ?, result = COALESCE(?, result) "
                           "WHERE action_id = ? AND status = ?",
                           (to_status, json.dumps(result) if result else None,
                            action_id, from_status))
        return cur.rowcount == 1


# ---------------------------------------------------------------- executing

def _execute(action, pnr, params, booking):
    if action == "cancel_ticket":
        return backend.cancel_ticket(pnr)
    option = next(o for o in _entitlements(booking)["options"] if o["id"] == params["option_id"])
    if option["kind"] == "rebook":
        return backend.rebook(pnr, option["flight"], voucher=option["voucher"])
    if option["kind"] == "refund":
        return backend.refund(pnr, option["amount"])
    if option["kind"] == "voucher":
        return backend.issue_voucher(pnr, option["amount"])
    raise ValueError(f"unknown option kind {option['kind']!r}")


@observe(name="actions.confirm", as_type="tool")
def confirm(user_id, session_id):
    """Run the caller's pending action. Every check happens here, at execution time."""
    action = pending_action(user_id, session_id)
    if not action:
        return {"ok": False, "reason": "nothing_pending"}
    aid, pnr, kind, params = action["action_id"], action["pnr"], action["action"], action["params"]
    verb = VERB[kind]

    if now() >= datetime.fromisoformat(action["expires_at"]):
        if _transition(aid, "pending", "expired"):
            audit(f"{verb}_expired", user_id, session_id, aid, pnr)
        return {"ok": False, "reason": "expired", "pnr": pnr, "action": kind}

    # Claim it. If two "yes" requests race, exactly one gets rowcount == 1.
    if not _transition(aid, "pending", "executing"):
        return {"ok": False, "reason": "nothing_pending"}

    # Re-read and re-validate: the booking, the seats or the ownership may have changed.
    booking = _owned_booking(user_id, pnr)
    reason = "not_found" if not booking else _validate(kind, booking, params)
    if reason:
        _transition(aid, "executing", "failed", {"reason": reason})
        audit(f"{verb}_failed", user_id, session_id, aid, pnr, reason=reason)
        return {"ok": False, "reason": reason, "pnr": pnr, "action": kind}

    result = _execute(kind, pnr, params, booking)
    status = "executed" if result.get("ok") else "failed"
    _transition(aid, "executing", status, result)
    audit(f"{verb}_{status}", user_id, session_id, aid, pnr, result=result)
    if not result.get("ok"):
        return {"ok": False, "reason": "backend_rejected", "pnr": pnr, "action": kind,
                "result": result}
    return {"ok": True, "action_id": aid, "action": kind, "pnr": pnr, "result": result}


@observe(name="actions.decline", as_type="tool")
def decline(user_id, session_id):
    """Customer said no (or moved on): close the pending action without running it."""
    action = pending_action(user_id, session_id)
    if not action:
        return {"ok": False, "reason": "nothing_pending"}
    aid, pnr = action["action_id"], action["pnr"]
    expired = now() >= datetime.fromisoformat(action["expires_at"])
    status = "expired" if expired else "declined"
    if _transition(aid, "pending", status):
        audit(f"{VERB[action['action']]}_{status}", user_id, session_id, aid, pnr)
    return {"ok": True, "pnr": pnr, "status": status}


# ---------------------------------------------------------------- human handoff

@observe(name="actions.escalate", as_type="tool")
def escalate(user_id, session_id, pnr, reason, summary):
    """Hand the case to a human agent. Not irreversible for the passenger, so no confirmation."""
    if pnr and not _owned_booking(user_id, pnr):
        pnr = None  # never attach someone else's booking to a ticket
    ref = f"ESC-{uuid.uuid4().hex[:6].upper()}"
    with _db() as conn:
        conn.execute("INSERT INTO escalations (ref, user_id, session_id, pnr, reason, summary, "
                     "status, created_at) VALUES (?, ?, ?, ?, ?, ?, 'open', ?)",
                     (ref, user_id, session_id, pnr, reason, summary, now().isoformat()))
    audit("escalated", user_id, session_id, pnr=pnr, ref=ref, reason=reason)
    return {"ok": True, "ref": ref, "reason": reason}


def escalations(user_id=None):
    with _db() as conn:
        sql, args = "SELECT * FROM escalations", []
        if user_id:
            sql, args = sql + " WHERE user_id = ?", [user_id]
        return [dict(r) for r in conn.execute(sql + " ORDER BY created_at", args)]
