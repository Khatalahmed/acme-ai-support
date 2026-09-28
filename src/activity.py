"""'My activity' and 'My requests' on the site: the audit log and hand-off tickets, readable.

The tool layer already writes an append-only audit log and an escalations table; this turns the
signed-in customer's own rows into plain sentences, written by code from the recorded facts, so
the safety design is something a visitor can SEE: every proposal, every "yes" and "no", every
refused request and every hand-off to a person.

Only the caller's own rows, ever. Raw messages are not repeated back (an escalation's summary
holds the customer's words; the page shows the reason, not the text).
"""

import json

import actions

KIND = {"proposed": "pending", "executed": "done", "declined": "stopped", "expired": "stopped",
        "failed": "stopped", "access_denied": "blocked", "escalated": "handoff",
        "demo_reset": "info"}

REASONS = {"requested_human": "You asked to speak to a person",
           "angry": "Flagged for priority help",
           "demands_exception": "Needs a decision beyond standard policy"}


def _option(option_id):
    if not option_id:
        return "an option"
    if option_id.startswith("rebook:"):
        return f"rebooking onto {option_id.split(':', 1)[1]}"
    return {"refund": "a full refund", "voucher": "a travel voucher"}.get(option_id, option_id)


def _done(verb, pnr, result):
    if verb == "cancel":
        amount = result.get("refund_amount")
        return (f"Cancelled {pnr} - Rs {amount:,} refund to your original payment method"
                if amount else f"Cancelled {pnr} - {result.get('note', 'no refund due')}")
    if "rebooked_from" in result:
        extra = f", plus a Rs {result['voucher_amount']:,} voucher" if result.get("voucher_amount") else ""
        return f"Rebooked {pnr} onto {result.get('flight')}{extra}"
    if "refund_amount" in result:
        return f"Refunded Rs {result['refund_amount']:,} for {pnr}"
    if "voucher_amount" in result:
        return f"Rs {result['voucher_amount']:,} travel voucher added to {pnr}"
    return f"Change to {pnr} completed"


def describe(row):
    """One audit row -> {"kind", "text"} for the page, or None for rows a customer needn't see."""
    event, pnr = row["event"], row["pnr"]
    detail = json.loads(row["detail"]) if row["detail"] else {}
    if event == "access_denied":
        return {"kind": "blocked",
                "text": f"A request about {pnr} was refused - it isn't on your account"}
    if event == "escalated":
        return {"kind": "handoff", "text": f"Passed to our team ({detail.get('ref')})"}
    if event == "demo_reset":
        return {"kind": "info", "text": "Demo data was reset"}
    verb, _, stage = event.partition("_")
    if verb not in ("cancel", "resolve") or stage not in KIND:
        return None
    what = "cancellation" if verb == "cancel" else _option(detail.get("option_id"))
    text = {
        "proposed": (f"You asked to cancel {pnr} - waiting for your yes" if verb == "cancel"
                     else f"You chose {what} for {pnr} - waiting for your yes"),
        "executed": _done(verb, pnr, detail.get("result") or {}),
        "declined": f"You said no - nothing changed on {pnr}",
        "expired": f"Not confirmed within 5 minutes - nothing changed on {pnr}",
        "failed": (f"Couldn't go ahead ({str(detail.get('reason', 'unavailable')).replace('_', ' ')})"
                   f" - nothing changed on {pnr}"),
    }[stage]
    return {"kind": KIND[stage], "text": text}


def for_user(user_id, limit=30):
    """The customer's recent activity (newest first) and their support requests."""
    events = []
    for row in reversed(actions.audit_trail(user_id=user_id)):
        item = describe(row)
        if item:
            events.append({"ts": row["ts"], "pnr": row["pnr"], **item})
        if len(events) >= limit:
            break
    requests = [{"ref": e["ref"], "pnr": e["pnr"], "status": e["status"],
                 "reason": REASONS.get(e["reason"], e["reason"].replace("_", " ")),
                 "created_at": e["created_at"]}
                for e in reversed(actions.escalations(user_id))]
    return {"activity": events, "requests": requests}
