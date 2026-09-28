"""Live aggregates for the public /insights page - numbers only, never message text.

Every chat request adds one row to request_log: its route, which router decided, timings and
token counts. No message, no user id, no session id: the page is public, so what it can show
is limited by what is stored. Actions, escalations and blocked access come from the audit log
the tool layer already writes; router agreement comes from shadow mode's log.

The container scales to zero and its SQLite file lives inside it, so these are "since the app
last started" numbers. The measured evaluation results (data/insights/measured.json) are the
long-term record.
"""

import statistics
from collections import Counter
from datetime import datetime, timezone

import actions
import shadow

SCHEMA = """
CREATE TABLE IF NOT EXISTS request_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ts               TEXT NOT NULL,
    route            TEXT NOT NULL,
    router           TEXT NOT NULL,
    confidence       REAL,
    latency_ms       INTEGER NOT NULL,
    router_ms        INTEGER,
    llm_calls        INTEGER NOT NULL,
    llm_ms           INTEGER NOT NULL,
    tokens           INTEGER NOT NULL,
    reasoning_tokens INTEGER NOT NULL
);
"""

STARTED = datetime.now(timezone.utc)

# Route -> the family a chart shows (the raw routes are too fine-grained for a reader).
FAMILIES = [("rag", "Policy answer"), ("tool:get_flight_status", "Booking status"),
            ("agent:", "Disruption agent"), ("confirm:", "Confirmations"),
            ("tool:", "Confirmations"), ("escalated", "Handed to a person"),
            ("clarify", "Clarifying question")]


def family(route):
    return next((name for prefix, name in FAMILIES if route.startswith(prefix)), "Other")


def record(route, decision, latency_ms, calls):
    """One row per chat request. Never raises: metrics must not break a reply."""
    try:
        with actions._db() as conn:
            conn.executescript(SCHEMA)
            conn.execute(
                "INSERT INTO request_log (ts, route, router, confidence, latency_ms, router_ms,"
                " llm_calls, llm_ms, tokens, reasoning_tokens) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (actions.now().isoformat(), route, decision.get("router", "?"),
                 decision.get("confidence"), latency_ms, decision.get("router_ms"), len(calls),
                 sum(c["ms"] for c in calls),
                 sum((c["input_tokens"] or 0) + (c["output_tokens"] or 0) for c in calls),
                 sum(c["reasoning_tokens"] or 0 for c in calls)))
    except Exception as e:
        print(f"[insights] could not record a request: {e!r}")


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * p))]


def summary():
    with actions._db() as conn:
        conn.executescript(SCHEMA)
        rows = [dict(r) for r in conn.execute("SELECT * FROM request_log ORDER BY id")]
        events = Counter(r["event"] for r in conn.execute(
            "SELECT event FROM audit_log WHERE event != 'demo_reset'"))

    by_family = {}
    for r in rows:
        by_family.setdefault(family(r["route"]), []).append(r["latency_ms"])
    routers = Counter(r["router"] for r in rows if r["router"] in ("jev", "jev->llm", "llm"))
    confidences = [r["confidence"] for r in rows if r["router"] == "jev" and r["confidence"]]
    shadowed = shadow.summary()

    def count(suffix):
        return sum(n for e, n in events.items() if e.endswith(suffix))

    return {
        "since": STARTED.isoformat(timespec="seconds"),
        "requests": len(rows),
        "routes": [{"family": f, "count": len(v), "p50_ms": percentile(v, .5),
                    "p95_ms": percentile(v, .95)}
                   for f, v in sorted(by_family.items(), key=lambda kv: -len(kv[1]))],
        "latency": {"p50_ms": percentile([r["latency_ms"] for r in rows], .5),
                    "p95_ms": percentile([r["latency_ms"] for r in rows], .95)},
        "no_model_replies": sum(1 for r in rows if r["llm_calls"] == 0),
        "tokens": sum(r["tokens"] for r in rows),
        "reasoning_tokens": sum(r["reasoning_tokens"] for r in rows),
        "routers": dict(routers),
        "jev_median_confidence": statistics.median(confidences) if confidences else None,
        "router_ms_p50": percentile([r["router_ms"] for r in rows if r["router_ms"]], .5),
        "actions": {"proposed": count("_proposed"), "executed": count("_executed"),
                    "declined": count("_declined"), "expired": count("_expired"),
                    "failed": count("_failed")},
        "escalations": events.get("escalated", 0),
        "blocked_access": events.get("access_denied", 0),
        "shadow": {"compared": shadowed["compared"], "agree": shadowed["agree"]},
    }
