"""Shadow routing: run the OTHER router on real traffic, silently, and record where they disagree.

    request -> primary router -> decision -> customer            (never waits for the shadow)
                        \-> background thread: shadow router -> compare outcomes
                              -> shadow_log (SQLite): every comparison
                              -> Langfuse, when on: router_agreement score on the trace, and each
                                 disagreement added to the "router-disagreements" dataset
                                 (linked to its trace) for a person to label -> benchmark v2

Rules:
  1. never slows the reply   - runs on a background thread after the decision is made
  2. never changes behaviour - its decision is only recorded; every error is caught
  3. costs are bounded       - ROUTER_SHADOW_RATE samples a fraction of traffic

Settings (.env):  ROUTER_SHADOW = llm | jev | off (default off)
                  ROUTER_SHADOW_RATE = 0..1 (default 1 = every eligible request)
"""

import contextvars
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, wait

from langfuse import get_client, observe

import actions
import router

DATASET = "router-disagreements"
SHADOW_ROUTERS = {"llm": router.route_llm, "jev": router.route_jev}

SCHEMA = """
CREATE TABLE IF NOT EXISTS shadow_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,
    trace_id        TEXT,
    user_id         TEXT,
    session_id      TEXT,
    message         TEXT NOT NULL,
    primary_router  TEXT NOT NULL,
    primary_outcome TEXT NOT NULL,
    shadow_router   TEXT NOT NULL,
    shadow_outcome  TEXT,
    agree           INTEGER,
    shadow_ms       INTEGER,
    error           TEXT
);
"""

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="shadow")
_pending = []
_dataset_ready = False


def _setting(name, default):
    return (os.environ.get(name) or default).strip().lower()


def langfuse_on():
    """True only when tracing is enabled AND keys exist (blank keys still try to export)."""
    return (_setting("LANGFUSE_TRACING_ENABLED", "true") != "false"
            and bool(os.environ.get("LANGFUSE_PUBLIC_KEY"))
            and bool(os.environ.get("LANGFUSE_SECRET_KEY")))


def maybe_run(message, decision, user_id, session_id, trace_id=None):
    """Schedule a shadow comparison if configured. Returns the shadow router's name, or None.

    Never raises and never blocks: the customer's request continues immediately.
    """
    shadow = _setting("ROUTER_SHADOW", "off")
    served_by = decision.get("router", "")
    if shadow not in SHADOW_ROUTERS:
        return None
    if served_by == "jev->llm" and decision.get("unsure_jev"):
        # Jev was unsure and the LLM decided: both opinions already exist - compare them at
        # no extra cost. These are the most interesting cases, so they're never sampled away.
        unsure = decision["unsure_jev"]
        _pending.append(_executor.submit(contextvars.copy_context().run, _run, "jev (unsure)",
                                         message, decision, user_id, session_id, trace_id,
                                         unsure))
        return "jev (unsure)"
    if served_by.endswith(shadow):
        return None                    # that router already made this decision
    if random.random() >= float(_setting("ROUTER_SHADOW_RATE", "1")):
        return None                    # not sampled
    ctx = contextvars.copy_context()   # keeps the request's trace as the shadow's parent
    _pending.append(_executor.submit(ctx.run, _run, shadow, message, decision,
                                     user_id, session_id, trace_id))
    return shadow


def drain(timeout=30):
    """Wait for scheduled shadow comparisons (tests and scripts; the API never waits)."""
    done, _ = wait(list(_pending), timeout=timeout)
    for f in done:
        _pending.remove(f)


def _fmt(o):
    return f"{o[0]} {o[1]}" if o[1] else o[0]


@observe(name="router.shadow")
def _run(shadow, message, decision, user_id, session_id, trace_id, known=None):
    """Compare the served decision with the shadow's. `known`: a shadow decision that already
    exists (Jev's unsure answer) - then no router is called."""
    try:
        t0, error, shadow_out = time.perf_counter(), None, None
        try:
            shadow_decision = known if known is not None else SHADOW_ROUTERS[shadow](message)
            shadow_out = router.outcome(shadow_decision)
        except Exception as e:  # the shadow failing must never matter to anyone
            error = repr(e)
        ms = round((time.perf_counter() - t0) * 1000)
        primary_out = router.outcome(decision)
        agree = None if error else primary_out == shadow_out

        with actions._db() as conn:
            conn.executescript(SCHEMA)
            conn.execute(
                "INSERT INTO shadow_log (ts, trace_id, user_id, session_id, message, primary_router,"
                " primary_outcome, shadow_router, shadow_outcome, agree, shadow_ms, error)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (actions.now().isoformat(), trace_id, user_id, session_id, message,
                 decision.get("router", "?"), _fmt(primary_out), shadow,
                 _fmt(shadow_out) if shadow_out else None,
                 None if agree is None else int(agree), ms, error))

        if agree is not None and trace_id and langfuse_on():
            _report(message, decision, primary_out, shadow, shadow_out, agree, trace_id)
    except Exception as e:
        print(f"[shadow] comparison failed and was ignored: {e!r}")


def _report(message, decision, primary_out, shadow, shadow_out, agree, trace_id):
    global _dataset_ready
    lf = get_client()
    lf.create_score(trace_id=trace_id, name="router_agreement", value=1 if agree else 0,
                    data_type="BOOLEAN",
                    comment=f"{decision.get('router')}: {_fmt(primary_out)} | "
                            f"{shadow}: {_fmt(shadow_out)}")
    if agree:
        return
    if not _dataset_ready:
        lf.create_dataset(name=DATASET, description=(
            "Real messages where the served router and the shadow router chose different "
            "outcomes. Label expected_output with the correct route to grow the benchmark."))
        _dataset_ready = True
    lf.create_dataset_item(
        dataset_name=DATASET, input={"message": message}, source_trace_id=trace_id,
        metadata={"served_by": decision.get("router"), "served_outcome": _fmt(primary_out),
                  "shadow_router": shadow, "shadow_outcome": _fmt(shadow_out),
                  "confidence": decision.get("confidence")})


def summary():
    """Agreement stats from shadow_log (local, works without Langfuse)."""
    with actions._db() as conn:
        conn.executescript(SCHEMA)
        rows = [dict(r) for r in conn.execute("SELECT * FROM shadow_log ORDER BY id")]
    compared = [r for r in rows if r["agree"] is not None]
    return {"compared": len(compared), "agree": sum(r["agree"] for r in compared),
            "errors": sum(1 for r in rows if r["error"]),
            "disagreements": [r for r in compared if not r["agree"]], "rows": rows}
