"""The "How I decided" panel under each chat reply - built by code from what actually happened.

Every line comes from the request's real facts: the router's decision and timing, the route the
API took, the sources it used, the pending action it created, and each model call's measured
time and token counts (llm_backend.calls). Nothing is generated or estimated, so the panel can
be trusted as a debugging view - and it shows visitors what a black-box chat hides.
"""

from router import RISK_THRESHOLD

INTENTS = {"get_flight_status": "a booking status check", "cancel_ticket": "a cancel request",
           "disruption_help": "help with a disrupted flight", "human_agent": "a request for a person",
           "ask_pnr": "a booking question without a PNR", None: "a policy question"}

SAFETY = {
    "rag": ["Answered only from the retrieved policy text"],
    "tool:get_flight_status": ["Ownership checked: only your own bookings are visible",
                               "Policy section looked up by name from the booking's state"],
    "confirm:cancel_ticket": ["Nothing changes until you reply YES",
                              "The confirmation is stored on the server and expires in 5 minutes"],
    "tool:cancel_ticket": ["Ran only after your YES", "Re-checked at the moment of execution",
                           "Recorded in the audit log"],
    "tool:resolve_disruption": ["Ran only after your YES", "Re-checked at the moment of execution",
                                "Recorded in the audit log"],
    "agent:presented": ["Options computed by policy code - the model can't add or change one",
                        "Choosing an option changes nothing yet"],
    "agent:proposed": ["Nothing changes until you reply YES"],
    "agent:resolved": ["Written from the booking record - no model call"],
    "agent:explained": ["Entitlement decided by policy code; the model only explains it"],
    "agent:not_found": ["Someone else's booking and a missing one get the same answer"],
    "escalated": ["Handed to a person with a reference", "Nothing on the booking was changed"],
    "agent:escalated": ["Handed to a person with the policy entitlement attached",
                        "Nothing on the booking was changed"],
}
NOTHING_CHANGED = ["Nothing was changed"]


def _understand(decision, route):
    router, conf = decision.get("router"), decision.get("confidence")
    if router == "confirmation":
        return "Your message answered a pending confirmation"
    if router == "agent":
        return "Your message answered the disruption agent's options"
    intent = INTENTS.get(decision.get("tool"), "a policy question")
    if router == "jev":
        pct = f" ({conf:.0%} confident)" if conf is not None else ""
        return f"Jev classifier read this as {intent}{pct}"
    if router == "jev->llm":
        unsure = (decision.get("unsure_jev") or {}).get("confidence")
        why = f"Jev was unsure ({unsure:.0%})" if unsure is not None else "Jev was unavailable"
        return f"{why}, so the LLM router decided: {intent}"
    return f"LLM router read this as {intent}"


def _act(route, sources, pending):
    backend = [s.removeprefix("backend:") for s in sources if s.startswith("backend:")]
    policy = [s for s in sources if not s.startswith("backend:")]
    if route == "rag":
        return f"Searched the policy documents: {len(policy)} sections used"
    if pending:
        return f"Proposed {pending['action'].replace('_', ' ')} on {pending['pnr']} - waiting for YES"
    if route.startswith("agent:"):
        return f"Disruption agent step: {route.split(':', 1)[1].replace('_', ' ')}"
    if backend:
        return "Called " + ", ".join(backend)
    if route == "escalated":
        return "Created a hand-off ticket for a person"
    if route.startswith("confirm:"):
        return f"Confirmation {route.split(':', 1)[1].replace('_', ' ')}"
    return "Asked a clarifying question instead of guessing"


def _risk(decision):
    risk = decision.get("risk") or {}
    flagged = [k.replace("_", " ") for k, v in risk.items() if v >= RISK_THRESHOLD]
    return {"angry": risk.get("angry"), "demands_exception": risk.get("demands_exception"),
            "flagged": flagged} if risk else None


def build(route, decision, sources, pending, calls):
    steps = [{"step": "Understand", "detail": _understand(decision, route),
              "ms": decision.get("router_ms")},
             {"step": "Act", "detail": _act(route, sources, pending), "ms": None}]
    for c in calls:
        tokens = (c["input_tokens"] or 0) + (c["output_tokens"] or 0)
        detail = f"Model call {c['name']}: {tokens:,} tokens"
        if c.get("reasoning_tokens"):
            detail += f" ({c['reasoning_tokens']:,} spent reasoning)"
        steps.append({"step": "Write", "detail": detail, "ms": c["ms"]})
    if not calls:
        steps.append({"step": "Write", "detail": "Written by code - no model call", "ms": 0})

    safety = SAFETY.get(route) or (NOTHING_CHANGED if route.startswith("confirm:") else [])
    return {"router": decision.get("router"), "confidence": decision.get("confidence"),
            "risk": _risk(decision), "steps": steps, "safety": safety,
            "policy": [s for s in sources if not s.startswith("backend:")],
            "model_calls": len(calls)}
