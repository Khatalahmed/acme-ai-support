"""Disruption-recovery agent: a LangGraph state machine over the tool layer.

    assess ──┬─► escalate ─────────────────────────────► END  (angry / demands an exception)
             ├─► explain ──────────────────────────────► END  (nothing to choose)
             └─► present ─► await_choice ─► propose ───► END  (pending action created)
                               │ (pauses)
                               └─ not a choice ────────► END  (moved on: API routes normally)

Rules this file follows:
  - The agent never changes a booking. It PROPOSES through actions.py; the customer's "yes"
    is handled by the same server-side confirmation as every other change.
  - Entitlements come from policy.py via actions.disruption_options - the LLM only writes a
    short, empathetic intro. The options list itself is appended by code.
  - await_choice holds the interrupt() and nothing else with side effects: LangGraph re-runs
    the interrupted node from its start when the customer replies.
  - Conversation state lives in SQLite (one thread per user+session), so a paused
    conversation survives an API restart.
"""

import os
import re
import sqlite3
from pathlib import Path
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from langfuse import get_client, observe

import actions
import policy
from llm_backend import chat

RISK_THRESHOLD = 0.7  # angry / demands_exception at or above this -> a human takes over

INTRO_PROMPT = """You are an empathetic ACME Bharat Airlines support agent.
Customer's message: {message}
Their booking: {booking}
Policy note: {note}

Write ONE or TWO short sentences acknowledging their situation. Do NOT list options, amounts,
refunds or promises of any kind - the exact options are added after your text."""

EXPLAIN_PROMPT = """You are an empathetic ACME Bharat Airlines support agent.
Customer's message: {message}
Their booking: {booking}
What they are entitled to (decided by policy - state it exactly, add nothing): {facts}

Reply in 2-4 sentences: acknowledge them, state exactly what they are entitled to, and end with
one helpful next-step question. Never offer anything not listed above."""

ORDINALS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
            "fourth": 4, "4th": 4}
CARDINALS = {"one": 1, "two": 2, "three": 3, "four": 4}


class State(TypedDict):
    user_id: str
    session_id: str
    pnr: str
    message: str
    risk: dict
    booking: dict
    options: list
    automatic: list
    note: str
    choice: str | None        # chosen option id
    outcome: str              # not_found | escalated | explained | presented | proposed | ...
    reply: str
    pending: dict | None


def llm(prompt, name="agent.llm"):
    """The only model call in the agent (tests replace it)."""
    return chat([{"role": "user", "content": prompt}], name=name).strip()


# ---------------------------------------------------------------- nodes

@observe(name="agent.assess")
def assess(state):
    """Read-only: load the booking and what policy says the passenger is owed."""
    out = actions.disruption_options(state["user_id"], state["session_id"], state["pnr"])
    if not out["ok"]:
        return {"outcome": "not_found", "reply": (
            f"I couldn't find a booking with PNR {state['pnr']} on your account. "
            "Could you re-check it?")}
    return {"booking": out["booking"], "options": out["options"],
            "automatic": out["automatic"], "note": out["note"]}


def after_assess(state):
    if state.get("outcome") == "not_found":
        return END
    risk = state["risk"]
    if max(risk.get("angry", 0), risk.get("demands_exception", 0)) >= RISK_THRESHOLD:
        return "escalate"
    return "present" if state["options"] else "explain"


@observe(name="agent.escalate")
def escalate(state):
    risk = state["risk"]
    reason = ("demands_exception" if risk.get("demands_exception", 0) >= RISK_THRESHOLD
              else "angry")
    entitled = [policy.describe(o) for o in state["options"] + state["automatic"]]
    summary = (f"Customer message: {state['message']!r}. Policy entitlement: "
               f"{'; '.join(entitled) or 'none'} ({state['note']})")
    ticket = actions.escalate(state["user_id"], state["session_id"], state["pnr"], reason, summary)
    return {"outcome": "escalated", "reply": (
        f"I'm sorry this has been so difficult. I've passed your case to a senior member of our "
        f"team (reference {ticket['ref']}) with everything you've told me, and they will follow "
        f"up with you directly. Nothing on your booking {state['pnr']} has been changed.")}


@observe(name="agent.explain")
def explain(state):
    facts = [policy.describe(a) for a in state["automatic"]] or [state["note"] or "No compensation"]
    reply = llm(EXPLAIN_PROMPT.format(message=state["message"], booking=state["booking"],
                                      facts="; ".join(facts) + f" ({state['note']})"),
                name="agent.explain.llm")
    return {"outcome": "explained", "reply": reply}


SAFE_INTRO = "I'm sorry about the disruption to your journey."
# The intro must not promise anything: money, refunds, vouchers or compensation belong only in
# the code-generated options list below it.
PROMISE = re.compile(r"(rs\.?|₹|inr|rupees?)\s*\d|\d[\d,]{2,}|refund|voucher|compensat",
                     re.IGNORECASE)


@observe(name="guardrail.intro", as_type="guardrail")
def guard_intro(text):
    """Output guardrail: drop an LLM intro that makes promises the options list doesn't."""
    replaced = not text or bool(PROMISE.search(text))
    get_client().update_current_span(metadata={"replaced": replaced})  # how often does it fire?
    return SAFE_INTRO if replaced else text


@observe(name="agent.present")
def present(state):
    intro = guard_intro(llm(INTRO_PROMPT.format(message=state["message"],
                                                booking=state["booking"], note=state["note"]),
                            name="agent.intro.llm"))
    lines = [f"{i}. {policy.describe(o)}" for i, o in enumerate(state["options"], 1)]
    reply = (f"{intro}\n\nHere is what you can choose for booking {state['pnr']}:\n"
             + "\n".join(lines) + f"\n\nReply with the option number (1-{len(lines)}).")
    return {"outcome": "presented", "reply": reply}


def parse_choice(message, options):
    """Option id from the customer's reply, or None. Pure - safe to re-run on resume."""
    text = message.lower()
    words = re.findall(r"[a-z0-9]+", text)
    in_range = range(1, len(options) + 1)
    ordinals = {ORDINALS[w] for w in words if ORDINALS.get(w) in in_range}
    numbers = {int(w) for w in words if w.isdigit() and int(w) in in_range}
    numbers |= {CARDINALS[w] for w in words if CARDINALS.get(w) in in_range}
    # "the second one": an ordinal wins over the filler "one"
    picks = ordinals or numbers
    if len(picks) == 1:
        return options[picks.pop() - 1]["id"]
    by_keyword = [o["id"] for o in options
                  if (o["kind"] == "rebook" and o["flight"].lower() in text)
                  or (o["kind"] in ("refund", "voucher") and o["kind"] in words)]
    return by_keyword[0] if len(by_keyword) == 1 else None


def await_choice(state):
    # Nothing with side effects before interrupt(): this line runs again on resume.
    answer = interrupt({"type": "choose_option", "options": [o["id"] for o in state["options"]]})
    choice = parse_choice(answer, state["options"])
    if not choice:
        return {"choice": None, "message": answer, "outcome": "moved_on", "reply": ""}
    return {"choice": choice, "message": answer}


def after_choice(state):
    return "propose" if state.get("choice") else END


@observe(name="agent.propose")
def propose(state):
    option = next(o for o in state["options"] if o["id"] == state["choice"])
    out = actions.propose_option(state["user_id"], state["session_id"], state["pnr"],
                                 option["id"], source="agent")
    if not out["ok"]:
        return {"outcome": "option_unavailable", "pending": None, "reply": (
            "Sorry - that option has just become unavailable (for example, the last seat was "
            f"taken). Nothing has been changed on booking {state['pnr']}. Would you like to see "
            "the current options?")}
    return {"outcome": "proposed", "reply": (
        f"You chose: {policy.describe(option)}. Shall I go ahead for booking {state['pnr']}? "
        "Reply YES to confirm, or NO to keep things as they are."),
        "pending": {k: out[k] for k in ("action_id", "action", "pnr", "expires_at")}
        | {"option_id": option["id"]}}


# ---------------------------------------------------------------- graph

def _build():
    g = StateGraph(State)
    for name, fn in [("assess", assess), ("escalate", escalate), ("explain", explain),
                     ("present", present), ("await_choice", await_choice), ("propose", propose)]:
        g.add_node(name, fn)
    g.add_edge(START, "assess")
    g.add_conditional_edges("assess", after_assess, ["escalate", "explain", "present", END])
    g.add_edge("escalate", END)
    g.add_edge("explain", END)
    g.add_edge("present", "await_choice")
    g.add_conditional_edges("await_choice", after_choice, ["propose", END])
    g.add_edge("propose", END)
    return g


_graphs = {}


def graph():
    """Compiled graph, checkpointed next to the actions database (one per database path)."""
    path = Path(os.environ.get("ACME_DB_PATH") or actions.ROOT / "data" / "acme.db")
    path.parent.mkdir(parents=True, exist_ok=True)
    agent_db = path.with_name(path.stem + "-agent.db")   # e.g. data/acme-agent.db
    if agent_db not in _graphs:
        conn = sqlite3.connect(agent_db, check_same_thread=False)
        _graphs[agent_db] = _build().compile(checkpointer=SqliteSaver(conn))
    return _graphs[agent_db]


def _config(user_id, session_id):
    return {"configurable": {"thread_id": f"{user_id}:{session_id}"}}


def _result(user_id, session_id):
    values = graph().get_state(_config(user_id, session_id)).values
    return {"outcome": values.get("outcome"), "reply": values.get("reply"),
            "pending": values.get("pending"), "options": values.get("options", [])}


@observe(name="agent.start", as_type="agent")
def start(user_id, session_id, pnr, message, risk):
    """Begin a new disruption case (resets any finished one in this session)."""
    fresh = State(user_id=user_id, session_id=session_id, pnr=pnr.upper(), message=message,
                  risk=risk or {}, booking={}, options=[], automatic=[], note="", choice=None,
                  outcome="", reply="", pending=None)
    graph().invoke(fresh, _config(user_id, session_id))
    return _result(user_id, session_id)


def waiting(user_id, session_id):
    """True while this session's agent is paused, waiting for the customer to pick an option."""
    return bool(graph().get_state(_config(user_id, session_id)).next)


@observe(name="agent.resume", as_type="agent")
def resume(user_id, session_id, message):
    graph().invoke(Command(resume=message), _config(user_id, session_id))
    return _result(user_id, session_id)
