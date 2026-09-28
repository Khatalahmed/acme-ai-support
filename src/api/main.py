"""Day 10 - Capstone: the hybrid ACME support API.

POST /v1/chat routes every message: RAG (policies) | tools (live bookings) | clarify,
all composed by the fine-tuned local model (or Azure, via LLM_BACKEND), with an output
cleaner and a trace log.

Every request carries a bearer token (who) and a session_id (which conversation). Booking
tools go through src/actions.py, which enforces ownership, server-side confirmation with
expiry, idempotent execution and the audit log. A cancel request only PROPOSES; the next
message in the same session answers it, and only a clear yes cancels.
"""

import copy
import json
import os
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

import chromadb
from langfuse import get_client, observe, propagate_attributes
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
import actions  # the safety boundary: every booking read/write goes through here
import agent  # disruption-recovery agent (LangGraph); proposes changes via actions
import backend  # the mock airline (read here only to snapshot/reset it in demo mode)
import decision_trace  # the "How I decided" panel, built from what actually happened
import insights  # numbers-only request metrics for the public /insights page
import llm_backend
import ratelimit  # abuse controls for the public demo
import shadow  # shadow routing: the other router's opinion, recorded, never used
from auth import user_from_token
from llm_backend import chat as llm_chat
from router import RISK_THRESHOLD, TOOLS, demands_exception, find_pnr, high_risk
from router import route as route_message  # LLM or Jev, picked by ROUTER_BACKEND

# Policy sections given to the LLM. 3 -> 5 after the C4 eval: two questions (non-refundable
# refunds, a Hinglish one) had their answer ranked 4th/5th, so the LLM correctly refused;
# with 5 (and RAG_REASONING_EFFORT=low) fact recall went 95% -> 100% over repeated runs.
TOP_K = 5

app = FastAPI(title="ACME Bharat Airlines Support AI", version="1.1")

# The last line used to be "Structure the response with empathy, the policy answer, and a
# next-step question" (the fine-tuning template). Seen live on Azure, and then measured: the
# model printed those as headings in 33/41 answers and opened 32/41 with an apology - including
# "What is the checked baggage allowance?". Accuracy metrics were all 100%, so only a style
# check in rag_eval.py caught it.
RAG_SYSTEM = (
    "You are a friendly, professional customer support assistant for ACME Bharat Airlines. "
    "Answer ONLY using the policy context provided by the user. Quote exact numbers and "
    "timelines from the context. If the context does not contain the answer, say: "
    "'I don't have sufficient information on this. Kindly contact AcmeConnect for assistance.' "
    "Write plain, natural sentences (a short bullet list is fine for several rules) with no "
    "headings or labels. Apologise only if the customer describes a problem they are having; "
    "for a plain question, answer it directly. End with one short, relevant follow-up question."
)

RAG_PROMPT = """Policy context:
{context}

Customer question: {question}"""

TOOL_RESPONSE_PROMPT = """The customer asked: {question}

Verified backend result for their booking:
{result}

Relevant ACME policy context:
{context}

Reply in 2-4 short sentences. Use ONLY the booking facts above, in plain words, and ONLY the
policy context for any policy statement - NEVER state a policy that is not in the context.
Do not offer options, services or next steps, and do not ask a question: a closing line is
added after your text."""

# Status replies get their own system prompt instead of the shared persona: the persona's
# "end with a question / list options" rules made the model invent services (C5 eval).
STATUS_SYSTEM = (
    "You are a polite, empathetic ACME Bharat Airlines support assistant. Acknowledge the "
    "customer briefly, then state the booking facts you were given in plain words. Mention "
    "refunds, vouchers or compensation only if the policy context says they apply.")
STATUS_CLOSING = "Is there anything else I can help you with?"

CLARIFY_PNR = ("Could you please share your 6-character PNR (for example ACX123) "
               "so I can look up your booking?")

# Answers to "are you sure?". A yes needs an affirmative first word and no negation anywhere
# ("yes but don't" is not a yes); a no is a negative first word.
AFFIRM = {"yes", "y", "yeah", "yep", "yup", "ok", "okay", "sure", "confirm", "confirmed",
          "haan", "han", "ha", "ji", "hanji"}
NEGATE = {"no", "not", "don", "dont", "nahi", "nahin", "mat", "wait", "stop", "hold"}
DECLINE_START = {"no", "nope", "nah", "nahi", "nahin", "na", "don", "dont", "keep"}


class ChatRequest(BaseModel):
    message: str
    # Client-generated conversation id; a pending "are you sure?" belongs to one session.
    session_id: str = Field(min_length=1, max_length=100)


# Declared as a security scheme (not a plain header parameter), so the OpenAPI docs show an
# "Authorize" button - Swagger UI ignores header parameters named "Authorization".
bearer = HTTPBearer(auto_error=False, description="Demo tokens: demo-asha, demo-ravi (src/auth.py)")


def current_user(creds: HTTPAuthorizationCredentials | None = Depends(bearer)):
    """Authentication: bearer token -> user_id (simulated; see src/auth.py)."""
    user_id = user_from_token(creds.credentials) if creds else None
    if not user_id:
        raise HTTPException(401, "Missing or invalid token", headers={"WWW-Authenticate": "Bearer"})
    return user_id


@lru_cache(maxsize=1)
def _policies():
    # Opened on first use, so importing the API (e.g. in tests) needs no index on disk.
    return chromadb.PersistentClient(path=str(ROOT / "data" / "chroma")).get_collection("acme_policies")


@observe(name="rag.retrieve", as_type="retriever")
def retrieve(question, k=None):
    res = _policies().query(query_texts=[question], n_results=k or TOP_K)
    return res["documents"][0], res["metadatas"][0]


def llm(prompt, system=None, name="api.llm", persona=False, reasoning_effort=None):
    messages = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": prompt})
    return llm_chat(messages, name=name, persona=persona, reasoning_effort=reasoning_effort)


def rag_answer(question, reasoning_effort=None):
    """Retrieve policy sections and answer from them only. The API and src/evals/rag_eval.py
    both call this, so the evaluation measures exactly what customers get."""
    effort = reasoning_effort or os.environ.get("RAG_REASONING_EFFORT") or None
    docs, metas = retrieve(question)
    reply = clean(llm(RAG_PROMPT.format(context="\n\n---\n\n".join(docs), question=question),
                      system=RAG_SYSTEM, name="rag.answer", reasoning_effort=effort))
    return reply, docs, metas


def clean(text):
    """Output validator: strip tool_call tags AND lone gibberish first tokens."""
    text = re.sub(r"^(\s*</?tool_call>\s*)+", "", text.strip())
    lines = text.strip().splitlines()
    if len(lines) > 1 and " " not in lines[0].strip() and len(lines[0].strip()) <= 15:
        lines = lines[1:]                      # drops junk like '줫' or 'ontvangst'
    return "\n".join(lines).strip()


def valid_pnr(pnr):
    return isinstance(pnr, str) and re.fullmatch(r"[A-Za-z]{3}\d{3}", pnr)


def _words(text):
    return re.findall(r"[a-z]+", text.lower())


def affirmative(text):
    words = _words(text)
    return bool(words) and words[0] in AFFIRM and not NEGATE.intersection(words)


def declines(text):
    words = _words(text)
    return bool(words) and words[0] in DECLINE_START


# Which policy section a booking's state makes relevant. Status replies used to SEARCH with the
# customer's words ("What's the status of ACX123?"), which say nothing about which policy matters:
# the right section was found for 7 of 21 test replies. Searching with a query built from the
# booking reached 12/21 - delays still ranked "Processing Claims" above the tiers. But here we
# KNOW the section, so we look it up by name: search is for when you don't know what you need.
DELAY_TIERS = ("delay-compensation-policy.md", "2. Delay Compensation Tiers")
DELAY_EXCLUSIONS = ("delay-compensation-policy.md", "4. Policy Exclusions")
AIRLINE_CANCEL = ("refund-cancellation-policy.md", "4. Airline-Initiated Cancellations")
WEATHER_CANCEL = ("refund-cancellation-policy.md", "5. Weather and Force Majeure Cancellations")
POLICY_SECTIONS = (DELAY_TIERS, DELAY_EXCLUSIONS, AIRLINE_CANCEL, WEATHER_CANCEL)
INTERNAL_FIELDS = {"ok", "flight_status", "delay_min", "cause", "seats_available"}


def policy_section(booking):
    """(source, section) that applies to this booking's disruption, or None (operating normally)."""
    status, external = booking.get("flight_status"), booking.get("cause") in ("weather", "atc")
    if status == "cancelled":
        return WEATHER_CANCEL if external else AIRLINE_CANCEL
    if status == "delayed":
        return DELAY_EXCLUSIONS if external else DELAY_TIERS
    return None


@observe(name="rag.lookup", as_type="retriever")
def status_context(booking, question):
    """Policy text for a booking status reply: a lookup by section name, not a search.
    Empty when the flight is operating normally (no policy applies)."""
    key = policy_section(booking)
    if not key:
        return [], []
    got = _policies().get(where={"$and": [{"source": key[0]}, {"section": key[1]}]},
                          include=["documents", "metadatas"])
    if got["documents"]:
        return got["documents"], got["metadatas"]
    # Section renamed in the policy docs? Degrade to search rather than answer without policy.
    return retrieve(f"{key[1]} {question}", k=2)


def customer_view(booking):
    """Booking facts as a customer should read them - what the LLM sees.

    Raw fields are for code: the LLM recited JSON it was given ("(flight_status: scheduled",
    "refundable: false; resolution: null; vouchers: []") straight to customers. So it gets
    plain labels and values, and nothing empty or internal.
    """
    lines = {"PNR": booking.get("pnr"), "Flight": booking.get("flight"),
             "Route": booking.get("route"), "Departure": booking.get("departure"),
             "Status": booking.get("status")}
    if booking.get("fare"):
        lines["Fare"] = f"Rs {booking['fare']:,}"
        lines["Fare type"] = "refundable" if booking.get("refundable") else "non-refundable"
    if booking.get("vouchers"):
        lines["Travel vouchers on this booking"] = ", ".join(f"Rs {v:,}" for v in booking["vouchers"])
    return "\n".join(f"- {k}: {v}" for k, v in lines.items() if v not in (None, "", []))


def phrase_result(tool, pnr, result, question):
    """Let the LLM phrase a verified backend result, grounded in retrieved policy."""
    docs, metas = status_context(result, question)
    context = ("\n\n---\n\n".join(docs) if docs else
               "(none - the flight is operating normally, so make no policy statements)")
    reply = clean(llm(TOOL_RESPONSE_PROMPT.format(
        question=question, result=customer_view(result), context=context),
        system=STATUS_SYSTEM, name="tool.reply",
        reasoning_effort=os.environ.get("REPLY_REASONING_EFFORT") or None))
    reply = f"{reply}\n\n{STATUS_CLOSING}"   # code, not LLM: nothing can be offered that we don't do
    return reply, [f"backend:{tool}({pnr})"] + [m["source"] for m in metas]


def receipt(action, pnr, result):
    """Exact, code-written confirmation of a completed change - never LLM-phrased.

    A receipt is a record: an LLM here drifted into re-offering options the passenger had
    already used (seen in a live run), so completed changes are stated by code, not a model.
    """
    if action == "cancel_ticket":
        money = (f"A refund of Rs {result['refund_amount']:,} will go back to your original "
                 "payment method within 7 business days." if result["refund_amount"]
                 else result["note"] + ".")
        text = f"Done - booking {pnr} is cancelled. {money}"
    elif "rebooked_from" in result:
        extra = (f" A Rs {result['voucher_amount']:,} travel voucher has been added to your "
                 "booking." if result["voucher_amount"] else "")
        text = (f"Done - booking {pnr} is now on {result['flight']} ({result['route']}), "
                f"departing {result['departure']}.{extra}")
    elif "refund_amount" in result:
        text = (f"Done - a full refund of Rs {result['refund_amount']:,} for booking {pnr} will "
                "go back to your original payment method within 7 business days. The booking "
                "is now cancelled.")
    else:
        text = (f"Done - a Rs {result['voucher_amount']:,} travel voucher has been added to "
                f"booking {pnr}. You keep your current flight.")
    return text + " Is there anything else I can help you with?"




def not_found(pnr):
    # Identical for "doesn't exist" and "not yours", so PNRs can't be probed.
    return f"I couldn't find a booking with PNR {pnr} on your account. Could you re-check it?"


def answer_pending(user_id, req):
    """The customer is replying to our "are you sure?". Returns a response, or None to route."""
    if affirmative(req.message):
        out = actions.confirm(user_id, req.session_id)
        pnr = out.get("pnr")
        if out["ok"]:
            return (f"tool:{out['action']}", receipt(out["action"], pnr, out["result"]),
                    [f"backend:{out['action']}({pnr})"])
        replies = {
            "expired": (f"That confirmation expired, so nothing has been changed on booking "
                        f"{pnr}. If you still want to go ahead, just ask again."),
            "already_cancelled": f"Booking {pnr} is already cancelled.",
            "option_unavailable": (f"Sorry - that option for booking {pnr} is no longer "
                                   "available (for example, the last seat was taken). Nothing "
                                   "has been changed. Would you like to see the current options?"),
            "backend_rejected": (f"Sorry - I couldn't complete that for booking {pnr}, and "
                                 "nothing has been changed. Would you like to see your options?"),
            "not_found": not_found(pnr),
            "nothing_pending": "There's nothing waiting for your confirmation right now.",
        }
        return f"confirm:{out['reason']}", replies[out["reason"]], []

    declined = actions.decline(user_id, req.session_id)
    if declines(req.message):
        return ("confirm:declined",
                f"No problem - I haven't made any changes to booking {declined.get('pnr')}. "
                "Is there anything else I can help you with?", [])
    return None  # moved on to something else: pending closed, handle the message normally


WEB = ROOT / "src" / "web"


# Shared stylesheet and assets for the two pages (plain files, no build step).
app.mount("/static", StaticFiles(directory=WEB / "static"), name="static")


@app.get("/", include_in_schema=False)
def chat_page():
    """The demo chat page (plain HTML/CSS/JS, no build step)."""
    return FileResponse(WEB / "index.html")


@app.get("/insights", include_in_schema=False)
def insights_page():
    """Public page: measured evaluation results plus live, numbers-only traffic stats."""
    return FileResponse(WEB / "insights.html")


MEASURED = ROOT / "data" / "insights" / "measured.json"


@app.get("/v1/insights")
def insights_data():
    """Aggregates only - no messages, users or sessions are stored or returned (public)."""
    measured = json.loads(MEASURED.read_text(encoding="utf-8")) if MEASURED.exists() else None
    return {"live": insights.summary(), "measured": measured}


@app.get("/v1/bookings")
def bookings(user_id: str = Depends(current_user)):
    """The signed-in user's own bookings, for the demo page's sidebar."""
    return [{**{k: b[k] for k in ("pnr", "flight", "route", "departure", "status")},
             **airports(b["flight"]), "state": booking_state(b), "outcome": booking_outcome(b)}
            for b in actions.my_bookings(user_id)]


def airports(flight_no):
    """Airport codes and cities, for the boarding-pass style trip cards."""
    f = backend.FLIGHTS[flight_no]
    return {"origin": f["origin"], "origin_city": backend.CITIES[f["origin"]],
            "dest": f["dest"], "dest_city": backend.CITIES[f["dest"]]}


def booking_state(b):
    """ok | delayed | cancelled - the page shows it as colour + icon + the status text."""
    if b["booking_status"] == "cancelled" or b["flight_status"] == "cancelled":
        return "cancelled"
    return "delayed" if b["flight_status"] == "delayed" else "ok"


def booking_outcome(b):
    """What was done to this booking, so the sidebar visibly changes after an action."""
    r = b.get("resolution") or {}
    if r.get("kind") == "refund":
        return f"Refunded Rs {r['amount']:,}"
    if r.get("kind") == "voucher":
        return f"Rs {r['amount']:,} voucher added"
    if r.get("kind") == "rebook":
        return "Rebooked" + (f" + Rs {r['voucher']:,} voucher" if r.get("voucher") else "")
    return "Cancelled by you" if b["booking_status"] == "cancelled" else None


def rate_limited(request: Request):
    """Abuse control for the public demo: every chat message can spend real API credit."""
    blocked = ratelimit.check(ratelimit.client_ip(request))
    if blocked:
        reason, retry_after = blocked
        raise HTTPException(429, f"Demo limit: {reason}. Please try again later.",
                            headers={"Retry-After": str(retry_after)})


# The mock airline as it was at startup - what a demo reset restores.
_DEMO_SNAPSHOT = copy.deepcopy((backend.FLIGHTS, backend.BOOKINGS))


def demo_mode():
    return (os.environ.get("DEMO_MODE") or "").strip().lower() in ("1", "true", "yes")


@app.get("/v1/demo")
def demo_info():
    """Tells the page whether demo controls (reset) are available."""
    return {"demo_mode": demo_mode()}


@app.post("/v1/demo/reset")
def demo_reset(user_id: str = Depends(current_user), _: None = Depends(rate_limited)):
    """Restore the mock airline so every visitor gets a working demo. Off unless DEMO_MODE."""
    if not demo_mode():
        raise HTTPException(404, "Not Found")
    for live, saved in zip((backend.FLIGHTS, backend.BOOKINGS), copy.deepcopy(_DEMO_SNAPSHOT)):
        live.clear()
        live.update(saved)
    actions.audit("demo_reset", user_id)
    return {"ok": True}


@app.post("/v1/chat")
def chat(req: ChatRequest, user_id: str = Depends(current_user),
         _: None = Depends(rate_limited)):
    """One Langfuse trace per request, tagged with user and session (off when tracing is off)."""
    with propagate_attributes(user_id=user_id, session_id=req.session_id, trace_name="chat"):
        with get_client().start_as_current_observation(
                name="POST /v1/chat", input={"message": req.message}) as root:
            token = llm_backend.calls.set([])   # this request's model calls, for its trace
            try:
                out = handle(req, user_id)
            finally:
                llm_backend.calls.reset(token)
            root.update(output=out, metadata={"route": out["route"], "router": out["router"]})
            return out


def handle(req, user_id):
    t0 = time.time()
    pending = None

    # 1. Answering our "are you sure?" (server-side pending action)
    if actions.pending_action(user_id, req.session_id):
        answered = answer_pending(user_id, req)
        if answered:
            route, reply, sources = answered
            decision = {"router": "confirmation", "confidence": None}
            return respond(t0, user_id, decision, route, reply, sources, pending)

    # 2. Picking one of the options the disruption agent offered
    if agent.waiting(user_id, req.session_id):
        out = agent.resume(user_id, req.session_id, req.message)
        if out["outcome"] != "moved_on":
            decision = {"router": "agent", "confidence": None}
            return respond(t0, user_id, decision, f"agent:{out['outcome']}", out["reply"],
                           [], out["pending"])

    # 3. A new request
    started = time.perf_counter()
    decision = route_message(req.message)
    decision = {**decision, "router_ms": round((time.perf_counter() - started) * 1000)}
    # Second opinion on a background thread; its answer is only recorded (see src/shadow.py).
    shadow.maybe_run(req.message, decision, user_id, req.session_id,
                     get_client().get_current_trace_id())
    tool = decision.get("tool")
    pnr_in_msg = find_pnr(req.message)

    if demands_exception(decision) and tool not in ("human_agent", "disruption_help"):
        # Only a person can grant an exception, whatever the route. (The disruption agent
        # escalates these itself, with the passenger's entitlements in the handover.)
        ticket = actions.escalate(user_id, req.session_id, pnr_in_msg, "demands_exception",
                                  f"Customer asked for an exception: {req.message!r}")
        route, sources = "escalated", []
        reply = (f"I understand. That needs a decision from a member of our team, so I've passed "
                 f"your request to them (reference {ticket['ref']}); they will get back to you "
                 f"directly. Nothing on your booking has been changed.")

    elif tool == "ask_pnr":
        route, reply, sources = "clarify", CLARIFY_PNR, []

    elif tool == "human_agent":
        ticket = actions.escalate(user_id, req.session_id, find_pnr(req.message),
                                  "requested_human", f"Customer asked for a person: {req.message!r}")
        route, sources = "escalated", []
        reply = (f"Of course - I've passed your conversation to a member of our team "
                 f"(reference {ticket['ref']}). They will get back to you directly.")

    elif tool in TOOLS:
        pnr = (decision.get("arguments") or {}).get("pnr", "")
        if not valid_pnr(pnr):
            route, reply, sources = "clarify", f"'{pnr}' does not look like a valid PNR. Could you re-check it?", []
        elif tool == "cancel_ticket" and high_risk(decision):
            # Angry, or demanding an exception: a cancel "yes" wouldn't give them what they
            # want and can't be undone - a person should handle it.
            risk = decision["risk"]
            reason = "demands_exception" if risk.get("demands_exception", 0) >= \
                RISK_THRESHOLD else "angry"
            ticket = actions.escalate(user_id, req.session_id, pnr, reason,
                                      f"High-risk cancel request: {req.message!r}")
            route, sources = "escalated", []
            reply = (f"I'm sorry this has been so frustrating. I haven't changed anything on "
                     f"booking {pnr.upper()}; I've passed your case to a senior member of our "
                     f"team (reference {ticket['ref']}), who will contact you directly.")
        elif tool == "disruption_help":
            out = agent.start(user_id, req.session_id, pnr, req.message, decision.get("risk"))
            route, reply, sources, pending = (f"agent:{out['outcome']}", out["reply"], [],
                                              out["pending"])
        elif tool == "cancel_ticket":
            proposal = actions.propose_cancel(user_id, req.session_id, pnr,
                                              router=decision["router"],
                                              confidence=decision.get("confidence"))
            if proposal["ok"]:
                b = proposal["booking"]
                route, sources = "confirm:cancel_ticket", [f"backend:get_flight_status({b['pnr']})"]
                reply = (f"Just to confirm: do you want me to cancel booking {b['pnr']} "
                         f"({b['route']}, scheduled {b['scheduled']})? This can't be undone. "
                         f"Reply YES to cancel, or NO to keep your booking.")
                pending = {k: proposal[k] for k in ("action_id", "action", "pnr", "expires_at")}
            elif proposal["reason"] == "already_cancelled":
                route, reply, sources = "clarify", f"Booking {pnr.upper()} is already cancelled.", []
            else:
                route, reply, sources = "clarify", not_found(pnr.upper()), []
        else:  # get_flight_status
            result = actions.flight_status(user_id, req.session_id, pnr)
            if result["ok"]:
                reply, sources = phrase_result(tool, pnr.upper(), result, req.message)
                route = f"tool:{tool}"
            else:
                route, reply, sources = "clarify", not_found(pnr.upper()), []

    else:  # policy / general -> RAG
        reply, docs, metas = rag_answer(req.message)
        route = "rag"
        sources = [f"{m['source']} [{m['section']}]" for m in metas]

    return respond(t0, user_id, decision, route, reply, sources, pending)


def respond(t0, user_id, decision, route, reply, sources, pending):
    latency_ms = round((time.time() - t0) * 1000)
    print(f"[trace] user={user_id} router={decision['router']} confidence={decision['confidence']} "
          f"route={route} latency_ms={latency_ms} sources={sources}")
    calls = llm_backend.calls.get() or []
    insights.record(route, decision, latency_ms, calls)
    return {"reply": reply, "route": route, "router": decision["router"],
            "sources": sources, "pending_action": pending, "latency_ms": latency_ms,
            "trace": decision_trace.build(route, decision, sources, pending, calls)}
