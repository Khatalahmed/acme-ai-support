"""Day 10 - Capstone: the hybrid ACME support API.

POST /v1/chat routes every message: RAG (policies) | tools (live bookings) | clarify,
all composed by the fine-tuned local model (or Azure, via LLM_BACKEND), with an output
cleaner and a trace log.

Every request carries a bearer token (who) and a session_id (which conversation). Booking
tools go through src/actions.py, which enforces ownership, server-side confirmation with
expiry, idempotent execution and the audit log. A cancel request only PROPOSES; the next
message in the same session answers it, and only a clear yes cancels.
"""

import json
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

import chromadb
from langfuse import get_client, observe, propagate_attributes
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
import actions  # the safety boundary: every booking read/write goes through here
import agent  # disruption-recovery agent (LangGraph); proposes changes via actions
from auth import user_from_token
from llm_backend import chat as llm_chat
from router import TOOLS, find_pnr
from router import route as route_message  # LLM or Jev, picked by ROUTER_BACKEND

TOP_K = 3

app = FastAPI(title="ACME Bharat Airlines Support AI", version="1.1")

RAG_SYSTEM = (
    "You are a polite and empathetic customer support executive of ACME Bharat Airlines. "
    "Answer ONLY using the policy context provided by the user. Quote exact numbers and "
    "timelines from the context. If the context does not contain the answer, say: "
    "'I don't have sufficient information on this. Kindly contact AcmeConnect for assistance.' "
    "Structure the response with empathy, the policy answer, and a next-step question."
)

RAG_PROMPT = """Policy context:
{context}

Customer question: {question}"""

TOOL_RESPONSE_PROMPT = """The customer asked: {question}

Verified backend result for their booking:
{result}

Relevant ACME policy context:
{context}

Reply to the customer. Use ONLY the backend result for booking facts, and ONLY the
policy context for any policy statements - NEVER state a policy that is not in the
context. Be empathetic, state facts exactly, end with one next-step question."""

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
def retrieve(question, k=TOP_K):
    res = _policies().query(query_texts=[question], n_results=k)
    return res["documents"][0], res["metadatas"][0]


def llm(prompt, system=None, name="api.llm"):
    messages = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": prompt})
    return llm_chat(messages, name=name)


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


def phrase_result(tool, pnr, result, question):
    """Let the LLM phrase a verified backend result, grounded in retrieved policy."""
    docs, metas = retrieve(question)
    reply = clean(llm(TOOL_RESPONSE_PROMPT.format(
        question=question, result=json.dumps(result), context="\n\n---\n\n".join(docs)),
        name="tool.reply"))
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


def high_risk(decision):
    risk = decision.get("risk") or {}
    return max(risk.get("angry", 0), risk.get("demands_exception", 0)) >= agent.RISK_THRESHOLD


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


@app.post("/v1/chat")
def chat(req: ChatRequest, user_id: str = Depends(current_user)):
    """One Langfuse trace per request, tagged with user and session (off when tracing is off)."""
    with propagate_attributes(user_id=user_id, session_id=req.session_id, trace_name="chat"):
        with get_client().start_as_current_observation(
                name="POST /v1/chat", input={"message": req.message}) as root:
            out = handle(req, user_id)
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
    decision = route_message(req.message)
    tool = decision.get("tool")

    if tool == "ask_pnr":
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
                agent.RISK_THRESHOLD else "angry"
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
        docs, metas = retrieve(req.message)
        reply = clean(llm(RAG_PROMPT.format(
            context="\n\n---\n\n".join(docs), question=req.message), system=RAG_SYSTEM,
            name="rag.answer"))
        route = "rag"
        sources = [f"{m['source']} [{m['section']}]" for m in metas]

    return respond(t0, user_id, decision, route, reply, sources, pending)


def respond(t0, user_id, decision, route, reply, sources, pending):
    latency_ms = round((time.time() - t0) * 1000)
    print(f"[trace] user={user_id} router={decision['router']} confidence={decision['confidence']} "
          f"route={route} latency_ms={latency_ms} sources={sources}")
    return {"reply": reply, "route": route, "router": decision["router"],
            "sources": sources, "pending_action": pending, "latency_ms": latency_ms}
