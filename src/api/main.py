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
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
import actions  # the safety boundary: every booking read/write goes through here
from auth import user_from_token
from llm_backend import chat as llm_chat
from router import TOOLS
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


def current_user(authorization: str | None = Header(default=None)):
    """Authentication: bearer token -> user_id (simulated; see src/auth.py)."""
    token = authorization.removeprefix("Bearer ").strip() if authorization else None
    user_id = user_from_token(token)
    if not user_id:
        raise HTTPException(401, "Missing or invalid token", headers={"WWW-Authenticate": "Bearer"})
    return user_id


@lru_cache(maxsize=1)
def _policies():
    # Opened on first use, so importing the API (e.g. in tests) needs no index on disk.
    return chromadb.PersistentClient(path=str(ROOT / "data" / "chroma")).get_collection("acme_policies")


def retrieve(question, k=TOP_K):
    res = _policies().query(query_texts=[question], n_results=k)
    return res["documents"][0], res["metadatas"][0]


def llm(prompt, system=None):
    messages = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": prompt})
    return llm_chat(messages)


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
        question=question, result=json.dumps(result), context="\n\n---\n\n".join(docs))))
    return reply, [f"backend:{tool}({pnr})"] + [m["source"] for m in metas]


def not_found(pnr):
    # Identical for "doesn't exist" and "not yours", so PNRs can't be probed.
    return f"I couldn't find a booking with PNR {pnr} on your account. Could you re-check it?"


def answer_pending(user_id, req):
    """The customer is replying to our "are you sure?". Returns a response, or None to route."""
    if affirmative(req.message):
        out = actions.confirm(user_id, req.session_id)
        if out["ok"]:
            reply, sources = phrase_result("cancel_ticket", out["pnr"], out["result"],
                                           f"Please cancel my booking {out['pnr']}.")
            return "tool:cancel_ticket", reply, sources
        replies = {
            "expired": (f"That confirmation expired, so booking {out.get('pnr')} has NOT been "
                        "cancelled. If you still want to cancel, just ask again."),
            "already_cancelled": f"Booking {out.get('pnr')} is already cancelled.",
            "not_found": not_found(out.get("pnr")),
            "nothing_pending": "There's nothing waiting for your confirmation right now.",
        }
        return f"confirm:{out['reason']}", replies[out["reason"]], []

    declined = actions.decline(user_id, req.session_id)
    if declines(req.message):
        pnr = declined.get("pnr")
        return ("confirm:declined",
                f"No problem - I have not cancelled booking {pnr}. "
                "Is there anything else I can help you with?", [])
    return None  # moved on to something else: pending closed, handle the message normally


@app.post("/v1/chat")
def chat(req: ChatRequest, user_id: str = Depends(current_user)):
    t0 = time.time()
    pending = None

    if actions.pending_action(user_id, req.session_id):
        answered = answer_pending(user_id, req)
        if answered:
            route, reply, sources = answered
            decision = {"router": "confirmation", "confidence": None}
            return respond(t0, user_id, decision, route, reply, sources, pending)

    decision = route_message(req.message)
    tool = decision.get("tool")

    if tool == "ask_pnr":
        route, reply, sources = "clarify", CLARIFY_PNR, []

    elif tool in TOOLS:
        pnr = (decision.get("arguments") or {}).get("pnr", "")
        if not valid_pnr(pnr):
            route, reply, sources = "clarify", f"'{pnr}' does not look like a valid PNR. Could you re-check it?", []
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
            context="\n\n---\n\n".join(docs), question=req.message), system=RAG_SYSTEM))
        route = "rag"
        sources = [f"{m['source']} [{m['section']}]" for m in metas]

    return respond(t0, user_id, decision, route, reply, sources, pending)


def respond(t0, user_id, decision, route, reply, sources, pending):
    latency_ms = round((time.time() - t0) * 1000)
    print(f"[trace] user={user_id} router={decision['router']} confidence={decision['confidence']} "
          f"route={route} latency_ms={latency_ms} sources={sources}")
    return {"reply": reply, "route": route, "router": decision["router"],
            "sources": sources, "pending_action": pending, "latency_ms": latency_ms}
