"""Day 10 - Capstone: the hybrid ACME support API.

POST /v1/chat routes every message: RAG (policies) | tools (live bookings) | clarify,
all composed by the fine-tuned local model (or Azure, via LLM_BACKEND), with an output
cleaner and a trace log.

cancel_ticket is irreversible, so it never runs on a routing decision alone: the reply asks
"are you sure?" and returns pending_action; the client sends the customer's answer back with
confirm_cancel=<PNR>, and only a clear yes cancels.
"""

import json
import re
import sys
import time
from pathlib import Path

import chromadb
from fastapi import FastAPI
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import backend  # the mock reservation system (proper packaging = polish task)
from llm_backend import chat as llm_chat
from router import route as route_message  # LLM or Jev, picked by ROUTER_BACKEND

TOP_K = 3

app = FastAPI(title="ACME Bharat Airlines Support AI", version="1.0")

TOOL_REGISTRY = {
    "get_flight_status": backend.get_flight_status,
    "cancel_ticket": backend.cancel_ticket,
}

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

# Irreversible tools never run on a routing decision alone: the customer must confirm.
# First word must be one of these, and no word may be a negation ("yes but don't").
AFFIRM = {"yes", "y", "yeah", "yep", "yup", "ok", "okay", "sure", "confirm", "confirmed",
          "haan", "han", "ha", "ji", "hanji"}
NEGATE = {"no", "not", "don", "dont", "nahi", "nahin", "mat", "wait", "stop", "hold"}

_chroma = chromadb.PersistentClient(path=str(ROOT / "data" / "chroma"))
_policies = _chroma.get_collection("acme_policies")


class ChatRequest(BaseModel):
    message: str
    # Set to the PNR from a previous reply's pending_action to answer "cancel it? yes/no".
    confirm_cancel: str | None = None


def retrieve(question, k=TOP_K):
    res = _policies.query(query_texts=[question], n_results=k)
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


def affirmative(text):
    words = re.findall(r"[a-z]+", text.lower())
    return bool(words) and words[0] in AFFIRM and not NEGATE.intersection(words)


def run_tool(tool, pnr, question):
    """Execute a validated tool call and let the LLM phrase the verified result."""
    result = TOOL_REGISTRY[tool](pnr)
    docs, metas = retrieve(question)
    reply = clean(llm(TOOL_RESPONSE_PROMPT.format(
        question=question, result=json.dumps(result),
        context="\n\n---\n\n".join(docs))))
    return reply, [f"backend:{tool}({pnr})"] + [m["source"] for m in metas]


def ask_cancel_confirmation(pnr):
    """Deterministic 'are you sure?' - no LLM, so nothing can talk it into cancelling."""
    booking = backend.get_flight_status(pnr)
    if not booking["ok"]:
        return "clarify", f"I couldn't find a booking with PNR {pnr}. Could you re-check it?", None
    reply = (f"Just to confirm: do you want me to cancel booking {pnr} "
             f"({booking['route']}, scheduled {booking['scheduled']})? This can't be undone. "
             f"Reply YES to cancel, or NO to keep your booking.")
    return "confirm:cancel_ticket", reply, {"tool": "cancel_ticket", "pnr": pnr}


def answer_confirmation(req):
    pnr = req.confirm_cancel.strip().upper()
    if not valid_pnr(pnr):
        return "clarify", f"'{pnr}' does not look like a valid PNR. Could you re-check it?", []
    if not affirmative(req.message):
        return "confirm:declined", (f"No problem - I have not cancelled booking {pnr}. "
                                    "Is there anything else I can help you with?"), []
    reply, sources = run_tool("cancel_ticket", pnr, f"Please cancel my booking {pnr}.")
    return "tool:cancel_ticket", reply, sources


@app.post("/v1/chat")
def chat(req: ChatRequest):
    t0 = time.time()
    pending = None

    if req.confirm_cancel is not None:  # the customer is answering our "are you sure?"
        decision = {"router": "confirmation", "confidence": None}
        route, reply, sources = answer_confirmation(req)
        return respond(t0, decision, route, reply, sources, pending)

    decision = route_message(req.message)
    tool = decision.get("tool")

    if tool == "ask_pnr":
        route, reply, sources = "clarify", CLARIFY_PNR, []

    elif tool in TOOL_REGISTRY:
        pnr = decision.get("arguments", {}).get("pnr", "")
        if not valid_pnr(pnr):
            route, reply, sources = "clarify", f"'{pnr}' does not look like a valid PNR. Could you re-check it?", []
        elif tool == "cancel_ticket":
            route, reply, pending = ask_cancel_confirmation(pnr.upper())
            sources = [f"backend:get_flight_status({pnr.upper()})"] if pending else []
        else:
            reply, sources = run_tool(tool, pnr, req.message)
            route = f"tool:{tool}"

    else:  # policy / general -> RAG
        docs, metas = retrieve(req.message)
        reply = clean(llm(RAG_PROMPT.format(
            context="\n\n---\n\n".join(docs), question=req.message), system=RAG_SYSTEM))
        route = "rag"
        sources = [f"{m['source']} [{m['section']}]" for m in metas]

    return respond(t0, decision, route, reply, sources, pending)


def respond(t0, decision, route, reply, sources, pending):
    latency_ms = round((time.time() - t0) * 1000)
    print(f"[trace] router={decision['router']} confidence={decision['confidence']} "
          f"route={route} latency_ms={latency_ms} sources={sources}")
    return {"reply": reply, "route": route, "router": decision["router"],
            "sources": sources, "pending_action": pending, "latency_ms": latency_ms}