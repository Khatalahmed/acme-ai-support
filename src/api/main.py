"""Day 10 - Capstone: the hybrid ACME support API.

POST /v1/chat routes every message: RAG (policies) | tools (live bookings) | clarify,
all composed by the fine-tuned local model, with an output cleaner and a trace log.
"""

import json
import re
import sys
import time
from pathlib import Path

import chromadb
import ollama
from fastapi import FastAPI
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "tools"))
import backend  # the mock reservation system (proper packaging = polish task)

MODEL = "acme-support"
TOP_K = 3

app = FastAPI(title="ACME Bharat Airlines Support AI", version="1.0")

TOOL_REGISTRY = {
    "get_flight_status": backend.get_flight_status,
    "cancel_ticket": backend.cancel_ticket,
}

ROUTER_PROMPT = """You are a request router for ACME Bharat Airlines.
Decide if the customer message requires calling a backend tool.

Available tools:
1. get_flight_status(pnr) - live status of a booking. Needs a PNR (3 letters + 3 digits, e.g. ACX123).
2. cancel_ticket(pnr) - cancel a booking and compute the refund. Needs a PNR.

Rules - output ONLY one JSON object, nothing else:
- Tool needed, PNR present:  {{"tool": "<tool_name>", "arguments": {{"pnr": "<PNR>"}}}}
- Tool needed, PNR missing:  {{"tool": "ask_pnr"}}
- No tool needed (policy or general question): {{"tool": null}}

Customer message: {question}
JSON:"""

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

_chroma = chromadb.PersistentClient(path=str(ROOT / "data" / "chroma"))
_policies = _chroma.get_collection("acme_policies")


class ChatRequest(BaseModel):
    message: str


def retrieve(question, k=TOP_K):
    res = _policies.query(query_texts=[question], n_results=k)
    return res["documents"][0], res["metadatas"][0]


def llm(prompt, system=None):
    messages = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": prompt})
    return ollama.chat(model=MODEL, messages=messages)["message"]["content"]


def clean(text):
    """Output validator: strip tool_call tags AND lone gibberish first tokens."""
    text = re.sub(r"^(\s*</?tool_call>\s*)+", "", text.strip())
    lines = text.strip().splitlines()
    if len(lines) > 1 and " " not in lines[0].strip() and len(lines[0].strip()) <= 15:
        lines = lines[1:]                      # drops junk like '줫' or 'ontvangst'
    return "\n".join(lines).strip()


def extract_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"tool": None}
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return {"tool": None}


def valid_pnr(pnr):
    return isinstance(pnr, str) and re.fullmatch(r"[A-Za-z]{3}\d{3}", pnr)


@app.post("/v1/chat")
def chat(req: ChatRequest):
    t0 = time.time()
    decision = extract_json(llm(ROUTER_PROMPT.format(question=req.message)))
    tool = decision.get("tool")

    if tool == "ask_pnr":
        route, reply, sources = "clarify", CLARIFY_PNR, []

    elif tool in TOOL_REGISTRY:
        pnr = decision.get("arguments", {}).get("pnr", "")
        if not valid_pnr(pnr):
            route, reply, sources = "clarify", f"'{pnr}' does not look like a valid PNR. Could you re-check it?", []
        else:
            result = TOOL_REGISTRY[tool](pnr)
            docs, metas = retrieve(req.message)
            reply = clean(llm(TOOL_RESPONSE_PROMPT.format(
                question=req.message, result=json.dumps(result),
                context="\n\n---\n\n".join(docs))))
            route = f"tool:{tool}"
            sources = [f"backend:{tool}({pnr})"] + [m["source"] for m in metas]

    else:  # policy / general -> RAG
        docs, metas = retrieve(req.message)
        reply = clean(llm(RAG_PROMPT.format(
            context="\n\n---\n\n".join(docs), question=req.message), system=RAG_SYSTEM))
        route = "rag"
        sources = [f"{m['source']} [{m['section']}]" for m in metas]

    latency_ms = round((time.time() - t0) * 1000)
    print(f"[trace] route={route} latency_ms={latency_ms} sources={sources}")
    return {"reply": reply, "route": route, "sources": sources, "latency_ms": latency_ms}