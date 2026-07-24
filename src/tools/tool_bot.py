"""Day 9: tool calling - the LLM REQUESTS actions as JSON; Python validates + executes."""

import argparse
import json
import re

import ollama
from backend import get_flight_status, cancel_ticket

MODEL = "acme-support"

TOOL_REGISTRY = {
    "get_flight_status": get_flight_status,
    "cancel_ticket": cancel_ticket,
}

ROUTER_PROMPT = """You are a request router for ACME Bharat Airlines.
Decide if the customer message requires calling a backend tool.

Available tools:
1. get_flight_status(pnr) - live status of a booking. Needs a PNR (format: 3 letters + 3 digits, e.g. ACX123).
2. cancel_ticket(pnr) - cancel a booking and compute the refund. Needs a PNR.

Rules - output ONLY one JSON object, nothing else:
- Tool needed, PNR present:  {{"tool": "<tool_name>", "arguments": {{"pnr": "<PNR>"}}}}
- Tool needed, PNR missing:  {{"tool": "ask_pnr"}}
- No tool needed (policy or general question): {{"tool": null}}

Customer message: {question}
JSON:"""

FINAL_PROMPT = """The customer asked: {question}

The backend system returned this verified result:
{result}

Reply to the customer using ONLY the facts in the backend result. Be empathetic,
state the facts exactly (statuses, amounts), and end with one next-step question."""


def extract_json(text):
    """Pull the first {...} out of model output (junk-token tolerant)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"tool": None}
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return {"tool": None}


def valid_pnr(pnr):
    """NEVER execute with unvalidated arguments - the security half of the iron rule."""
    return isinstance(pnr, str) and re.fullmatch(r"[A-Za-z]{3}\d{3}", pnr)


def chat(prompt):
    resp = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    return resp["message"]["content"].strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("question")
    args = p.parse_args()

    print("[1] ROUTER deciding...")
    decision = extract_json(chat(ROUTER_PROMPT.format(question=args.question)))
    print(f"    decision: {json.dumps(decision)}")

    tool = decision.get("tool")

    if tool == "ask_pnr":
        print("\n[bot] Could you please share your 6-character PNR (e.g. ACX123) "
              "so I can look up your booking?")
        return

    if tool not in TOOL_REGISTRY:
        print("\n[bot] No tool needed - this is a policy/general question. "
              "(Day 10 will route this to the RAG layer.)")
        return

    pnr = decision.get("arguments", {}).get("pnr", "")
    if not valid_pnr(pnr):
        print(f"\n[bot] '{pnr}' is not a valid PNR format. Could you re-check it?")
        return

    print(f"[2] EXECUTING {tool}('{pnr}') in Python (not in the model)...")
    result = TOOL_REGISTRY[tool](pnr)
    print(f"    backend result: {json.dumps(result)}")

    print("[3] RESPONDER composing the customer reply...")
    reply = chat(FINAL_PROMPT.format(question=args.question, result=json.dumps(result)))
    reply = re.sub(r"^(\s*</?tool_call>\s*)+", "", reply).strip()
    print("\n" + "=" * 70)
    print(reply)


if __name__ == "__main__":
    main()