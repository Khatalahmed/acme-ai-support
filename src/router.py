"""Request router: decide RAG vs tool vs clarify for each customer message.

Two interchangeable routers, picked by ROUTER_BACKEND in .env:
    llm  (default) - the LLM writes a JSON decision, we parse it (llm_backend model)
    jev            - TypeSafe's Jev answers one typed `choice` question; Python extracts the PNR
                     (called directly with TYPESAFE_API_KEY, or via Vercel AI Gateway with
                     AI_GATEWAY_API_KEY)

Both return the same decision shape the API already understands:
    {"tool": "get_flight_status" | "cancel_ticket", "arguments": {"pnr": "ACX123"}}
    {"tool": "ask_pnr"}
    {"tool": None}
plus "router" (which router answered) and "confidence" (Jev only).
"""

import json
import os
import re

import httpx

from llm_backend import chat

# Same request/response shape on both; only URL, key and model name differ.
JEV_PROVIDERS = {
    "TYPESAFE_API_KEY": ("https://api.typesafe.ai/v1/systemone", "jev-latest"),
    "AI_GATEWAY_API_KEY": ("https://ai-gateway.vercel.sh/typesafe/v1/systemone", "typesafe-ai/jev"),
}
PNR_PATTERN = re.compile(r"\b([A-Za-z]{3}\d{3})\b")
# "acx 789" / "ACX-789" only right after "pnr"/"booking" - anywhere else, a 3-letter word
# before a 3-digit number ("for 500 rupees") would be misread as a PNR.
PNR_SPACED = re.compile(r"\b(?:pnr|booking)\W{0,3}(?:is\W+|no\W{0,2}|number\W+)?"
                        r"([A-Za-z]{3})[\s-](\d{3})\b", re.IGNORECASE)
TOOLS = ("get_flight_status", "cancel_ticket")


def find_pnr(message):
    """First PNR in the message, normalised to ACX123 form, or None."""
    m = PNR_PATTERN.search(message)
    if m:
        return m.group(1).upper()
    m = PNR_SPACED.search(message)
    return (m.group(1) + m.group(2)).upper() if m else None

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

# Jev picks ONE of these; it never writes JSON and never supplies tool arguments.
JEV_INTENT = {
    "type": "choice",
    "instructions": (
        "Classify what this ACME Bharat Airlines customer needs next. Pick "
        "get_flight_status or cancel_ticket ONLY when they want a lookup or action on "
        "their own specific booking, whether or not they gave a booking reference."
    ),
    "criteria": {
        "get_flight_status": "Wants the live status, timing, gate or delay of their own booked flight.",
        "cancel_ticket": "Wants to cancel their own booked flight now.",
        "policy": "A general question about rules, allowances, fees, compensation or refund "
                  "policy, or anything else that needs no lookup or action on a booking.",
    },
}


def extract_json(text):
    """Pull the first {...} out of model output (junk-token tolerant)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"tool": None}
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return {"tool": None}


def route_llm(message):
    decision = extract_json(chat([{"role": "user", "content": ROUTER_PROMPT.format(question=message)}]))
    args = decision.get("arguments")
    if isinstance(args, dict) and isinstance(args.get("pnr"), str):
        args["pnr"] = re.sub(r"[\s-]", "", args["pnr"]).upper()  # "acx 789" -> "ACX789"
    return {**decision, "router": "llm", "confidence": None}


def jev_provider():
    """(env var, url, default model) for the first Jev key found: TypeSafe direct, else Vercel."""
    for env, (url, model) in JEV_PROVIDERS.items():
        if os.environ.get(env):
            return env, url, model
    raise KeyError("no Jev key: set TYPESAFE_API_KEY or AI_GATEWAY_API_KEY in .env")


def route_jev(message, timeout=10.0):
    """One Jev call. Raises on HTTP errors - route() decides whether to fall back."""
    env, url, default_model = jev_provider()
    resp = httpx.post(
        url,
        headers={"Authorization": f"Bearer {os.environ[env]}"},
        json={
            "model": os.environ.get("JEV_MODEL") or default_model,  # blank in .env = default
            "state": {"customer_message": message},
            "questions": {"intent": JEV_INTENT},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    body = resp.json()
    answer = body["answers"]["intent"]
    intent, confidence = answer["choice"], answer.get("confidence")
    input_tokens = body.get("usage", {}).get("input_tokens")

    if intent not in TOOLS:
        decision = {"tool": None}
    else:
        # The PNR comes from a regex, never from a model: nothing unvalidated reaches a tool.
        pnr = find_pnr(message)
        decision = ({"tool": intent, "arguments": {"pnr": pnr}} if pnr
                    else {"tool": "ask_pnr"})
    return {**decision, "router": "jev", "confidence": confidence, "input_tokens": input_tokens}


def route(message):
    """The configured router. Jev falls back to the LLM when unsure or unreachable."""
    if (os.environ.get("ROUTER_BACKEND") or "llm").strip().lower() != "jev":
        return route_llm(message)
    try:
        decision = route_jev(message)
    except (httpx.HTTPError, KeyError) as e:
        print(f"[router] Jev failed ({e!r}) - falling back to LLM router")
        return {**route_llm(message), "router": "jev->llm"}
    if decision["confidence"] is not None and \
            decision["confidence"] < float(os.environ.get("JEV_MIN_CONFIDENCE") or "0.6"):
        return {**route_llm(message), "router": "jev->llm"}
    return decision
