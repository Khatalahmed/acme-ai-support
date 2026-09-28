"""Request router: decide RAG vs tool vs clarify for each customer message.

Two interchangeable routers, picked by ROUTER_BACKEND in .env:
    llm  (default) - the LLM writes a JSON decision, we parse it (llm_backend model)
    jev            - TypeSafe's Jev answers one typed `choice` question; Python extracts the PNR
                     (called directly with TYPESAFE_API_KEY, or via Vercel AI Gateway with
                     AI_GATEWAY_API_KEY)

Both return the same decision shape the API already understands:
    {"tool": "get_flight_status" | "cancel_ticket" | "disruption_help", "arguments": {"pnr": ...}}
    {"tool": "human_agent"}      - customer asks for a person
    {"tool": "ask_pnr"}          - status/cancel without a PNR
    {"tool": None}               - policy / general (disruption questions without a PNR too)
plus "router", "confidence" (Jev only) and "risk": {"angry", "demands_exception"} in 0..1,
which the disruption agent uses to decide when to hand over to a human.
"""

import json
import os
import re

import httpx
from langfuse import get_client, observe

from llm_backend import chat

JEV_USD_PER_TOKEN = 0.042 / 1_000_000   # list price, input tokens; output is free

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
TOOLS = ("get_flight_status", "cancel_ticket", "disruption_help")  # need a booking (PNR)
ASK_PNR_TOOLS = ("get_flight_status", "cancel_ticket")  # without a PNR: ask for it
NO_RISK = {"angry": 0.0, "demands_exception": 0.0}


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
3. disruption_help(pnr) - their own flight is delayed or cancelled and they want their options:
   compensation, rebooking, a voucher or a refund for that disruption. Needs a PNR.
4. human_agent - they explicitly ask for a human, a manager or a supervisor.

Rules - output ONLY one JSON object, nothing else:
- Tool needed, PNR present:  {{"tool": "<tool_name>", "arguments": {{"pnr": "<PNR>"}}}}
- Tool needed, PNR missing:  {{"tool": "ask_pnr"}}
- Asks for a human:          {{"tool": "human_agent"}}
- No tool needed (policy or general question): {{"tool": null}}
Always also add "angry": true/false (hostile or very upset, not just disappointed) and
"demands_exception": true/false (demands something beyond normal policy, e.g. a refund on a
non-refundable fare or compensation "or else").

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
        "disruption_help": "Their own flight is delayed or cancelled and they want their options: "
                           "compensation, rebooking, a voucher or a refund for that disruption.",
        "human_agent": "Explicitly asks to talk to a human, a manager or a supervisor.",
        "policy": "A general question about rules, allowances, fees, compensation or refund "
                  "policy, or anything else that needs no lookup or action on a booking.",
    },
}
# Extra questions in the SAME Jev call: answered in parallel, so they add almost no latency.
JEV_RISK = {
    "angry": {"type": "noul", "instructions": "Is the customer angry, hostile or threatening - "
                                              "not just mildly disappointed?"},
    "demands_exception": {"type": "noul", "instructions": (
        "Is the customer demanding something beyond normal airline policy - for example a refund "
        "on a non-refundable ticket, compensation 'or else', or an exception to the rules?")},
}


def outcome(decision):
    """Decision -> (route, pnr): what the customer would actually get, mirroring the API.

    The single definition of "same routing result", shared by the benchmark (router_compare)
    and shadow mode, so the two can never disagree about what "agree" means.
    """
    tool = decision.get("tool")
    if tool == "ask_pnr":
        return "clarify", None
    if tool == "human_agent":
        return "escalate", None
    if tool in TOOLS:
        pnr = (decision.get("arguments") or {}).get("pnr") or ""
        if not re.fullmatch(r"[A-Za-z]{3}\d{3}", pnr):
            return "clarify", None
        return f"tool:{tool}", pnr.upper()
    return "rag", None


def _finish(decision, message):
    """Shared post-processing: PNR rules per tool, so both routers behave identically."""
    tool = decision.get("tool")
    if tool == "disruption_help" and not decision.get("arguments", {}).get("pnr"):
        return {"tool": None}                     # general question: answer from policy (RAG)
    if tool in ASK_PNR_TOOLS and not decision.get("arguments", {}).get("pnr"):
        return {"tool": "ask_pnr"}
    return decision


def extract_json(text):
    """Pull the first {...} out of model output (junk-token tolerant)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"tool": None}
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return {"tool": None}


@observe(name="router.llm")
def route_llm(message):
    raw = extract_json(chat([{"role": "user", "content": ROUTER_PROMPT.format(question=message)}],
                            name="router.llm"))
    risk = {k: 1.0 if raw.get(k) is True else 0.0 for k in NO_RISK}
    decision = {k: v for k, v in raw.items() if k in ("tool", "arguments")}
    args = decision.get("arguments")
    if isinstance(args, dict) and isinstance(args.get("pnr"), str):
        args["pnr"] = re.sub(r"[\s-]", "", args["pnr"]).upper()  # "acx 789" -> "ACX789"
    elif decision.get("tool") in TOOLS:
        decision.pop("arguments", None)
    return {**_finish(decision, message), "router": "llm", "confidence": None, "risk": risk}


def jev_provider():
    """(env var, url, default model) for the first Jev key found: TypeSafe direct, else Vercel."""
    for env, (url, model) in JEV_PROVIDERS.items():
        if os.environ.get(env):
            return env, url, model
    raise KeyError("no Jev key: set TYPESAFE_API_KEY or AI_GATEWAY_API_KEY in .env")


@observe(name="router.jev", as_type="generation")
def route_jev(message, timeout=10.0):
    """One Jev call. Raises on HTTP errors - route() decides whether to fall back."""
    env, url, default_model = jev_provider()
    resp = httpx.post(
        url,
        headers={"Authorization": f"Bearer {os.environ[env]}"},
        json={
            "model": os.environ.get("JEV_MODEL") or default_model,  # blank in .env = default
            "state": {"customer_message": message},
            "questions": {"intent": JEV_INTENT, **JEV_RISK},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    body = resp.json()
    answers = body["answers"]
    intent, confidence = answers["intent"]["choice"], answers["intent"].get("confidence")
    input_tokens = body.get("usage", {}).get("input_tokens")
    risk = {k: float(answers.get(k, {}).get("noul") or 0.0) for k in NO_RISK}
    get_client().update_current_generation(
        model=body.get("model"), usage_details={"input": input_tokens or 0},
        cost_details={"input": (input_tokens or 0) * JEV_USD_PER_TOKEN},
        metadata={"intent": intent, "confidence": confidence, "risk": risk})

    if intent == "human_agent":
        decision = {"tool": "human_agent"}
    elif intent not in TOOLS:
        decision = {"tool": None}
    else:
        # The PNR comes from a regex, never from a model: nothing unvalidated reaches a tool.
        pnr = find_pnr(message)
        decision = {"tool": intent, **({"arguments": {"pnr": pnr}} if pnr else {})}
    return {**_finish(decision, message), "router": "jev", "confidence": confidence,
            "input_tokens": input_tokens, "risk": risk}


@observe(name="router")
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
        # Keep Jev's unsure answer: shadow mode compares it with the LLM's for free.
        return {**route_llm(message), "router": "jev->llm", "unsure_jev": decision}
    return decision
