"""Chat backend switch for the runtime bots: local Ollama (default) or Azure OpenAI.

Set in .env:
    LLM_BACKEND=ollama   -> OLLAMA_MODEL (default "acme-support", the fine-tuned GGUF)
    LLM_BACKEND=azure    -> AZURE_OPENAI_DEPLOYMENT

Every call is recorded in Langfuse as a "generation" (model, tokens, cost) when tracing is on.
"""

import contextvars
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langfuse import get_client, observe

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

BACKEND = (os.environ.get("LLM_BACKEND") or "ollama").strip().lower()  # blank in .env = default

# acme-support has its own persona baked into its Modelfile; any other model gets this one.
# It used to say "structure every response with empathy, options, a policy note, and a
# next-step question" (the fine-tuning template). The C5 tool-call eval showed what that does
# when there are no real options: an on-time status reply invented "set up a status alert,
# online check-in, seat requests" and mentioned refunds/vouchers (3 of 4 runs). So the persona
# now lists what the assistant can actually do, and money is mentioned only when policy says so.
PERSONA = (
    "You are a polite and empathetic customer support assistant for ACME Bharat Airlines. "
    "Keep replies short: acknowledge the customer, state the facts you were given exactly, "
    "and end with one helpful question. Only offer things this assistant can actually do: "
    "check a booking's status, explain ACME policy, cancel a booking, show the options for a "
    "delayed or cancelled flight, or connect the customer to a person. Never mention refunds, "
    "vouchers or compensation unless the policy text you were given says they apply."
)

if BACKEND == "azure":
    # Langfuse's drop-in OpenAI client: identical API, and each call becomes a traced
    # generation with model, token counts and cost - no hand-written tracing needed.
    from langfuse.openai import OpenAI

    # Azure OpenAI v1 API: plain OpenAI client pointed at <endpoint>/openai/v1/, no api-version.
    _client = OpenAI(
        base_url=os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/") + "/openai/v1/",
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
    )
    MODEL = os.environ["AZURE_OPENAI_DEPLOYMENT"]
elif BACKEND == "ollama":
    import ollama

    MODEL = os.environ.get("OLLAMA_MODEL") or "acme-support"
else:
    raise ValueError(f"LLM_BACKEND must be 'ollama' or 'azure', got {BACKEND!r}")


# Per-request record of model calls, for the chat page's "How I decided" panel and /insights.
# The API sets a fresh list per request; shadow mode's background calls set None so they never
# land in a customer's request.
calls = contextvars.ContextVar("llm_calls", default=None)


def _record(name, started, usage):
    log = calls.get()
    if log is None:
        return
    details = getattr(usage, "completion_tokens_details", None)
    log.append({"name": name, "ms": round((time.perf_counter() - started) * 1000),
                "input_tokens": getattr(usage, "prompt_tokens", None),
                "output_tokens": getattr(usage, "completion_tokens", None),
                "reasoning_tokens": getattr(details, "reasoning_tokens", None)})


def chat(messages, name="llm", persona=False, reasoning_effort=None):
    """Send a chat-format message list, return the reply text. `name` labels the trace step.

    persona=True adds the airline persona as the system message - ONLY for customer-facing
    replies that don't bring their own. It used to be added to every call without a system
    message, and tracing showed it contradicting task prompts ("options, a policy note…" vs.
    "don't list options") and padding the router's JSON call.
    """
    if (persona and MODEL != "acme-support"
            and not any(m["role"] == "system" for m in messages)):
        messages = [{"role": "system", "content": PERSONA}] + messages
    if BACKEND == "azure":
        # reasoning_effort (minimal | low | medium | high): how much hidden "thinking" a
        # reasoning model does before answering - traced at up to 94% of a call's cost (C1)
        extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
        started = time.perf_counter()
        resp = _client.chat.completions.create(model=MODEL, messages=messages, name=name, **extra)
        _record(name, started, resp.usage)
        return resp.choices[0].message.content
    started = time.perf_counter()
    reply, usage = _ollama_chat(messages, name)
    _record(name, started, usage)
    return reply


@observe(as_type="generation")
def _ollama_chat(messages, name):
    resp = ollama.chat(model=MODEL, messages=messages)
    get_client().update_current_generation(
        name=name, model=MODEL,
        usage_details={"input": resp.get("prompt_eval_count") or 0,
                       "output": resp.get("eval_count") or 0})
    usage = type("Usage", (), {"prompt_tokens": resp.get("prompt_eval_count"),
                               "completion_tokens": resp.get("eval_count")})
    return resp["message"]["content"], usage
