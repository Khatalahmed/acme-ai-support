"""Chat backend switch for the runtime bots: local Ollama (default) or Azure OpenAI.

Set in .env:
    LLM_BACKEND=ollama   -> OLLAMA_MODEL (default "acme-support", the fine-tuned GGUF)
    LLM_BACKEND=azure    -> AZURE_OPENAI_DEPLOYMENT

Every call is recorded in Langfuse as a "generation" (model, tokens, cost) when tracing is on.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langfuse import get_client, observe

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

BACKEND = (os.environ.get("LLM_BACKEND") or "ollama").strip().lower()  # blank in .env = default

# acme-support has this persona baked into its Modelfile; any other model needs it sent.
PERSONA = (
    "You are a polite and empathetic customer support executive of ACME Bharat Airlines. "
    "Always follow company SOP. Structure every response with empathy, options, a policy "
    "note, and a next-step question."
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


def chat(messages, name="llm"):
    """Send a chat-format message list, return the reply text. `name` labels the trace step."""
    if MODEL != "acme-support" and not any(m["role"] == "system" for m in messages):
        messages = [{"role": "system", "content": PERSONA}] + messages
    if BACKEND == "azure":
        resp = _client.chat.completions.create(model=MODEL, messages=messages, name=name)
        return resp.choices[0].message.content
    return _ollama_chat(messages, name)


@observe(as_type="generation")
def _ollama_chat(messages, name):
    resp = ollama.chat(model=MODEL, messages=messages)
    get_client().update_current_generation(
        name=name, model=MODEL,
        usage_details={"input": resp.get("prompt_eval_count") or 0,
                       "output": resp.get("eval_count") or 0})
    return resp["message"]["content"]
