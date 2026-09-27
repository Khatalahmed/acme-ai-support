"""Chat backend switch for the runtime bots: local Ollama (default) or Azure OpenAI.

Set in .env:
    LLM_BACKEND=ollama   -> OLLAMA_MODEL (default "acme-support", the fine-tuned GGUF)
    LLM_BACKEND=azure    -> AZURE_OPENAI_DEPLOYMENT
"""

import os
from pathlib import Path

from dotenv import load_dotenv

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
    from openai import OpenAI

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


def chat(messages):
    """Send a chat-format message list, return the reply text."""
    if MODEL != "acme-support" and not any(m["role"] == "system" for m in messages):
        messages = [{"role": "system", "content": PERSONA}] + messages
    if BACKEND == "azure":
        resp = _client.chat.completions.create(model=MODEL, messages=messages)
        return resp.choices[0].message.content
    return ollama.chat(model=MODEL, messages=messages)["message"]["content"]
