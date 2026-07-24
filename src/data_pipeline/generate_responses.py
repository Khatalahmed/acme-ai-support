"""Day 2 - Stage 2: generate one AER-template response per complaint.

Resilience stack (each layer catches what the previous one can't):
  1. pacing (sleep)           -> avoid hitting the rate limit at all
  2. retry + backoff          -> absorb transient 429/500/503 errors
  3. fallback model           -> survive the primary model being unavailable
  4. resume file (load_done)  -> survive the whole script dying mid-run
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

# This file lives at src/data_pipeline/, so project root is 3 levels up
ROOT = Path(__file__).resolve().parent.parent.parent

IN_PATH = ROOT / "data" / "raw" / "complaints.jsonl"
OUT_PATH = ROOT / "data" / "raw" / "pairs.jsonl"

RESPONSE_PROMPT = """You are the ideal ACME Bharat Airlines customer care executive.
Write ONE reply to the customer complaint below.

The reply MUST follow the AER structure, in this exact order:
1. Empathy: begin with a sincere apology/acknowledgement (e.g. "We sincerely apologize...").
2. Options: offer 2-3 concrete options (refund / rebooking / voucher / escalation, as appropriate).
3. Policy note: one policy-safe sentence containing the phrase "As per ACME Bharat Airlines policy".
4. Next step: end with ONE clarifying question (e.g. "May I know your preference?").

Rules:
- 60 to 120 words. Professional, warm, Indian English.
- Address the customer ONLY as "Sir/Ma'am". NEVER invent a customer name.
- NEVER invent reference numbers, PNRs, flight numbers, or amounts not present in the complaint.
- Output ONLY the reply text, nothing else.

Customer complaint:
{complaint}
"""


def load_done():
    """Resume support: if a previous run was interrupted, skip finished complaints."""
    done = set()
    if OUT_PATH.exists():
        for line in open(OUT_PATH, encoding="utf-8"):
            done.add(json.loads(line)["complaint"])
    return done


def generate_with_retry(client, model, prompt, config, max_retries=3):
    """Call one model; on 429/500/503 wait and retry with doubling delays."""
    delay = 10
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model, contents=prompt, config=config
            )
        except errors.APIError as e:
            retryable = getattr(e, "code", None) in (429, 500, 503)
            if retryable and attempt < max_retries - 1:
                print(f"  {model}: {e.code} - waiting {delay}s (retry {attempt + 1}/{max_retries})")
                time.sleep(delay)
                delay *= 2
            else:
                raise


def generate_with_fallback(client, models, prompt, config):
    """Try each model in order. Returns (response, model_that_answered)."""
    last_err = None
    for m in models:
        try:
            return generate_with_retry(client, m, prompt, config), m
        except errors.APIError as e:
            print(f"  {m} exhausted retries ({getattr(e, 'code', '?')}) - falling back")
            last_err = e
    raise last_err


def main():
    load_dotenv(ROOT / ".env")
    client = genai.Client(
        vertexai=True,
        project=os.environ["GCP_PROJECT_ID"],
        location=os.environ.get("VERTEX_LOCATION", "global"),
    )
    models = [
        os.environ.get("VERTEX_MODEL", "gemini-3.5-flash"),
        os.environ.get("VERTEX_MODEL_FALLBACK", "gemini-2.5-flash"),
    ]
    config = types.GenerateContentConfig(temperature=0.4)

    done = load_done()
    records = [json.loads(line) for line in open(IN_PATH, encoding="utf-8")]
    print(f"{len(records)} complaints | {len(done)} already answered (will skip)")
    print(f"primary model: {models[0]} | fallback: {models[1]}")

    with open(OUT_PATH, "a", encoding="utf-8") as f:
        for i, rec in enumerate(records, 1):
            if rec["complaint"] in done:
                continue
            resp, used = generate_with_fallback(
                client, models,
                RESPONSE_PROMPT.format(complaint=rec["complaint"]),
                config,
            )
            time.sleep(1.5)  # pacing: ~40 requests/min, stays under quota
            rec["response"] = resp.text.strip()
            rec["model"] = used
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 10 == 0:
                print(f"progress: {i}/{len(records)}")

    print(f"\nAll pairs saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()