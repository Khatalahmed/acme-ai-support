"""Day 2 - Stage 1: generate realistic raw complaints (text only, NO JSON yet)."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

# This file lives at src/data_pipeline/, so project root is 3 levels up
ROOT = Path(__file__).resolve().parent.parent.parent

# The 12 scenario buckets from the 02 March lesson
BUCKETS = {
    "flight-delay":          "flight delays (short, long, overnight, weather, technical)",
    "flight-cancellation":   "flight cancellations (airline-initiated, weather, passenger)",
    "missed-connection":     "missed connecting flights (same PNR, different PNR)",
    "rebooking":             "rebooking requests (same day, different city)",
    "refunds":               "refunds (refundable, non-refundable, partial, voucher)",
    "baggage":               "baggage issues (delayed, damaged, lost, oversize)",
    "seat-changes":          "seat changes (upgrade, downgrade, special request)",
    "special-assistance":    "special assistance (wheelchair, medical, minor, pregnancy)",
    "loyalty-points":        "loyalty program issues (missing points, tier upgrade, expiry)",
    "payment-failures":      "payment failures (double charge, failed transaction)",
    "complaints-escalation": "poor service, rude staff behaviour, supervisor requests",
    "compensation":          "compensation demands for disruptions",
}

PER_BUCKET = 20    # 12 buckets x 20 = 240 raw complaints (over-generate on purpose)
MIN_CHARS = 150    # class rule: shorter than this is not a realistic complaint

PROMPT_TEMPLATE = """You are an expert data generator.

Generate {n} realistic airline customer complaints for ACME Bharat Airlines.
Topic for this batch: {topic}.
Context: Indian passengers, domestic travel. Vary cities, tone, and severity.
Each complaint must be at least 150 characters and sound like a real frustrated human.

Example complaint 1 (delay + missed connection):
My flight was supposed to land in Delhi at 1 PM but it arrived at 6 PM. Because of
this delay I missed my connecting flight and had to spend Rs 8000 on another ticket.
I need a refund. Please help.

Example complaint 2 (baggage damage):
My baggage was damaged after my flight from Amritsar. Customer care asked me to send
emails and evidence but after 10 days there is still no resolution. This is mentally
frustrating.

Output rules:
- Output ONLY the complaints. No numbering, no headings, no commentary.
- Separate each complaint with a line containing exactly: ###
"""


def parse_complaints(text):
    """Split model output into complaints. Primary: '###'. Fallback: blank lines."""
    parts = [p.strip() for p in text.split("###") if p.strip()]
    if len(parts) <= 1:
        # Model ignored our separator - fall back to blank-line splitting
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    return parts


def main():
    load_dotenv(ROOT / ".env")
    client = genai.Client(
        vertexai=True,
        project=os.environ["GCP_PROJECT_ID"],
        location=os.environ.get("VERTEX_LOCATION", "global"),
    )
    model = os.environ.get("VERTEX_MODEL", "gemini-2.5-flash")

    out_path = ROOT / "data" / "raw" / "complaints.jsonl"
    kept, dropped = 0, 0
    with open(out_path, "w", encoding="utf-8") as f:
        for bucket, topic in BUCKETS.items():
            prompt = PROMPT_TEMPLATE.format(n=PER_BUCKET, topic=topic)
            resp = client.models.generate_content(model=model, contents=prompt)
            bucket_kept = 0
            for complaint in parse_complaints(resp.text):
                if len(complaint) < MIN_CHARS:
                    dropped += 1
                    continue
                f.write(json.dumps({"bucket": bucket, "complaint": complaint}) + "\n")
                kept += 1
                bucket_kept += 1
            warn = "  <-- WARNING: low yield, check this bucket!" if bucket_kept < 10 else ""
            print(f"{bucket}: kept {bucket_kept}{warn}")

    print(f"\nSaved {kept} complaints -> {out_path}  ({dropped} dropped as too short)")


if __name__ == "__main__":
    main()