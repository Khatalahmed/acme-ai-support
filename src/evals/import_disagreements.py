"""Import labelled shadow-mode disagreements from Langfuse into the routing benchmark.

In Langfuse (Datasets -> router-disagreements), set an item's Expected output to the correct
route(s), e.g.:
    rag
    clarify|rag
    tool:get_flight_status ACX123
    tool:cancel_ticket ACX456 | clarify
Routes: rag, clarify, escalate, tool:get_flight_status, tool:cancel_ticket, tool:disruption_help.

Labelled items are validated and written to data/evals/routing_set_real.jsonl (category
"real_traffic"), which router_compare.py includes automatically. Unlabelled items are skipped;
invalid labels are reported, never guessed.

Usage:
    uv run python src/evals/import_disagreements.py
"""

import json
import re
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")
from langfuse import get_client  # noqa: E402

DATASET = "router-disagreements"
OUT = ROOT / "data" / "evals" / "routing_set_real.jsonl"
ROUTES = {"rag", "clarify", "escalate", "tool:get_flight_status", "tool:cancel_ticket",
          "tool:disruption_help"}
PNR = re.compile(r"\b([A-Za-z]{3}\d{3})\b")


def parse_label(label):
    """'tool:get_flight_status ACX123 | clarify' -> (["tool:get_flight_status", "clarify"], "ACX123")."""
    if isinstance(label, dict):
        label = label.get("route") or label.get("expected") or ""
    text = str(label).strip()
    routes = [r for r in re.split(r"[|,\s]+", PNR.sub(" ", text)) if r]
    bad = [r for r in routes if r not in ROUTES]
    if not routes or bad:
        raise ValueError(f"unknown route(s) {bad or '(none)'} in label {text!r}")
    pnrs = {p.upper() for p in PNR.findall(text)}
    if len(pnrs) > 1:
        raise ValueError(f"more than one PNR in label {text!r}")
    needs_pnr = any(r.startswith("tool:") for r in routes)
    if needs_pnr and not pnrs:
        raise ValueError(f"tool route without a PNR in label {text!r}")
    return routes, (pnrs.pop() if pnrs else None)


def main():
    items = get_client().get_dataset(DATASET).items
    rows, skipped, invalid = [], 0, []
    for it in items:
        if not it.expected_output:
            skipped += 1
            continue
        try:
            routes, pnr = parse_label(it.expected_output)
        except ValueError as e:
            invalid.append((it.input.get("message", ""), str(e)))
            continue
        rows.append({"id": f"rt-{it.id[:8]}", "category": "real_traffic",
                     "message": it.input["message"],
                     "expected": routes[0] if len(routes) == 1 else routes, "pnr": pnr,
                     "note": f"from shadow mode; trace {it.source_trace_id}"})

    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                   encoding="utf-8")
    print(f"{len(items)} items: {len(rows)} imported, {skipped} unlabelled, {len(invalid)} invalid")
    for msg, err in invalid:
        print(f"  INVALID  {msg[:60]!r}: {err}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
