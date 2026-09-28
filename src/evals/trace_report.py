"""Break down a conversation's Langfuse traces: time, tokens and cost per step.

Uses the v2 observations API (the legacy traces API is off for organisations created after
2026-09-16). Needs LANGFUSE_* keys in .env; traces appear a few seconds after a request.

Usage:
    uv run python src/evals/trace_report.py <session_id> [--hours 24]
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
from langfuse import get_client  # noqa: E402

FIELDS = "core,basic,time,io,metadata,model,usage"


def seconds(o):
    return (o.end_time - o.start_time).total_seconds() if o.end_time else 0.0


def report(session_id, hours):
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    obs = get_client().api.observations.get_many(
        session_id=session_id, fields=FIELDS, expand_metadata="replaced",
        from_start_time=since, limit=1000).data
    if not obs:
        print(f"No observations for session {session_id!r} in the last {hours} h "
              "(tracing off, wrong session, or not ingested yet - retry in a few seconds).")
        return

    traces = defaultdict(list)
    for o in obs:
        traces[o.trace_id].append(o)

    grand = 0.0
    for items in sorted(traces.values(), key=lambda xs: min(o.start_time for o in xs)):
        items.sort(key=lambda o: o.start_time)
        root = next((o for o in items if o.name == "POST /v1/chat"), items[0])
        data = root.input
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except ValueError:
                pass
        message = data.get("message") if isinstance(data, dict) else data
        cost = sum((o.cost_details or {}).get("total", 0) for o in items)
        grand += cost
        print(f"\n== {str(message)[:60]!r}  {seconds(root):.2f}s  ${cost:.6f}")
        for o in items:
            if o is root:
                continue
            line = f"   {seconds(o):6.2f}s  {str(o.type):11} {o.name}"
            if o.usage_details:
                u = o.usage_details
                reasoning = u.get("output_reasoning_tokens")
                line += (f"  {o.model}  in={u.get('input')} out={u.get('output')}"
                         + (f" reasoning={reasoning}" if reasoning else "")
                         + f"  ${(o.cost_details or {}).get('total', 0):.6f}")
            if o.name.startswith("guardrail."):
                line += f"  replaced={(o.metadata or {}).get('replaced')}"
            print(line)
    print(f"\n{len(traces)} requests, total ${grand:.6f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("session_id")
    p.add_argument("--hours", type=int, default=24)
    args = p.parse_args()
    report(args.session_id, args.hours)
