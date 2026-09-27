"""Router comparison: LLM JSON router vs Jev typed-decision router.

Runs every message in data/evals/routing_set.jsonl through each router, resolves the
decision exactly as src/api/main.py does (invalid PNR -> clarify), and reports accuracy,
per-route accuracy, latency, Jev cost and every miss. A record's "expected" may be a list
of acceptable routes (e.g. the prompt-injection case: any route except a tool).

Usage:
    uv run python src/evals/router_compare.py                 # both routers
    uv run python src/evals/router_compare.py --routers llm   # one router
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from router import TOOLS, jev_provider, route_jev, route_llm  # also loads .env via llm_backend

SET_PATH = ROOT / "data" / "evals" / "routing_set.jsonl"
OUT_PATH = ROOT / "data" / "evals" / "router_compare.jsonl"
ROUTERS = {"llm": route_llm, "jev": route_jev}
JEV_USD_PER_MTOK = 0.042  # vendor list price, input tokens; output is free


def resolve(decision):
    """Decision -> (route, pnr), mirroring the branches in src/api/main.py.

    Grades the ROUTER: a cancel decision counts as tool:cancel_ticket even though the API
    now asks the customer to confirm before running it."""
    tool = decision.get("tool")
    if tool == "ask_pnr":
        return "clarify", None
    if tool in TOOLS:
        pnr = (decision.get("arguments") or {}).get("pnr") or ""
        if not re.fullmatch(r"[A-Za-z]{3}\d{3}", pnr):
            return "clarify", None
        return f"tool:{tool}", pnr.upper()
    return "rag", None


def pct(n, d):
    return f"{100 * n / d:.0f}%" if d else "-"


def run(name, fn, records):
    rows = []
    for i, rec in enumerate(records):
        t0 = time.perf_counter()
        try:
            decision = fn(rec["message"])
            got, pnr = resolve(decision)
            error = None
        except Exception as e:
            if i == 0:  # failing on the very first call = bad key/endpoint, not a routing miss
                sys.exit(f"{name} router failed on its first call - check .env:\n  {e!r}")
            decision, got, pnr, error = {}, "error", None, repr(e)  # later failures count as misses
        ms = (time.perf_counter() - t0) * 1000
        # "expected" is one route, or a list of acceptable routes
        allowed = rec["expected"] if isinstance(rec["expected"], list) else [rec["expected"]]
        # the PNR only matters when a tool actually runs; rag/clarify never carry one
        correct = got in allowed and (pnr == rec["pnr"] if got.startswith("tool:") else True)
        rows.append({"router": name, "id": rec["id"], "set": rec.get("set", "base"),
                     "message": rec["message"],
                     "expected": "|".join(allowed), "got": got, "pnr": pnr,
                     "expected_pnr": rec["pnr"], "correct": correct, "latency_ms": round(ms),
                     "confidence": decision.get("confidence"),
                     "input_tokens": decision.get("input_tokens"), "error": error})
        print(f"  {name:3} {rec['id']}  {'ok ' if correct else 'MISS'}  {round(ms):6d} ms  "
              f"{got}{' ' + pnr if pnr else ''}")
    return rows


def report(name, rows):
    n, hits = len(rows), sum(r["correct"] for r in rows)
    lat = sorted(r["latency_ms"] for r in rows)
    print("\n" + "=" * 70)
    print(f"{name.upper()} ROUTER  accuracy {hits}/{n} ({pct(hits, n)})")
    by_set = defaultdict(list)
    for r in rows:
        by_set[r["set"]].append(r["correct"])
    print("  " + " | ".join(f"{s} set {sum(o)}/{len(o)} ({pct(sum(o), len(o))})"
                            for s, o in by_set.items()))
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["expected"]].append(r["correct"])
    for label, oks in sorted(by_label.items()):
        print(f"  {label:24} {sum(oks)}/{len(oks)} ({pct(sum(oks), len(oks))})")
    print(f"  latency ms: mean {statistics.mean(lat):.0f} | p50 {lat[n // 2]} | "
          f"p95 {lat[min(n - 1, int(n * 0.95))]} | max {lat[-1]}")

    confs = [(r["confidence"], r["correct"]) for r in rows if r["confidence"] is not None]
    if confs:
        parts = []
        for label, keep in (("hits", True), ("misses", False)):
            vals = [c for c, ok in confs if ok == keep]
            parts.append(f"mean on {label} {statistics.mean(vals):.2f}" if vals else f"no {label}")
        print("  confidence: " + " | ".join(parts))
    tokens = sum(r["input_tokens"] or 0 for r in rows)
    if tokens:
        print(f"  input tokens {tokens} -> ${tokens * JEV_USD_PER_MTOK / 1e6:.6f} at list price")

    misses = [r for r in rows if not r["correct"]]
    if misses:
        print("  misses:")
        for r in misses:
            extra = f"  [{r['error']}]" if r["error"] else ""
            print(f"    {r['id']} expected {r['expected']} {r['expected_pnr'] or ''}| "
                  f"got {r['got']} {r['pnr'] or ''}| {r['message']}{extra}")
    # the costliest error: choosing cancel_ticket for a message that didn't ask for it
    # (the API's confirmation step stops these reaching the backend)
    bad_cancels = [r for r in rows if r["got"] == "tool:cancel_ticket"
                   and "tool:cancel_ticket" not in r["expected"].split("|")]
    print(f"  wrongful cancel decisions (blocked by API confirmation): {len(bad_cancels)}"
          + "".join(f"\n    {r['id']} {r['message']}" for r in bad_cancels))

    hard = by_set.get("hard", [])
    return {"router": name, "accuracy": hits / n, "hits": hits, "n": n,
            "hard_hits": sum(hard), "hard_n": len(hard), "bad_cancels": len(bad_cancels),
            "mean_ms": statistics.mean(lat), "p50_ms": lat[n // 2]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--routers", nargs="+", choices=list(ROUTERS), default=list(ROUTERS))
    args = p.parse_args()

    routers = list(args.routers)
    if "jev" in routers:
        try:
            env, url, _ = jev_provider()
            print(f"Jev via {url} ({env})")
        except KeyError:
            print("No TYPESAFE_API_KEY or AI_GATEWAY_API_KEY in .env - skipping the Jev router.\n")
            routers.remove("jev")
    if not routers:
        return

    records = [json.loads(l) for l in open(SET_PATH, encoding="utf-8")]
    print(f"{len(records)} labelled messages | routers: {', '.join(routers)}\n")

    all_rows, summaries = [], []
    for name in routers:
        rows = run(name, ROUTERS[name], records)
        all_rows += rows
        summaries.append(report(name, rows))

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")

    print("\n" + "=" * 70)
    print("| Router | Accuracy | Hard set | Wrongful cancel decisions | Mean latency | p50 latency |")
    print("|---|---|---|---|---|---|")
    for s in summaries:
        print(f"| {s['router']} | {s['hits']}/{s['n']} ({s['accuracy']:.0%}) | "
              f"{s['hard_hits']}/{s['hard_n']} | {s['bad_cancels']} | "
              f"{s['mean_ms']:.0f} ms | {s['p50_ms']} ms |")
    print(f"\nper-message results -> {OUT_PATH}")


if __name__ == "__main__":
    main()
