"""Tool-call evaluation: whole conversations, judged by what the system actually DID.

Each scenario in data/evals/tool_scenarios.jsonl runs through the real API - real router, real
LLM, real safety layer, real backend - against a fresh copy of the airline and a throwaway
database. It is then judged on:
  - actions:    executed actions (from the pending_actions table) vs expected - unwanted
                actions are the number that must stay at 0
  - escalation: did the conversation reach a person when (and only when) it should?
  - replies:    must / must-not phrases (facts stated, nothing promised)
  - state:      the booking afterwards (e.g. now on AC314)

Usage:
    uv run python src/evals/tool_eval.py                        # 1 run
    uv run python src/evals/tool_eval.py --runs 2 --reply-effort low
"""

import argparse
import copy
import json
import os
import re
import statistics
import sys
import tempfile
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if "--trace" not in sys.argv:
    os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
os.environ["ROUTER_SHADOW"] = "off"            # the second opinion isn't what we're measuring
# One "visitor" sends every eval message: the public-demo limits would otherwise block the run.
os.environ["RATE_LIMIT_PER_IP"] = "1000000"
os.environ["DAILY_REQUEST_CAP"] = "1000000"
for p in (ROOT, ROOT / "src", ROOT / "src" / "tools"):
    sys.path.insert(0, str(p))

from fastapi.testclient import TestClient  # noqa: E402

import actions  # noqa: E402
import backend  # noqa: E402
from src.api.main import app  # noqa: E402

SCENARIOS = ROOT / "data" / "evals" / "tool_scenarios.jsonl"


def norm(text):
    """Lower-case, thousands separators removed ("5,400" == "5400")."""
    return re.sub(r"(?<=\d),(?=\d{3})", "", text.lower())


def executed_actions():
    with actions._db() as conn:
        rows = conn.execute("SELECT action, pnr, params FROM pending_actions "
                            "WHERE status = 'executed'").fetchall()
    return Counter((r["action"], r["pnr"], json.loads(r["params"] or "{}").get("option_id"))
                   for r in rows)


def run_scenario(s, snapshot, tmp):
    for live, saved in zip((backend.FLIGHTS, backend.BOOKINGS), copy.deepcopy(snapshot)):
        live.clear()
        live.update(saved)
    os.environ["ACME_DB_PATH"] = str(Path(tmp) / f"{s['id']}-{uuid.uuid4().hex[:6]}.db")

    client, session, turns = TestClient(app), f"eval-{uuid.uuid4().hex[:8]}", []
    for message in s["turns"]:
        t0 = time.perf_counter()
        r = client.post("/v1/chat", json={"message": message, "session_id": session},
                        headers={"Authorization": f"Bearer demo-{s['user']}"})
        body = r.json() if r.status_code == 200 else {"route": f"http_{r.status_code}", "reply": ""}
        turns.append({"message": message, "route": body["route"], "reply": body.get("reply", ""),
                      "ms": round((time.perf_counter() - t0) * 1000)})

    expected = Counter((e["action"], e["pnr"], e["option"]) for e in s["expect_executed"])
    got = executed_actions()
    escalated = any(e == "escalated" for e in (row["event"] for row in actions.audit_trail()))

    problems = []
    missing, unwanted = expected - got, got - expected
    problems += [f"MISSING action {a}" for a in missing.elements()]
    problems += [f"UNWANTED action {a}" for a in unwanted.elements()]
    if escalated != s["expect_escalation"]:
        problems.append(f"escalation expected={s['expect_escalation']} got={escalated}")
    for c in s["reply_checks"]:
        reply = norm(turns[c["turn"]]["reply"])
        if c["any"] and not any(norm(x) in reply for x in c["any"]):
            problems.append(f"turn {c['turn']} reply lacks any of {c['any']}")
        bad = [x for x in c["none"] if norm(x) in reply]
        if bad:
            problems.append(f"turn {c['turn']} reply contains {bad}")
    for st in s["state"]:
        value = backend.BOOKINGS[st["pnr"]][st["field"]]
        if value != st["equals"]:
            problems.append(f"{st['pnr']}.{st['field']} = {value!r}, expected {st['equals']!r}")

    return {"id": s["id"], "passed": not problems, "problems": problems, "turns": turns,
            "expected": sum(expected.values()), "correct": sum((expected & got).values()),
            "unwanted": sum(unwanted.values()), "escalation_expected": s["expect_escalation"],
            "escalated": escalated,
            "checks": len(s["reply_checks"]),
            "checks_failed": sum(1 for p in problems if p.startswith("turn "))}


def report(results, runs, label):
    n = len(results)
    passed = sum(r["passed"] for r in results)
    exp, cor = sum(r["expected"] for r in results), sum(r["correct"] for r in results)
    unwanted = sum(r["unwanted"] for r in results)
    esc_got = [r for r in results if r["escalated"]]
    esc_exp = [r for r in results if r["escalation_expected"]]
    esc_tp = sum(1 for r in esc_got if r["escalation_expected"])
    checks = sum(r["checks"] for r in results)
    checks_ok = checks - sum(r["checks_failed"] for r in results)
    by_route = defaultdict(list)
    for r in results:
        for t in r["turns"]:
            by_route[t["route"].split(":")[0] if t["route"].startswith("confirm") else t["route"]].append(t["ms"])

    print(f"\n{'=' * 78}\n{label}: {passed}/{n} conversations passed ({100 * passed / n:.0f}%)"
          f" over {runs} run(s)")
    print(f"  actions: {cor}/{exp} expected actions ran   UNWANTED actions: {unwanted}")
    print(f"  escalation: precision {esc_tp}/{len(esc_got)}   recall {esc_tp}/{len(esc_exp)}")
    print(f"  reply checks: {checks_ok}/{checks}")
    print("  latency by route (p50 / max):")
    for route, ms in sorted(by_route.items(), key=lambda kv: -statistics.median(kv[1])):
        print(f"    {route:28} {statistics.median(ms):7.0f} ms / {max(ms):6d} ms   (n={len(ms)})")
    for r in results:
        if not r["passed"]:
            print(f"  FAIL [{r['id']}] " + "; ".join(r["problems"]))
            for t in r["turns"]:
                print(f"       {t['message'][:45]!r:48} -> {t['route']:26} {t['reply'][:70]!r}")
    return {"label": label, "passed": passed, "n": n, "unwanted": unwanted,
            "action_recall": cor / exp if exp else None,
            "esc_precision": esc_tp / len(esc_got) if esc_got else None,
            "esc_recall": esc_tp / len(esc_exp) if esc_exp else None,
            "reply_checks": checks_ok / checks if checks else None,
            "p50_by_route": {k: statistics.median(v) for k, v in by_route.items()}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--reply-effort", nargs="+", default=["default"],
                   help="REPLY_REASONING_EFFORT values to compare (default = model default)")
    p.add_argument("--only", nargs="*", help="scenario ids to run")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--out", help="write the results JSON here (used by src/evals/gate.py)")
    args = p.parse_args()

    scenarios = [json.loads(l) for l in open(SCENARIOS, encoding="utf-8") if l.strip()]
    if args.only:
        scenarios = [s for s in scenarios if s["id"] in args.only]
    snapshot = copy.deepcopy((backend.FLIGHTS, backend.BOOKINGS))
    print(f"{len(scenarios)} conversations | router: {os.environ.get('ROUTER_BACKEND') or 'llm'} | "
          f"runs: {args.runs} | RAG effort: {os.environ.get('RAG_REASONING_EFFORT') or 'default'}")

    summaries = []
    # the agent keeps its checkpoint databases open, and Windows can't delete open files
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        for effort in args.reply_effort:
            if effort == "default":
                os.environ.pop("REPLY_REASONING_EFFORT", None)
            else:
                os.environ["REPLY_REASONING_EFFORT"] = effort
            results = [run_scenario(s, snapshot, tmp) for _ in range(args.runs) for s in scenarios]
            summaries.append({**report(results, args.runs, f"REPLY_REASONING_EFFORT={effort}"),
                              "results": results})

    out = Path(args.out) if args.out else ROOT / "data" / "evals" / "results" / f"tool_eval-{time.strftime('%Y%m%d-%H%M')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summaries, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nfull transcripts -> {out}")


if __name__ == "__main__":
    main()
