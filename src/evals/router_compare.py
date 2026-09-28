"""Router benchmark: LLM router vs Jev vs production (Jev with LLM fallback).

Judges what the CUSTOMER would get (router.customer_outcome: the routing outcome plus the API's
rule that high-risk change requests go to a person), repeated over several runs because LLM
routers are not deterministic. Reports separate metrics - a wrongful cancel is a different
failure from a missed PNR - and a label-review list: cases where every router disagrees with the
label are checked first, because the label may be what's wrong.

Usage:
    uv run python src/evals/router_compare.py                         # v2 set, all routers, 1 run
    uv run python src/evals/router_compare.py --runs 3 --routers llm prod
    uv run python src/evals/router_compare.py --set data/evals/routing_set.jsonl   # v1 set

Tracing is OFF by default (a 150-case run is thousands of Langfuse units); add --trace to keep it.
"""

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if "--trace" not in sys.argv:
    os.environ["LANGFUSE_TRACING_ENABLED"] = "false"   # before langfuse is imported
sys.path.insert(0, str(ROOT / "src"))
import router  # noqa: E402  (also loads .env via llm_backend)
from router import customer_outcome, jev_provider, outcome, route_jev, route_llm  # noqa: E402

SET_V2 = ROOT / "data" / "evals" / "routing_set_v2.jsonl"
SET_REAL = ROOT / "data" / "evals" / "routing_set_real.jsonl"   # labelled shadow disagreements
RESULTS_DIR = ROOT / "data" / "evals" / "results"
JEV_USD_PER_MTOK = 0.042

# v1 name kept for older callers/tests: the routing outcome without the risk rule.
resolve = outcome


def route_prod(message):
    """Production configuration: Jev, falling back to the LLM router when unsure/unreachable."""
    return router.route(message)


ROUTERS = {"llm": route_llm, "jev": route_jev, "prod": route_prod}


def allowed(case):
    exp = case["expected"]
    return exp if isinstance(exp, list) else [exp]


def judge(case, decision):
    route, pnr = customer_outcome(decision)
    ok = route in allowed(case) and (pnr == case.get("pnr") if route.startswith("tool:") else True)
    return route, pnr, ok


def run_case(name, fn, case, run_no):
    t0 = time.perf_counter()
    try:
        decision, error = fn(case["message"]), None
    except Exception as e:  # a failed call is a miss, not a crash
        decision, error = {}, repr(e)
    ms = round((time.perf_counter() - t0) * 1000)
    route, pnr, ok = judge(case, decision) if not error else ("error", None, False)
    return {"router": name, "run": run_no, "id": case["id"],
            "category": case.get("category", case.get("set", "base")),
            "message": case["message"], "expected": "|".join(allowed(case)),
            "expected_pnr": case.get("pnr"), "got": route, "pnr": pnr, "correct": ok,
            "latency_ms": ms, "served_by": decision.get("router"),
            "confidence": decision.get("confidence"), "risk": decision.get("risk"),
            "input_tokens": decision.get("input_tokens"), "error": error}


def pct(n, d):
    return f"{100 * n / d:.0f}%" if d else "-"


def summarise(name, rows, runs):
    by_run = defaultdict(list)
    for r in rows:
        by_run[r["run"]].append(r)
    accs = [sum(r["correct"] for r in rr) / len(rr) for rr in by_run.values()]
    n = len(by_run[1])

    # consistency: same customer outcome on every run
    outcomes = defaultdict(set)
    for r in rows:
        outcomes[r["id"]].add((r["got"], r["pnr"]))
    consistent = sum(1 for s in outcomes.values() if len(s) == 1)

    wrong_cancel = [r for r in rows if r["got"] == "tool:cancel_ticket"
                    and "tool:cancel_ticket" not in r["expected"].split("|")]
    # the downside check: genuine cancel requests must still reach cancel
    must_cancel = [r for r in rows if r["expected"] == "tool:cancel_ticket"]
    cancel_hit = sum(1 for r in must_cancel if r["got"] == "tool:cancel_ticket")
    esc = [r for r in rows if r["got"] == "escalate"]
    esc_ok = [r for r in esc if "escalate" in r["expected"].split("|")]
    must_esc = [r for r in rows if r["expected"] == "escalate"]
    fallback = [r for r in rows if r["served_by"] == "jev->llm"]
    lat = sorted(r["latency_ms"] for r in rows)
    tokens = sum(r["input_tokens"] or 0 for r in rows if name == "jev")

    print("\n" + "=" * 78)
    acc_txt = (f"{100 * statistics.mean(accs):.1f}%" + (f" (runs {100 * min(accs):.0f}-"
               f"{100 * max(accs):.0f}%)" if runs > 1 else ""))
    print(f"{name.upper()}  accuracy {acc_txt}  over {n} cases x {runs} run(s)")
    if runs > 1:
        print(f"  consistency: {consistent}/{n} cases got the same outcome on every run")
    print(f"  wrongful cancels: {len(wrong_cancel)} ({len(wrong_cancel) / runs:.1f}/run)   "
          f"cancel recall {pct(cancel_hit, len(must_cancel))} ({cancel_hit}/{len(must_cancel)})   "
          f"escalation precision {pct(len(esc_ok), len(esc))} ({len(esc_ok)}/{len(esc)})   "
          f"recall {pct(sum(1 for r in must_esc if r['got'] == 'escalate'), len(must_esc))}")
    if name == "prod":
        print(f"  Jev -> LLM fallback rate: {pct(len(fallback), len(rows))}")
    print(f"  latency p50 {lat[len(lat) // 2]} ms   p95 {lat[min(len(lat) - 1, int(len(lat) * .95))]} ms"
          + (f"   Jev cost ${tokens * JEV_USD_PER_MTOK / 1e6 / runs:.5f}/run" if tokens else ""))
    cats = defaultdict(list)
    for r in rows:
        cats[r["category"]].append(r["correct"])
    print("  by category: " + " | ".join(f"{c} {pct(sum(v), len(v))}" for c, v in cats.items()))
    for r in wrong_cancel[:10]:
        print(f"    WRONGFUL CANCEL [{r['id']}] {r['message'][:70]!r}")
    return {"router": name, "accuracy": statistics.mean(accs), "acc_min": min(accs),
            "acc_max": max(accs), "consistent": consistent, "n": n, "runs": runs,
            "wrongful_cancels": len(wrong_cancel),
            "cancel_recall": cancel_hit / len(must_cancel) if must_cancel else None,
            "esc_precision": (len(esc_ok) / len(esc)) if esc else None,
            "esc_recall": (sum(1 for r in must_esc if r["got"] == "escalate") / len(must_esc))
            if must_esc else None,
            "fallback_rate": len(fallback) / len(rows) if name == "prod" else None,
            "p50_ms": lat[len(lat) // 2], "p95_ms": lat[min(len(lat) - 1, int(len(lat) * .95))]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", nargs="+",
                   default=[str(SET_V2)] + ([str(SET_REAL)] if SET_REAL.exists() else []),
                   help="one or more JSONL case files (default: v2 + imported real traffic)")
    p.add_argument("--routers", nargs="+", choices=list(ROUTERS), default=list(ROUTERS))
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--trace", action="store_true", help="keep Langfuse tracing on")
    p.add_argument("--out", help="write the summary JSON here (used by src/evals/gate.py)")
    args = p.parse_args()

    routers = list(args.routers)
    try:
        env, _, _ = jev_provider()
    except KeyError:
        print("No Jev key in .env - skipping jev and prod.")
        routers = [r for r in routers if r == "llm"]
    if "prod" in routers:
        os.environ["ROUTER_BACKEND"] = "jev"
    cases = [json.loads(line) for path in args.set
             for line in open(path, encoding="utf-8") if line.strip()]
    print(f"{len(cases)} cases | routers: {', '.join(routers)} | runs: {args.runs} | "
          f"workers: {args.workers}")

    all_rows, summaries = [], []
    for name in routers:
        jobs = [(name, ROUTERS[name], c, run) for run in range(1, args.runs + 1) for c in cases]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(lambda j: run_case(*j), jobs))
        print(f"  {name}: {len(rows)} calls in {time.perf_counter() - t0:.0f}s, "
              f"{sum(1 for r in rows if r['error'])} errors")
        all_rows += rows
        summaries.append(summarise(name, rows, args.runs))

    # Label review: every router, every run disagrees with the label -> check the label first.
    wrong_by = defaultdict(set)
    seen_by = defaultdict(set)
    for r in all_rows:
        seen_by[r["id"]].add(r["router"])
        if not r["correct"]:
            wrong_by[r["id"]].add(r["router"])
    unanimous = [c for c in cases if wrong_by[c["id"]] and wrong_by[c["id"]] == seen_by[c["id"]]
                 and all(not r["correct"] for r in all_rows if r["id"] == c["id"])]
    if unanimous and len(routers) > 1:
        print("\n" + "=" * 78)
        print(f"LABEL REVIEW - every router disagreed with these {len(unanimous)} labels "
              "(check the label before blaming the routers):")
        for c in unanimous:
            got = Counter(r["got"] for r in all_rows if r["id"] == c["id"]).most_common(1)[0][0]
            print(f"  [{c['id']}] label={'|'.join(allowed(c))}  routers said={got}  "
                  f"{c['message'][:60]!r}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    out = RESULTS_DIR / f"router_compare-{stamp}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (Path(args.out) if args.out else RESULTS_DIR / f"summary-{stamp}.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("| Router | Accuracy | Consistent | Wrongful cancels/run | Cancel recall | Esc. precision "
          "| Esc. recall | Fallback | p50 | p95 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for s in summaries:
        f = lambda v: "-" if v is None else f"{100 * v:.0f}%"  # noqa: E731
        acc = f"{100 * s['accuracy']:.1f}%" + (f" ({100 * s['acc_min']:.0f}-{100 * s['acc_max']:.0f})"
                                              if s["runs"] > 1 else "")
        print(f"| {s['router']} | {acc} | {s['consistent']}/{s['n']} | "
              f"{s['wrongful_cancels'] / s['runs']:.1f} | {f(s['cancel_recall'])} | "
              f"{f(s['esc_precision'])} | {f(s['esc_recall'])} | {f(s['fallback_rate'])} | "
              f"{s['p50_ms']} ms | {s['p95_ms']} ms |")
    print(f"\nper-case results -> {out}")


if __name__ == "__main__":
    main()
