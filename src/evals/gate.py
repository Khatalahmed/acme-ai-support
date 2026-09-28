"""AI quality gate (CI gate 2): run every evaluation, compare with data/evals/gate.json, fail loudly.

Runs the production configuration (Jev router with LLM fallback, the .env/CI settings):
  tool_eval       20 conversations - actions, escalations, replies
  rag_eval        41 questions     - retrieval and answers
  router_compare  178 messages     - the 150-case set + both held-out sets, production router
Writes a table to stdout and, in GitHub Actions, to the run's summary page; exits 1 if any
limit is crossed. Costs roughly $0.15 and ~10 minutes per run - hence weekly, not per commit.

Usage:
    uv run python src/evals/gate.py
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
# Load the same settings production uses BEFORE deciding what to run. A first local run printed
# "router=llm" and passed no reply effort, because only the child evaluations read .env.
# In CI there's no .env: the workflow sets the variables, and these never override them.
load_dotenv(ROOT / ".env")
EVALS = ROOT / "data" / "evals"
LIMITS = json.loads((EVALS / "gate.json").read_text(encoding="utf-8"))


def run(script, out, *args):
    """Run one evaluation as its own process (as a person would), results written to `out`."""
    t0 = time.perf_counter()
    cmd = [sys.executable, str(ROOT / "src" / "evals" / script), "--out", str(out), *args]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    print(f"  {script:20} {time.perf_counter() - t0:5.0f}s  exit {proc.returncode}")
    if proc.returncode != 0 or not out.exists():
        print(proc.stdout[-3000:], proc.stderr[-3000:], sep="\n")
        raise SystemExit(f"{script} did not produce results - gate cannot judge, failing")
    return json.loads(out.read_text(encoding="utf-8"))


def metrics(tool, rag, router):
    t = tool[0]
    ranks = [r["rank"] for r in rag["retrieval"]]
    a = rag["answers"][0]
    rows = a["rows"]
    unans = [r for r in rows if not r["answerable"]]
    p = router[0]
    return {
        "tool": {"unwanted_actions": t["unwanted"], "action_recall": t["action_recall"],
                 "escalation_recall": t["esc_recall"], "escalation_precision": t["esc_precision"],
                 "conversation_pass_rate": t["passed"] / t["n"]},
        "rag": {"grounded_numbers": sum(not r["ungrounded"] for r in rows) / len(rows),
                "unanswerable_refused": sum(r["refused"] for r in unans) / len(unans),
                "retrieval_hit_at_5": sum(1 for k in ranks if k and k <= 5) / len(ranks),
                "fact_recall": a["fact_recall"]},
        "router": {"accuracy": p["accuracy"],
                   "wrongful_cancels_per_run": p["wrongful_cancels"] / p["runs"],
                   "cancel_recall": p["cancel_recall"], "escalation_recall": p["esc_recall"]},
    }


def judge(measured):
    rows, failed = [], []
    for group, checks in LIMITS.items():
        if group.startswith("_"):
            continue
        for name, rule in checks.items():
            value = measured[group][name]
            ok = value is not None and (value <= rule["max"] if "max" in rule else value >= rule["min"])
            limit = f"<= {rule['max']}" if "max" in rule else f">= {rule['min']}"
            shown = "-" if value is None else (f"{value:.1%}" if isinstance(value, float) and value <= 1
                                               and "per_run" not in name else f"{value:g}")
            rows.append((group, name, shown, limit, rule["critical"], ok))
            if not ok:
                failed.append(f"{group}.{name}")
    return rows, failed


def main():
    stamp = time.strftime("%Y%m%d-%H%M")
    out_dir = EVALS / "results" / f"gate-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"AI quality gate | router={os.environ.get('ROUTER_BACKEND') or 'llm'} | "
          f"LLM={os.environ.get('LLM_BACKEND') or 'ollama'} | "
          f"RAG effort={os.environ.get('RAG_REASONING_EFFORT') or 'default'} | "
          f"reply effort={os.environ.get('REPLY_REASONING_EFFORT') or 'default'}\n"
          f"results -> {out_dir}")

    tool = run("tool_eval.py", out_dir / "tool.json", "--reply-effort",
               os.environ.get("REPLY_REASONING_EFFORT") or "default")
    rag = run("rag_eval.py", out_dir / "rag.json", "--answers", "--effort",
              os.environ.get("RAG_REASONING_EFFORT") or "default")
    router = run("router_compare.py", out_dir / "router.json", "--routers", "prod", "--set",
                 str(EVALS / "routing_set_v2.jsonl"), str(EVALS / "routing_set_holdout.jsonl"),
                 str(EVALS / "routing_set_holdout_exception.jsonl"))

    rows, failed = judge(metrics(tool, rag, router))
    table = ["| | Area | Metric | Measured | Limit |", "|---|---|---|---|---|"]
    for group, name, shown, limit, critical, ok in rows:
        icon = "✅" if ok else ("🛑" if critical else "❌")
        table.append(f"| {icon} | {group} | {name}{' (critical)' if critical else ''} | "
                     f"{shown} | {limit} |")
    verdict = ("**PASSED** - every metric within its limit" if not failed else
               f"**FAILED** - {', '.join(failed)}")
    report = "\n".join(["## AI quality gate", "", verdict, "", *table])
    print("\n" + report)
    (out_dir / "gate.md").write_text(report, encoding="utf-8")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(report + "\n")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
