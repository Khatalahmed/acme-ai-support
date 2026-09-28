"""Build data/insights/measured.json - the evaluation results the public /insights page shows.

Every number is computed here from result files committed in data/evals/results/, or (for the
few measured from traces and live requests) copied from where it was recorded, with that source
named next to it. tests/test_insights_data.py re-runs this and fails if the committed file has
drifted, so the page can't quietly show numbers no run produced.

    uv run python src/evals/build_insights.py
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "data" / "evals" / "results"
OUT = ROOT / "data" / "insights" / "measured.json"

ROUTER_BEFORE = "summary-routing_set_v2-20260928-0928.json"   # C3 first run
ROUTER_AFTER = "summary-20260928-1002.json"                    # after the C3b fixes
ROUTER_CASES = "router_compare-20260928-1002.jsonl"
HOLDOUT = ("summary-20260928-0935.json", "summary-20260928-0937.json")
EXCEPTION_HOLDOUT = ("summary-20260928-0949.json", "summary-20260928-0952.json")
RAG_BEFORE = ("rag_eval-20260928-1014.json", "default")        # top 3, default effort
RAG_AFTER = ("rag_eval-20260928-1021.json", "low")             # top 5, low effort (shipped)
STYLE = ("rag_eval-20260928-style-before.json", "rag_eval-20260928-style-after1.json",
         "rag_eval-20260928-style-after2.json")
TOOL_BEFORE, TOOL_AFTER = "tool_eval-20260928-1043.json", "tool_eval-20260928-1048.json"

CATEGORIES = {"normal": "Normal", "typos": "Typos / noisy", "hinglish": "Hinglish",
              "negation": "Negation", "ambiguous": "Ambiguous", "multi_intent": "Multi-intent",
              "injection": "Prompt injection", "tool_manipulation": "Tool manipulation",
              "cancel_refund_edge": "Cancel / refund edge cases"}
NAMES = {"llm": "LLM router", "jev": "Jev classifier", "prod": "Production (Jev + LLM fallback)"}


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def by_router(name):
    return {r["router"]: r for r in load(name)}


def answers(name, effort):
    return next(a for a in load(name)["answers"] if a["effort"] == effort)


def pct(x):
    return round(100 * x, 1)


def router_section():
    after = by_router(ROUTER_AFTER)
    rows = [json.loads(line) for line in (RESULTS / ROUTER_CASES).read_text(
        encoding="utf-8").splitlines() if line.strip()]
    per_cat = defaultdict(lambda: defaultdict(list))
    jev_tokens = []
    for r in rows:
        per_cat[r["category"]][r["router"]].append(r["correct"])
        if r["router"] == "jev" and r["input_tokens"]:
            jev_tokens.append(r["input_tokens"])
    categories = [{"category": CATEGORIES.get(c, c), "cases": len(v["prod"]) // after["prod"]["runs"],
                   **{k: pct(sum(v[k]) / len(v[k])) for k in ("llm", "jev", "prod")}}
                  for c, v in per_cat.items()]
    return {
        "about": (f"{after['prod']['n']} labelled customer messages in {len(categories)} "
                  f"categories, {after['prod']['runs']} runs per router, judged on what the "
                  "customer actually gets"),
        "source": f"data/evals/results/{ROUTER_AFTER}, {ROUTER_CASES}",
        "routers": [{"key": k, "name": NAMES[k], "accuracy": pct(r["accuracy"]),
                     "consistent": r["consistent"], "n": r["n"],
                     "p50_ms": r["p50_ms"], "p95_ms": r["p95_ms"],
                     "wrongful_cancels_per_run": round(r["wrongful_cancels"] / r["runs"], 1),
                     "escalation_precision": pct(r["esc_precision"]),
                     "escalation_recall": pct(r["esc_recall"]),
                     "fallback_rate": pct(r["fallback_rate"]) if r["fallback_rate"] else None}
                    for k, r in after.items()],
        "categories": sorted(categories, key=lambda c: c["prod"]),
        "jev_input_tokens_median": statistics.median(jev_tokens) if jev_tokens else None,
    }


def rag_section():
    a = answers(*RAG_AFTER)
    rows = a["rows"]
    styled = [load(f)["answers"][0]["rows"] for f in STYLE[1:]]
    return {
        "about": (f"{len(rows)} policy questions ({sum(not r['answerable'] for r in rows)} "
                  "deliberately unanswerable), deterministic checks, no LLM judge"),
        "source": f"data/evals/results/{RAG_AFTER[0]}, {', '.join(STYLE[1:])}",
        "fact_recall": pct(a["fact_recall"]), "grounded_numbers": pct(a["grounded"]),
        "unanswerable_refused": pct(a["refusals"]), "p50_ms": a["p50_ms"], "p95_ms": a["p95_ms"],
        "leaked_headings": max(sum(bool(r["labels"]) for r in s) for s in styled),
        "questions": len(rows),
    }


def tool_section():
    t = load(TOOL_AFTER)[0]
    return {
        "about": "20 multi-turn conversations x 3 runs through the real router, LLM, safety "
                 "layer and mock airline; judged on the final booking state",
        "source": f"data/evals/results/{TOOL_AFTER}",
        "conversations": t["n"], "passed": t["passed"], "unwanted_actions": t["unwanted"],
        "action_recall": pct(t["action_recall"]), "escalation_recall": pct(t["esc_recall"]),
        "escalation_precision": pct(t["esc_precision"]),
    }


def fixes():
    """Before -> after for each problem found and fixed, in the order they were found."""
    r0, r1 = by_router(ROUTER_BEFORE)["prod"], by_router(ROUTER_AFTER)["prod"]
    h0, h1 = (by_router(f)["prod"] for f in HOLDOUT)
    e0, e1 = (by_router(f)["prod"] for f in EXCEPTION_HOLDOUT)
    g0, g1 = answers(*RAG_BEFORE), answers(*RAG_AFTER)
    s0, s1 = (load(f)["answers"][0]["rows"] for f in STYLE[:2])
    t0, t1 = load(TOOL_BEFORE)[0], load(TOOL_AFTER)[0]
    results = "data/evals/results/"
    return [
        {"phase": "C1", "title": "Opening reply for a disrupted flight",
         "problem": "An LLM-written intro took 15.7 s of the request and the guardrail discarded "
                    "it every time", "fix": "Intro written by code from the booking",
         "metric": "reply time", "unit": "s", "before": 16.25, "after": 0.49, "better": "lower",
         "source": "Langfuse trace (PLAN.md C1)"},
        {"phase": "C3b", "title": "Cancel lookalikes",
         "problem": "Questions like 'what happens if I cancel?' were routed as cancellations",
         "fix": "Cancel only on a clear instruction; exception demands go to a person",
         "metric": "held-out accuracy", "unit": "%", "before": pct(h0["accuracy"]),
         "after": pct(h1["accuracy"]), "better": "higher",
         "source": results + " -> ".join(HOLDOUT)},
        {"phase": "C3b", "title": "Exception demands",
         "problem": "'Refund me or I'll sue' reached a person only when it was misrouted",
         "fix": "Demanding an exception escalates on any route",
         "metric": "held-out accuracy", "unit": "%", "before": pct(e0["accuracy"]),
         "after": pct(e1["accuracy"]), "better": "higher",
         "source": results + " -> ".join(EXCEPTION_HOLDOUT)},
        {"phase": "C3b", "title": "Wrong cancel decisions",
         "problem": "About 5 per run on the 150-case set (all stopped by the YES step)",
         "fix": "Same routing fix, measured on the full benchmark",
         "metric": "per run", "unit": "", "before": round(r0["wrongful_cancels"] / r0["runs"], 1),
         "after": round(r1["wrongful_cancels"] / r1["runs"], 1), "better": "lower",
         "source": f"{results}{ROUTER_BEFORE} -> {ROUTER_AFTER}"},
        {"phase": "C4", "title": "Policy answers",
         "problem": "The right section ranked 4th-5th for two questions, so the model refused",
         "fix": "Top 5 sections and low reasoning effort",
         "metric": "fact recall", "unit": "%", "before": pct(g0["fact_recall"]),
         "after": pct(g1["fact_recall"]), "better": "higher",
         "extra": f"p50 {g0['p50_ms'] / 1000:.1f} s -> {g1['p50_ms'] / 1000:.1f} s",
         "source": f"{results}{RAG_BEFORE[0]} -> {RAG_AFTER[0]}"},
        {"phase": "C4", "title": "Status replies used the wrong policy",
         "problem": "Searching with the customer's words found the right section 7 of 21 times",
         "fix": "Look the section up by name from the booking's state",
         "metric": "right section", "unit": "/21", "before": 7, "after": 21, "better": "higher",
         "source": "rag_eval.py status-context check (PLAN.md C4)"},
        {"phase": "C5", "title": "Invented services in status replies",
         "problem": "An on-time reply offered 'status alerts' and recited raw JSON",
         "fix": "Plain-language facts, own system prompt, closing line written by code",
         "metric": "conversations passing", "unit": "/60", "before": t0["passed"],
         "after": t1["passed"], "better": "higher",
         "source": f"{results}{TOOL_BEFORE} -> {TOOL_AFTER}"},
        {"phase": "D", "title": "Replies read like a form",
         "problem": "Answers printed 'Policy answer:' headings and opened with an apology",
         "fix": "Prompt rewritten; the quality gate now checks for headings",
         "metric": "answers with headings", "unit": "/41",
         "before": sum(bool(r["labels"]) for r in s0), "after": sum(bool(r["labels"]) for r in s1),
         "better": "lower", "source": results + " -> ".join(STYLE[:2])},
        {"phase": "D", "title": "Already-resolved booking",
         "problem": "Asking again offered options that no longer existed",
         "fix": "Reply written from the booking record",
         "metric": "reply time", "unit": "ms", "before": 3949, "after": 311, "better": "lower",
         "source": "live Azure deployment, 2026-09-28 (commit 5ca66c6)"},
    ]


def build():
    return {"router": router_section(), "rag": rag_section(), "tool": tool_section(),
            "fixes": fixes(),
            "gate": {"checks": len([1 for g, c in json.loads(
                (ROOT / "data" / "evals" / "gate.json").read_text(encoding="utf-8")).items()
                if not g.startswith("_") for _ in c])}}


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
