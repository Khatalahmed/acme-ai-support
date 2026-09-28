"""RAG evaluation: retrieval and answers measured SEPARATELY, with deterministic metrics.

  retrieval (free, instant): did we fetch the labelled section?  hit@1, hit@3, MRR
  answers   (calls the LLM): fact recall, grounded numbers (every number in the answer appears
            in the retrieved text), correct refusals on unanswerable questions, latency, and style:
            leaked prompt headings ("Policy answer:") and replies that open with an apology

Both call the production code (api.main.retrieve / rag_answer), so the numbers are what customers
get. No LLM grades another LLM here: every check is a plain string/number comparison.

Usage:
    uv run python src/evals/rag_eval.py                         # retrieval only
    uv run python src/evals/rag_eval.py --answers               # + answers (default effort)
    uv run python src/evals/rag_eval.py --answers --effort minimal low default
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if "--trace" not in sys.argv:
    os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import src.api.main as api  # noqa: E402
from src.api.main import rag_answer, retrieve  # noqa: E402

SET = ROOT / "data" / "evals" / "rag_set.jsonl"
UNITS = r"(?:\s*(?:kg|kgs|hours?|hrs?|days?|weeks?|months?|years?|miles|cm|%))"


def norm(text):
    """Lower-case, no whitespace, currency unified, thousands separators removed."""
    t = text.lower().replace("₹", "rs").replace("inr", "rs").replace("rs.", "rs")
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)
    return re.sub(r"\s+", "", t)


def fact_hit(fact, answer_n):
    options = fact if isinstance(fact, list) else [fact]
    return any(norm(o) in answer_n for o in options)


def significant_numbers(text):
    """Numbers a customer could act on: 2+ digits, or any number with a unit ("7 days")."""
    out = set()
    for m in re.finditer(r"\d[\d,]*(?:\.\d+)?(" + UNITS + r")?", text, re.IGNORECASE):
        num = m.group(0)
        value = re.sub(r"[^\d.]", "", re.match(r"[\d,.]+", num).group(0)).rstrip(".")
        if len(value) >= 2 or m.group(1):
            out.add(value)
    return out


def all_numbers(text):
    return {re.sub(r"[^\d.]", "", n).rstrip(".") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}


def refused(answer_n):
    """The scripted refusal. ("AcmeConnect" alone isn't one: good answers often end by offering
    it - an earlier version of this check counted those as refusals.)"""
    return "sufficientinformation" in answer_n


# The old system prompt said "structure the response with empathy, the policy answer, and a
# next-step question"; live on Azure the model printed those as headings. A line that starts
# with one of these words and a colon is a leaked template, not writing a customer should see.
LABEL = re.compile(r"^[\s*#_-]*(empathy|acknowledg\w*|policy answer|answer|next[- ]step"
                   r"(?: question)?|follow[- ]up(?: question)?)\s*[*_]*\s*:", re.I | re.M)


def leaked_labels(reply):
    return sorted({m.group(1).lower() for m in LABEL.finditer(reply)})


def opens_with_apology(reply):
    first = re.split(r"(?<=[.!?])\s", reply.strip(), maxsplit=1)[0].lower()
    return "sorry" in first or "apolog" in first


def section_key(meta):
    return f"{meta['source']}::{meta['section']}"


# ---------------------------------------------------------------- retrieval

def eval_retrieval(cases, k=5):
    rows = []
    for c in cases:
        if not c["answerable"]:
            continue
        _, metas = retrieve(c["question"], k=k)
        keys = [section_key(m) for m in metas]
        rank = next((i + 1 for i, key in enumerate(keys) if key in c["expected"]), None)
        rows.append({**c, "retrieved": keys, "rank": rank})
    n = len(rows)
    hit = lambda k_: sum(1 for r in rows if r["rank"] and r["rank"] <= k_)  # noqa: E731
    print(f"RETRIEVAL ({n} answerable questions, top {k})")
    print(f"  hit@1 {hit(1)}/{n} ({100 * hit(1) / n:.0f}%)   hit@3 {hit(3)}/{n} "
          f"({100 * hit(3) / n:.0f}%)   hit@5 {hit(5)}/{n}   "
          f"MRR {statistics.mean(1 / r['rank'] if r['rank'] else 0 for r in rows):.2f}")
    misses = [r for r in rows if not r["rank"] or r["rank"] > 1]
    if misses:
        print("  not ranked first (the API uses the top 3):")
        for r in misses:
            top = r["retrieved"][0].replace("-policy.md", "")
            print(f"    [{r['id']}] rank={r['rank'] or '>' + str(k)}  top={top:45} {r['question'][:55]!r}")
    return rows


# ---------------------------------------------------------------- status-reply context

# Which policy section a status reply needs, by booking (None = no policy applies).
STATUS_EXPECTED = {
    "ACX123": "delay-compensation-policy.md::2. Delay Compensation Tiers",    # 5 h, technical
    "ACX321": "delay-compensation-policy.md::2. Delay Compensation Tiers",    # 2.5 h, technical
    "ACX246": "delay-compensation-policy.md::2. Delay Compensation Tiers",    # 6 h 40, technical
    "ACX987": "delay-compensation-policy.md::4. Policy Exclusions",           # weather delay
    "ACX789": "refund-cancellation-policy.md::4. Airline-Initiated Cancellations",
    "ACX654": "refund-cancellation-policy.md::5. Weather and Force Majeure Cancellations",
    "ACX456": None,                                                            # on time
}
STATUS_PHRASINGS = ["What's the status of {p}?", "Is {p} on time?", "Any update on booking {p}?"]


def eval_status_context():
    """Old (search with the customer's words, top 3) vs new (query from the booking state)."""
    from src.api.main import status_context
    from tools.backend import get_flight_status

    def score(get_sections):
        found = noise = n = 0
        for pnr, expected in STATUS_EXPECTED.items():
            booking = get_flight_status(pnr)
            for phrasing in STATUS_PHRASINGS:
                keys = get_sections(booking, phrasing.format(p=pnr))
                n += 1
                if expected is None:
                    found += not keys                   # right answer: no policy at all
                    noise += len(keys)
                else:
                    found += expected in keys
                    noise += sum(1 for k in keys if k != expected)
        return found, noise, n

    old = score(lambda b, q: [section_key(m) for m in retrieve(q, k=3)[1]])
    new = score(lambda b, q: [section_key(m) for m in status_context(b, q)[1]])
    print("\nSTATUS-REPLY CONTEXT (7 bookings x 3 phrasings)")
    for label, (found, noise, n) in (("old: customer's words, top 3", old),
                                     ("new: section lookup from booking", new)):
        print(f"  {label:32} right context {found}/{n}   irrelevant sections passed to the LLM "
              f"{noise} ({noise / n:.1f} per reply)")


# ---------------------------------------------------------------- answers

def answer_one(c, effort):
    t0 = time.perf_counter()
    reply, docs, metas = rag_answer(c["question"], reasoning_effort=effort)
    ms = round((time.perf_counter() - t0) * 1000)
    ans_n, context = norm(reply), "\n".join(docs)
    ungrounded = sorted(significant_numbers(reply) - all_numbers(context)
                        - all_numbers(c["question"]))
    facts = [fact_hit(f, ans_n) for f in c["facts"]]
    return {"id": c["id"], "answerable": c["answerable"], "ms": ms, "reply": reply,
            "facts_ok": all(facts) if c["answerable"] else None,
            "refused": refused(ans_n), "ungrounded": ungrounded,
            "labels": leaked_labels(reply), "apology": opens_with_apology(reply)}


def eval_answers(cases, effort, workers):
    label = effort or "default"
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda c: answer_one(c, effort), cases))
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    lat = sorted(r["ms"] for r in rows)
    facts = sum(r["facts_ok"] for r in ans)
    grounded = sum(not r["ungrounded"] for r in rows)
    good_refusals = sum(r["refused"] for r in unans)
    # wrong refusal = declined AND didn't give the answer (a partial caveat after the fact is fine)
    false_refusals = sum(r["refused"] and not r["facts_ok"] for r in ans)
    print(f"\nANSWERS  reasoning_effort={label}")
    print(f"  fact recall {facts}/{len(ans)} ({100 * facts / len(ans):.0f}%)   "
          f"grounded numbers {grounded}/{len(rows)}   "
          f"refused unanswerable {good_refusals}/{len(unans)}   "
          f"wrongly refused answerable {false_refusals}/{len(ans)}")
    labelled = sum(bool(r["labels"]) for r in rows)
    apologies = sum(r["apology"] for r in rows)
    print(f"  style: leaked prompt headings {labelled}/{len(rows)}   "
          f"opens with an apology {apologies}/{len(rows)}")
    print(f"  latency p50 {lat[len(lat) // 2]} ms   p95 {lat[min(len(lat) - 1, int(len(lat) * .95))]} ms")
    by_id = {c["id"]: c for c in cases}
    for r in rows:
        problems = []
        if r["answerable"] and not r["facts_ok"]:
            problems.append("missing fact " + str(by_id[r["id"]]["facts"]))
        if r["ungrounded"]:
            problems.append(f"UNGROUNDED numbers {r['ungrounded']}")
        if r["answerable"] and r["refused"] and not r["facts_ok"]:
            problems.append("refused an answerable question")
        if not r["answerable"] and not r["refused"]:
            problems.append("did NOT refuse (may have invented an answer)")
        if r["labels"]:
            problems.append(f"leaked headings {r['labels']}")
        if problems:
            print(f"    [{r['id']}] {'; '.join(problems)}  <- {r['reply'][:90]!r}")
    return {"effort": label, "fact_recall": facts / len(ans), "grounded": grounded / len(rows),
            "refusals": good_refusals / len(unans) if unans else None,
            "false_refusals": false_refusals, "no_labels": 1 - labelled / len(rows),
            "apology_openings": apologies, "p50_ms": lat[len(lat) // 2],
            "p95_ms": lat[min(len(lat) - 1, int(len(lat) * .95))], "rows": rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--answers", action="store_true", help="also evaluate generated answers (LLM)")
    p.add_argument("--effort", nargs="+", default=["default"],
                   help="reasoning_effort values to compare: minimal low medium high default")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--k", type=int, help="sections given to the LLM (default: the API's TOP_K)")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--out", help="write the results JSON here (used by src/evals/gate.py)")
    args = p.parse_args()
    if args.k:
        api.TOP_K = args.k
    print(f"sections given to the LLM: top {api.TOP_K}")

    cases = [json.loads(line) for line in open(SET, encoding="utf-8") if line.strip()]
    retrieval = eval_retrieval(cases)
    eval_status_context()
    if not args.answers:
        return
    summaries = [eval_answers(cases, None if e == "default" else e, args.workers)
                 for e in args.effort]

    out_dir = ROOT / "data" / "evals" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else out_dir / f"rag_eval-{time.strftime('%Y%m%d-%H%M')}.json"
    out.write_text(json.dumps({"retrieval": [{k: r[k] for k in ("id", "rank", "retrieved")}
                                             for r in retrieval],
                               "answers": summaries}, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nfull replies -> {out}")
    print("\n| reasoning_effort | Fact recall | Grounded numbers | Refused unanswerable "
          "| Wrong refusals | p50 | p95 |")
    print("|---|---|---|---|---|---|---|")
    for s in summaries:
        refusals = "-" if s["refusals"] is None else f"{100 * s['refusals']:.0f}%"
        print(f"| {s['effort']} | {100 * s['fact_recall']:.0f}% | {100 * s['grounded']:.0f}% | "
              f"{refusals} | {s['false_refusals']} | {s['p50_ms']} ms | {s['p95_ms']} ms |")


if __name__ == "__main__":
    main()
