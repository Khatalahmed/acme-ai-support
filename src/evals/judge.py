"""Eval harness Phase 2+3: LLM-as-judge scoring + report.

--calibrate scores the GOLD reference answers (expect ~100% after rubric fix).
Free-tier quota is 5 req/min -> 13s pacing + retry + resume.
"""

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors

ROOT = Path(__file__).resolve().parent.parent.parent
IN_PATH = ROOT / "data" / "evals" / "outputs.jsonl"

CRITERIA = ["empathy", "options", "policy_note", "next_step", "no_invented_customer_data"]

JUDGE_PROMPT = """You are a strict QA evaluator for ACME Bharat Airlines customer support.

Customer complaint:
{complaint}

Agent response to evaluate:
{response}

Score each criterion 1 (clearly satisfied) or 0 (not satisfied):
- empathy: begins with a sincere acknowledgement or apology
- options: offers 2-3 concrete options (refund / rebooking / voucher / escalation)
- policy_note: contains a policy-safe statement (like "As per ACME Bharat Airlines policy")
- next_step: ends with exactly one clarifying question
- no_invented_customer_data: does NOT invent customer names, PNR numbers, reference or
  case numbers, or booking details absent from the complaint. General policy amounts,
  voucher values, and timelines ARE allowed - do NOT penalize those.

Output ONLY this JSON, nothing else:
{{"empathy":0,"options":0,"policy_note":0,"next_step":0,"no_invented_customer_data":0,"comment":"<max 15 words>"}}"""


def judge_one(client, model, complaint, response, max_retries=5):
    # Some errors are only temporary. We wait and try again for these:
    #   429 = we sent requests too fast (rate limit on our side).
    #   503 = the model is busy on Google's side (high demand).
    # For any other error we stop, because a retry will not help.
    RETRY_CODES = {429, 503}
    delay = 15
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=JUDGE_PROMPT.format(complaint=complaint, response=response))
            text = resp.text.strip()
            # Keep only the JSON part: from the first { to the last }.
            return json.loads(text[text.find("{"):text.rfind("}") + 1])
        except errors.APIError as e:
            code = getattr(e, "code", None)
            if code in RETRY_CODES and attempt < max_retries - 1:
                print(f"    temporary error {code} - waiting {delay}s then retry")
                time.sleep(delay)
                delay *= 2   # wait longer each round: 15s, 30s, 60s, 120s
            else:
                raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--calibrate", action="store_true",
                   help="judge the GOLD reference answers (judge sanity check)")
    args = p.parse_args()

    load_dotenv(ROOT / ".env")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    judge_model = os.environ.get("VERTEX_MODEL", "gemini-2.5-flash")
    field = "reference" if args.calibrate else "response"
    scores_path = ROOT / "data" / "evals" / f"scores_{field}.jsonl"

    done = {}
    if scores_path.exists():
        for line in open(scores_path, encoding="utf-8"):
            row = json.loads(line)
            done[row["id"]] = row

    records = [json.loads(l) for l in open(IN_PATH, encoding="utf-8")]
    print(f"judging {len(records)} records | field={field} | judge={judge_model} "
          f"| {len(done)} already scored (resume)\n")

    with open(scores_path, "a", encoding="utf-8") as f:
        for i, rec in enumerate(records, 1):
            if rec["id"] in done:
                continue
            scores = judge_one(client, judge_model, rec["complaint"], rec[field])
            time.sleep(13)  # free tier: 5 req/min -> stay under at ~4.6/min
            row = {"id": rec["id"], "scenario": rec["scenario"], **scores}
            f.write(json.dumps(row) + "\n")
            f.flush()
            done[rec["id"]] = row
            total = sum(scores.get(c, 0) for c in CRITERIA)
            print(f"{i:2d}/{len(records)}  {rec['id']}  {total}/5  {scores.get('comment', '')}")

    # report (from the full saved file, so it works even after resumed runs)
    n = len(done)
    totals = [sum(r.get(c, 0) for c in CRITERIA) for r in done.values()]
    per_criterion = Counter()
    for r in done.values():
        for c in CRITERIA:
            per_criterion[c] += r.get(c, 0)

    print("\n" + "=" * 60)
    print(f"OVERALL ({field}): {sum(totals)}/{n * 5}  ({100 * sum(totals) / (n * 5):.1f}%)")
    for c in CRITERIA:
        print(f"  {c:28} {per_criterion[c]}/{n}  ({100 * per_criterion[c] / n:.0f}%)")
    print("\nWORST 3:")
    ranked = sorted(done.values(), key=lambda r: sum(r.get(c, 0) for c in CRITERIA))
    for r in ranked[:3]:
        t = sum(r.get(c, 0) for c in CRITERIA)
        print(f"  {r['id']} [{r['scenario']}] {t}/5 - {r.get('comment', '')}")


if __name__ == "__main__":
    main()