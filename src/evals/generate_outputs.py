"""Eval harness Phase 1: run the held-out test set through the local model."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from llm_backend import chat

TEST_PATH = ROOT / "data" / "processed" / "test.jsonl"
OUT_PATH = ROOT / "data" / "evals" / "outputs.jsonl"


def load_done():
    done = set()
    if OUT_PATH.exists():
        for line in open(OUT_PATH, encoding="utf-8"):
            done.add(json.loads(line)["id"])
    return done


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = [json.loads(l) for l in open(TEST_PATH, encoding="utf-8")]
    done = load_done()
    print(f"{len(records)} test records | {len(done)} already generated (skip)")

    with open(OUT_PATH, "a", encoding="utf-8") as f:
        for i, rec in enumerate(records, 1):
            if rec["id"] in done:
                continue
            turns = {t["from"]: t["value"] for t in rec["conversations"]}
            reply = chat([
                {"role": "system", "content": turns["system"]},
                {"role": "user", "content": turns["human"]},
            ])
            f.write(json.dumps({
                "id": rec["id"],
                "scenario": rec["meta"]["scenario"],
                "complaint": turns["human"],
                "reference": turns["gpt"],            # the gold answer from Day 3
                "response": reply.strip(),
            }) + "\n")
            f.flush()
            print(f"generated {i}/{len(records)}")

    print(f"\noutputs -> {OUT_PATH}")


if __name__ == "__main__":
    main()