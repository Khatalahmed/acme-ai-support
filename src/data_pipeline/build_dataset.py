"""Day 3 - Stage 3: pairs.jsonl -> ShareGPT train.jsonl + test.jsonl."""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

IN_PATH = ROOT / "data" / "raw" / "pairs.jsonl"
TRAIN_PATH = ROOT / "data" / "processed" / "train.jsonl"
TEST_PATH = ROOT / "data" / "processed" / "test.jsonl"

SEED = 3407          # same seed the course uses everywhere - reproducible splits
TRAIN_FRACTION = 0.9

# This EXACT text is also what we will use at inference time (Day 6 Modelfile).
# It appears in every training record so the model binds the persona to it.
SYSTEM_PROMPT = (
    "You are a polite and empathetic customer support executive of "
    "ACME Bharat Airlines. Always follow company SOP. Structure every "
    "response with empathy, options, a policy note, and a next-step question."
)


def to_sharegpt(rec, idx):
    """Wrap one complaint/response pair in the ShareGPT structure."""
    return {
        "id": f"conv-{idx:04d}",
        "meta": {"scenario": rec["bucket"], "gen_model": rec.get("model", "gemini-3.5-flash")},
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": rec["complaint"]},
            {"from": "gpt", "value": rec["response"]},
        ],
    }


def main():
    pairs = [json.loads(line) for line in open(IN_PATH, encoding="utf-8")]
    print(f"loaded {len(pairs)} pairs")

    # 1. Deduplicate on the actual training content (complaint + response)
    seen, unique = set(), []
    for rec in pairs:
        h = hashlib.md5((rec["complaint"] + rec["response"]).encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        unique.append(rec)
    print(f"after dedup: {len(unique)} ({len(pairs) - len(unique)} duplicates removed)")

    # 2. Wrap in ShareGPT format
    records = [to_sharegpt(rec, i) for i, rec in enumerate(unique, 1)]

    # 3. Shuffle (seeded) and split 90/10
    random.seed(SEED)
    random.shuffle(records)
    cut = int(TRAIN_FRACTION * len(records))
    train, test = records[:cut], records[cut:]

    # 4. Write + self-validate (every line must round-trip through json.loads)
    for path, subset in [(TRAIN_PATH, train), (TEST_PATH, test)]:
        with open(path, "w", encoding="utf-8") as f:
            for r in subset:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for line in open(path, encoding="utf-8"):
            json.loads(line)  # raises if any line is malformed
        print(f"wrote + validated {len(subset):3d} records -> {path.name}")

    # 5. Coverage report: every bucket should appear in train
    train_buckets = Counter(r["meta"]["scenario"] for r in train)
    print("\ntrain bucket coverage:")
    for bucket, n in sorted(train_buckets.items()):
        print(f"  {bucket:24} {n}")
    missing = set(r["meta"]["scenario"] for r in records) - set(train_buckets)
    print(f"\nbuckets missing from train: {missing or 'none'}")


if __name__ == "__main__":
    main()