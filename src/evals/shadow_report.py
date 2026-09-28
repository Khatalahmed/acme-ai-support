"""Shadow-mode report: how often the served and shadow routers agree on real traffic.

Reads the local shadow_log (data/acme.db, or ACME_DB_PATH). The same comparisons also appear in
Langfuse as the `router_agreement` score, and disagreements in the "router-disagreements"
dataset, when tracing is on.

Usage:
    uv run python src/evals/shadow_report.py [--last 50]
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
import shadow  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--last", type=int, default=0, help="only the most recent N comparisons")
    args = p.parse_args()

    s = shadow.summary()
    rows = s["rows"][-args.last:] if args.last else s["rows"]
    compared = [r for r in rows if r["agree"] is not None]
    if not rows:
        print("No shadow comparisons yet. Set ROUTER_SHADOW=llm (or jev) in .env and send requests.")
        return

    agree = sum(r["agree"] for r in compared)
    print(f"{len(rows)} shadowed requests | {len(compared)} compared | "
          f"agreement {agree}/{len(compared)}"
          + (f" ({100 * agree / len(compared):.0f}%)" if compared else "")
          + f" | shadow errors {sum(1 for r in rows if r['error'])}")
    ms = sorted(r["shadow_ms"] for r in rows if r["shadow_ms"] is not None)
    if ms:
        print(f"shadow router time (off the customer's path): p50 {ms[len(ms) // 2]} ms, "
              f"max {ms[-1]} ms")

    pairs = Counter((r["primary_outcome"].split()[0], r["shadow_outcome"].split()[0])
                    for r in compared if not r["agree"])
    if pairs:
        print("\nDisagreement patterns (served -> shadow):")
        for (a, b), n in pairs.most_common():
            print(f"  {n:3}  {a:24} -> {b}")
        print("\nDisagreements (review these; label them in Langfuse to grow the benchmark):")
        for r in compared:
            if not r["agree"]:
                print(f"  [{r['primary_router']}] {r['primary_outcome']:28} "
                      f"[{r['shadow_router']}] {r['shadow_outcome']:28} {r['message'][:70]!r}")


if __name__ == "__main__":
    main()
