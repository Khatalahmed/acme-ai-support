"""Day 8 - Stage 2: chunk the policy canon and build the Chroma vector index."""

import re
from pathlib import Path

import chromadb

ROOT = Path(__file__).resolve().parent.parent.parent
POLICY_DIR = ROOT / "data" / "policies"
DB_DIR = ROOT / "data" / "chroma"


def chunk_markdown(text, source):
    """Structure-aware chunking: one chunk per '## section', doc title prepended."""
    title_match = re.search(r"^# (.+)$", text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else source

    parts = re.split(r"(?m)^## ", text)
    chunks = []
    for part in parts[1:]:                      # parts[0] = preamble before first ##
        heading, _, body = part.partition("\n")
        chunk_text = f"{title} - {heading.strip()}\n{body.strip()}"
        chunks.append((heading.strip(), chunk_text))
    return chunks


def main():
    client = chromadb.PersistentClient(path=str(DB_DIR))

    # start fresh every run: policy edit -> rerun this script -> index updated
    try:
        client.delete_collection("acme_policies")
    except Exception:
        pass
    collection = client.create_collection("acme_policies")

    n = 0
    for md_file in sorted(POLICY_DIR.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        for heading, chunk in chunk_markdown(text, md_file.stem):
            collection.add(
                ids=[f"{md_file.stem}::{n}"],
                documents=[chunk],
                metadatas=[{"source": md_file.name, "section": heading}],
            )
            n += 1
        print(f"indexed {md_file.name}")

    print(f"\n{n} chunks in collection 'acme_policies' -> {DB_DIR}")

    # smoke test: does semantic search land in the RIGHT document?
    for q in [
        "How much compensation do I get for a 5 hour flight delay?",
        "What is the checked baggage weight limit?",
    ]:
        res = collection.query(query_texts=[q], n_results=2)
        print(f"\nQUERY: {q}")
        for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
            print(f"  <- {meta['source']} [{meta['section']}]  {doc[:80]}...")


if __name__ == "__main__":
    main()