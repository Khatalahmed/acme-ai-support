"""Day 8 - Stage 3: RAG bot = Chroma retrieval + local acme-support generation."""

import argparse
import re
from pathlib import Path

import chromadb
import ollama

ROOT = Path(__file__).resolve().parent.parent.parent
DB_DIR = ROOT / "data" / "chroma"

MODEL = "acme-support"
TOP_K = 3

RAG_SYSTEM = (
    "You are a polite and empathetic customer support executive of ACME Bharat Airlines. "
    "Answer ONLY using the policy context provided by the user. Quote exact numbers and "
    "timelines from the context. If the context does not contain the answer, say: "
    "'I don't have sufficient information on this. Kindly contact AcmeConnect for assistance.' "
    "Structure the response with empathy, the policy answer, and a next-step question."
)

RAG_PROMPT = """Policy context:
{context}

Customer question: {question}"""


def retrieve(question, k=TOP_K):
    client = chromadb.PersistentClient(path=str(DB_DIR))
    col = client.get_collection("acme_policies")
    res = col.query(query_texts=[question], n_results=k)
    return res["documents"][0], res["metadatas"][0]


def clean(text):
    """App-layer filter for tool_call junk (opening and/or closing tags)."""
    return re.sub(r"^(\s*</?tool_call>\s*)+", "", text).strip()


def ask(question, use_rag):
    if use_rag:
        docs, metas = retrieve(question)
        messages = [
            {"role": "system", "content": RAG_SYSTEM},
            {"role": "user", "content": RAG_PROMPT.format(
                context="\n\n---\n\n".join(docs), question=question)},
        ]
        sources = [f"{m['source']} [{m['section']}]" for m in metas]
    else:
        # no system message -> Ollama falls back to the Modelfile's baked-in persona
        messages = [{"role": "user", "content": question}]
        sources = []

    resp = ollama.chat(model=MODEL, messages=messages)
    return clean(resp["message"]["content"]), sources


def main():
    p = argparse.ArgumentParser()
    p.add_argument("question", help="the customer question")
    p.add_argument("--compare", action="store_true",
                   help="also show the ungrounded (memory-only) answer")
    args = p.parse_args()

    if args.compare:
        print("=" * 70)
        print("WITHOUT RAG - model memory only (watch it invent policy):")
        print("=" * 70)
        text, _ = ask(args.question, use_rag=False)
        print(text, "\n")

    print("=" * 70)
    print("WITH RAG - grounded in the policy canon:")
    print("=" * 70)
    text, sources = ask(args.question, use_rag=True)
    print(text)
    print("\nSOURCES:", " | ".join(sources))


if __name__ == "__main__":
    main()