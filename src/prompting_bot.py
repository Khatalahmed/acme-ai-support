"""Project 1 — the prompting-only ACME bot.

The entire "product" is one system prompt (config/bot.yaml). There is no RAG,
no fine-tuning, no tools. Run --demo to see exactly what that buys you (tone)
and what it cannot buy you (knowledge).

Usage:
    python src/prompting_bot.py --demo    # run the two Day-1 test cases
    python src/prompting_bot.py --chat    # interactive chat loop
"""

import argparse
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parent.parent

DEMO_TESTS = [
    (
        "TEST A - tone control (prompting's strength)",
        "I have travelled in your airlines. It was dirty.",
        "Expect: polite, empathetic, on-brand apology asking for flight details.\n"
        "Lesson: the system prompt successfully controls BEHAVIOR.",
    ),
    (
        "TEST B - knowledge gap (prompting's limit)",
        "What is ACME's exact compensation amount for a 4-hour flight delay?",
        "Expect: a plausible-sounding INVENTED policy, or a vague dodge.\n"
        "Lesson: the model has never seen ACME's real policy - no prompt can add\n"
        "knowledge it doesn't have. This is the problem RAG solves on Day 8.",
    ),
]


def load_settings():
    load_dotenv(ROOT / ".env")
    import os

    cfg = yaml.safe_load((ROOT / "config" / "bot.yaml").read_text(encoding="utf-8"))
    return {
        "project": os.environ["GCP_PROJECT_ID"],
        "location": os.environ.get("VERTEX_LOCATION", "global"),
        "model": os.environ.get("VERTEX_MODEL", "gemini-2.5-flash"),
        "system_prompt": cfg["system_prompt"],
        "temperature": float(cfg.get("temperature", 0.7)),
    }


def make_client(settings):
    return genai.Client(
        vertexai=True, project=settings["project"], location=settings["location"]
    )


def make_chat(client, settings):
    # The client must outlive the chat: if the Client object is garbage-collected,
    # its HTTP connection closes and every send_message fails.
    return client.chats.create(
        model=settings["model"],
        config=types.GenerateContentConfig(
            system_instruction=settings["system_prompt"],
            temperature=settings["temperature"],
        ),
    )


def run_demo(settings):
    print(f"Model: {settings['model']} | temperature: {settings['temperature']}")
    print("Persona loaded from config/bot.yaml\n")
    client = make_client(settings)
    for title, question, commentary in DEMO_TESTS:
        chat = make_chat(client, settings)  # fresh conversation per test
        print("=" * 70)
        print(title)
        print("=" * 70)
        print(f"CUSTOMER: {question}\n")
        reply = chat.send_message(question)
        print(f"BOT: {reply.text.strip()}\n")
        print(commentary)
        print()


def run_chat(settings):
    print("ACME support bot - interactive mode. Type 'exit' to quit.\n")
    client = make_client(settings)
    chat = make_chat(client, settings)
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user or user.lower() in {"exit", "quit"}:
            break
        reply = chat.send_message(user)
        print(f"Bot: {reply.text.strip()}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true", help="run the Day-1 test cases")
    mode.add_argument("--chat", action="store_true", help="interactive chat loop")
    args = parser.parse_args()

    settings = load_settings()
    if args.demo:
        run_demo(settings)
    else:
        run_chat(settings)


if __name__ == "__main__":
    sys.exit(main())
