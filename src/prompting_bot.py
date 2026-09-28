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
from openai import OpenAI

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
        "endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
        "api_key": os.environ["AZURE_OPENAI_API_KEY"],
        "model": os.environ["AZURE_OPENAI_DEPLOYMENT"],
        "system_prompt": cfg["system_prompt"],
    }


def make_client(settings):
    # Azure OpenAI v1 API: plain OpenAI client pointed at <endpoint>/openai/v1/, no api-version.
    return OpenAI(
        base_url=settings["endpoint"].rstrip("/") + "/openai/v1/",
        api_key=settings["api_key"],
    )


def make_chat(settings):
    # Azure OpenAI is stateless: the "chat" is just the message history we resend each turn.
    return [{"role": "system", "content": settings["system_prompt"]}]


def send_message(client, settings, chat, text):
    chat.append({"role": "user", "content": text})
    # No temperature: reasoning deployments (gpt-5+/o-series) only accept the default.
    resp = client.chat.completions.create(model=settings["model"], messages=chat)
    reply = resp.choices[0].message.content
    chat.append({"role": "assistant", "content": reply})
    return reply


def run_demo(settings):
    print(f"Model: {settings['model']}")
    print("Persona loaded from config/bot.yaml\n")
    client = make_client(settings)
    for title, question, commentary in DEMO_TESTS:
        chat = make_chat(settings)  # fresh conversation per test
        print("=" * 70)
        print(title)
        print("=" * 70)
        print(f"CUSTOMER: {question}\n")
        reply = send_message(client, settings, chat, question)
        print(f"BOT: {reply.strip()}\n")
        print(commentary)
        print()


def run_chat(settings):
    print("ACME support bot - interactive mode. Type 'exit' to quit.\n")
    client = make_client(settings)
    chat = make_chat(settings)
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user or user.lower() in {"exit", "quit"}:
            break
        reply = send_message(client, settings, chat, user)
        print(f"Bot: {reply.strip()}\n")


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
