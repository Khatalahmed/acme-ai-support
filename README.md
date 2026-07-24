# ACME Airlines Support AI

A hybrid customer-support AI for a fictional airline (ACME Bharat Airlines), built end-to-end:
a **fine-tuned local LLM** for brand tone, **RAG** for policy facts, and **tool calling** for live
actions — served through one FastAPI endpoint, with an **LLM-as-judge eval harness** to prove it works.

> **Why hybrid?** Fine-tuning teaches the model *how to speak* (tone, structure), but it also
> invents facts. RAG pins answers to real policy documents. Tools handle live data (flight
> status, cancellations). Each layer fixes the failure mode of the previous one — and this
> repo demonstrates that failure → fix chain deliberately.

## Architecture

```
                      ┌──────────────────────────────┐
User complaint ──────>│  FastAPI  POST /v1/chat      │
                      │  Router (intent detection)   │
                      └──────┬───────────┬───────────┘
                     policy Q│           │action (PNR)
                             v           v
                      ┌────────────┐  ┌──────────────────┐
                      │ RAG        │  │ Tool calling      │
                      │ Chroma     │  │ get_flight_status │
                      │ 28 chunks  │  │ cancel_ticket     │
                      │ 5 policies │  │ (mock backend)    │
                      └─────┬──────┘  └────────┬─────────┘
                            └────────┬─────────┘
                                     v
                      ┌──────────────────────────────┐
                      │ Fine-tuned Qwen3-4B (QLoRA)  │
                      │ served locally via Ollama     │
                      │ (Q5_K_M GGUF, CPU)            │
                      └──────────────────────────────┘
```

## The pipeline (what was actually built)

| Stage | What | Key detail |
|---|---|---|
| 1. Prompting bot | Baseline Gemini bot with persona config | Shows tone control + knowledge gaps |
| 2. Synthetic data | 240 complaint→response pairs, 12 scenario buckets | AER 4-part response template, ShareGPT JSONL, resilience stack (pacing / 429 backoff / fallback model / resume) |
| 3. Fine-tune | **QLoRA on Colab T4**: Qwen3-4B-Instruct 4-bit, r=16, all 7 modules, 33M trainable params (0.81%) | Learns the AER response format from 216 examples |
| 4. Quantize + serve | Merge 16-bit → GGUF f16 → **Q5_K_M (2.7 GB)** → Ollama on Windows CPU | Fully local inference, no API cost |
| 5. RAG | 5 policy docs → structure-aware chunking → Chroma + MiniLM embeddings | Without RAG the tuned model invents policies; with RAG it cites canon (Rs 3,000 voucher, 7-day refunds) |
| 6. Tool calling | Router prompt → JSON decision → Python validates & executes → responder grounds the reply | The model never executes anything — code does |
| 7. Hybrid API | FastAPI `/v1/chat` routing RAG / tools / clarify + output cleaning + trace logging | The capstone: all layers in one endpoint |
| 8. Eval harness | Held-out test set → local model outputs → **Gemini LLM-judge** scoring 5 AER criteria | `--calibrate` mode sanity-checks the judge on gold answers first |

## Eval harness

Five binary criteria per response, scored by an LLM judge (`src/evals/judge.py`):

- **empathy** — opens with a sincere acknowledgement
- **options** — offers 2–3 concrete options
- **policy_note** — contains a policy-safe statement
- **next_step** — ends with exactly one clarifying question
- **no_invented_customer_data** — no fabricated PNRs/names/case numbers

The judge is calibrated on gold reference answers before scoring the model
(a judge that can't score the answer key ~100% can't be trusted to grade the student).
Includes free-tier survival: request pacing, 429/503 retry with exponential backoff, resume files.

## Honest limitations (kept on purpose — they motivate the roadmap)

- The 4B model sometimes **blends retrieved facts with invented policy details** on the tool
  path → next: reflection agent + stricter grounding
- Occasional **garbage leading tokens** from the quantized model → filtered at the app layer
- One response asked a customer for "bank details" → motivates an **output guardrail blocklist**
- CPU latency: deterministic clarify path ~5 s vs LLM paths 36–86 s → quantifies why
  deterministic routing matters

## Run it

```bash
# 1. Environment
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# 2. Local model (needs Ollama + the GGUF built in stage 4)
ollama create acme-support -f data/processed/Modelfile

# 3. Build the RAG index
python src/rag/build_index.py

# 4. Serve
uvicorn src.api.main:app --port 8000
# Swagger UI: http://localhost:8000/docs
```

## Stack

Qwen3-4B-Instruct · Unsloth QLoRA · llama.cpp GGUF/Q5_K_M · Ollama · ChromaDB ·
MiniLM embeddings · Gemini (synthetic data + judge) · FastAPI · Windows CPU serving

## Roadmap

Eval harness (in progress) → output guardrails → ACME MCP server → LangGraph reflection
agent → feedback-to-DPO loop.
