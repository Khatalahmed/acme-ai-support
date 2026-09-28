# ACME Disruption Agent — Plan

> An airline disruption-recovery agent where irreversible actions need server-side, expiring,
> authorised confirmation; routing is benchmarked (LLM vs Jev, trialled in shadow mode); and
> every decision is traced, costed and gated in CI.

**Status:** Phases 0, A and B done · Phase C next · **Scope:** frozen (see bottom) · **README:** rewritten last, in Phase D

## Where we are (start of plan)

- Runtime LLM switchable via `LLM_BACKEND` (Azure OpenAI now; local fine-tuned Ollama model when the GGUF is rebuilt)
- Router switchable via `ROUTER_BACKEND` (LLM JSON router or TypeSafe Jev, with confidence fallback)
- Router benchmark: 60 labelled messages (39 base + 21 hard) — Jev 58/60 at ~0.5 s p50, LLM 57/60 at ~2.4 s p50
- Cancellation requires customer confirmation; cancel is idempotent (no double refund)

## Architecture target

```
Passenger ─► API (auth: user token) ─► Conversation state (SQLite)
                     │
                     ▼
        Router: Jev (System 1) ──uncertain──► gpt-5-mini (System 2)
                     │                 └── other router runs in shadow, logged
                     ▼
        LangGraph agent: disruption recovery · RAG · policy · escalation
                     │  (can only PROPOSE actions)
                     ▼
 ┌─ Tool layer: the safety boundary ────────────────────────────────┐
 │ pending action → confirm → expiry → ownership → re-read booking   │
 │ → policy check → idempotency key → execute → audit log            │
 └───────────────▲──────────────────────────────▲────────────────────┘
          in-process agent              MCP server (same rules)

Langfuse: traces · cost · latency · datasets     GitHub Actions: 2 gates
```

**Design rules**
1. Safety lives in the tool layer, not the agent — every caller (agent, MCP client, script) gets the same checks.
2. The LLM never decides entitlement. Code decides what a passenger is owed from the policy rules; the LLM explains it.
3. In LangGraph, the confirmation `interrupt()` is its own node with no side effects; execution is a separate node
   (on resume, LangGraph re-runs the interrupted node from the start).
4. Only measured numbers go in the README.

## Phase 0 — Setup (½ day)

- [x] Connect this folder to GitHub (`Khatalahmed/acme-ai-support`, **public**) and open a PR from a new branch
      — [PR #1](https://github.com/Khatalahmed/acme-ai-support/pull/1), branch `azure-jev-safety`
- [x] `.gitignore` covers `.env`, audit logs and SQLite databases before the first push
- [x] `pytest` as a dev dependency
- [x] This plan committed

**Done when:** branch pushed, PR open, `uv run pytest` runs.

## Phase A — Foundation and safety (days 1–3)

- [x] **A1 Tests in the repo** — PNR, yes/no, cancel-flow and Jev-router tests in `tests/`, fully offline
      (model calls faked; RAG index opened lazily). 76 tests, ~3 s; mutation-checked (removing the
      ownership check or the compare-and-set claim makes tests fail)
- [x] **A2 Mock users + ownership** — bookings have owners; each request carries a user token;
      another user's PNR gets the same reply as a non-existent one (`src/auth.py`, `src/actions.py`)
- [x] **A3 Server-side pending actions** — SQLite: `action_id, user_id, session_id, pnr, action,
      created_at, expires_at (+5 min), status`; "yes" confirms only the caller's newest pending action in
      that session; `confirm_cancel` removed from the API
- [x] **A4 Execution checks** — pending exists → not expired → owned by caller → re-read booking →
      still allowed → claim → execute. Idempotency: the pending action is the idempotency key, claimed
      with one conditional UPDATE (pending → executing), so only one "yes" can win; the backend is
      idempotent too. `tool_bot.py` is now read-only for cancellations
- [x] **A5 Audit log** — every proposed / declined / expired / executed / failed action and every denied
      access, with router and confidence; append-only (`audit_log` table)
- [x] **A6 CI gate 1** — pytest on every push and PR; no model calls, no cost
      (`.github/workflows/tests.yml`; first run: 76 passed in 2.4 s on ubuntu, no `.env`, no index)

**Done when:** tests prove an injected cancel, another user's booking, an expired confirmation, a replayed
"yes" and a double-click cannot cancel or refund.

## Phase B — Disruption recovery agent (days 4–9)

- [x] **B1 Mock airline data** — flights and bookings separated; one day of operations with a case for
      every policy tier, full flights, weather and technical causes (`src/tools/backend.py`)
- [x] **B2 Tools** — `disruption_options`, `rebook`, `refund`, `issue_voucher`, `escalate`; every write
      goes through the A4 lifecycle; options re-computed at execution (last-seat race fails safely)
- [x] **B3 LangGraph agent** — assess → escalate | explain | present → await_choice (interrupt, own node)
      → propose. The agent only proposes; the "yes" reuses the A3 server-side confirmation (one
      mechanism, not two). SQLite checkpointer: paused conversations survive a restart (`src/agent.py`)
- [x] **B4 Three-way decision** — simple → options; nothing to choose → explain; angry / demands an
      exception → human queue with summary. Risk signals are extra questions in the same Jev call
      (or JSON fields from the LLM router); also applied to high-risk cancel requests
- [x] **B5 Policy in code** — `src/policy.py`, pure functions citing policy sections; the "2–4 h" /
      "4–6 h" boundary overlap resolved explicitly (4 h and 6 h → 4–6 h tier)
- [x] **B6 MCP server** — `src/mcp_server.py` (MCP SDK 2.x); no confirm tool exists — every change asks
      the human via elicitation, then runs through actions.py

**Done when:** scripted run passes — cancelled flight → options → pick rebook → confirm → rebooked + audit
trail — and the escalation path works. ✅ Tests (140) and live runs with both routers.

**Found in live runs (fixed):** LLM-phrased confirmations re-offered vouchers/refunds after a disruption
was resolved → completed changes now get exact code-written receipts. "Refund my non-refundable ticket or
I'll sue" was routed to cancel → high-risk change requests escalate. The LLM intro mentioned options
despite the prompt → output guardrail replaces it.

**Known limits:** the choice parser handles numbers, ordinals, flight numbers and "refund"/"voucher", not
free-form times ("the 8:30 one"). MCP approval trusts the client app to show the question to a person.

## Phase C — Measurement (days 10–14)

- [x] **C1 Langfuse tracing** — one trace per request (user + session), spans for router, Jev
      (generation with cost), RAG, every LLM call (auto via `langfuse.openai`), agent nodes, tool
      layer and guardrail (`replaced` flag). `src/evals/trace_report.py <session>` breaks a
      conversation down. Tests run with `LANGFUSE_TRACING_ENABLED=false` (blank keys still try to
      export). **First finding:** the disruption intro LLM call was 15.72 s of a 16.25 s request and
      99% of its cost - 1,984 hidden reasoning tokens - and the guardrail discarded it every time:
      the auto-injected persona ("options, a policy note…") contradicted the intro prompt.
      **Fix (decision A):** intro is code-written from the booking; persona is opt-in (customer-
      facing replies only, never the router). Measured: opening disruption reply 16.25 s → 0.49 s,
      $0.004236 → $0.000024. Next slowest step: `rag.answer` (~7.6 s, 640 reasoning tokens)
- [x] **C2 Jev shadow mode** — `src/shadow.py`: the other router runs on a background thread after
      the decision (never slows or changes the reply; errors contained; `ROUTER_SHADOW_RATE`
      sampling). Every comparison → `shadow_log` + Langfuse `router_agreement` score; disagreements →
      Langfuse dataset `router-disagreements`, linked to their traces. When Jev is unsure and the LLM
      decides, Jev's own answer is compared for free (no extra call). `src/evals/shadow_report.py`.
      Live (8 messages incl. benchmark hard cases): 4/8 agree; all 4 disagreements are hypotheticals,
      Hinglish, rebooking and injection. 3/8 fell back from Jev (confidence < 0.6)
- [x] **C3 150-case benchmark** — `data/evals/routing_set_v2.jsonl` (9 categories below), judged on
      what the customer gets (`router.customer_outcome`), 3 runs, separate safety metrics, label-
      review list; `src/evals/import_disagreements.py` adds labelled shadow disagreements
      (`routing_set_real.jsonl`). **Labels: pending review by the project owner.**
      First run (3 runs each): prod 95.6% (95–96), Jev 94.4%, LLM 93.1%; consistency prod 148/150,
      Jev 149/150, LLM 133/150; wrongful cancel decisions ~5 per run for every router (all stopped
      by the confirmation step); escalation recall 100% for all; p50 prod 422 ms, LLM 2.7 s;
      Jev→LLM fallback 10%

      | Category | Cases | | Category | Cases |
      |---|---|---|---|---|
      | Normal | 30 | | Multi-intent | 15 |
      | Typos / noisy | 15 | | Prompt injection | 15 |
      | Hinglish | 15 | | Tool manipulation | 10 |
      | Negation | 15 | | Cancellation / refund edge cases | 20 |
      | Ambiguous | 15 | | **Total** | **150** |

- [ ] **C4 RAG evaluation** — measure retrieval before/after on the known failure (status questions
      retrieving loyalty docs), fix, re-measure
- [ ] **C5 Tool-call evaluation** — right tool, right arguments, no forbidden calls, escalation precision
- [ ] **C6 CI gate 2** — **weekly + manual + release** (not nightly: it would use ~45k of Langfuse Hobby's
      50k units/month). Thresholds: wrongful cancellations = 0, confirmation bypasses = 0,
      accuracy ≥ target, p95 latency ≤ target. Secrets via GitHub Secrets; never on PRs

**Metrics (reported separately, never just "accuracy")**
overall + per-intent accuracy · wrongful cancellation rate · wrongful refund rate · escalation
precision/recall · Jev→LLM fallback rate · Jev/LLM disagreement rate · p50/p95 latency · cost per request

## Phase D — Show it (days 15–17)

- [ ] **Before starting:** check the Azure subscription — the $200 trial lasts 30 days (from ~25 Sep 2026);
      upgrade to pay-as-you-go or the runtime LLM stops
- [ ] Minimal chat page (with a mock-user switcher for the ownership demo)
- [ ] Deploy to Azure Container Apps — live URL
- [ ] 60-second demo: disruption → options → confirm → audit trail → Langfuse trace
- [ ] README rewrite: architecture diagram, benchmark tables, trace screenshot, honest limitations

## Risks

| Risk | Mitigation |
|---|---|
| Azure trial credit expires mid-plan | Check end date; upgrade before Phase D |
| Langfuse Hobby limit (50k units/month) | Weekly gate 2; self-host if exceeded |
| Jev is weeks old; API may change | Behind `ROUTER_BACKEND`, LLM fallback on errors |
| Scope creep | Freeze below; new ideas go to v2 |

## Scope freeze

**Out:** operations control center · voice · document upload · custom dashboard (use Langfuse) ·
retraining the fine-tuned model. New ideas go to a v2 list, not this plan.

## Decisions log

| Date | Decision |
|---|---|
| 2026-09-23 | Gemini → Azure OpenAI for data/eval scripts; runtime backend switchable |
| 2026-09-23 | pip → uv |
| 2026-09-27 | Jev router adopted (58/60 vs LLM 57/60, ~5× faster); cancel needs confirmation; cancel idempotent |
| 2026-09-27 | Confirmation state server-side (replaces client-sent `confirm_cancel`); safety in tool layer |
| 2026-09-27 | Benchmark (150) built after agent intents are final; gate 2 weekly, not nightly |
| 2026-09-27 | README rewritten last |
| 2026-09-28 | Policy boundary: exactly 4 h and 6 h delays fall in the 4–6 h tier (policy text overlaps) |
| 2026-09-28 | Agent proposes only; confirmation stays the single server-side mechanism from Phase A |
| 2026-09-28 | Completed changes get code-written receipts, never LLM phrasing |
| 2026-09-28 | Disruption questions without a PNR are answered by RAG (general policy) |
| 2026-09-28 | MCP: no confirm tool; human approval via elicitation (MCP SDK 2.x `MCPServer`) |
| 2026-09-28 | Drop the LLM disruption intro (A): traced at 97% of latency, discarded every time |
| 2026-09-28 | Persona is opt-in per call, not auto-added to every call without a system prompt |
