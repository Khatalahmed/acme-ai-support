# ACME Disruption Agent — Plan

> An airline disruption-recovery agent where irreversible actions need server-side, expiring,
> authorised confirmation; routing is benchmarked (LLM vs Jev, trialled in shadow mode); and
> every decision is traced, costed and gated in CI.

**Status:** Phases 0, A, B and C done · Phase D next · **Scope:** frozen (see bottom) · **README:** rewritten last, in Phase D

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
      Jev→LLM fallback 10%. Label review done (i15 kept; tm09 accepts disruption or escalate).
- [x] **C3b Fix the wrong-cancel pattern (measure → change → re-measure)** — held-out sets written
      and baselined BEFORE each change (`routing_set_holdout.jsonl`: 8 genuine cancels + 12
      lookalikes; `routing_set_holdout_exception.jsonl`: 8 exception/anger cases). Changes: cancel
      only on a clear instruction (questions, conditions, undo, refund-status, claimed approvals are
      not cancels) in both routers; demanding an exception → a person on ANY route (the old rule only
      caught it when the router misrouted it to cancel - fixing the misroute exposed that).
      **Production, before → after:** held-out 81.7% → 100% (wrongful cancels 3.7 → 0/run, genuine
      cancels still 100%); exception held-out 91.7% → 100%; 150-case set 95.6% → 98.4%, wrongful
      cancels 4.7 → 1.3/run, escalation precision 92% → 100%, recall 100%. Remaining: the JSON-
      injection case (i02) and "refund to my other card" (tm07) - both still stopped by the
      confirmation step; "canel ACX-456" (t13) still misses (hyphen PNR regex, known limit)

      | Category | Cases | | Category | Cases |
      |---|---|---|---|---|
      | Normal | 30 | | Multi-intent | 15 |
      | Typos / noisy | 15 | | Prompt injection | 15 |
      | Hinglish | 15 | | Tool manipulation | 10 |
      | Negation | 15 | | Cancellation / refund edge cases | 20 |
      | Ambiguous | 15 | | **Total** | **150** |

- [x] **C4 RAG evaluation** — `src/evals/rag_eval.py` + `data/evals/rag_set.jsonl` (41 questions, 4
      unanswerable). Retrieval and answers measured separately, all deterministic (hit@k, MRR, fact
      recall, grounded numbers, correct refusals). Baseline (top 3, default effort): hit@3 95%, fact
      recall 95%, p50 8.1 s; both misses were RETRIEVAL (answer ranked 4th/5th) and the LLM correctly
      refused. Effort sweep: default/low/minimal equal quality, minimal 3.2x faster; top 5 fixes the
      misses but is flaky with minimal. **Chosen: TOP_K 5 + RAG_REASONING_EFFORT=low → 100% fact
      recall on 2/2 runs, 0 invented numbers, p50 ~4.1 s.** Status replies: searching with the
      customer's words found the right policy 7/21 (2.7 irrelevant sections/reply); query from
      booking 12/21; **lookup by section name from the booking state 21/21, 0 irrelevant**. Also
      stopped internal fields (`flight_status: scheduled`) reaching customer replies. Known limit:
      Hinglish retrieval (English-only MiniLM) - covered by top 5, a multilingual embedding is v2
- [x] **C5 Tool-call evaluation** — `src/evals/tool_eval.py` + `data/evals/tool_scenarios.jsonl`
      (20 multi-turn conversations): real router, LLM, safety layer and backend on a fresh airline
      per conversation; judged on executed actions (pending_actions), escalations, reply must/must-not
      phrases and final booking state. First run (2x): actions 12/12, **0 unwanted**, escalation 6/6,
      but the on-time status reply invented services ("status alerts", "check-in") and recited raw
      JSON ("resolution: null; vouchers: []") - a persona rewrite alone didn't fix it (0/3). Fixed
      structurally: the LLM gets plain-language facts (`customer_view`), status replies get their own
      system prompt, and the closing question is code. **Final (3 runs, REPLY_REASONING_EFFORT=low):
      60/60 conversations, 18/18 actions, 0 unwanted, escalation 9/9, reply checks 27/27; status
      reply p50 10.5 s → 4.4 s, explanations 12.1 s → 5.6 s**
- [x] **C6 CI gate 2** — `.github/workflows/ai-quality-gate.yml` (weekly Mon 03:30 UTC + manual +
      `v*` tags; never on PRs) runs `src/evals/gate.py`: tool eval, RAG eval, production router on
      178 messages, against the versioned limits in `data/evals/gate.json` (critical: 0 unwanted
      actions, 100% action + escalation recall, 100% grounded numbers, 100% refusals on unanswerable;
      quality floors below measured values for noise). Tracing off (Langfuse quota). Results to the
      run summary page + 90-day artifact. Gate logic has its own 13 tests. **Local run with the
      production configuration: PASSED 13/13 in ~3.2 min.** Activation needs the repo secrets and
      the workflow on `main` (GitHub only runs scheduled/manual workflows from the default branch)

**Metrics (reported separately, never just "accuracy")**
overall + per-intent accuracy · wrongful cancellation rate · wrongful refund rate · escalation
precision/recall · Jev→LLM fallback rate · Jev/LLM disagreement rate · p50/p95 latency · cost per request

## Phase D — Show it (days 15–17)

- [x] **Before starting:** check the Azure subscription — the $200 trial lasts 30 days (from ~25 Sep 2026);
      upgrade to pay-as-you-go or the runtime LLM stops (checked: Enabled)
- [x] **D1 Chat page** (with a mock-user switcher for the ownership demo) — `src/web/index.html`
- [x] **D2 Safe to put online** — per-visitor rate limit + daily cap, demo reset, non-root Docker image
- [x] **D3 Deploy to Azure Container Apps — live URL** (keys as secrets, 1 replica; free trial blocks
      ACR Tasks, so images are built locally and pushed with a new tag each release). Found live:
      re-asking about a resolved booking offered options that no longer existed (3.9 s LLM reply →
      0.3 s code reply); RAG answers leaked prompt headings 33/41 → 0/41 (now a gated metric)
- [x] **D3b Glass-box chat + public /insights** (scope change, see decisions log) — every reply has a
      "How I decided" panel built by code from what happened (router + confidence, action, each model
      call's real time and tokens, risk signals, safety checks, sources). `/insights`: measured results
      computed from committed eval files (`build_insights.py`, drift-tested) + numbers-only live stats
- [ ] 60-second demo: disruption → options → confirm → audit trail → Langfuse trace
- [x] **D5 README rewrite:** architecture diagram, measured results with sources, benchmark tables,
      live-site screenshots, honest limitations. AI quality gate first run on GitHub: 14/14 passed

## Risks

| Risk | Mitigation |
|---|---|
| Azure trial credit expires mid-plan | Check end date; upgrade before Phase D |
| Langfuse Hobby limit (50k units/month) | Weekly gate 2; self-host if exceeded |
| Jev is weeks old; API may change | Behind `ROUTER_BACKEND`, LLM fallback on errors |
| Scope creep | Freeze below; new ideas go to v2 |

## Scope freeze

**Out:** operations control center · voice · document upload · internal ops dashboard (use Langfuse) ·
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
| 2026-09-28 | Router changes are validated on held-out cases written before the change (no tuning to the 150) |
| 2026-09-28 | Demanding an exception escalates on any route; anger alone only escalates change requests |
| 2026-09-28 | RAG: top 5 sections; reasoning effort is an .env setting (low), not code (model-dependent) |
| 2026-09-28 | Status replies look policy up by section name from booking state - search only when unknown |
| 2026-09-28 | LLMs get facts in plain language, never raw JSON; reply closings that offer help are code |
| 2026-09-28 | Public /insights page added (reverses "no custom dashboard"): Langfuse is private, so the measured results were invisible to anyone evaluating the project. Numbers only, every figure traceable to a committed run |
