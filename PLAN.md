# ACME Disruption Agent — Plan

> An airline disruption-recovery agent where irreversible actions need server-side, expiring,
> authorised confirmation; routing is benchmarked (LLM vs Jev, trialled in shadow mode); and
> every decision is traced, costed and gated in CI.

**Status:** Phase 0 done · Phase A next · **Scope:** frozen (see bottom) · **README:** rewritten last, in Phase D

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
- [ ] **A6 CI gate 1** — pytest on every push and PR; no model calls, no cost

**Done when:** tests prove an injected cancel, another user's booking, an expired confirmation, a replayed
"yes" and a double-click cannot cancel or refund.

## Phase B — Disruption recovery agent (days 4–9)

- [ ] **B1 Mock airline data** — flights (times, status, seats), bookings (fare class, refundability, owner),
      disruptions, alternatives with availability
- [ ] **B2 Tools** — `get_booking`, `get_flight_status`, `find_alternatives`, `rebook`, `refund`,
      `issue_voucher`, `escalate_to_human`; every write tool goes through A4
- [ ] **B3 LangGraph agent** — route → load booking → check disruption → retrieve policy → eligible options →
      present 2–4 → passenger picks → pending action → confirm (own node) → execute (own node);
      state persisted with the SQLite checkpointer
- [ ] **B4 Three-way decision** — simple → handle; ambiguous → clarify; outside policy / high-risk → human
      queue with summary. Risk signals (`angry`, `demands_exception`, …) are extra questions in the
      **same** Jev routing call — no added latency
- [ ] **B5 Policy in code** — eligibility computed from the policy rules, never by the LLM
- [ ] **B6 MCP server** — exposes the same tool layer with the same checks

**Done when:** scripted run passes — cancelled flight → options → pick rebook → confirm → rebooked + audit
trail — and the escalation path works.

## Phase C — Measurement (days 10–14)

- [ ] **C1 Langfuse tracing** — router, retrieval, LLM, tools; cost and latency per step
- [ ] **C2 Jev shadow mode** — serve with Jev, run the LLM router silently, log disagreements to a
      Langfuse dataset for review
- [ ] **C3 150-case benchmark** — built against the final intents, every label reviewed; shadow
      disagreements feed dataset v2

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
