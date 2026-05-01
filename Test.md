# MediCortex AI 2.0 — Comprehensive Test Plan

**Purpose:** End-to-end browser testing via Claude in Chrome.  
**Scope:** All RAG-1 features (multi-entity retrieval, user clarification, reactive re-retrieval) plus regression coverage of existing pipeline behaviour.  
**Pre-requisites:** Backend running on `http://localhost:8001`, frontend on `http://localhost:5173`, PostgreSQL and MinIO reachable.

---

## Setup Checklist

Before running any test, verify:

- [ ] `python orchestrator.py` is running — confirm `Orchestrator Graph Compiled` in terminal logs
- [ ] **Wait for `Gemma 4 warmup complete — model is hot` in the orchestrator log before running the first test.** `gemma4:e2b` on homeserver Ollama loads quickly (seconds, not minutes). If warmup fails, check homeserver connectivity — the model will still serve requests, just without pre-warming.
- [ ] Confirm homeserver Ollama is running with `OLLAMA_FLASH_ATTENTION=0 OLLAMA_HOST=0.0.0.0 ollama serve`. Not strictly required for `gemma4:e2b` (2B model, unaffected by the Flash Attention bug) but good practice.
- [ ] `npm run dev` is running in `frontend/` — confirm `http://localhost:5173` loads
- [ ] Structlog output is visible in the orchestrator terminal (used to verify routing decisions)
- [ ] Open Chrome DevTools → Network tab → filter by `EventSource` to watch SSE events
- [ ] Open Chrome DevTools → Console tab to catch any frontend errors
- [ ] Open a Postgres client (or `psql`) to inspect `chat_messages.message_metadata` JSONB after each test

> **API testing note (2026-04-15):** When testing via `curl` against `/chat/stream` (SSE), use `--max-time 600` or longer. Requests routinely take 4–5 minutes end-to-end. If the client disconnects mid-stream, FastAPI cancels the generator and the assistant message is **not saved to DB**. Use `POST /chat` (synchronous) for automated testing — it blocks until the full response is ready, guarantees DB persistence, and returns `session_id`, `metadata`, `thinking`, and `response` in one JSON object.

---

## Test Suite 1 — Multi-Entity KB Retrieval (RAG-1 Part A)

### T1.1 — Dual-entity drug interaction query ⚠️ PARTIAL (2026-04-17)

**Goal:** Verify that two separate `[KB: ...]` context sections are generated instead of one.

> **Result (2026-04-17):** Routed to pharmacology, `retrieval_ambiguous=false` (both entities extracted). Response correctly identifies the moderate Metformin+Lisinopril interaction. Judge 4/5. However, response is brief — only one tool call (`check_drug_interactions`) rather than two KB lookup log lines. Separate `[KB: metformin]` and `[KB: lisinopril]` sections could not be verified via API (terminal log verification required).

**Steps:**
1. Open `http://localhost:5173` in Chrome
2. Start a new chat session
3. Send: `What are the interactions between metformin and lisinopril?`
4. Wait for the full response to stream

**Expected results:**
- [x] Response addresses both drugs, not just one
- [x] In DB: `message_metadata->'retrieval_ambiguous'` = `false`
- [x] In DB: `message_metadata->'retrieval_iterations'` = `0`
- [ ] Orchestrator terminal shows two `KB lookup` log lines (one per entity) — requires terminal verification

**DB verification query:**
```sql
SELECT message_metadata->>'retrieval_ambiguous', message_metadata->>'retrieval_iterations'
FROM chat_messages
WHERE role = 'assistant'
ORDER BY created_at DESC LIMIT 1;
```

---

### T1.2 — Three-entity query (circuit breaker on KB lookups) ✅ PASSED (2026-04-17)

**Goal:** Verify all three entities get KB lookups, pipeline stays stable.

> **Result (2026-04-17):** Both `diagnosis` and `pharmacology` agents ran. Pharmacology made 3 separate `recommend_drugs` calls (T2D, HTN, HF). Diagnosis crawled all 3 conditions. Response covers all three. Judge 4/5, 95% confidence. No errors.

**Steps:**
1. New chat session
2. Send: `Compare symptoms, treatment options and drug interactions for Type 2 Diabetes, Hypertension, and Heart Failure`
3. Wait for full response

**Expected results:**
- [x] Response covers all three conditions without omitting any
- [x] `agents_used = ["diagnosis", "pharmacology"]`
- [x] No pipeline errors
- [ ] Orchestrator terminal shows three `KB lookup` log lines — requires terminal verification

---

### T1.3 — Topic-shift follow-up (implicit entity injection) ✅ PASSED (2026-04-18)

**Goal:** Verify that a vague follow-up after a specific query injects the prior entity into KB lookup.

> **Result (2026-04-18):** BUG-4 fixed. Turn 2 ("What are the side effects?") correctly injected `metformin` from `routing_context` — `is_clarification=False`, pharmacology ran. Orchestrator log: `Topic-shift detected, injecting entity from routing_context entity=metformin`. KB lookup fired for metformin. Judge 2/5 on turn 2 (MedGemma produced generic overview rather than side-effect-focused answer — KB content quality gap, not a routing failure).

**Steps:**
1. New chat session
2. Send: `Tell me about metformin`
3. Wait for full response
4. In the same session, send: `What are the side effects?` (no explicit entity)

**Expected results:**
- [x] Turn 2 resolves "side effects" to metformin (topic-shift injection)
- [x] Response discusses metformin (pharmacology agent ran, KB lookup for metformin)
- [x] `routing_context` includes prior turn's routing decision

---

## Test Suite 2 — User Clarification (New Feature)

### T2.1 — Genuinely vague query triggers clarification ✅ PASSED (2026-04-12)

**Goal:** Verify that an underspecified query produces a clarifying question, not a hallucinated answer.

> **Result (2026-04-12):** Gemma 4 warmup confirmed. Clarification fired correctly in ~12s. All expected DB fields verified.

**Steps:**
1. New chat session
2. Send: `my heart feels weird`
3. Wait for the response

**Expected results:**
- [x] AI responds with a clarifying question (e.g. *"Could you describe the sensation in more detail — is it a sharp pain, palpitations, or shortness of breath?"*)
- [x] Response is SHORT — just the question, no full clinical answer
- [x] Thinking Process accordion is ABSENT or EMPTY (no agent ran)
- [x] No "Verification & Metadata" judge score section (reviewer was skipped)
- [x] In DB: `message_metadata->>'is_clarification'` = `true`
- [x] In DB: `message_metadata->>'agents_used'` contains only `["clarification"]`

**DB verification query:**
```sql
SELECT message_metadata->>'is_clarification', message_metadata->>'agents_used'
FROM chat_messages
WHERE role = 'assistant'
ORDER BY created_at DESC LIMIT 1;
```

---

### T2.2 — User answers the clarification → full pipeline runs ✅ PASSED (2026-04-17)

**Goal:** Verify that the follow-up answer to a clarification triggers a normal full-pipeline run.

> **Result (2026-04-17):** MedGemma ran clean after `temperature=0.4` fix (no CUDA OOM, no fallback). Judge 5/5, 100% confidence. Pleurisy, ACS, PE correctly identified as differentials.

**Steps:**
1. Continue the session from T2.1 (same session)
2. Send: `It feels like a sharp stabbing pain when I breathe deeply`
3. Wait for the full response

**Expected results:**
- [x] Response is a full clinical answer (not another clarifying question)
- [x] Thinking Process accordion shows agent tool calls (diagnosis agent ran)
- [x] "Verification & Metadata" judge score section is present — **score: 5/5, 100% confidence**
- [x] In DB: `message_metadata->>'is_clarification'` = `false`
- [x] In DB: `message_metadata->>'agents_used'` includes `["diagnosis"]`
- [x] Content addresses pleuritic chest pain / ACS / PE (consistent with the symptom described)

---

### T2.3 — No double clarification on consecutive vague messages ✅ PASSED (2026-04-17)

**Goal:** Verify the system never asks for clarification twice in a row (second turn should attempt a best-effort answer).

> **Result (2026-04-17):** Turn 1 clarification fired correctly. Turn 2 routed to diagnosis (`is_clarification=false`), judge 4/5. Anti-double-clarification suppression working.

**Steps:**
1. New chat session
2. Send: `I feel unwell`
3. Observe the clarification response
4. Send: `I don't know, just generally bad`

**Expected results:**
- [x] Turn 1: clarification question returned, `is_clarification = true` in DB
- [x] Turn 2: a full-pipeline response is returned (even if vague), NOT another clarification question
- [x] Turn 2 orchestrator logs do NOT show "Clarification question generated"
- [x] In DB for turn 2: `message_metadata->>'is_clarification'` = `false`

---

### T2.4 — Specific query bypasses clarification entirely ⚠️ PARTIAL (2026-04-17)

**Goal:** Verify that a query with clear entities never triggers clarification.

> **Result (2026-04-17):** Clarification correctly bypassed (`is_clarification=false`, `retrieval_ambiguous=false`). However judge scored 3/5 — response did not directly list symptoms, instead led with "no explicit symptoms reported" and pivoted to investigation/referral. KB context returned classifications rather than symptom facts. Pipeline routing correct; response quality suboptimal for this query shape.

**Steps:**
1. New chat session
2. Send: `What are the symptoms of Type 2 Diabetes?`

**Expected results:**
- [x] Response is a full clinical answer immediately (no question asked)
- [x] In DB: `message_metadata->>'is_clarification'` = `false`
- [x] In DB: `message_metadata->>'retrieval_ambiguous'` = `false`
- [ ] Response directly lists T2D symptoms (polyuria, polydipsia, fatigue, etc.) — judge 3/5, content pivoted to management

---

### T2.5 — Clarification does not expose PII in the question ✅ PASSED (2026-04-17)

**Goal:** HIPAA — verify the clarification question does not echo any PII the user may have included.

> **Result (2026-04-17):** Clarification fired. Response: "Could you tell me where the pain is located and what kind of pain you are feeling?" — no PII present. `is_clarification=true`.

**Steps:**
1. New chat session
2. Send: `John Smith at 42 Oak Street has some pain somewhere`

**Expected results:**
- [x] Clarification triggered: question does NOT contain "John Smith" or "42 Oak Street"
- [x] No PII placeholders (`<PERSON_1>`, `<LOCATION_1>`) visible in the response
- [x] `is_clarification = true` in DB

---

## Test Suite 3 — Reactive Re-Retrieval (RAG-1 Part B)

> **Note:** This suite requires queries where the KB returns thin/empty results, which depends on ArangoDB connectivity and knowledge graph content. If ArangoDB is offline, `retrieval_feedback` will still be set (agents will flag low_context due to empty KB context), but re-retrieval will also produce empty results — both cases are valid to verify.

### T3.1 — Low-context signal propagates when KB is sparse ✅ PASSED (2026-04-17)

**Goal:** Verify that `low_context` and `retrieval_feedback` are captured in metadata.

> **Result (2026-04-17):** Pipeline stable, judge 4/5, `retrieval_feedback` populated with `{"agent": "pharmacology", "refined_query": "Erdheim-Chester disease treatment protocols"}`. Note: `retrieval_iterations` shows 0 in metadata even when re-retrieval ran — the inline v2 call in the aggregator doesn't propagate its state increment back through LangGraph (minor metadata tracking gap).

**Steps:**
1. New chat session
2. Send: `Tell me about Erdheim-Chester disease treatment protocols`
3. Wait for the full response

**Expected results:**
- [x] Response is generated (pipeline doesn't crash) — judge 4/5
- [x] `message_metadata->>'retrieval_feedback'` is non-empty (low-context signal captured)
- [ ] `message_metadata->>'retrieval_iterations'` = `1` — shows `0` due to inline v2 call not propagating state

**DB verification query:**
```sql
SELECT
  message_metadata->>'retrieval_iterations',
  message_metadata->>'retrieval_feedback'
FROM chat_messages
WHERE role = 'assistant'
ORDER BY created_at DESC LIMIT 1;
```

---

### T3.2 — Re-retrieved context appears in enriched response (or no-op guard fires)

**Goal:** Verify re-retrieval either enriches the response (real KB data found) or skips cleanly (empty KB).

**Steps:**
1. Identify a query from T3.1 where `retrieval_iterations = 1`
2. Open a second new session and send the same query again, but this time check terminal logs
3. Look for `Reactive re-retrieval triggered` in orchestrator logs

**Expected results (ArangoDB populated):**
- [ ] Terminal shows `Reactive re-retrieval triggered` with the `feedback` payload logged
- [ ] Terminal shows `Re-run agent [name] succeeded after re-retrieval`
- [ ] `node_retrieve_knowledge_v2` log line appears with a refined search term
- [ ] The final response section has a `(re-retrieved)` agent output merged in

**Expected results (ArangoDB empty / offline — no-op guard):**
- [ ] Terminal shows `Reactive re-retrieval triggered`
- [ ] Terminal shows `Re-retrieval KB lookup also returned empty — skipping agent re-run`
- [ ] Terminal shows `Agent re-run skipped — re-retrieval returned no new KB data`
- [ ] `retrieval_iterations = 1` in DB but NO `(re-retrieved)` label in response (re-run was skipped)
- [ ] Response quality is at least as good as the first pass (no repetition loop)

---

## Test Suite 4 — Existing Pipeline Regression

### T4.1 — Normal diagnosis query (baseline) ✅ PASSED (2026-04-15)

**Steps:**
1. New chat session
2. Send: `What are the symptoms and causes of Type 2 Diabetes?`

**Expected results:**
- [x] Response covers symptoms (polyuria, polydipsia, fatigue, etc.) and causes (insulin resistance, obesity, genetics)
- [x] Thinking Process accordion shows `diagnosis` agent tool calls
- [x] "Verification & Metadata" section present with judge score ≥ 3 — **score: 4/5**, reason: "accurately addresses query, evidence-based references"
- [x] Sources accordion present with relevant medical URLs (Mayo Clinic)
- [x] `agents_used = ["diagnosis"]` in DB

> **Note (2026-04-15):** MedGemma repetition loop fired and was correctly caught — `"MedGemma loop detected — falling back to Gemma 4"` logged. Gemma 4 fallback produced a clean response. BUG-1 repetition guard working as designed.

---

### T4.2 — Drug interaction (pharmacology agent) ✅ PASSED (2026-04-15)

**Steps:**
1. New chat session
2. Send: `What are the drug interactions between Metformin and Ibuprofen?`

**Expected results:**
- [x] Response addresses the interaction (renal risk, lactic acidosis concern)
- [x] `agents_used = ["pharmacology"]` in DB
- [x] Sources accordion shows drug reference URLs
- [x] Judge score: **4/5**

---

### T4.3 — Multi-agent routing (diagnosis + pharmacology) ⚠️ PARTIAL (2026-04-15)

**Steps:**
1. New chat session
2. Send: `What are the treatment options and medications for Hypertension?`

**Expected results:**
- [ ] Thinking Process shows BOTH `diagnosis` and `pharmacology` agents running (possibly in parallel)
- [ ] `agents_used = ["diagnosis", "pharmacology"]` (or reverse order) in DB
- [ ] Response covers both clinical background and specific drug options (ACE inhibitors, beta-blockers, etc.)

> **Result (2026-04-15):** `agents_used = ["pharmacology"]` only — diagnosis agent not routed. Judge score: 4/5. Response content was clinically appropriate (covered treatment options and drug classes) but came from pharmacology only. Router decided pharmacology alone was sufficient for a medication-focused query. This may be acceptable behaviour (the prompt emphasises *medications*) but warrants re-testing with a more clearly split query like T4.1-style. **Not a blocking failure — routing judgment call.**

---

### T4.4 — Multi-turn pronoun resolution ✅ PASSED (2026-04-17)

**Steps:**
1. New chat session
2. Send: `Tell me about metformin`
3. Wait for response
4. Send: `What are the most dangerous side effects?`
5. Wait for response

> **Result (2026-04-17):** Turn 2 routed to pharmacology, response discusses metformin specifically (lactic acidosis, GI effects). Judge 4/5. Note: resolved via anti-double-clarification suppression + router context (turn 2 in this session had already triggered clarification, suppressing it for turn 3). Direct topic-shift entity injection is still broken (BUG-4).

**Expected results:**
- [x] Turn 2 correctly resolves "side effects" as referring to metformin
- [x] Response discusses metformin-specific side effects (lactic acidosis, GI effects)
- [x] `routing_context` includes `Routed to: [pharmacology]` from prior turn

---

### T4.5 — Streaming and UI: Thinking Process accordion behavior

**Steps:**
1. New chat session
2. Send a query that triggers MedGemma inference (e.g. `Explain the pathophysiology of myocardial infarction`)
3. Watch the UI while the response streams

**Expected results:**
- [ ] Bouncing dots appear immediately while thinking steps stream (before first response token)
- [ ] Thinking Process accordion auto-expands while streaming is in progress
- [ ] Accordion auto-collapses once the full response is rendered
- [ ] Auto-scroll keeps the latest content visible (if user is near bottom)
- [ ] No duplicate message bubbles appear

---

### T4.6 — Session persistence across page refresh

**Steps:**
1. Open a chat session with a few messages
2. Note the URL (e.g. `/chat/some-uuid`)
3. Hard refresh the page (Ctrl+F5)

**Expected results:**
- [ ] Chat history reloads correctly — no blank empty state
- [ ] All prior messages (user and assistant) render with Markdown formatting
- [ ] Sidebar shows the correct session in the list with preview text
- [ ] Sources and thinking accordions are present on assistant messages that had them

---

### T4.7 — Sidebar preview strips Markdown

**Steps:**
1. Send a query that produces a Markdown-heavy response (headers, bold, bullets)
2. Check the sidebar preview for that session

**Expected results:**
- [ ] Preview shows plain text (e.g. `Type 2 Diabetes: Overview, Man...`), NOT `### Type 2 Diabetes...`
- [ ] No `**`, `*`, `#`, or backtick symbols visible in the sidebar preview

---

### T4.8 — PubMed agent routing ✅ PASSED (2026-04-17)

**Steps:**
1. New chat session
2. Send: `What does recent research say about GLP-1 agonists and cardiovascular outcomes?`

> **Result (2026-04-17):** `agents_used = ["pubmed"]`, response covers GLP-1 + SGLT2 cardiovascular evidence from systematic reviews and meta-analyses. Judge 4/5.

**Expected results:**
- [x] `agents_used = ["pubmed"]` in DB
- [x] Response cites research/studies
- [x] Sources populated

---

## Test Suite 5 — HIPAA & Privacy

### T5.1 — PII redacted before LLM, restored in output ✅ PASSED (2026-04-17)

**Steps:**
1. New chat session
2. Send: `My name is Sarah Johnson and I have been diagnosed with diabetes. What should I know?`

> **Result (2026-04-17):** "Sarah Johnson" absent from response (MedGemma addressed condition generically). No `<PERSON_1>` placeholder visible in response. `agents_used = ["diagnosis"]`, judge 4/5. Note: `node_restore_privacy` had nothing to restore since MedGemma didn't reference the patient by name in its output — this is correct behaviour.

**Expected results:**
- [x] "Sarah Johnson" never reaches MedGemma (redaction confirmed — no name in response)
- [x] No `<PERSON_1>` placeholder leaks into the response
- [x] Response is about diabetes management
- [ ] Final response contains "Sarah Johnson" — MedGemma didn't use the name in output so nothing to restore (expected for generic clinical responses)

---

### T5.2 — History re-injection strips names but preserves dates ✅ PASSED (2026-04-18)

**Steps:**
1. New chat session
2. Send: `I'm John Doe, diagnosed on March 15th 2024 with Type 2 Diabetes`
3. Wait for response
4. Send: `What medications were introduced in 2024 for this condition?`

> **Result (2026-04-18):** BUG-4 fixed. Turn 2 routed to pharmacology, `is_clarification=False`. "medications" now in `followup_indicators`, topic-shift entity injection fired. Presidio tokens (PERSON) correctly filtered from extracted entity. Privacy correct — "John Doe" not in any response; no `<PERSON_N>` placeholders visible; March 15th date preserved in response. Judge 2/5 (MedGemma gave generic T2D overview rather than 2024-specific drugs — KB content gap).

**Expected results:**
- [x] No `<PERSON_1>` or `<DATE_TIME_N>` placeholders in any response
- [x] "John Doe" not leaked to LLM
- [x] Turn 2 full pipeline ran (pharmacology), not clarification
- [x] `message_metadata->>'is_clarification'` = `false` for turn 2

---

## Test Suite 6 — Error Handling & Resilience

### T6.1 — Empty message

**Steps:**
1. New chat session
2. Try to send an empty message (input bar should be disabled, but if accessible: submit blank)

**Expected results:**
- [ ] Send button is disabled when input is empty
- [ ] No request is sent (verify in Network tab)

---

### T6.2 — Very long query ✅ PASSED (2026-04-17)

**Steps:**
1. New chat session
2. Paste a 2000-character medical question
3. Submit

> **Result (2026-04-17):** 1600-char complex multi-comorbidity query (T2DM + HTN + LVH + CKD + dyslipidaemia) processed cleanly. `agents_used = ["diagnosis", "pharmacology"]`, 4479-char response, judge 4/5. No errors.

**Expected results:**
- [x] Pipeline processes it without a 500 error
- [x] Response generated (4479 chars)
- [x] No truncation artifacts

---

### T6.3 — Switching sessions mid-stream

**Steps:**
1. Start a long-running query in Session A (e.g. a complex multi-agent query)
2. While the response is streaming, click on a different session (Session B) in the sidebar
3. Switch back to Session A

**Expected results:**
- [ ] Session B loads its history correctly (no blank state)
- [ ] Switching back to Session A: the full streamed response is visible, all thinking steps intact
- [ ] No duplicate message bubbles
- [ ] Verification & Metadata section (judge score) is present

---

## Test Suite 7 — Production Readiness (added 2026-05-01, refined 2026-05-01 post-implementation)

> **Pre-flight for Suite 7:**
> - `.env` must define `DEBUG=true` for local testing — config.py now refuses to start in prod with default secrets.
> - `pip install slowapi` (added to requirements.txt) — needed for T7.7.
> - Backend on `http://localhost:8001`, frontend on `http://localhost:5173`.
> - Each test specifies the **exact** payload, log line to grep, and DB query — no trial-and-error.

### T7.1 — Mid-stream disconnect persists assistant message (BUG-5)

**Goal:** Verify the assistant turn is saved to DB even when the client disconnects mid-stream.

**Setup:** Open Chrome DevTools → Network → filter "chat". Have a `psql` shell open.

**Steps:**
1. New chat. Send: `Explain the full pathophysiology of acute myocardial infarction including ischemic cascade, biomarker timing, ECG progression, and reperfusion injury — give a comprehensive textbook-level answer.`
2. Wait until at least one `data: {"type":"thought"...}` SSE event arrives in the Network tab (~5–10 s).
3. Close the browser tab (or hit DevTools → "Stop loading" and then reload). **Do not click "Stop streaming" — close the tab.**
4. Wait 90 s in the `psql` shell, then run:
   ```sql
   SELECT role, LENGTH(content) AS chars, message_metadata->>'agents_used' AS agents
   FROM chat_messages
   WHERE session_id = (SELECT id FROM chat_sessions ORDER BY created_at DESC LIMIT 1)
   ORDER BY created_at;
   ```

**Expected results:**
- [ ] Two rows: `user` (the question) and `assistant` (LENGTH > 200 chars)
- [ ] Orchestrator stdout contains exactly: `client disconnected — persisting completed graph result if available` followed by `post-disconnect save completed`
- [ ] No `unhandled CancelledError` traceback in the log

**Failure signature (what it looked like before BUG-5 fix):** only the `user` row exists; the assistant turn is gone.

---

### T7.2 — CORS allowlist enforcement (DEPLOY-3)

**Goal:** Verify CORS rejects foreign origins and accepts configured ones.

**Setup:** In `.env` set `ALLOWED_ORIGINS=http://localhost:5173`. Restart orchestrator.

**Steps:**

1. **Foreign origin test** — From any HTML file on disk (`file://`) or a different localhost port, run in Chrome DevTools console:
   ```js
   fetch('http://localhost:8001/chats', { credentials: 'include' })
     .then(r => r.text()).then(console.log).catch(console.error)
   ```
2. **Allowed origin test** — Open `http://localhost:5173` (the SPA). Run the same fetch in DevTools console.
3. **Wildcard rejection test** (prod safety check) — Set `DEBUG=false ALLOWED_ORIGINS=*` and try to start the orchestrator.

**Expected results:**
- [ ] Step 1: Console shows `TypeError: Failed to fetch` and a CORS error in the Network panel
- [ ] Step 1: Response header `Access-Control-Allow-Origin` is **absent** (NOT `*`)
- [ ] Step 2: Request succeeds with HTTP 200
- [ ] Step 2: Response header `Access-Control-Allow-Origin: http://localhost:5173`
- [ ] Step 3: Orchestrator refuses to start with: `Refusing to start with insecure production configuration: ALLOWED_ORIGINS contains '*'`

---

### T7.3 — RunPod cold-start fallback timing (DEPLOY-2)

**Goal:** Verify the Gemma 4 fallback fires within 30 s when MedGemma is unreachable.

**Setup:** Stop the local MedGemma server (`Ctrl+C` the `medgemma-host` / RunPod worker).

**Steps:**
1. New chat. Send: `Explain the pathophysiology of myocardial infarction.`
2. Start a stopwatch when you press Enter. Stop it when the first non-thought `data: {"type":"response"...}` SSE event arrives.

**Expected results:**
- [ ] Stopwatch reading: ≤ **35 s** (was up to 120 s before fix)
- [ ] Orchestrator log contains: `MedGemma server unreachable` followed by `Falling back to Gemma 4`
- [ ] Final response is coherent (Gemma 4 produces fluent output)

**Optional verification (RunPod-only):** If `MEDGEMMA_KEEPWARM_URL` is set, after the orchestrator starts you should see `MedGemma keepwarm scheduled` in the lifespan log, and `MedGemma keepwarm ping ok` every 4 minutes.

---

### T7.4 — Upload size cap (SEC-1)

**Goal:** Verify oversize uploads are rejected with HTTP 413 without OOMing.

**Setup:** `dd if=/dev/zero of=/tmp/big.pdf bs=1M count=60` (creates a 60 MB file > the 50 MB default cap).

**Steps:**
1. Create the oversize file as above. Then:
   ```bash
   curl -i -F file=@/tmp/big.pdf http://localhost:8001/upload
   ```
2. Create a small valid PDF (`/tmp/small.pdf`, any 1–10 MB PDF) and upload it the same way.
3. Watch the orchestrator's RSS in `top -p $(pgrep -f orchestrator.py)` during step 1.

**Expected results:**
- [ ] Step 1: HTTP `413` response, body `{"detail":"File too large (max 50 MB)"}`
- [ ] Step 1: Orchestrator RSS does **not** spike by 60 MB
- [ ] Step 2: HTTP `200` with `{"url": "...", "filename": "small.pdf"}`

---

### T7.5 — DB pool sizing under 10-user load (OPS-1)

**Goal:** Verify the pool sustains 10 concurrent requests without exhaustion.

**Setup:** `pip install httpx anyio` in the test venv.

**Test script** (`/tmp/load10.py`):
```python
import asyncio, httpx, time
async def hit(i):
    async with httpx.AsyncClient(timeout=600) as c:
        r = await c.post("http://localhost:8001/chat", json={"message":f"List 5 symptoms of condition #{i}"})
        return r.status_code, r.elapsed.total_seconds()
async def main():
    t0=time.time()
    results = await asyncio.gather(*[hit(i) for i in range(10)])
    print("done in", time.time()-t0, "s")
    for i,(s,e) in enumerate(results): print(i, s, round(e,1))
asyncio.run(main())
```

**Steps:**
1. Run `python /tmp/load10.py`.
2. While it runs, in a second terminal: `psql -c "SELECT count(*) FROM pg_stat_activity WHERE datname='medicortex';"`.
3. Inspect orchestrator log for `QueuePool` warnings.

**Expected results:**
- [ ] All 10 responses return HTTP 200
- [ ] Peak `pg_stat_activity` count ≤ 30 (pool_size 20 + overflow 10)
- [ ] **No** log line containing `QueuePool limit` or `pool_timeout`
- [ ] p95 latency ≤ 2× the single-request baseline (run script with `range(1)` first to capture baseline)

---

### T7.6 — `retrieval_iterations` metadata propagation (BUG-6)

**Goal:** Verify `retrieval_iterations` reads `1` after inline re-retrieval, not `0`.

**Steps:**
1. New chat. Send: `Tell me about Erdheim-Chester disease treatment protocols`
2. After the response arrives, run in psql:
   ```sql
   SELECT
     message_metadata->>'retrieval_iterations' AS iters,
     message_metadata->>'retrieval_feedback' AS feedback
   FROM chat_messages
   WHERE role='assistant'
   ORDER BY created_at DESC LIMIT 1;
   ```
3. Grep orchestrator log for `Reactive re-retrieval triggered`.

**Expected results:**
- [ ] `iters = 1` (was `0` before BUG-6 fix)
- [ ] `feedback` is non-empty JSON array
- [ ] Log line `Reactive re-retrieval triggered` appears for this turn

---

### T7.7 — Rate limit enforcement (OPS-5)

**Goal:** Verify `slowapi` returns 429 after the threshold.

**Setup:** Confirm `RATELIMIT_ENABLED=true` and `RATELIMIT_CHAT=30/minute` in config.

**Test script** (`/tmp/rate.sh`):
```bash
for i in $(seq 1 35); do
  code=$(curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8001/chat \
    -H "Content-Type: application/json" \
    -d '{"message":"What is diabetes?"}')
  echo "req $i -> $code"
done
```

**Steps:**
1. Run `bash /tmp/rate.sh` from a single host.

**Expected results:**
- [ ] First ~30 requests: HTTP 200 (or 504 if MedGemma is slow — the limiter still passes the request through)
- [ ] Requests 31–35: HTTP **429** with header `Retry-After`

---

### T7.8 — Robust clarification sentinel parse (BUG-7)

**Goal:** Verify the router's clarification path triggers reliably even when the LLM emits malformed/quoted output.

**Steps:**
1. New chat. Send: `my heart feels weird` (the established T2.1 vague query).
2. Run:
   ```sql
   SELECT message_metadata->>'is_clarification' AS clar
   FROM chat_messages WHERE role='assistant'
   ORDER BY created_at DESC LIMIT 1;
   ```
3. New chat. Send: `i'm confused about what's going on with me` (apostrophe — previously broke the naive `replace("'",'"')`).
4. Re-run the SQL.

**Expected results:**
- [ ] Both turns: `clar = true`
- [ ] No `JSONDecodeError` in orchestrator log
- [ ] Response is a single empathetic question, not a multi-paragraph clinical answer

---

### T7.9 — MinIO presigned URL TTL reduced + host swap (SEC-2)

**Goal:** Verify presigned URLs expire in 1 hour and embed the public host.

**Steps:**
1. Set `MINIO_PRESIGN_TTL_SECONDS=3600` (default). Optionally set `MINIO_PUBLIC_URL=http://files.example/` and restart.
2. Upload a small PDF via the SPA's paperclip icon.
3. Inspect the URL in the Network tab response: `X-Amz-Expires` query parameter.
4. If `MINIO_PUBLIC_URL` is set, confirm the URL host equals it.
5. Wait 65 minutes. Try to fetch the URL via `curl`.

**Expected results:**
- [ ] `X-Amz-Expires=3600` in the URL query string (was `604800` before)
- [ ] Step 4: URL host matches `MINIO_PUBLIC_URL` (when set)
- [ ] Step 5: HTTP 403 with `<Code>AccessDenied</Code>` and `<Message>Request has expired</Message>`

---

### T7.10 — `/livez` and `/readyz` probes (OBS-2)

**Goal:** Verify split health endpoints distinguish process-up from dependency-up.

**Steps:**
1. With everything running:
   ```bash
   curl -i http://localhost:8001/livez
   curl -i http://localhost:8001/readyz
   ```
2. Stop ArangoDB OR Redis on the homeserver (whichever is easier). Re-run both.
3. Restart the dependency. Re-run.

**Expected results:**
- [ ] Step 1: `/livez` → 200 `{"status":"alive"}`. `/readyz` → 200 `{"status":"ready","components":{"postgres":true,"ollama":true,"medgemma":true,"redis":true}}`.
- [ ] Step 2: `/livez` still 200. `/readyz` → **503** with the failed component flagged `false`.
- [ ] Step 3: Both return 200 again (no orchestrator restart needed).

---

### T7.11 — Frontend env-driven API base (DEPLOY-4)

**Goal:** Verify the SPA reads `VITE_API_BASE_URL` and never hits hardcoded `localhost:8001`.

**Steps:**
1. Set in `frontend/.env.development`: `VITE_API_BASE_URL=http://localhost:8001`. Run `npm run dev`. Open `http://localhost:5173`. Confirm chat works.
2. Stop the dev server. Set `VITE_API_BASE_URL=http://192.0.2.99:9999` (a known-bad host). `npm run dev`. Reload.
3. Open Chrome DevTools → Network. Click "New chat" and send any message.
4. Inspect the failed request URL.

**Expected results:**
- [ ] Step 1: chat works end-to-end.
- [ ] Step 4: failed request URL is `http://192.0.2.99:9999/chat/stream` (proves the build picked up the env, not the fallback). Console shows `Failed to fetch`.
- [ ] Confirms a production build will use the GitHub Actions secret `VITE_API_BASE_URL`.

---

### T7.12 — DEBUG=False refuses insecure config (SEC-3)

**Goal:** Verify the Pydantic validator blocks startup with default secrets in prod mode.

**Steps:**
1. In `.env` set:
   ```
   DEBUG=false
   MINIO_ACCESS_KEY=minioadmin
   MINIO_SECRET_KEY=minioadmin
   ARANGODB_PASSWORD=
   GROQ_API_KEY=
   ```
2. `python orchestrator.py`.

**Expected results:**
- [ ] Process exits immediately with `ValidationError` containing `Refusing to start with insecure production configuration`
- [ ] Each unset/default secret is listed (`MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `ARANGODB_PASSWORD`, `GROQ_API_KEY`)
- [ ] After supplying real values, orchestrator starts cleanly

---

### T7.13 — Caching planner reduces ChatOllama instantiations (OPS-4)

**Goal:** Verify Phase-1 planner is built once per agent, not per request.

**Steps:**
1. Restart orchestrator with debug logging on `langchain_ollama` (set `LOGLEVEL=DEBUG`, optional).
2. Send three different queries that all route to `pharmacology` (e.g. dosage of Metformin, dosage of Lisinopril, dosage of Atorvastatin).
3. Grep orchestrator log for `ChatOllama` initialization lines (or look for the agent's `__init__` log).

**Expected results:**
- [ ] First query: planner is built (one initialization log per agent).
- [ ] Second and third queries: **no new** ChatOllama init, only `Tool call` logs.
- [ ] Total request latency on the second/third query should be at least slightly lower than the first (warm planner).

---

---

## Test Suite 8 — Query Expansion (QEX-1, added 2026-05-01)

> Query expansion uses Gemma4 to generate up to 4 synonyms/related terms per extracted entity before KG lookup, capped at 10 total terms. Both `node_retrieve_knowledge` and `node_retrieve_knowledge_v2` (re-retrieval) apply expansion.

### T8.1 — Expansion log lines appear for a clinical query

**Steps:**
1. Restart orchestrator: `python orchestrator.py`
2. New chat. Send: `What are the drug interactions between warfarin and aspirin?`
3. Watch `tail -f /tmp/orchestrator.log` in a second terminal.

**Expected results:**
- [ ] Log contains `KB query expansion` with `original` showing `["warfarin", "aspirin"]` and `expanded` showing 6–10 terms (e.g. `["warfarin", "coumadin", "vitamin K antagonist", "anticoagulant", "aspirin", "salicylate", ...]`)
- [ ] Multiple `KB lookup` log lines follow (one per expanded term)
- [ ] Response is richer than a single-entity query — covers drug class interactions, not just the two named drugs

### T8.2 — Expansion falls back cleanly when Gemma4 returns malformed JSON

**Steps:**
1. Temporarily disconnect the homeserver Ollama (`OLLAMA_CLOUD_URL` pointed at unreachable host).
2. Send: `Tell me about metformin`
3. Restore Ollama connectivity.

**Expected results:**
- [ ] Log contains `Query expansion failed for entity` (warning, not error)
- [ ] Pipeline continues — original entity `["metformin"]` is still looked up
- [ ] Response is generated (graceful degradation, no 500)

### T8.3 — Cap at 10 terms is respected for a multi-entity query

**Steps:**
1. Send: `Compare metformin, lisinopril, atorvastatin, and aspirin interactions`

**Expected results:**
- [ ] Log shows `KB query expansion` with `expanded` list length ≤ 10
- [ ] `KB lookup` log lines count ≤ 10 (not 4 × 5 = 20)
- [ ] Response covers all four drugs

---

## Automated Test Suites (pytest)

> These run against the live backend without a browser. Ensure the orchestrator is running on `http://localhost:8001` before executing.

### EVAL-2 Component Tests (Layer 1)

Files created in the repo — run with:
```bash
pytest tests/unit/ tests/integration/ -v --tb=short -m "not stress"
```

| File | What it tests |
|---|---|
| `tests/unit/test_privacy_node.py` | HIPAA PII redaction/restoration — 6 test cases |
| `tests/integration/test_retrieval_node.py` | Entity extraction, KB degradation, synonym resolution — 6 test cases |
| `tests/integration/test_router_accuracy.py` | 50-query routing ground truth (`tests/resources/routing_ground_truth.json`) |
| `tests/integration/test_reviewer_calibration.py` | Judge score determinism, PII detection, sample rate — 6 test cases |
| `tests/integration/test_repetition_guard.py` | BUG-1 regression — repetition fallback, KB placeholder stripping — 3 test cases |

### EVAL-1 Thesis Experiments (Layer 2)

Scripts created — require `tests/resources/eval_test_set.json` (50 queries, write manually) and running backend:
```bash
pip install ragas pingouin
python tests/evaluation/run_ragas.py              # → Table III
python tests/evaluation/run_ablation.py           # → Table VI
python tests/evaluation/run_judge_calibration.py  # → Table IV
python tests/evaluation/plots/generate_all.py     # → Section 6.2 figures
```

### EVAL-3 Stress / Adversarial Tests (Layer 3, post-submission)

```bash
pytest tests/stress/ -v --tb=short -m stress
```

| File | What it tests |
|---|---|
| `tests/stress/test_failure_injection.py` | ArangoDB/Ollama/Groq/MinIO offline resilience |
| `tests/stress/test_adversarial.py` | PII injection, jailbreak prompts, path traversal uploads |

---

## Verification Queries (Reference)

Use these SQL queries in psql or a DB client to inspect metadata after any test:

```sql
-- Latest assistant message full metadata
SELECT
  content,
  message_metadata
FROM chat_messages
WHERE role = 'assistant'
ORDER BY created_at DESC LIMIT 1;

-- RAG-1 fields for all recent assistant messages
SELECT
  created_at,
  message_metadata->>'agents_used' AS agents,
  message_metadata->>'retrieval_ambiguous' AS ambiguous,
  message_metadata->>'retrieval_iterations' AS re_retrieval_count,
  message_metadata->>'is_clarification' AS clarification,
  message_metadata->>'retrieval_feedback' AS feedback
FROM chat_messages
WHERE role = 'assistant'
ORDER BY created_at DESC LIMIT 10;

-- Find all clarification turns
SELECT created_at, content
FROM chat_messages
WHERE role = 'assistant'
  AND message_metadata->>'is_clarification' = 'true'
ORDER BY created_at DESC;

-- Find all re-retrieval turns
SELECT created_at, content, message_metadata->>'retrieval_feedback' AS feedback
FROM chat_messages
WHERE role = 'assistant'
  AND (message_metadata->>'retrieval_iterations')::int > 0
ORDER BY created_at DESC;
```

---

## Pass / Fail Summary

| Suite | Test | Description | Status |
|---|---|---|---|
| 1 | T1.1 | Dual-entity drug interaction | ⚠️ 2026-04-17 — routing correct, response brief; terminal KB log verification pending |
| 1 | T1.2 | Three-entity complex query | ✅ 2026-04-17 — both agents ran, all 3 conditions covered, score 4/5 |
| 1 | T1.3 | Topic-shift follow-up entity injection | ✅ 2026-04-18 — BUG-4 fixed: entity=metformin injected from routing_context, is_clarification=False |
| 2 | T2.1 | Vague query → clarification question | ✅ 2026-04-12 |
| 2 | T2.2 | Answer to clarification → full pipeline | ✅ 2026-04-17 — score 5/5, 100% confidence, MedGemma ran clean |
| 2 | T2.3 | No double clarification | ✅ 2026-04-17 — diagnosis ran on turn 2, score 4/5 |
| 2 | T2.4 | Specific query bypasses clarification | ⚠️ 2026-04-17 — routing correct but response quality 3/5 (KB returned classifications not symptoms) |
| 2 | T2.5 | Clarification does not leak PII | ✅ 2026-04-17 — no PII in clarification question |
| 3 | T3.1 | Low-context signal captured in metadata | ✅ 2026-04-17 — retrieval_feedback populated, score 4/5 |
| 3 | T3.2 | Re-retrieved context enriches response | ⬜ Requires terminal log verification |
| 4 | T4.1 | Baseline diagnosis query | ✅ 2026-04-15 — score 4/5, MedGemma fallback guard fired correctly |
| 4 | T4.2 | Drug interaction query | ✅ 2026-04-15 — score 4/5 |
| 4 | T4.3 | Multi-agent routing | ⚠️ 2026-04-15 — pharmacology only (not diagnosis+pharmacology); score 4/5; routing judgment call |
| 4 | T4.4 | Multi-turn pronoun resolution | ✅ 2026-04-17 — metformin resolved, score 4/5 (via clarification suppression + router context) |
| 4 | T4.5 | Streaming UI / thinking accordion | ✅ 2026-03-25 (UI-1 verified) |
| 4 | T4.6 | Session persistence on refresh | ✅ 2026-03-25 (UI-4 verified) |
| 4 | T4.7 | Sidebar preview strips Markdown | ✅ 2026-03-26 (SB-1 verified) |
| 4 | T4.8 | PubMed routing | ✅ 2026-04-17 — agents_used=["pubmed"], score 4/5 |
| 5 | T5.1 | PII redacted before LLM, restored in output | ✅ 2026-04-17 — no leaks, no placeholders in output |
| 5 | T5.2 | History re-injection: names stripped, dates kept | ✅ 2026-04-18 — BUG-4 fixed: pharmacology ran on turn 2, no PII leaks, is_clarification=False |
| 6 | T6.1 | Empty message blocked | ⬜ API-level blocking not implemented (UI button disabled only) |
| 6 | T6.2 | Long query handled | ✅ 2026-04-17 — 1600-char query, score 4/5, 4479-char response |
| 6 | T6.3 | Session switch mid-stream | ✅ 2026-03-25 (UI-6 verified) |
| 7 | T7.1 | Mid-stream disconnect persists turn (BUG-5) | 🟢 Ready to test (fix shipped 2026-05-01) |
| 7 | T7.2 | CORS allowlist enforcement (DEPLOY-3) | 🟢 Ready to test |
| 7 | T7.3 | RunPod cold-start fallback (DEPLOY-2) | 🟢 Ready to test |
| 7 | T7.4 | Upload size cap (SEC-1) | 🟢 Ready to test |
| 7 | T7.5 | DB pool under 10-user load (OPS-1) | 🟢 Ready to test |
| 7 | T7.6 | `retrieval_iterations` propagation (BUG-6) | 🟢 Ready to test |
| 7 | T7.7 | Rate limit enforcement (OPS-5) | 🟢 Ready to test (`pip install slowapi` first) |
| 7 | T7.8 | Robust clarification parse (BUG-7) | 🟢 Ready to test |
| 7 | T7.9 | MinIO presigned URL TTL + host swap (SEC-2) | 🟢 Ready to test |
| 7 | T7.10 | `/livez` and `/readyz` probes (OBS-2) | 🟢 Ready to test |
| 7 | T7.11 | Frontend env-driven API base (DEPLOY-4) | 🟢 Ready to test |
| 7 | T7.12 | DEBUG=False refuses insecure config (SEC-3) | 🟢 Ready to test |
| 7 | T7.13 | Cached planner ChatOllama (OPS-4) | 🟢 Ready to test |
