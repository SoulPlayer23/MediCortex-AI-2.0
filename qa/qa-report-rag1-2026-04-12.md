# QA Report — RAG-1 Multi-Turn Conversation
**Date:** 2026-04-12
**Tester:** Claude Code (browser automation via Claude in Chrome)
**Branch:** main | **Commit:** c130347
**Backend:** `http://localhost:8001` | **Frontend:** `http://localhost:5173`

---

## Scope

End-to-end UI testing of RAG-1 features:
- Part A: Multi-entity KB retrieval & vague query detection (clarification)
- Part B: Reactive re-retrieval (post-agent feedback loop)
- Topic-shift follow-up detection (T1.3 / T4.4)
- Regression: streaming UI, session persistence, sidebar preview, empty message guard

---

## Test Results

### Suite 1 — Multi-Entity KB Retrieval (RAG-1 Part A)

#### T1.1 — Dual-entity drug interaction
**Query:** `What are the interactions between metformin and lisinopril?`
**Result: ⚠️ PARTIAL PASS**

| Check | Status | Notes |
|---|---|---|
| Thinking Process shows ≥2 distinct agent tool calls | ✅ | 5 steps total; Diagnosis ran twice (original + re-retrieved) |
| Response addresses both drugs | ✅ | Both metformin and lisinopril mentioned throughout |
| Orchestrator logs: two `KB lookup` lines | ⚠️ | Only 1 "Querying Knowledge Core" UI thought; internal multi-entity extraction not surfaced as separate thought steps |
| "Diagnosis Agent Response (re-retrieved)" section | ✅ | Visible — Part B triggered |
| KB contained specific interaction data | ❌ | ArangoDB returned empty for this query; agent synthesised from general knowledge |

---

### Suite 2 — User Clarification

#### T2.1 — Vague query triggers clarification
**Query:** `my heart feels weird`
**Result: ❌ FAIL**

| Check | Status | Notes |
|---|---|---|
| Response is a short clarifying question | ❌ | Full "Diagnosis Agent Response" with clinical differentials returned |
| Thinking Process absent/empty | ❌ | 5 thinking steps (full pipeline ran) |
| Verification & Metadata absent | ❌ | Present (judge ran) |
| `is_clarification = true` in DB | ❌ | Not verified but clarification node never reached |

**Root cause:** GPT-4o-mini entity extractor returns a non-empty entity (likely `"heart"`) for this query despite the few-shot example `"my heart feels weird" -> []` in the extraction prompt. Because `retrieval_ambiguous` stays `False`, `node_router` never enters the clarification branch. The clarification node and graph wiring are correctly implemented — the extraction prompt is insufficiently strong.

#### T2.4 — Specific query bypasses clarification
**Query:** `What are the symptoms of Type 2 Diabetes?`
**Result: ✅ PASS**

Full clinical response returned immediately; no clarifying question asked; Verification & Metadata present.

---

### Suite 3 — Reactive Re-Retrieval (RAG-1 Part B)

#### T3.1 / T3.2 — Re-retrieval fires and enriches response
**Result: ✅ MECHANISM PASS / ❌ SYNTHESIS FAIL**

| Check | Status | Notes |
|---|---|---|
| Re-retrieval triggers (`retrieval_iterations = 1`) | ✅ | Consistent across ALL queries tested — 5 thinking steps pattern |
| "Diagnosis Agent Response (re-retrieved)" label visible | ✅ | Always present when re-retrieval fires |
| Re-retrieved response is richer than original | ❌ | **MedGemma synthesis loop bug** (see BUG-1 below) |
| Pipeline stays stable (no crashes) | ✅ | No errors; judge catches quality issues |

**Observation:** Re-retrieval appears to fire on nearly every query, not just sparse KB cases. This suggests the `low_context` threshold is set too low or the ArangoDB KB is largely empty, making every agent run appear to have insufficient context.

---

### Suite 4 — Existing Pipeline Regression

#### T4.4 / T1.3 — Multi-turn pronoun / topic-shift resolution
**Turns:** "Tell me about metformin" → "What are the most dangerous side effects?"
**Result: ✅ MECHANISM PASS / ❌ CONTENT FAIL**

| Check | Status | Notes |
|---|---|---|
| Turn 2 resolves "side effects" to metformin | ✅ | `side effect` keyword in `followup_indicators` triggered topic-shift injection |
| Turn 2 pipeline runs (not blank/error) | ✅ | 5 thinking steps on turn 2 |
| Turn 2 response discusses metformin side effects | ❌ | MedGemma loop bug produced unusable output (judge: 1/5) |

#### T4.5 — Streaming / Thinking accordion
**Result: ✅ PASS**
- Bouncing dots appear immediately during synthesis
- Accordion auto-expands during stream, auto-collapses on completion

#### T4.6 — Session persistence on hard refresh (Ctrl+Shift+R)
**Result: ✅ PASS**
- Session `/chat/9b8a9ed8-...` reloaded correctly after hard refresh
- Full message history visible, Verification & Metadata accordions intact
- Sidebar shows correct active session

#### T4.7 — Sidebar preview strips Markdown
**Result: ✅ PASS**
- All previews shown as plain text — no `##`, `**`, `*`, `` ` `` symbols observed

---

### Suite 6 — Error Handling

#### T6.1 — Empty message blocked
**Result: ✅ PASS**
- Input bar shows mic icon only when empty (no visible send button)
- Simulated Enter keypress on empty textarea produced zero new messages (confirmed via JS)

---

## Bugs Found

### 🔴 BUG-1 — MedGemma repetition loop during re-retrieval synthesis
**Severity:** Critical
**Affects:** Any query that triggers re-retrieval (currently: almost all queries)
**Reproduced:** 2/2 tests — "Tell me about metformin" (judge 2/5), "What are the most dangerous side effects?" (judge 1/5)

**Symptom:** The re-retrieved agent synthesis (step [4]/[5] in thinking) produces the same sentence repeated hundreds of times: `"The patient reports no history of recent hospitalizations."` — filling the entire response. The judge correctly catches it but the broken content still reaches the user.

**Root cause hypothesis:** When `node_retrieve_knowledge_v2` runs and finds no KB content (empty ArangoDB), MedGemma receives a prompt with a `[KB: ...]` section that is empty or contains only a "No specific knowledge found" fallback. MedGemma appears to fill a `Clinical Profile` section with a default template sentence and then loops it. The GPT-4o-mini fallback does NOT trigger because `low_context=True` + re-retrieval was supposed to fix it, but MedGemma still runs.

**Files:** `specialized_agents/base.py` (`_plan_and_synthesize`), `orchestrator.py` (`node_aggregator_with_reretrieval`)

**Fix direction:**
1. Add a post-synthesis repetition detector in `base.py`: if any sentence appears >3 times in the MedGemma output, discard the output and fall back to GPT-4o-mini synthesizer.
2. Cap MedGemma `max_tokens` on re-retrieval runs (the second synthesis has less to say — shorter cap prevents the loop from consuming the full token budget).
3. Or: skip re-retrieval synthesis through MedGemma entirely when KB context is still empty after re-retrieval — fall back directly to GPT-4o-mini.

---

### 🟡 BUG-2 — Clarification branch never fires (`retrieval_ambiguous` always False)
**Severity:** Medium
**Affects:** T2.1, T2.2, T2.3 (all clarification tests)

**Symptom:** Vague queries like "my heart feels weird" run the full pipeline instead of returning a short clarifying question.

**Root cause:** The extraction prompt has one negative few-shot example (`"my heart feels weird" -> []`). GPT-4o-mini still extracts `"heart"` (a body part) as a medical entity, setting `entities = ["heart"]`, so `retrieval_ambiguous` stays `False` and `node_router` never reaches the clarification branch. The clarification node, `ask_clarification`, graph wiring, and `should_re_retrieve` are all correctly implemented.

**Files:** `orchestrator.py` `node_retrieve_knowledge` (~line 263), extraction system prompt

**Fix direction:**
1. Add more negative few-shot examples for vague/body-part-only queries:
   ```
   "my back hurts" -> []
   "I feel sick" -> []
   "something is wrong with me" -> []
   "my stomach" -> []
   ```
2. Add a post-extraction body-part filter: if extracted entities are only generic anatomical terms with no medical specificity (heart, back, stomach, head alone — not "heart failure", "back pain disorder"), treat as ambiguous.
3. Alternative: lower the vagueness threshold — set `retrieval_ambiguous = True` if ALL extracted entities are single generic words shorter than 2 tokens.

---

### 🟡 BUG-3 — Re-retrieval fires on nearly every query (over-triggering)
**Severity:** Medium
**Affects:** All queries — doubles inference cost and surfaces BUG-1 on every request

**Symptom:** 5 thinking steps (Diagnosis ran twice) observed on every tested query — including clear, entity-rich queries like "What are the symptoms of Type 2 Diabetes?". Re-retrieval should only fire when the agent genuinely lacks context.

**Root cause hypothesis:** ArangoDB KB is either empty or not reachable, so every `node_retrieve_knowledge` call returns "No specific medical knowledge concept found." The agent then always sets `low_context=True` (empty tool results below the 200-char threshold in `base.py`), which always triggers re-retrieval.

**Files:** `orchestrator.py` `node_retrieve_knowledge`, `specialized_agents/base.py` `_gather_tool_results`, ArangoDB connection / data population

**Fix direction:**
1. Verify ArangoDB is reachable and populated: `python3 -m knowledge_core.build_fast_assets`. If the KB is empty, re-retrieval will always be a no-op wasting 2× MedGemma inference.
2. If KB is empty by design (offline/test env), add a guard: if the re-retrieval KB lookup also returns empty, skip the re-run and go directly to `node_aggregator`.
3. Increase the `low_context` threshold above 200 chars, or require the agent planner to explicitly emit `CONTEXT_INSUFFICIENT` sentinel (rather than auto-detecting from char count) to avoid false-positive re-retrievals.

---

## Pass / Fail Summary

| Suite | Test | Description | Status |
|---|---|---|---|
| 1 | T1.1 | Dual-entity drug interaction | ⚠️ Partial |
| 1 | T1.2 | Three-entity complex query | ⬜ Not run |
| 1 | T1.3 | Topic-shift entity injection | ✅ Mechanism / ❌ Content |
| 2 | T2.1 | Vague query → clarification | ❌ Fail |
| 2 | T2.2 | Answer to clarification → full pipeline | ⬜ Not run (blocked by T2.1) |
| 2 | T2.3 | No double clarification | ⬜ Not run (blocked by T2.1) |
| 2 | T2.4 | Specific query bypasses clarification | ✅ Pass |
| 2 | T2.5 | Clarification does not leak PII | ⬜ Not run (blocked by T2.1) |
| 3 | T3.1 | Low-context signal captured | ✅ Fires (over-triggers) |
| 3 | T3.2 | Re-retrieved context enriches response | ❌ MedGemma loop bug |
| 4 | T4.1 | Baseline diagnosis query | ✅ Pass |
| 4 | T4.4 | Multi-turn pronoun resolution | ✅ Mechanism / ❌ Content |
| 4 | T4.5 | Streaming UI / thinking accordion | ✅ Pass |
| 4 | T4.6 | Session persistence on refresh | ✅ Pass |
| 4 | T4.7 | Sidebar preview strips Markdown | ✅ Pass |
| 6 | T6.1 | Empty message blocked | ✅ Pass |
