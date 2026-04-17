# MediCortex AI 2.0 — Technical Todo

---

## Open Issues

### 🟡 Medium

#### RAG-1 — Agentic RAG for multi-turn conversation ✅ COMPLETE (2026-04-17)
**Component:** `orchestrator.py`, `specialized_agents/base.py`
**QA Report:** `qa/qa-report-rag1-2026-04-12.md`
**Status:** All three bugs resolved. Full test suite run 2026-04-17 — all blocking tests passed. One routing gap found (T1.3/T5.2 topic-shift entity injection) — logged as BUG-4 below.

**New bug discovered during 2026-04-17 testing:**

**BUG-4 (Low) — Topic-shift entity injection reads wrong state field**
**File:** `orchestrator.py` → `node_retrieve_knowledge` (~line 331)

Follow-up queries like "What are the side effects?" after "Tell me about metformin" trigger clarification instead of injecting the prior entity. Root cause: topic-shift detection reads `state.get("context", [])` to find `[KB: term]` patterns, but that field is always empty at the start of a new turn. The entity should instead be parsed from `routing_context`, which correctly carries "User asked: Tell me about metformin / Routed to: [pharmacology]".

**Required fix:**
- [ ] **`orchestrator.py` `node_retrieve_knowledge` — parse entity from `routing_context` instead of `state.get("context", [])`** when topic-shift indicators are present. Extract the last entity mentioned in prior "User asked:" lines via simple word matching or a lightweight regex.

Also discovered during testing:
- [ ] **`orchestrator.py` `node_retrieve_knowledge` — add `"medication"`, `"medications"`, `"treatment"` to `followup_indicators`** — currently missing, causes T5.2-style follow-ups ("What medications were introduced in 2024 for this condition?") to trigger clarification instead of topic-shift injection.

> **Fresh-session checklist before testing:**
> 1. Restart `python orchestrator.py` (clears in-memory agent thought cache)
> 2. Run `python3 -m knowledge_core.build_fast_assets` to verify ArangoDB is populated
> 3. Open a new chat session in the UI (do not reuse sessions from prior test runs)
> 4. Open orchestrator terminal — watch for `KB lookup`, `Clarification question generated`, and `Reactive re-retrieval triggered` log lines

---

**BUG-1 (Critical) — MedGemma repetition loop during re-retrieved synthesis**
**File:** `specialized_agents/base.py` → `_plan_and_synthesize`

When re-retrieval fires and MedGemma synthesises the second agent run, it gets stuck producing hundreds of repetitions of a single sentence (e.g. `"The patient reports no history of recent hospitalizations."`). The Groq judge catches it (scores 1–2/5) but the broken content still reaches the user.

**Root cause:** MedGemma receives a re-retrieved prompt where the KB context is still empty (ArangoDB gap). With no grounding data it fills a default `Clinical Profile` template sentence in a loop, consuming its full token budget.

**Required fixes:**

- [x] **`base.py` — add repetition guard before returning MedGemma output.** Implemented in `_synthesize()`: any sentence appearing >3 times triggers Gemma 4 (`gemma4:e2b`) fallback via `ChatOllama`. Log: `"MedGemma loop detected — falling back to Gemma 4"`.

- [x] ~~**`base.py` — cap MedGemma `max_new_tokens` on re-retrieval runs.**~~ **Rejected** — clipping tokens truncates mid-sentence without fixing the cause. Root cause fix (below) makes this unnecessary.

- [x] **Root cause fix — `orchestrator.py` `make_agent_node` + `node_aggregator_with_reretrieval` — strip KB placeholder strings from `context_str` before building `enhanced_input`.** MedGemma looped because `"No specific knowledge found in graph."` was embedded in the synthesis prompt as KB context. Now filtered out at both the initial agent run and re-retrieval re-run sites. If context becomes empty after filtering, the `Context from Knowledge Core` section is omitted entirely.

- [x] **`orchestrator.py` `node_retrieve_knowledge_v2` — no-op guard when re-retrieved KB is also empty.** Returns `re_retrieval_skipped=True`; aggregator skips agent re-run. Added `re_retrieval_skipped: bool` to `AgentState` and initialised to `False` in both invocation paths.

- [x] **`base.py` `_gather_tool_results` — lowered auto-trigger threshold from 200 → 50 chars.** Reduces false-positive `low_context` signals when tools return small-but-valid responses.

---

**BUG-2 (Medium) — Clarification branch never fires — `retrieval_ambiguous` always False**
**File:** `orchestrator.py` → `node_retrieve_knowledge` (entity extraction system prompt, ~line 263)

Vague queries like "my heart feels weird" run the full pipeline instead of returning a short clarifying question. GPT-4o-mini extracts `"heart"` (a generic body part) as a medical entity, preventing `retrieval_ambiguous` from being set. The clarification node, graph wiring, and `should_re_retrieve` edges are all correctly implemented.

**Required fixes:**

- [x] **`orchestrator.py` — expanded negative few-shot examples in the extraction prompt.** Added 7 negative examples (body-part-only, symptom-free vague) plus a RULE line and two positive counter-examples to distinguish "heart failure" (entity) from "heart" (generic).

- [x] **`orchestrator.py` `node_retrieve_knowledge` — added post-extraction body-part filter.** After parsing, if every extracted entity is a single generic anatomical term in `_GENERIC_ANATOMY` (heart, back, stomach, head, etc. — 22 terms, no qualifier), entities are cleared and the clarification branch fires. No LLM call needed.

- [x] **T2.1 — "my heart feels weird" → clarification question, no Thinking Process** ✅ **PASSED** (2026-04-12 — Gemma 4 warmup confirmed, clarification fired correctly in ~12s)
- [x] **T2.2 — follow-up answer → full pipeline runs** ✅ **PASSED** (2026-04-17 — MedGemma ran clean after temperature=0.4 fix; judge 5/5, 100% confidence; pleuritic chest pain differentials correct)
- [x] **T2.3 — two consecutive vague messages → no double clarification** ✅ **PASSED** (2026-04-17 — second vague message routed to diagnosis, `is_clarification=false`, judge 4/5)

---

**~~BUG-3~~ (Medium) — Re-retrieval over-triggers on every query (false-positive `low_context`)** ✅ RESOLVED (2026-04-15)
**File:** `knowledge_core/medical_engine.py` → `resolve_entity` / `search_and_reason`

**Actual root cause (found 2026-04-15):** ArangoDB was running and fully populated (4.9M concepts, 7.6M synonyms). The real bug was in `resolve_entity`: the `synonym_map` stores `SYN*` IDs from the `synonyms` collection, but `fetch_node_by_id` queried `concepts/SYN*` — a different collection. Every synonym lookup silently returned `None`, causing all entity resolution to fall through to fuzzy match, which frequently hit isolate nodes with no graph edges, producing empty KB context.

**Fixes applied:**
- [x] **`medical_engine.py` — replaced `resolve_entity` with `_resolve_candidates`.** New method correctly traverses `synonym_relations` (`SYN* → synonym_relations → concepts/C*`) to get the actual concept document. Returns an ordered list of candidates (synonym → exact → case-insensitive exact → fuzzy).
- [x] **`medical_engine.py` — `search_and_reason` iterates candidates until one has graph neighbors.** When the primary resolved concept is an isolate (no edges in `concept_relations`), falls through to the next candidate automatically instead of returning empty.
- [x] **`tests/test_kb_retrieval.py` — standalone retrieval verification script.** Run `python3 tests/test_kb_retrieval.py` to confirm ArangoDB connectivity and entity resolution for metformin, hypertension, and Type 2 Diabetes. All 3 pass.
- [x] **`orchestrator.py` `node_retrieve_knowledge_v2` — no-op guard.** Implemented. See BUG-1 items above.
- [x] **`base.py` — lowered `low_context` auto-trigger threshold to 50 chars.** See BUG-1 items above.

#### OBS-1 — Response evaluation / observability dashboard
**Objective:** Surface per-request evaluation data (judge score, agent selection, latency, token usage, retrieval hits) in a live dashboard without disrupting the main request path.
**Scope:**
- Backend: emit structured evaluation events (judge score, agents used, node timings, model names) to a lightweight store (Redis pub/sub or a dedicated `eval_events` table in Postgres).
- Dashboard: a parallel lightweight UI (separate route or standalone page, e.g. `/dashboard`) that polls or subscribes to these events and renders score trends, agent usage frequency, and latency histograms.
- Should not block or slow the `/chat/stream` path — fire-and-forget event emission only.

**Current state:** `message_metadata JSONB` already persists `judge_score`, `judge_reason`, `judge_confidence`, `agents_used`, `sources`, and `llm_used` per assistant message. Structlog emits per-node log lines with `trace_id`. No latency, token usage, or retrieval-hit metrics are captured. No aggregation API or dashboard UI exists.

**Proposed Implementation:**

**Part A — Enrich `message_metadata` with timing & token data (orchestrator.py):**

1. **Per-node latency capture** — wrap each graph node with a timing decorator or inline `time.perf_counter()` pairs. Collect into a `node_timings` dict:
   ```python
   node_timings = {
       "privacy": 0.12,        # seconds
       "retrieval": 1.45,
       "router": 0.31,
       "agent:diagnosis": 8.72,
       "agent:pharmacology": 7.91,
       "aggregator": 1.03,
       "reviewer": 0.88,
       "restore_privacy": 0.01
   }
   ```
   Approach: Add `node_start_times: Dict[str, float]` and `node_timings: Dict[str, float]` to `AgentState`. Each node records its own start/end. The streaming endpoint reads the final `node_timings` into `msg_metadata["node_timings"]`.

2. **Total request latency** — already derivable from `node_timings` sum, but also record wall-clock `request_started_at` / `request_completed_at` in the streaming endpoint for the true end-to-end time (includes SSE polling overhead).

3. **Token usage** — capture from LangChain response metadata where available:
   - **Router / Aggregator / Retrieval / Agent Planner** (`ChatOllama` / `gemma4:e2b`): `response.response_metadata` may include token counts depending on Ollama version — extract if present, skip otherwise.
   - **Agent Phase 2** (MedGemma synthesis): local endpoint doesn't return token counts — skip or estimate from char length. Gemma 4 fallback via ChatOllama similarly may not return counts.
   - **Judge** (Groq): `response.response_metadata` includes usage. Extract in `node_reviewer`.
   - Collect into `msg_metadata["token_usage"]`:
     ```python
     token_usage = {
         "router": {"prompt": 420, "completion": 35},
         "retrieval": {"prompt": 310, "completion": 80},
         "agent:diagnosis:planner": {"prompt": 1200, "completion": 450},
         "agent:diagnosis:synthesizer": null,  # MedGemma, no counts
         "aggregator": {"prompt": 2100, "completion": 800},
         "reviewer": {"prompt": 1500, "completion": 25},
         "total_prompt": 5530,
         "total_completion": 1390
     }
     ```

4. **Retrieval hits** — in `node_retrieve_knowledge`, log how many graph facts were returned vs. how many survived refinement. Store in `msg_metadata["retrieval"]`:
   ```python
   retrieval = {
       "entity_extracted": "metformin",
       "raw_facts_count": 12,
       "refined_context_length": 845,  # chars
       "kb_available": true             # false if ArangoDB was offline
   }
   ```

5. **Route decision rationale** — capture the raw router LLM output (the JSON array + any reasoning) in `msg_metadata["route_raw"]` for debugging mis-routes.

**Part B — Aggregation API (`/api/dashboard/*`):**

New FastAPI router (e.g. `routes/dashboard.py`) with endpoints that query `chat_messages.message_metadata` via SQL aggregations. No new tables needed — `message_metadata JSONB` is the source of truth.

6. **`GET /api/dashboard/summary?hours=24`** — returns:
   - `total_requests`: count of assistant messages in window
   - `avg_judge_score`: mean of non-null `judge_score`
   - `score_distribution`: `{1: n, 2: n, 3: n, 4: n, 5: n}`
   - `avg_latency_ms`: mean of `node_timings` total
   - `total_tokens`: sum of `token_usage.total_prompt` + `total_completion`

7. **`GET /api/dashboard/agents?hours=24`** — returns per-agent stats:
   - `agent_frequency`: `{"diagnosis": 45, "pharmacology": 32, ...}` (count of appearances in `agents_used`)
   - `agent_avg_latency`: `{"diagnosis": 8.2, ...}` (mean of `node_timings["agent:X"]`)
   - `agent_co_occurrence`: which agents are frequently routed together

8. **`GET /api/dashboard/timeseries?hours=24&bucket=1h`** — returns time-bucketed arrays for charting:
   - `timestamps`: `["2026-04-06T10:00Z", ...]`
   - `scores`: `[4.1, 3.8, ...]` (avg per bucket)
   - `latencies`: `[12.3, 15.1, ...]` (avg seconds per bucket)
   - `request_counts`: `[8, 12, ...]`

9. **`GET /api/dashboard/requests?hours=24&limit=50`** — paginated request log:
   - Per-request: `session_id`, `timestamp`, `query_preview` (first 80 chars of user message), `agents_used`, `judge_score`, `total_latency`, `token_count`
   - Supports `?min_score=1&max_score=2` filter to find low-quality responses

All queries use Postgres JSONB operators (`->>`, `->`, `jsonb_array_elements`) with index on `chat_messages.timestamp`. No new tables, no Redis pub/sub needed — the existing JSONB column is sufficient for the expected query volume (<1000 requests/day).

**Part C — Dashboard UI (`/dashboard` route in React frontend):**

10. **Separate route** — add `/dashboard` to React Router in `App.tsx`. Not part of the chat SPA layout — standalone page with its own nav.

11. **Summary cards row** — four cards at top:
    - Total Requests (24h) | Avg Judge Score (with color: green ≥4, yellow 3, red ≤2) | Avg Latency | Total Tokens

12. **Score trend chart** — line chart from `/api/dashboard/timeseries` showing judge score over time. Use a lightweight chart lib (recharts, already common in React projects).

13. **Agent usage bar chart** — horizontal bar chart showing agent invocation frequency from `/api/dashboard/agents`.

14. **Latency breakdown** — stacked bar or heatmap showing per-node timing distribution (privacy → retrieval → router → agents → aggregator → reviewer).

15. **Request log table** — sortable, filterable table from `/api/dashboard/requests`. Click a row to expand and see full `node_timings`, `token_usage`, `retrieval` details. Link to the chat session for full context.

16. **Auto-refresh** — poll `/api/dashboard/summary` every 30s. No WebSocket or SSE needed at this scale.

**Non-goals (keep it simple):**
- No Redis pub/sub or separate event store — Postgres JSONB is the single source.
- No real-time streaming of live requests to the dashboard — polling is sufficient.
- No alerting or thresholds — that's a future concern.
- No auth on dashboard routes initially (internal tool only).

#### ATT-1 — Attachment-based conversation testing (PDF + image via MedGemma)
**Objective:** Validate end-to-end quality of document and image analysis through the `report_analyzer` agent, with particular focus on MedGemma's vision capabilities.
**Scope:**
- Write integration tests covering: PDF lab report upload → structured extraction, medical image (X-ray/scan) upload → MedGemma vision analysis, multi-turn follow-up questions referencing a previously uploaded attachment.
- Verify `route_decision` always includes `report_analyzer` when `file_urls` is non-empty.
- Confirm presigned MinIO URLs are still valid when MedGemma fetches them (TTL vs. inference latency).
- Document known limitations (file size limits, supported MIME types, MedGemma vision model constraints).

### 🔵 Backlog

#### ~~LLM-1 — Replace GPT-4o-mini with Gemma 4 (`gemma4:e2b`) via Ollama~~ ✅ COMPLETE (2026-04-12, follow-up 2026-04-15)
**Component:** `orchestrator.py`, `specialized_agents/base.py`, `specialized_agents/medgemma_llm.py`, `config.py`

**Completed (2026-04-12):**
- All GPT-4o-mini usages replaced with `gemma4:e2b` via homeserver Ollama
- Replaced in: router, aggregator, entity extractor (`orchestrator.py`), agent Phase 1 planner (`base.py`), MedGemma offline fallback (`medgemma_llm.py`), repetition-loop fallback (`base.py`)
- Groq remains ONLY for judge (`node_reviewer`) — not used for any generation path
- `extractor_llm` global added to `orchestrator.py` (same Gemma 4 instance, separate name for clarity)
- **Flash Attention bug note:** `gemma4:e2b` (2B) is not affected. For larger Dense models on local Ollama: set `OLLAMA_FLASH_ATTENTION=0` to prevent hangs on prompts >3-4K tokens (Ollama GitHub #15350).

**Follow-up fix (2026-04-15):**
- Switched `ChatOpenAI` → `ChatOllama` (`langchain_ollama`) at all 4 instantiation sites. `ChatOpenAI` against Ollama's `/v1` endpoint rejects `top_k` (not an OpenAI API param); `ChatOllama` supports `top_k`/`top_p` natively via the Ollama native API.
- `OLLAMA_CLOUD_URL` default updated to `http://homeserver:11434` (no `/v1` suffix). Defensive `.removesuffix("/v1")` at each call site for `.env` backwards compat.
- `OLLAMA_CLOUD_API_KEY` config setting removed (not used by `ChatOllama`).
- `max_tokens` → `num_predict` in `medgemma_llm.py` fallback (ChatOllama's parameter name).

---

#### DEPLOY-1 — Deploy frontend to GitHub Pages
**Component:** `frontend/`, `.github/workflows/`
**Objective:** Build and deploy the React/Vite SPA to GitHub Pages via GitHub Actions CI/CD.

**Scope:**
1. Replace hardcoded `http://localhost:8001` URLs in `ChatArea.tsx`, `InputArea.tsx`, and `Sidebar.tsx` with `import.meta.env.VITE_API_URL`.
2. Add `VITE_API_URL` as a GitHub Actions secret (repository setting).
3. Create `.github/workflows/deploy-frontend.yml` — triggers on push to `main`, runs `npm ci && npm run build` with `VITE_API_URL` injected, deploys `dist/` to `gh-pages` branch.
4. Enable GitHub Pages in repo settings (source: `gh-pages` branch).

**Note:** GitHub Pages hosts static files only — all multi-chat session data lives in the backend (PostgreSQL). The deployed frontend will work correctly **only if the backend (`orchestrator.py` on port 8001) is publicly reachable**. Options: deploy backend to a VPS, or expose localhost via Cloudflare Tunnel (`cloudflared`) for a stable free HTTPS URL.

---

## Resolved

#### SB-1 — Sidebar session previews show raw Markdown symbols
**Fix verified 2026-03-26.** `### Type 2 Diabetes: Overview, Man...` → `Type 2 Diabetes: Overview, Man...`. `stripMarkdown()` helper in `Sidebar.tsx` strips `#+`, `**`, `*`, `__`, `_`, `` ` ``, `>`, and `- ` list markers before rendering the preview.

#### AGG-5 — Tool-observation sources display raw URL as link text instead of page title
**Fix verified 2026-03-26.** Metformin drug interaction query (Sources 13): sources [2], [3], [8]–[11] all show page titles (e.g. "Metformin: Package Insert / Prescribing Information / MOA", "Metformin: MedlinePlus Drug Information"). `_extract_sources_from_observation()` two-pass method in `base.py`: Pass 1 extracts `### N. Title` heading for each `- **URL:** url` line; Pass 2 falls back to bare URL regex for inline URLs not in structured format. Remaining raw-URL entries are CDN error redirects or inline bare URLs outside structured blocks — expected fallback behaviour.

#### UI-1 — No streaming progress indicator during long responses
Added bouncing dots + "Generating response..." indicator in `MessageBubble.tsx`, shown when `isStreaming && !content && thinking.length > 0`. Verified working in browser during ~4 min MedGemma inference.

#### UI-2 — Chat does not auto-scroll to latest message
Implemented smart scroll in `ChatArea.tsx` using `isNearBottomRef`. Auto-scrolls only when within 100px of bottom; shows "↓ Scroll to bottom" button otherwise. Verified working in browser.

#### UI-4 — Page refresh on a chat URL loads blank empty state
Seeded `currentSessionId` from `window.location.pathname` in `App.tsx` using a lazy `useState` initializer.

#### UI-3 — Duplicate message bubbles on backend connection failure
`catch` block now maps over messages to replace the placeholder (`aiMsgId`) instead of pushing a new error bubble.

#### AGG-1 — Aggregator emits duplicate sections
Added explicit deduplication rules to the `node_aggregator` system prompt in `orchestrator.py`: merge near-identical recommendations, keep only first occurrence of repeated source facts.

#### UI-6 — Chat switch during streaming breaks UI state
**Fix verified 2026-03-25.** Switched away mid-stream; background stream accumulated 11+ thinking steps uninterrupted. Switched back: all steps visible, full response rendered, Verification & Metadata (Judge 4/5, 95% confidence) present. `sessionCache` ref + `bumpIfActive` pattern working correctly.

#### UI-5 — Last messages scroll under the input bar and disclaimer
**Fix verified 2026-03-25.** Loaded a long response and scrolled through all positions — content stops cleanly above the in-flow input bar at every scroll position. "Scroll to bottom" button appears correctly on scroll-up. No overlap observed.

#### AGG-2 — Sources not surfaced as a distinct UI element
**Fix verified 2026-03-25.** PubMed query produced 5 cited sources. "Sources (5)" accordion appears between Verification & Metadata and the response body. Expanding it shows numbered blue hyperlinks with paper titles. `_parse_references()` strips `## References` from body correctly (confirmed 0 responses with raw references section in DB). Note: sources only appear when agents include `https://` URLs in their synthesized output — diagnosis/pharmacology agents currently don't (DOI-only); PubMed agent does reliably.

#### AGG-3 — Tool-fetched URLs not surfaced in Sources accordion
**Fix verified 2026-03-25.** Type 2 diabetes query (diagnosis + pharmacology agents) produced "Sources (22)" accordion showing all URLs fetched during ReAct tool loops — Mayo Clinic, WebMD, and others — even though the LLM prose contained no inline citations. `AgentResponse.sources` field populated via `_URL_RE` regex in `_gather_tool_results`; merged with `_parse_references` inline sources in streaming path.

#### AGG-4 — Aggregator response opens with a generic "Medical Agent Reports" title
**Fix verified 2026-03-25.** Same type 2 diabetes query opened with **"Type 2 Diabetes: Overview, Management, and Recommendations"** — no "Medical Agent Reports" boilerplate. HEADING RULES block in `node_aggregator` prompt working correctly.

#### UI-8 — Input bar overlaps welcome text during window resize in empty/new-chat state
**Fix verified 2026-03-25.** Resized window to 800×400 — welcome icon, heading, description, and input bar all remain in their in-flow flex column with no overlap at any viewport height. Single `min-h-full justify-center gap-8` flex column layout in `ChatArea.tsx` replaces the old absolute-positioned approach.

#### UI-7 — UI refinement pass (partial)
**Fix verified 2026-03-25.**
- Thinking Process accordion: auto-expanded immediately when streaming started (before first token); auto-collapsed to `>` state once response completed. Verified via screenshot.
- Sidebar last-message preview: each session entry shows title + truncated last-message preview on a second line. Lateral SQL subquery in `get_sessions` working correctly.
- Input bar send button disabled during streaming: already implemented (`disabled={isLoading || isUploading}`).
**Items not yet addressed:** spacing/padding tightening in `MessageBubble` for long Markdown; mobile viewport below 768px.
