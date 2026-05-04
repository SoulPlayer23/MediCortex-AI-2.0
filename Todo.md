# MediCortex AI 2.0 — Technical Todo

---

## Open Issues

### 🔴 Critical (production blockers — surfaced by 2026-05-01 review)

#### BUG-5 — Mid-stream client disconnect silently drops the assistant message to DB
**File:** `orchestrator.py:1119–1272` (`event_generator` in `/chat/stream`)
**Symptom:** When the browser tab is closed or the network drops mid-stream, FastAPI stops iterating the SSE generator. The LangGraph result is fully computed in `final_output_container`, but `chat_service.add_message(...)` at ~line 1262 never runs. The `finally` block only cleans `ACTIVE_STREAMS`. On reload the turn is missing from history.
**Fix:**
- Move the DB save into the `finally` block, guarded by a "graph completed" sentinel.
- Wrap `session_id` reference with `locals().get("session_id")` since the disconnect can happen before binding.
- Add a `try/except (asyncio.CancelledError, GeneratorExit)` to log disconnect cleanly.
**Priority:** Critical — reproduces on every mid-stream tab close at the 10-user scale.

#### DEPLOY-3 — CORS wildcard with credentials is invalid and a security hole ✅ RESOLVED (2026-05-03)
**Fix applied:** `ALLOWED_ORIGINS` env var wired into `CORSMiddleware` in `orchestrator.py:1333`. `.env` on homeserver set to `["http://localhost:5173","https://soulplayer23.github.io"]`. GitHub Pages frontend access confirmed working.

#### DEPLOY-4 — Frontend hardcodes `http://localhost:8001` (blocks GH Pages SPA) ✅ RESOLVED (2026-05-03)
**Fix applied:** `VITE_API_BASE_URL` injected via GitHub Actions secret. All components (`ChatArea.tsx`, `InputArea.tsx`, `Sidebar.tsx`) use `import.meta.env.VITE_API_BASE_URL` with `localhost:8001` fallback for dev.

#### DEPLOY-2 — RunPod Serverless cold-start handling for MedGemma ⚠️ PARTIAL (2026-05-04)
**Resolved:**
- ✅ `MEDGEMMA_API_URL` set to RunPod `/runsync` in `.env`
- ✅ `RUNPOD_API_KEY` set — Bearer auth working
- ✅ `MEDGEMMA_TIMEOUT_SECONDS=120` — raised from 30s (2026-05-04); 30s was too aggressive for warm-pod inference (2+ min synthesis on complex queries)
- ✅ Request/response shape correct (`_build_payload` wraps in `{"input":…}`, `_unwrap_response` handles `{"output":{"response":"…"}}`)

**Remaining — OPS-8 (keepwarm):**
The keepwarm pinger (`_start_keepwarm_task`) sends a `GET` to the RunPod `/runsync` URL every 240s. Both issues make it ineffective:
1. RunPod `/runsync` only accepts `POST` — a GET does nothing to wake a worker.
2. RunPod idle timeout is **10 seconds** — a 240s interval can never prevent cold starts.

See **OPS-8** below for the fix.

#### SEC-1 — Unbounded `file.read()` on `/upload` and `/extract` (DoS / OOM) ✅ RESOLVED (2026-05-04)
- `orchestrator.py:1661` reads `cap + 1` bytes and rejects with HTTP 413 if oversize.
- `tools/document_extraction_tools.py:173` streams the download and accumulates up to `MAX_PDF_BYTES`.
- `config.py`: `MAX_UPLOAD_BYTES=50MB`, `MAX_PDF_BYTES=100MB`.

---

### 🟠 High

#### OPS-1 — DB connection pool sizing for 10 concurrent users ✅ RESOLVED (2026-05-04)
- `database/connection.py`: `pool_size=20`, `max_overflow=10`, `pool_timeout=30`, `pool_pre_ping=True`.
- `config.py`: `SQLALCHEMY_POOL_SIZE=20`, `SQLALCHEMY_MAX_OVERFLOW=10`, `SQLALCHEMY_POOL_TIMEOUT=30` (all env-overridable).

#### OPS-2 — Sync graph nodes block the asyncio event loop
**File:** `orchestrator.py:665–768` (`node_aggregator_with_reretrieval`), all `make_agent_node`-built nodes, `node_retrieve_knowledge`, `node_router`
**Symptom:** Sync `def` graph nodes call `agent_executor.process(envelope)` which calls `requests.post(...)` (blocking). LangGraph's `ainvoke` only offloads sync nodes to a thread pool if explicitly configured. With one event-loop thread, all 10 users effectively serialize.
**Fix (pick one):**
- Convert all graph nodes to `async def` and switch `MedGemmaLLM` + planner code to `httpx.AsyncClient`.
- Or pass `config={"run_in_executor": True}` to `ainvoke` (verify LangGraph version supports it).
**Verification:** Hit `/chat/stream` from two clients simultaneously; second request's first SSE event should arrive within ~1s, not after the first request completes.
**Note (QEX-1):** Query expansion (added 2026-05-01) adds up to N sequential Gemma4 LLM calls inside `node_retrieve_knowledge` (one expansion call per extracted entity, capped at 10 total terms). These are sync blocking calls on the event loop. Once OPS-2 is tackled, replace these with `asyncio.gather` across all expansion calls to run them in parallel.

#### OPS-3 — `ACTIVE_STREAMS` is a process-local dict (breaks with `--workers >1`)
**File:** `orchestrator.py:191,1165,1270`
**Symptom:** Module-level dict; agents in worker A cannot publish thoughts to an SSE consumer in worker B. Even with `--workers 1`, future scale-out silently breaks thought streaming.
**Fix:**
- Short-term: assert at startup that `os.environ.get("WEB_CONCURRENCY", "1") == "1"`, document the constraint.
- Medium-term: move `ACTIVE_STREAMS` to Redis pub/sub or Redis lists keyed by `streams:{session_id}`. Agents `RPUSH`, SSE poller `BLPOP`.

#### SEC-2 — MinIO presigned URL TTL is 7 days for HIPAA-protected medical docs ✅ RESOLVED (2026-05-04)
- `services/minio_service.py`: `ExpiresIn=settings.MINIO_PRESIGN_TTL_SECONDS` (default 3600, was 604800).
- `generate_download_url()` issues fresh short-lived URLs on demand.
- `config.py`: `MINIO_PRESIGN_TTL_SECONDS=3600`, `MINIO_PUBLIC_URL` for host rewriting.

#### SEC-3 — Default credentials and `DEBUG=True` in `config.py` ✅ RESOLVED (2026-05-04)
- `config.py`: `DEBUG: bool = False` default; all secret fields default to `""`.
- `_validate_prod_secrets` model validator raises `ValueError` on startup if `DEBUG=False` and any of `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `ARANGODB_PASSWORD`, `GROQ_API_KEY` are empty/insecure, or `ALLOWED_ORIGINS` contains `"*"`.

---

### 🟡 Medium

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

---

#### REP-1 — LangExtract structured pre-extraction for `report_analyzer` agent
**Objective:** Replace the current raw-text dump into MedGemma's context with a LangExtract-powered structured pre-extraction step, giving the `report_analyzer` agent typed, schema-validated, hallucination-flagged entities before the ReAct synthesis loop runs.

**Why this is worth doing:**
The current pipeline (`extract_document_text` → raw text → MedGemma ReAct loop) has three critical gaps: (1) no structured schema — lab values, units, and reference range flags arrive as prose; (2) no hallucination tracing — fabricated values cannot be distinguished from extracted ones; (3) no multi-entity relationship resolution — anaphoric references ("the latter", "the former") in medication/diagnosis text are silently dropped. LangExtract + MedGemma 1.5 4B (already deployed locally) closes all three. MedGemma 1.5 benchmarks on this exact task: PDF→JSON lab extraction at Micro F1 88% / Macro F1 91%, and an 18% macro F1 gain over MedGemma 1.0. The model and the serving infrastructure are already in place.

**Scope:**

**Part A — New `tools/langextract_tools.py` pre-extraction layer:**

1. **Install and configure LangExtract** (`pip install langextract`). Add to `requirements.txt`. Register a custom Ollama provider plugin pointing to `http://homeserver:11434` (using `@router.register()` as documented). Use `temperature=0.0`, `use_schema_constraints=False`, `fence_output=False` (required for local Gemma/MedGemma — cloud JSON-mode is unavailable locally).

2. **Define three Pydantic extraction schemas** as LangExtract class sets:
   - `LabReportExtraction`: `lab_test_name`, `value`, `unit`, `reference_range`, `flag` (H/L/Critical), `specimen_type`
   - `RadiologyExtraction`: `finding`, `anatomic_location`, `laterality`, `severity`, `impression_line`
   - `DischargeSummaryExtraction`: `medication_name`, `dosage`, `route`, `frequency`, `duration`, `indication`, `diagnosis`, `procedure`

3. **`langextract_structured_extract(text: str, doc_type: str) → dict`** — main tool function. Detects `doc_type` from content heuristics (lab panel → `LabReportExtraction`, radiology keywords → `RadiologyExtraction`, else `DischargeSummaryExtraction`). Runs `lx.extract()` with 2–3 few-shot `ExampleData` objects per schema. Returns:
   ```python
   {
       "doc_type": "lab_report",
       "entities": [...],           # list of typed Extraction objects as dicts
       "grounded": [...],           # entities with non-None char_interval (verified)
       "ungrounded": [...],         # entities with None char_interval → hallucination suspects
       "extraction_passes": 2,
       "model_used": "medgemma-1.5-4b-it" | "gemma4:e2b"
   }
   ```

4. **Few-shot examples** — write 2–3 `lx.data.ExampleData` instances per schema using synthetic (de-identified) clinical text. Examples must use `extraction_text` that is a verbatim substring of the example source text (LangExtract requirement for char-grounding to work). Store in `tools/langextract_examples.py`.

5. **Multi-pass recall** — set `extraction_passes=2` to catch secondary findings (e.g. incidental abnormalities buried after the primary impression, PRN medications in discharge notes).

**Part B — Wire into `report_agent.py`:**

6. **Add `langextract_structured_extract` to the agent's tool list** alongside the existing `extract_document_text`, `extract_image_findings`, and `analyze_report`. Update `report_card.capabilities` to include `"langextract-structured-extraction"`.

7. **Update `_SYSTEM_PROMPT`** to instruct the agent to call `langextract_structured_extract` first on any text-based report, then use the structured JSON output as the authoritative source for the synthesis step. Flag any `ungrounded` entities explicitly in the **Abnormalities** section with a ⚠️ provenance note.

8. **Fallback chain**: if LangExtract raises `ResolverParsingError` (JSON malformed from local model), fall back silently to current `analyze_report` tool. Log the failure to `structlog` with `event="langextract_parse_error"`.

**Part C — Downstream enrichment:**

9. **Feed structured entities to sibling agents** — when `report_analyzer` runs in parallel with `pharmacology` or `diagnosis`, store the `LabReportExtraction` / `DischargeSummaryExtraction` result in `AgentState["structured_report"]`. The aggregator node can inject this structured context into the combined synthesis prompt, allowing pharmacology/diagnosis agents to reason over clean typed data rather than re-parsing prose.

10. **OBS-1 integration** — log `ungrounded_count`, `grounded_count`, `doc_type`, and `extraction_passes` into `msg_metadata["retrieval"]` (already planned in OBS-1) for dashboard visibility on extraction quality.

**Configuration notes (from research):**
- Local Gemma/MedGemma via Ollama: `use_schema_constraints=False`, `fence_output=False`, `temperature=0.0` — mandatory.
- MedGemma system prompt must stay concise: `"You are a helpful medical assistant."` — verbose system prompts degrade its performance.
- For image-based lab reports (PNG/JPEG scans), LangExtract multimodal support is tracked in upstream issue #270 — use the existing `extract_image_findings` MedGemma vision tool for images; LangExtract pre-extraction applies to text-based PDFs only in this implementation.
- `OLLAMA_FLASH_ATTENTION=0` already set on homeserver — no additional Ollama config needed.

**Dependencies:** LangExtract (`pip install langextract`). No new infrastructure required — MedGemma and Ollama are already running.

**Non-goals:**
- No vLLM migration for this ticket — Ollama with the custom provider plugin is sufficient.
- No FHIR mapping output — structured JSON stored in `AgentState` is sufficient for inter-agent communication.
- No change to the image analysis path (`extract_image_findings`) — LangExtract text-mode only in v1.

---

#### ATT-1 — Attachment-based conversation testing (PDF + image via MedGemma)
**Objective:** Validate end-to-end quality of document and image analysis through the `report_analyzer` agent, with particular focus on MedGemma's vision capabilities.
**Scope:**
- Write integration tests covering: PDF lab report upload → structured extraction, medical image (X-ray/scan) upload → MedGemma vision analysis, multi-turn follow-up questions referencing a previously uploaded attachment.
- Verify `route_decision` always includes `report_analyzer` when `file_urls` is non-empty.
- Confirm presigned MinIO URLs are still valid when MedGemma fetches them (TTL vs. inference latency).
- Document known limitations (file size limits, supported MIME types, MedGemma vision model constraints).

---

#### EVAL-2 — Component test suite (Layer 1 — run now, prerequisite for EVAL-1)
**Full spec:** `docs/evaluation-test-plan.md` §Layer 1
**Priority:** Complete before running EVAL-1 — confirms individual nodes behave correctly so full-pipeline numbers are trustworthy.

**Status (2026-05-01): Test files created ✅ — need to be run and passing.**

| File | Status | Tests | Covers |
|---|---|---|---|
| `tests/unit/test_privacy_node.py` | ✅ Created | PRIV-01..06 | 18 HIPAA identifiers, round-trip restore, multi-patient, placeholder leak prevention |
| `tests/integration/test_retrieval_node.py` | ✅ Created | RET-01..06 | Entity extraction, generic anatomy suppression, KB offline degradation, synonym resolution, multi-turn continuity |
| `tests/integration/test_router_accuracy.py` | ✅ Created | ROUTE-01..50 + CAP/MALFORMED/UNKNOWN | 50-query ground truth set; uses `tests/resources/routing_ground_truth.json` |
| `tests/integration/test_reviewer_calibration.py` | ✅ Created | JUDGE-01..06 | Fabricated dosage caught, determinism at `temperature=0`, PII placeholder detection, sample rate suppression |
| `tests/integration/test_repetition_guard.py` | ✅ Created | REP-GUARD-01..03 | BUG-1 regression: repetition triggers fallback, KB placeholder stripped from `enhanced_input` |

**Resources:**
- `tests/resources/routing_ground_truth.json` ✅ — 50 labeled queries already present
- `tests/resources/human_ratings_template.csv` ✅ — rating sheet present

**Run command:**
```bash
pytest tests/unit/ tests/integration/ -v --tb=short -m "not stress"
```

**Pass targets:** PRIV 100% · RET 100% · ROUTE ≥ 80% · JUDGE 100% · REP-GUARD 100%

---

#### EVAL-1 — Thesis evaluation experiments (Layer 2 — Tables III–VI + Section 6.2 plots)
**Component:** `orchestrator.py`, evaluation scripts in `tests/evaluation/`
**Thesis sections:** Chapter 6, Tables III–VI (RAGAS, Judge calibration, End-to-end, Ablation)
**Full spec:** `docs/evaluation-test-plan.md` §Layer 2
**Priority:** Required before dissertation submission
**Prerequisite:** OBS-1 complete (node_timings in `message_metadata`) + EVAL-2 passing

**Run order:**
```
1. pip install ragas pingouin  (add to requirements-eval.txt)
2. Write 50-query test set → tests/resources/eval_test_set.json  (10 queries × 5 domains, with ground-truth answers citing primary sources)
3. python tests/evaluation/run_ragas.py              → Table III (RAGAS scores per domain)
4. EVAL_FORCE_NOAGENT=1 python tests/evaluation/run_ragas.py    → Table V baseline
5. python tests/evaluation/run_ablation.py           → Table VI (4 ablation configs)
6. Fill tests/resources/human_ratings.csv (30 queries, 2 raters, 4 dimensions each)
7. python tests/evaluation/run_judge_calibration.py  → Table IV (ICC + Cohen's kappa)
8. python tests/evaluation/plots/generate_all.py     → 4 PDF figures for Section 6.2
```

**Scripts (all created ✅ — need `eval_test_set.json` and a running backend to execute):**

- **`tests/evaluation/run_ragas.py`** ✅ — sends each test set query to live `/chat/stream`, collects response + `message_metadata.retrieval.refined_context`, feeds `{query, answer, context, ground_truth}` into RAGAS with Llama-3.3-70B evaluator. Writes `results/ragas_scores.json`.

- **`tests/evaluation/run_judge_calibration.py`** ✅ — reads `tests/resources/human_ratings.csv` (30 queries rated by two human experts on 1–5 scale: Clinical Accuracy, Completeness, Safety, Clarity), reads judge scores from `message_metadata`, computes ICC via `pingouin.intraclass_corr()` and weighted Cohen's kappa via `sklearn.metrics.cohen_kappa_score()`.

- **`tests/evaluation/run_ablation.py`** ✅ — runs the 50-query set 4 times with these env-flag configurations:
  1. `ARANGODB_HOST=""` → no KG traversal (vector search only)
  2. `JUDGE_ENABLED=False` → no LLM-as-judge gate
  3. `MAX_CONCURRENT_AGENTS=1` in code → sequential execution (latency comparison)
  4. `MEDGEMMA_MODEL=gemma3:4b` → no domain adaptation (base model swap)

- **`tests/evaluation/plots/generate_all.py`** ✅ — produces 4 PDF figures:
  1. RAGAS faithfulness bar chart per domain
  2. Latency box plot (full vs. sequential vs. non-agentic)
  3. R-GCN ROC curve (load from notebook output)
  4. Judge score distribution histogram (from test set `message_metadata.judge_score`)

**Still needed before running EVAL-1:**
- `tests/resources/eval_test_set.json` ✅ — file exists; verify it has 50 queries with `ground_truth` fields populated
- `tests/resources/human_ratings.csv` — must be filled in by two human raters (template at `human_ratings_template.csv` ✅)

**Human rating sheet:** `tests/resources/human_ratings_template.csv` columns:
`item_id, query_preview, response_preview, rater_a_accuracy, rater_a_completeness, rater_a_safety, rater_a_clarity, rater_b_accuracy, rater_b_completeness, rater_b_safety, rater_b_clarity, judge_score, judge_reason`

**Non-agentic baseline ablation flag** — add to `node_router` in `orchestrator.py`:
```python
import os
if os.getenv("EVAL_FORCE_NOAGENT"):
    return {"messages": [AIMessage(content="[]")]}  # skip routing → direct aggregator
```

**Dissertation targets:**

| Metric | Target |
|---|---|
| RAGAS Faithfulness (overall) | ≥ 0.75 |
| RAGAS Answer Relevance (overall) | ≥ 0.80 |
| ICC (Judge vs. Human Rater) | ≥ 0.75 (excellent) |
| Weighted Cohen's kappa | ≥ 0.60 (substantial) |
| Accuracy vs. non-agentic baseline | ≥ +10% improvement |

---

#### EVAL-3 — Reliability and adversarial test suite (Layer 3 — post-submission, continuous)
**Full spec:** `docs/evaluation-test-plan.md` §Layer 3
**Priority:** 🔵 Backlog — run weekly post-thesis as production quality signal
**Marker:** `@pytest.mark.stress` — excluded from standard `pytest` CI run

**New test files to write (post-submission):**

| File | Tests | Covers |
|---|---|---|
| `tests/stress/test_failure_injection.py` | FAIL-01..05 | ArangoDB offline, Ollama 503, Groq timeout, MinIO 403, all-agents-timeout |
| `tests/stress/test_input_variation.py` | VAR-01..N | Routing stability — 5 phrasings of the same clinical concept must route to the same agent |
| `tests/stress/test_adversarial.py` | ADV-01..04 | PII injection via patient notes, jailbreak prompts, path traversal in uploads |
| `tests/stress/test_multiturn_integrity.py` | MULTI-01..03 | Cross-turn entity consistency, session isolation, PII mapping stability over 10 turns |
| `tests/stress/test_concurrent_load.py` | LOAD-01 | 10 concurrent `/chat/stream` requests — no session bleed, latency p95 ≤ 2× single-request |

**Run command (weekly cron or manual):**
```bash
pytest tests/stress/ -v --tb=short -m stress
```

---

#### OPS-8 — Keepwarm pinger is a no-op for RunPod (wrong method + wrong interval) ⚠️ PARTIAL (2026-05-04)
**File:** `orchestrator.py:_start_keepwarm_task`
**Symptom:** The pinger sends `GET` to `/runsync` every 240s. RunPod only accepts `POST` on that endpoint, and the idle timeout is 10s — so the worker is always cold by the time it's pinged, and even a correct POST every 240s wouldn't help.

**Resolved (2026-05-04):**
- ✅ Switched from `GET /runsync` to `POST /run` (async fire-and-forget) — prevents queue buildup that occurred when `/runsync` blocked 2+ minutes per ping
- ✅ `MEDGEMMA_TIMEOUT_SECONDS` raised from 30s → 120s in both `config.py` default and homeserver `.env`
- ✅ Synthesis model now logged: `[agent] synthesis complete model=medgemma|gemma4_fallback chars=N` in `specialized_agents/base.py`

**Remaining — activity-aware burst keepwarm:**
- Track `_last_request_at: float` (module-level, updated at the start of each `/chat/stream` and `/chat` request).
- Change the ping interval to **8s** but only fire when `time.time() - _last_request_at < 300` (i.e. within 5 min of the last real request). Outside that window, sleep the full 300s and recheck — avoids burning RunPod credits during idle periods.
- Log `keepwarm: worker hot` vs `keepwarm: idle, skipping` so it's observable.

**Effect:** After any real user request, the worker stays warm for the next ~5 min at a cost of ~37 pings (each ~1s inference = negligible). Outside active windows, zero cost.

#### BUG-8 — Pharmacology agent has no dosage lookup tool → score=1 on dosage queries
**File:** `specialized_agents/drug_agent.py`
**Symptom:** Queries like "What is the recommended dosage of amoxicillin for adults?" route to `pharmacology` but the agent's only tools are `check_drug_interactions` and `recommend_drugs` — neither handles dosage lookup. The agent skips tool calls entirely and MedGemma synthesizes from KB context alone, producing off-topic output (reviewer score=1, "does not address the query"). Observed 2026-05-04 on amoxicillin dosage query.
**Fix:**
- Add a `lookup_drug_dosage(drug_name: str, population: str = "adult") → str` tool to `tools/pharmacology_tools.py` that queries Drugs.com or FDA label API for standard dosing.
- Add it to `drug_agent`'s tool list alongside the existing two tools.
- Update `drug_card.capabilities` to include `"dosage-lookup"`.
**Priority:** High — any direct dosage query currently returns a disclaimer-only response.

#### OPS-4 — Per-call `ChatOllama` instantiation + missing timeouts in agent planner
**File:** `specialized_agents/base.py:286–315`
**Symptom:** A new `ChatOllama` + `bind_tools()` is built on every `_gather_tool_results` call. With 10 users × up to 5 agents = 50 simultaneous instantiations and no `timeout` set on the planner — a stale homeserver hangs indefinitely.
**Fix:**
- Cache `planner_with_tools` per agent class (build once in `__init__`).
- Add `timeout=60` to the `ChatOllama` constructor.
- Same audit on every other `ChatOllama`/`requests.post` call site.

#### OPS-5 — Rate limiting + request-size middleware
**File:** `orchestrator.py` (no middleware today)
**Fix:** Add `slowapi` (`Limiter(key_func=get_remote_address)`):
- `/chat/stream`: 10/min/IP
- `/chat`: 30/min/IP
- `/upload`: 20/min/IP
Combine with the SEC-1 size cap. Required before any public exposure.

#### BUG-6 — `retrieval_iterations` metadata always 0 when inline re-retrieval ran
**File:** `orchestrator.py:1231` + `node_aggregator_with_reretrieval` at ~line 683
**Symptom:** Test.md T3.1 confirms `retrieval_iterations=0` even though re-retrieval fired. The inline call to `node_retrieve_knowledge_v2(state)` mutates a local dict, not `AgentState`.
**Fix:** Have `node_aggregator_with_reretrieval` return `{"final_output": …, "retrieval_iteration": iteration + 1}` so LangGraph's reducer propagates the increment to `graph_output`.

#### BUG-7 — Clarification sentinel parsing is fragile
**File:** `orchestrator.py:942–947`
**Symptom:** `json.loads(last_msg.replace("'", '"'))` — relies on the LLM emitting Python-list-style output and breaks on any apostrophe in agent keys or surrounding prose.
**Fix:** Detect `"__clarify__"` substring before attempting parse; fall back to clarification only on exact match `["__clarify__"]` after a tolerant parse (`ast.literal_eval` then JSON).

#### OPS-6 — Idempotency cache key is random UUID (never hits)
**File:** `specialized_agents/base.py:86`, `specialized_agents/protocols.py`
**Symptom:** `Envelope.idempotency_key` is a per-request UUID4, so `_redis_cache.get(...)` always misses. Either dedup is silently disabled, or the Redis writes are wasted I/O.
**Fix:** Either derive the key from `hash((sender_id, receiver_id, json.dumps(payload, sort_keys=True)))`, or remove the cache layer entirely.

#### OPS-7 — Blocking Redis `ping()` at agent registry import time
**File:** `specialized_agents/base.py:61–64`
**Symptom:** `redis.from_url(...)` + `.ping()` runs synchronously when `AGENT_REGISTRY` is built (imported at orchestrator boot). A slow/down Redis blocks startup until timeout.
**Fix:** Pass `socket_timeout=2, socket_connect_timeout=2` to `redis.from_url`. Defer the ping to the lifespan health check.

#### OBS-2 — Production-grade health & readiness probes ✅ RESOLVED (2026-05-04)
- `/livez` → 200 `{"status":"alive"}` (process up, no deps checked).
- `/readyz` → 200/503 with `{"status":"ready"|"degraded","components":{postgres,ollama,medgemma,redis}}`.
- `/health` retained as legacy alias.

---

### 🔵 Backlog

#### DEPLOY-1 — Deploy frontend to GitHub Pages ✅ RESOLVED (2026-05-03)
**Live at:** `https://soulplayer23.github.io/MediCortex-AI-2.0/`
**Workflow:** `.github/workflows/deploy-pages.yml` — triggers on push to `main` (paths: `frontend/**`), builds with `VITE_API_BASE_URL` secret injected, deploys via `actions/deploy-pages`.
**Backend:** Hosted on homeserver (`orchestrator.py` on port 8001), exposed publicly via Tailscale Funnel at `https://homeserver.tail7372e0.ts.net`. Full stack confirmed working end-to-end: knowledge engine (4.9M concepts), Presidio HIPAA layer, Gemma 4 via Ollama, PostgreSQL/MinIO/Redis/ArangoDB all on homeserver Docker containers.

---

## Deployment Architecture (updated 2026-05-04)

> **Homeserver is the permanent backend.** The Hetzner/Neon/Upstash/R2 migration plan has been superseded — the homeserver runs all backend services and is the long-term host.

### Current Stack

| Component | Service | Notes |
|---|---|---|
| **Frontend SPA** | GitHub Pages | `https://soulplayer23.github.io/MediCortex-AI-2.0/` — auto-deploys on push via GitHub Actions |
| **Backend (orchestrator)** | Homeserver (port 8001) | Exposed publicly via Tailscale Funnel. `VITE_API_BASE_URL` GitHub Actions secret points here. |
| **MedGemma 1.5 4B** | RunPod Serverless | `https://api.runpod.ai/v2/a4lkhuehu9yml4/runsync` — cold starts fall back to Gemma 4 within 30s (OPS-8 keepwarm fix pending) |
| **Gemma 4 (router/aggregator)** | Homeserver Ollama | `http://homeserver:11434/v1`, model `gemma4:e2b` |
| **PostgreSQL** | Homeserver Docker | `homeserver:5432`, db `medicortex_db` |
| **Redis** | Homeserver Docker | `homeserver:6379` |
| **MinIO** | Homeserver Docker | `http://homeserver:9000` |
| **ArangoDB** | Homeserver Docker | `http://homeserver:8529`, db `clinical_ontology` |

### Remaining Hardening Work (in priority order)

1. **BUG-5** — disconnect-safe DB save
2. **OPS-8** — activity-aware keepwarm for RunPod MedGemma
3. **OPS-5** — rate limiting (`slowapi`)
4. **BUG-6, BUG-7, OPS-4** — iteration fixes

✅ Done: OPS-1 (DB pool), SEC-1 (upload cap), SEC-2 (MinIO TTL), SEC-3 (prod secrets), OBS-2 (health probes)

---

~~## Production Deployment Plan (superseded — Hetzner/Neon/Upstash/R2 migration no longer planned)~~

~~### Target Tech Stack — superseded, Hetzner/Neon/Upstash/R2 migration no longer planned~~

---

## Resolved

#### DEPLOY-1 — Deploy frontend to GitHub Pages ✅ COMPLETE (2026-05-03)
Live at `https://soulplayer23.github.io/MediCortex-AI-2.0/`. Backend on homeserver via Tailscale Funnel. See Backlog section for full details.

#### DEPLOY-3 — CORS fix ✅ COMPLETE (2026-05-03)
`ALLOWED_ORIGINS` env-driven, GitHub Pages origin added.

#### DEPLOY-4 — Frontend `VITE_API_BASE_URL` ✅ COMPLETE (2026-05-03)
All hardcoded `localhost:8001` replaced with env var, injected via GitHub Actions secret.

#### RAG-1 — Agentic RAG for multi-turn conversation ✅ COMPLETE (2026-04-17)
**Component:** `orchestrator.py`, `specialized_agents/base.py`
**QA Report:** `qa/qa-report-rag1-2026-04-12.md`
All three bugs resolved. Full test suite run 2026-04-17 — all blocking tests passed.

#### BUG-4 (Low) — Topic-shift entity injection reads wrong state field ✅ COMPLETE
**File:** `orchestrator.py` → `node_retrieve_knowledge` (~line 331)
- Parsed entity from `routing_context` instead of `state.get("context", [])` on topic-shift.
- Added `"medication"`, `"medications"`, `"treatment"` to `followup_indicators`.

#### BUG-1 (Critical) — MedGemma repetition loop during re-retrieved synthesis ✅ COMPLETE
**File:** `specialized_agents/base.py` → `_plan_and_synthesize`
- Added repetition guard in `_synthesize()`; falls back to Gemma 4 (`gemma4:e2b`) when loop detected.
- Stripped KB placeholder strings from `context_str` before building `enhanced_input` at both agent run and re-retrieval sites.
- Added no-op guard in `node_retrieve_knowledge_v2` when re-retrieved KB is also empty (`re_retrieval_skipped=True`).
- Lowered `low_context` auto-trigger threshold from 200 → 50 chars in `_gather_tool_results`.

#### BUG-2 (Medium) — Clarification branch never fires ✅ COMPLETE
**File:** `orchestrator.py` → `node_retrieve_knowledge`
- Expanded negative few-shot examples in extraction prompt (7 examples + RULE line).
- Added post-extraction body-part filter using `_GENERIC_ANATOMY` set (22 terms).

#### BUG-3 (Medium) — Re-retrieval over-triggers on every query ✅ RESOLVED (2026-04-15)
**File:** `knowledge_core/medical_engine.py`
- Replaced `resolve_entity` with `_resolve_candidates`; correctly traverses `synonym_relations`.
- `search_and_reason` iterates candidates until one has graph neighbors.
- Added `tests/test_kb_retrieval.py` standalone verification script.

#### LLM-1 — Replace GPT-4o-mini with Gemma 4 (`gemma4:e2b`) via Ollama ✅ COMPLETE (2026-04-15)
**Component:** `orchestrator.py`, `specialized_agents/base.py`, `specialized_agents/medgemma_llm.py`, `config.py`
- All GPT-4o-mini usages replaced with `gemma4:e2b` via homeserver Ollama.
- Switched `ChatOpenAI` → `ChatOllama` at all 4 instantiation sites.
- Groq remains only for judge (`node_reviewer`).

#### SB-1 — Sidebar session previews show raw Markdown symbols ✅ RESOLVED (2026-03-26)
`stripMarkdown()` helper in `Sidebar.tsx` strips heading/bold/italic/code/list markers before rendering preview.

#### AGG-5 — Tool-observation sources display raw URL as link text ✅ RESOLVED (2026-03-26)
`_extract_sources_from_observation()` two-pass method in `base.py` extracts `### N. Title` headings; falls back to bare URL regex.

#### UI-1 — No streaming progress indicator ✅ RESOLVED
Bouncing dots + "Generating response..." indicator in `MessageBubble.tsx`.

#### UI-2 — Chat does not auto-scroll to latest message ✅ RESOLVED
Smart scroll in `ChatArea.tsx` using `isNearBottomRef`; shows "↓ Scroll to bottom" button when not near bottom.

#### UI-4 — Page refresh on chat URL loads blank empty state ✅ RESOLVED
Seeded `currentSessionId` from `window.location.pathname` via lazy `useState` initializer in `App.tsx`.

#### UI-3 — Duplicate message bubbles on backend connection failure ✅ RESOLVED
`catch` block maps over messages to replace placeholder instead of pushing new error bubble.

#### AGG-1 — Aggregator emits duplicate sections ✅ RESOLVED
Explicit deduplication rules added to `node_aggregator` system prompt.

#### UI-6 — Chat switch during streaming breaks UI state ✅ RESOLVED (2026-03-25)
`sessionCache` ref + `bumpIfActive` pattern.

#### UI-5 — Last messages scroll under the input bar ✅ RESOLVED (2026-03-25)
Single `min-h-full justify-center gap-8` flex column layout in `ChatArea.tsx`.

#### AGG-2 — Sources not surfaced as a distinct UI element ✅ RESOLVED (2026-03-25)
"Sources (N)" accordion between Verification & Metadata and response body.

#### AGG-3 — Tool-fetched URLs not surfaced in Sources accordion ✅ RESOLVED (2026-03-25)
`AgentResponse.sources` populated via `_URL_RE` regex in `_gather_tool_results`; merged with inline sources.

#### AGG-4 — Aggregator response opens with generic title ✅ RESOLVED (2026-03-25)
HEADING RULES block in `node_aggregator` prompt.

#### UI-8 — Input bar overlaps welcome text during resize ✅ RESOLVED (2026-03-25)
In-flow flex column layout replaces old absolute-positioned approach.

#### UI-7 — UI refinement pass ✅ RESOLVED (2026-03-25)
Thinking Process auto-expand/collapse, sidebar last-message preview, send button disabled during streaming.
