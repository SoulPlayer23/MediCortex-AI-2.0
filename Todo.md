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

#### DEPLOY-3 — CORS wildcard with credentials is invalid and a security hole
**File:** `orchestrator.py:1105–1111`
**Symptom:** `allow_origins=["*"]` combined with `allow_credentials=True` is rejected by every modern browser per the CORS spec — and even if it worked, allows any website to make authenticated cross-origin requests to a HIPAA-adjacent API.
**Fix:**
- Add `ALLOWED_ORIGINS: list[str]` to `config.py` (default `["http://localhost:5173"]`, prod overrides with the GitHub Pages domain).
- Replace the middleware `allow_origins=["*"]` with `settings.ALLOWED_ORIGINS`.
**Priority:** Critical — blocks GitHub Pages frontend access entirely.

#### DEPLOY-4 — Frontend hardcodes `http://localhost:8001` (blocks GH Pages SPA)
**Files:** `frontend/src/components/ChatArea.tsx:107,142`, `frontend/src/components/InputArea.tsx:34` (and Sidebar.tsx if present)
**Fix:**
- Add `VITE_API_BASE_URL` to `frontend/.env.production` (VPS domain) and `.env.development` (`http://localhost:8001`).
- Replace every `http://localhost:8001` with `${import.meta.env.VITE_API_BASE_URL}`.
- Wire the variable into the GitHub Actions deploy workflow (planned in DEPLOY-1).
**Priority:** Critical — blocks the existing DEPLOY-1 ticket.

#### DEPLOY-2 — RunPod Serverless cold-start handling for MedGemma
**Files:** `config.py:27`, `specialized_agents/medgemma_llm.py:53`, `specialized_agents/base.py:19`
**Symptom:** `MEDGEMMA_API_URL` defaults to `http://localhost:8000/predict` and `MedGemmaLLM` has a 120s timeout. RunPod Serverless workers spin down after 3–10 min of inactivity; first request after cold-start can hang for the full 120s before the Gemma 4 fallback fires.
**Fix:**
- Add `MEDGEMMA_KEEPWARM_URL` to `config.py`; ping it in `lifespan()` startup and optionally from a cron job (e.g. every 4 minutes).
- Reduce `MedGemmaLLM.timeout` from 120s → 30s so the Gemma 4 fallback triggers promptly on cold start.
- Add a startup health probe to `MEDGEMMA_API_URL.replace("/predict", "/health")` so cold-start surfaces at server init, not first user.
- Confirm `requests.exceptions.HTTPError` is a subclass of `RequestException` (it is — the existing fallback path catches 503s correctly).
**Priority:** Critical — without this, every cold start = 4-minute hang for the user.

#### SEC-1 — Unbounded `file.read()` on `/upload` and `/extract` (DoS / OOM)
**Files:** `orchestrator.py:1358–1369`, `tools/document_extraction_tools.py:173`
**Symptom:** `await file.read()` reads the entire file into RAM with no size cap. `httpx.Client.get(url).content` does the same on the document extraction path.
**Fix:**
- Add `MAX_UPLOAD_BYTES = 50 * 1024 * 1024` and `MAX_PDF_BYTES = 100 * 1024 * 1024` constants in `config.py`.
- In `/upload`: read a chunk first, raise `HTTPException(413)` if oversize.
- In `document_extraction_tools.py`: switch to `client.stream("GET", url)` and accumulate up to the byte cap.
**Priority:** Critical for prod — single user can OOM a worker.

---

### 🟠 High

#### OPS-1 — DB connection pool sizing for 10 concurrent users
**File:** `database/connection.py:9`
**Symptom:** `create_async_engine` defaults to `pool_size=5, max_overflow=10`. Each `/chat/stream` request holds a connection for the full LangGraph pipeline (30–120s during MedGemma synthesis). At 10 concurrent users the pool is exhausted, requests queue, latency spikes.
**Fix:**
```python
engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    pool_timeout=30,
    pool_pre_ping=True,  # detects stale connections after VPS restarts
    echo=False,
)
```
Expose `SQLALCHEMY_POOL_SIZE` via `config.py` for env override.

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

#### SEC-2 — MinIO presigned URL TTL is 7 days for HIPAA-protected medical docs
**File:** `services/minio_service.py:51` (`ExpiresIn=604800`)
**Symptom:** Anyone with the URL can download the document for 7 days, no auth.
**Fix:**
- Reduce `ExpiresIn` to 3600 (1 hour) on the document-extraction path.
- For UI display links, generate a fresh presigned URL on demand via a `/files/{key}/url` endpoint.
- Document the TTL/inference-latency relationship for ATT-1.

#### SEC-3 — Default credentials and `DEBUG=True` in `config.py`
**File:** `config.py:7–44`
**Symptom:** `DEBUG: bool = True`, `MINIO_ACCESS_KEY = "minioadmin"`, `MINIO_SECRET_KEY = "minioadmin"`, `DATABASE_URL` includes `postgres:postgres@…`, `ARANGODB_PASSWORD = ""`. If a VPS `.env` is missing or partial, these defaults silently apply.
**Fix:**
- Change `DEBUG: bool = False` as the default; require explicit `DEBUG=true` in dev.
- Empty all secret defaults (`""`) and add a Pydantic `@model_validator(mode="after")` that raises when `DEBUG=False` and any secret is empty/default.
- Move `Ollama API Key MediCortex-AI-2.0.txt` and any in-repo secrets out of the working tree (.gitignore).

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

#### OBS-2 — Production-grade health & readiness probes
**File:** `orchestrator.py` (`/health` exists; readiness/liveness split missing)
**Fix:** Split into `/livez` (process up) and `/readyz` (Postgres + MinIO + Redis + Ollama + MedGemma reachable). Use these for VPS uptime monitoring and RunPod readiness gating.

---

### 🔵 Backlog

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

## Production Deployment Plan (added 2026-05-01)

> Persistent reference for the production rollout. Read top-to-bottom before starting any DEPLOY-* ticket. Surrounds and supersedes the older DEPLOY-1 backlog item.

### Target Tech Stack (cheapest reliable for 10 users, HIPAA-adjacent)

| Component | Service | Plan | Cost / month | Why |
|---|---|---|---|---|
| **Frontend SPA** | **GitHub Pages** | Free | $0 | Static React/Vite build, custom domain via CNAME, HTTPS auto. CI via GitHub Actions. |
| **MedGemma 1.5 4B + Gemma 4 e2b** | **RunPod Serverless GPU** | A4000/A5000 worker, scale-to-zero | ~$5–15 (10 users, ~200 req/day, ~30s/req at $0.00026/sec on A4000) | Pay-per-second. Cold-start handled by DEPLOY-2. Both models share one worker — load Gemma 4 + MedGemma 1.5 4B in a single Ollama-backed container. |
| **Backend (FastAPI orchestrator)** | **Hetzner Cloud CPX21** | 3 vCPU, 4 GB RAM, 80 GB SSD, Falkenstein/Helsinki | **€7.55 (~$8.20)** | Single-tenant VM, full root, Docker Compose stack. Best price/perf in Europe. Alternative: CX22 €4.51 (2 vCPU, 4 GB) if budget tight. |
| **PostgreSQL** | **Neon Serverless** Free tier | 0.5 GB, 1 always-on branch | $0 (upgrade to Launch $19 if >0.5 GB) | Auto-scale, point-in-time recovery, async-friendly (asyncpg compatible). Alternative: self-host on the Hetzner box (saves $0 either way at this size). |
| **Redis** | **Upstash Redis** Free | 256 MB, 10k cmd/day soft cap | $0 | REST + native protocol. Used for tool cache + ACTIVE_STREAMS once OPS-3 lands. Alternative: Redis container on the Hetzner box. |
| **Object storage (uploads)** | **Cloudflare R2** | 10 GB free, **zero egress** | $0 (until 10 GB) | S3-compatible, presigned URLs work identically to MinIO. Drop-in via `boto3`. Egress-free is critical for downloading uploads back to RunPod. |
| **ArangoDB (KG)** | Self-host on Hetzner VM (Docker) | bundled | $0 | ArangoGraph cloud is $30+/mo and overkill. Single-container ArangoDB 3.12 with daily `arangodump` backup to R2 is sufficient for 10-user load. |
| **CDN / WAF / TLS** | **Cloudflare Free** | Proxy `api.medicortex.<your-domain>` → Hetzner IP | $0 | Free TLS, DDoS shielding, simple rate-limit rules, hides origin IP. |
| **Monitoring** | **Better Stack** Free / **Grafana Cloud** Free | Uptime + log ingest | $0 | Hit `/livez` every 60s. 50 GB/mo log ingest free on Grafana Cloud. |
| **Secrets** | **Doppler** Free / **GitHub Actions secrets** | — | $0 | `.env` rendered into Hetzner box via Doppler CLI; Actions secrets for SPA build. |

**Estimated total monthly cost: ~$13–25/mo** (dominated by RunPod usage + Hetzner CPX21). Falls to ~$8 if RunPod scales fully to zero between sessions.

> **HIPAA caveat:** Hetzner / Neon / Upstash / R2 are GDPR-compliant but none sign a HIPAA BAA on their free / lowest tiers. For a thesis/internal-tool stage targeting 10 users, this is acceptable; surface it to stakeholders explicitly. If real HIPAA is required later, migrate the backend + DB to AWS (`t4g.small` $13/mo + RDS + S3 with BAA, ~$60–80/mo).

### Component Layout on Hetzner CPX21

Single VM, Docker Compose, stack on a `medicortex` bridge network:

```
┌─ Cloudflare proxy (TLS) ──────────────────────────────────────┐
│   api.medicortex.<domain>  →  Hetzner :443                    │
└──────────────────────────────────────┬────────────────────────┘
                                       │
                  ┌──────── Caddy (TLS termination, reverse-proxy) ────────┐
                  │   /chat/*, /upload, /chats → orchestrator:8001         │
                  │   /metrics                  → grafana-agent (optional) │
                  └────────────────────────────────────────────────────────┘
                                       │
       ┌──────── orchestrator (FastAPI, uvicorn --workers 1) ────────┐
       │   .env: NEON_DSN, UPSTASH_URL, R2_*, RUNPOD_MEDGEMMA_URL,   │
       │         RUNPOD_OLLAMA_URL, ALLOWED_ORIGINS, DEBUG=false     │
       └──┬──────────────────┬──────────────┬───────────────────────┘
          │                  │              │
   ┌──────▼──────┐    ┌──────▼──────┐  ┌────▼────────────┐
   │ ArangoDB    │    │  (Neon)     │  │ (Upstash Redis) │
   │ container   │    │  external   │  │   external      │
   └─────────────┘    └─────────────┘  └─────────────────┘
          │
   ┌──────▼─────────────────────────┐    ┌──────────────────────────┐
   │ daily arangodump → R2 (cron)   │    │ RunPod Serverless        │
   └────────────────────────────────┘    │  • medgemma-1.5-4b-it    │
                                         │  • gemma4:e2b (Ollama)   │
                                         │  scale-to-zero, idle 5m  │
                                         └──────────────────────────┘
```

### Migration Checklist (in order)

The Critical and High tickets above already define the *what*. This is the *order* to execute them so nothing blocks deploy.

**Phase 0 — Code prep (local, 1 day)**
1. SEC-3 — strip default secrets, `DEBUG=False` default, add Pydantic validator. Move `Ollama API Key MediCortex-AI-2.0.txt` out of repo, add to `.gitignore`, rotate the key.
2. DEPLOY-3 — `ALLOWED_ORIGINS` env var + CORS middleware fix.
3. DEPLOY-4 — `VITE_API_BASE_URL` everywhere in frontend.
4. SEC-1 — upload + download size caps.
5. BUG-5 — disconnect-safe DB save in `event_generator`.

**Phase 1 — RunPod (½ day)**
6. Create RunPod Serverless template:
   - Base: `ollama/ollama:latest` + custom entrypoint that `ollama pull medgemma-1.5-4b-it && ollama pull gemma4:e2b` on first boot, then `ollama serve`.
   - Wrap with a thin FastAPI shim exposing `/predict` (MedGemma) on port 8000 — replicates current `medgemma-host` contract — and proxies `gemma4:e2b` calls through `OLLAMA_CLOUD_URL`.
   - GPU: A4000 (16 GB) is sufficient for both models.
   - Idle timeout: 5 min. Max workers: 2.
7. Capture worker URL; set `MEDGEMMA_API_URL` and `OLLAMA_CLOUD_URL` to the RunPod endpoint(s).
8. DEPLOY-2 — keepwarm ping in `lifespan()` (default every 4 min) + 30s timeout.

**Phase 2 — External data services (1 hr)**
9. Provision Neon project; copy DSN to `DATABASE_URL`. Run `python -m database.init_db`.
10. Provision Upstash Redis; copy URL to `REDIS_URL`.
11. Provision Cloudflare R2 bucket; create API token; replace MinIO settings:
    - `MINIO_URL` → R2 S3 endpoint (`https://<account>.r2.cloudflarestorage.com`)
    - `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` → R2 credentials
    - `services/minio_service.py` works unchanged (S3-compatible). SEC-2 (1h TTL) lands here.

**Phase 3 — Hetzner backend (½ day)**
12. Provision Hetzner CPX21, Ubuntu 24.04, Falkenstein. Add SSH key only, disable password auth.
13. Install Docker + Compose. Clone repo. Render `.env` from Doppler.
14. `docker-compose.yml` services: `orchestrator`, `arangodb`, `caddy`. Use `restart: unless-stopped`.
15. Caddyfile: auto-TLS on `api.medicortex.<domain>` → `orchestrator:8001`. (Or terminate at Cloudflare and use HTTP between Cloudflare and Caddy via a Cloudflare Tunnel for origin-IP hiding.)
16. Run `python3 -m knowledge_core.build_fast_assets` once on the box; persist `knowledge_core/assets/` via a Docker volume.
17. OPS-1 — apply pool sizing.
18. OPS-3 — assert `WEB_CONCURRENCY=1` until Redis-backed `ACTIVE_STREAMS` lands.
19. OPS-5 — `slowapi` rate limits.
20. OBS-2 — `/livez` and `/readyz` split.

**Phase 4 — Frontend on GitHub Pages (1 hr)**
21. Add `.github/workflows/deploy-frontend.yml`:
    ```yaml
    on:
      push: { branches: [main], paths: ['frontend/**'] }
    jobs:
      deploy:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v4
          - uses: actions/setup-node@v4
            with: { node-version: '20' }
          - run: cd frontend && npm ci && npm run build
            env:
              VITE_API_BASE_URL: ${{ secrets.VITE_API_BASE_URL }}
          - uses: peaceiris/actions-gh-pages@v3
            with:
              github_token: ${{ secrets.GITHUB_TOKEN }}
              publish_dir: ./frontend/dist
    ```
22. Set repo secret `VITE_API_BASE_URL=https://api.medicortex.<domain>`.
23. Configure custom domain on GitHub Pages (`medicortex.<domain>`) + CNAME at registrar.

**Phase 5 — Verification (½ day)**
24. Run Test Suite 7 (T7.1–T7.9) end-to-end against the deployed stack.
25. Run an `EVAL-2` smoke pass against the live API (`pytest tests/integration/ -v`).
26. Set up Better Stack heartbeat against `/livez` (60s interval) and Grafana Cloud log shipping from Caddy + orchestrator.

**Phase 6 — Hardening (post-launch, while users use it)**
27. OPS-2 — convert nodes to `async def` once a real concurrency issue is observed.
28. OPS-3 — Redis-backed `ACTIVE_STREAMS` once `--workers 2+` is needed.
29. OPS-4, OPS-6, OPS-7, BUG-6, BUG-7 — fix during normal iteration.
30. OBS-1 — observability dashboard (Phase 7).

### Rollback Plan

- `docker compose down && git checkout <previous-tag> && docker compose up -d` on Hetzner.
- GitHub Pages: revert via `actions/deploy-pages` history (one-click).
- Neon: PITR to last good timestamp (free tier supports 24 h history).
- RunPod: redeploy previous template version (templates are versioned).

### Cost Sensitivity

| Scenario | Monthly cost |
|---|---|
| 10 users, average usage (current target) | **$13–25** |
| 10 users, idle (RunPod fully scaled to zero, no chats for 24 h) | **$8** (just Hetzner) |
| 50 users (3× volume) | **$30–60** (RunPod scales linearly; Hetzner unchanged; Neon may need Launch tier) |
| Drop RunPod, run models on the Hetzner box | infeasible — CPX21 has no GPU; would need GPU instance ($150+/mo) |

**Recommendation:** start at the budget tier (CPX21 + Neon Free + Upstash Free + R2 Free + RunPod scale-to-zero), monitor `/readyz` latency and Neon storage; upgrade individual components only when measurably saturated.

---

## Resolved

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
