# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Commands

### Backend
```bash
# Install dependencies
pip install -r requirements.txt

# Initialize database schema (drops & recreates tables)
python -m database.init_db

# Build knowledge graph assets
python3 -m knowledge_core.build_fast_assets

# Start the orchestrator API (port 8001)
python orchestrator.py

# Start the MCP server (stdio transport)
python tools/mcp_server.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev       # http://localhost:5173
npm run build
npm run lint
```

### Testing
```bash
# Run all tests
pytest

# Run a single test file
pytest tests/integration/test_reviewer_node.py

# Run a specific test function
pytest tests/integration/test_reviewer_node.py::test_reviewer_low_score

# Health check (requires running services)
python tests/health_check.py
```

---

## Architecture

### Request Flow (LangGraph Pipeline)

Every `/chat/stream` or `/chat` request traverses this LangGraph graph in `orchestrator.py`:

```
node_analyze_privacy      (Presidio redacts 8 PII entity types → placeholders; file_urls extracted from attachments)
    → node_retrieve_knowledge   (GPT-4o-mini extracts entity, queries knowledge graph; gracefully skipped if ArangoDB offline)
    → node_router               (GPT-4o-mini selects agents; route_decision always adds report_analyzer if file_urls present)
    → [pubmed | diagnosis | report_analyzer | patient | pharmacology]   (parallel or single, ≤3)
    → node_aggregator           (GPT-4o-mini formats Markdown)
    → node_reviewer             (Groq llama-3.3-70b-versatile scores 1–5; appends disclaimer if < 3)
    → node_restore_privacy      (replaces <PERSON_1> placeholders with real names)
    → END
```

**HIPAA Privacy**: Presidio redacts 8 entity types (`PERSON`, `PHONE_NUMBER`, `EMAIL_ADDRESS`, `DATE_TIME`, `LOCATION`, `US_SSN`, `URL`, `IP_ADDRESS`) at entry. Placeholders are **never** sent to external LLMs. `pii_mapping_json` travels only inside `Envelope.payload` — never injected into the LLM prompt text — preventing real name leakage through the GPT fallback. `node_restore_privacy` is the only place real names are restored.

**File Inputs**: `/upload` stores files to MinIO and returns a presigned URL. The frontend sends these as a structured `attachments: [{url, filename, type}]` field in `ChatRequest`. The orchestrator extracts `file_urls` into `AgentState` and injects them into the `report_analyzer` agent's input as `Files to analyze:\n<urls>`. `route_decision` automatically includes `report_analyzer` whenever `file_urls` is non-empty.

**SSE Streaming**: The `/chat/stream` endpoint uses a global `ACTIVE_STREAMS` dict keyed by `session_id`. Agents append to a shared `live_thoughts` list during the ReAct loop; the endpoint polls and yields `thought` events while the LangGraph task runs in the background.

**Knowledge Core**: `node_retrieve_knowledge` queries ArangoDB (on homeserver via Tailscale VPN) through `MedicalReasoningEngine`. `_aql()` has a 10s timeout; asset load failures are caught at init. `medical_engine` is set to `None` if unavailable, producing empty context without crashing the request.

### A2A Protocol

All inter-agent communication is typed via Pydantic in `specialized_agents/protocols.py`:
- **`Envelope`**: wraps every request with `trace_id`, `idempotency_key`, `sender_id`, `receiver_id`, `payload`.
- **`AgentResponse`**: wraps output with `thinking` (list of ReAct steps), optional `error`, and `usage`.
- **`AgentCard`**: metadata manifest published at `GET /.well-known/agent-cards`.

The `A2ABaseAgent` (`specialized_agents/base.py`) implements the ReAct loop (`_execute_rect_loop`), idempotency caching (Redis → in-memory fallback, 24h TTL), thought emission, and transparent tool context injection via `tool_context`.

**Tool Context Injection**: `_execute_rect_loop` accepts `tool_context: Dict[str, Any]`. When invoking a tool, the base agent uses `inspect.signature` to discover if the tool accepts any `tool_context` keys (e.g. `pii_mapping_json`). Matching keys are injected as named arguments without the LLM ever seeing them. This is the mechanism that makes HIPAA-safe patient record lookups work — the LLM only passes `<PERSON_1>`, and `pii_mapping_json` is silently appended at call time.

### Specialized Agents

Each agent in `specialized_agents/` extends `A2ABaseAgent` and is registered in `AGENT_REGISTRY` (`agents.py`). The registry key, `A2ABaseAgent` name, and `AgentCard.name` **must all match**:

| Registry Key | Agent File | Tools Used |
|---|---|---|
| `pubmed` | `pubmed_agent.py` | `pubmed_search_tools`, `medical_webcrawler_tools` |
| `diagnosis` | `diagnosis_agent.py` | `symptom_analysis_tools`, `diagnosis_webcrawler_tools` |
| `report_analyzer` | `report_agent.py` | `document_extraction_tools`, `image_extraction_tools`, `report_analysis_tools` |
| `patient` | `patient_agent.py` | `patient_retriever_tools`, `patient_history_analyzer_tools`, `patient_vitals_tools`, `patient_medication_review_tools` |
| `pharmacology` | `drug_agent.py` | `drug_interaction_tools`, `drug_recommendation_tools` |

### LLM Stack

- **Router / Aggregator / Knowledge Refinement**: `gpt-4o-mini` via `langchain-openai`
- **Specialized Agents (default)**: MedGemma hosted locally at `http://100.107.2.102:8000/predict` (homeserver via Tailscale VPN — may be offline). URL is configured via `MEDGEMMA_API_URL` in `config.py`. Falls back to `gpt-4o-mini` automatically on connection failure. See `specialized_agents/medgemma_llm.py`.
- **Model-as-Judge**: Groq `llama-3.3-70b-versatile` (fallback: `llama-3.1-8b-instant`). Controlled by `JUDGE_ENABLED`, `JUDGE_SAMPLE_RATE`, `JUDGE_MAX_INPUT_TOKENS` in `config.py`.

### Tool Caching

External API tools (PubMed, web crawlers) use the `@redis_cache` decorator from `utils/cache_utils.py` (24h TTL, Redis-backed, gracefully skipped if Redis is unavailable).

### Data Layer

- **PostgreSQL** (async via `asyncpg` + SQLAlchemy): `chat_sessions` + `chat_messages`. The `chat_messages` table has `thinking JSONB` (agent ReAct steps) and `message_metadata JSONB` (judge score, model used). Schema source of truth is `database/schema.sql`. A `patients` table is planned (see `Todo.md`) to replace the current simulated in-memory store in `tools/patient_retriever_tools.py`.
- **MinIO**: Object storage for uploaded PDFs/images. `MINIO_URL` must be a full URL (e.g. `http://localhost:9000`). Accessed via `services/minio_service.py` which reads all config from `settings`.
- **Schema note**: The Pydantic schema alias `message_metadata` avoids collision with SQLAlchemy's internal `MetaData` registry.

### MCP Server

`tools/mcp_server.py` exposes 13 tools, agent card **Resources** (URI scheme `agents://medicortex/{name}/card`), and 3 workflow **Prompts** (`patient-full-review`, `drug-safety-check`, `medical-report-analysis`) via STDIO transport.

---

## Key Conventions

### A2A Standards (from `skills/A2A.md`)
- Every agent must publish an `AgentCard`. Registry key, agent `name`, and `AgentCard.name` must all be identical.
- All handoffs use `Envelope` / `AgentResponse` Pydantic types.
- `MAX_CONCURRENT_AGENTS=3` caps parallel agent calls per request. Individual agent `max_iterations` acts as the per-agent ReAct circuit breaker.
- `trace_id` must propagate through all graph nodes and agent calls.
- Side-effect tools check `idempotency_key` before executing.

### HIPAA Rules
- PII is **never** injected into `enhanced_input` (the text sent to any LLM). It travels only inside `Envelope.payload["pii_mapping_json"]`.
- `tool_context` in `_execute_rect_loop` is the only approved mechanism to pass sensitive data to tools without exposing it to the model.
- When adding a new tool that requires PII resolution, add its parameter name to the `tool_context` dict in `base.py:process()`.
- The patient agent system prompt must instruct the LLM to pass **only** the redacted placeholder to `retrieve_patient_records` — never to extract or forward `pii_mapping_json` itself. `pii_mapping_json` is injected silently at call time via `tool_context`.

### MCP Standards (from `skills/MCP.md`)
- Tool `description` fields are prompt instructions — keep them precise and instructional.
- Read-only data access uses MCP Resources, not Tools.
- All tool inputs are validated against JSON Schema before execution.
- Errors are returned as structured text (never crash the server).

### Config
All settings are loaded from `.env` via `config.py` (Pydantic `BaseSettings`). Required: `OPENAI_API_KEY`. Optional: `GROQ_API_KEY` (judge), `REDIS_URL`, `DATABASE_URL`, `MINIO_*`. A `.env` template is committed at the repo root.

### Infrastructure Resilience
All external services degrade gracefully when offline:
- **Redis**: Falls back to in-memory cache (idempotency + tool caching both have fallbacks).
- **ArangoDB / Knowledge Core** (homeserver via Tailscale): `_aql()` has a 10s timeout; missing asset files are caught at init. Knowledge context is empty but the request completes normally.
- **MedGemma** (homeserver via Tailscale): Automatically falls back to `gpt-4o-mini` on connection failure.

### Test Structure
Tests live in `tests/` with subdirectories: `agents/`, `integration/`, `mcp/`, `tools/`, `unit/`. `pytest.ini` sets `asyncio_mode = auto`. Mocking strategy for agents: mock `Envelope` payloads and assert `AgentResponse` output fields.
