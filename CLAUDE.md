# CLAUDE.md

> **Navigation Map** — before making any change, check [`docs/nav-map.md`](docs/nav-map.md) first. It maps every UI page and backend service to the exact file(s) to edit.

---

## Commands

### Backend
```bash
pip install -r requirements.txt
python -m database.init_db          # drops & recreates tables
python3 -m knowledge_core.build_fast_assets
python orchestrator.py              # API on port 8001
python tools/mcp_server.py          # MCP (stdio)
```

### Frontend
```bash
cd frontend
npm install
npm run dev    # http://localhost:5173
npm run build
npm run lint
```

> **Vite HMR limitation**: File changes by Claude Code on Windows `D:` drive do **not** trigger chokidar. After any frontend change, restart the dev server manually (`Ctrl-C` then `npm run dev`).

### Testing

> **CRITICAL — always use the repo `.venv` and run tests serially (one at a time).** Parallel test workers crash WSL.

```bash
# Always activate .venv first
source .venv/bin/activate

# Run all non-stress tests (serial, no xdist)
.venv/bin/python3 -m pytest tests/unit/ tests/integration/ -v --tb=short -m "not stress"

# Run a single test file
.venv/bin/python3 -m pytest tests/integration/test_reviewer_node.py -v --tb=short

# Run a single test
.venv/bin/python3 -m pytest tests/integration/test_reviewer_node.py::test_reviewer_low_score -v

# Stress tests (run individually, not as a batch)
.venv/bin/python3 -m pytest tests/stress/test_failure_injection.py -v --tb=short -m stress
.venv/bin/python3 -m pytest tests/stress/test_adversarial.py -v --tb=short -m stress

# Health check (requires running services)
.venv/bin/python3 tests/health_check.py
```

> **Never run** `pytest -n auto` or `pytest -n 4` — xdist is disabled in pytest.ini via `-p no:xdist`. Do not override it.

---

## Architecture Overview

Full details in [`docs/architecture.md`](docs/architecture.md).

```
node_analyze_privacy → node_retrieve_knowledge → node_router
    → [pubmed | diagnosis | report_analyzer | patient | pharmacology]  (≤3 parallel)
    → node_aggregator → node_reviewer → node_restore_privacy → END
```

All nodes live in `orchestrator.py`. Singletons initialized in `lifespan()` — never at module level.

**HIPAA**: PII is redacted by Presidio at entry into `<PERSON_N>` placeholders. Real names only restored in `node_restore_privacy`. `pii_mapping_json` travels only in `Envelope.payload`, never in LLM prompt text. When re-injecting history use `redact_identifying_pii()` not `redact_pii()` (preserves `DATE_TIME`/`LOCATION`).

---

## Agents

Full details in [`docs/agents.md`](docs/agents.md).

| Registry Key | Agent File |
|---|---|
| `pubmed` | `pubmed_agent.py` |
| `diagnosis` | `diagnosis_agent.py` |
| `report_analyzer` | `report_agent.py` |
| `patient` | `patient_agent.py` |
| `pharmacology` | `drug_agent.py` |

Registry key, `A2ABaseAgent.name`, and `AgentCard.name` **must all match**.

All agents extend `A2ABaseAgent` (`specialized_agents/base.py`) which implements the ReAct loop, idempotency (Redis → in-memory fallback), and tool context injection (HIPAA-safe PII passing via `inspect.signature`).

**LLM Stack**:
- **Router / Aggregator / Extractor / Agent-Planner** → `gemma4:31b-cloud` via homeserver Ollama (`http://homeserver:11434/v1`). A warmup call fires at `lifespan()` startup to pre-load the model and avoid cold-start hangs. Set `OLLAMA_FLASH_ATTENTION=0` on the Ollama host to prevent Flash Attention hangs on long prompts.
- **Agents (synthesis)** → MedGemma (`localhost:8000`, fallback `gemma4:31b-cloud` via homeserver Ollama).
- **Judge** → Groq `llama-3.3-70b-versatile` (evaluation only, not used for generation).

Web crawlers use **DuckDuckGo** (not Google — bot-detected). Tool results cached via `@redis_cache` (24h TTL).

---

## Data Layer

Full details in [`docs/data-layer.md`](docs/data-layer.md).

- **PostgreSQL** (asyncpg + SQLAlchemy): `chat_sessions`, `chat_messages`, `patients`. Schema: `database/schema.sql`.
- **MinIO**: file uploads. `MINIO_URL` must be a full URL (e.g. `http://localhost:9000`).
- **Config**: `.env` via `config.py` (Pydantic `BaseSettings`). All services degrade gracefully. `OPENAI_API_KEY` is no longer required — Gemma 4 handles all generation.

---

## Frontend

Full details in [`docs/frontend.md`](docs/frontend.md).

React 19 + Vite + Tailwind in `frontend/`. Key files:

| Component | File |
|---|---|
| Root layout + session state | `App.tsx` |
| Chat history sidebar | `Sidebar.tsx` |
| Message display + SSE streaming | `ChatArea.tsx` |
| Message rendering + thinking accordion | `MessageBubble.tsx` |
| Text/file/mic input | `InputArea.tsx` |

---

## Key Conventions

- **Registry match**: `agents.py` key = `A2ABaseAgent.name` = `AgentCard.name`
- **PII rule**: never put real names in `enhanced_input`. Use `tool_context` for PII-needing tools.
- **History**: use `redact_identifying_pii()` not `redact_pii()` when injecting DB history into prompts
- **New PII tool**: add its parameter name to `tool_context` dict in `base.py:process()`
- **MCP tools**: `description` fields are LLM-facing instructions — keep them precise
- **`ARANGO_URL`**: built from `settings.ARANGODB_HOST` in `knowledge_core/medical_engine.py` — never hardcode
- **Tests**: `tests/` with subdirs `agents/`, `integration/`, `mcp/`, `tools/`, `unit/`. `asyncio_mode = auto`. Mock `Envelope` payloads, assert `AgentResponse` fields.
