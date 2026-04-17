# Data Layer — MediCortex AI 2.0

## PostgreSQL

Async via `asyncpg` + SQLAlchemy. Tables: `chat_sessions`, `chat_messages`, `patients`.

- `chat_messages` has `thinking JSONB` (agent ReAct steps) and `message_metadata JSONB` (judge score, model used)
- `patients` stores demographics, diagnoses, medications, allergies, vitals history as JSONB columns
- Seeded with 14,803 synthetic patients from Synthea CSV datasets (APR2020, NOV2021, COVID19; Apache 2.0)
- Schema source of truth: `database/schema.sql`
- Re-seed anytime: `python -m tools.migrate_db` (idempotent upsert)
- Patient lookup in `tools/patient_retriever_tools.py` uses `asyncpg` with a per-call event loop (safe from LangGraph's sync thread-pool nodes)

**Schema note**: Pydantic alias `message_metadata` avoids collision with SQLAlchemy's internal `MetaData` registry.

## MinIO

Object storage for uploaded PDFs/images. `MINIO_URL` must be a full URL (e.g. `http://localhost:9000`). Accessed via `services/minio_service.py` which reads all config from `settings`.

## MCP Server

`tools/mcp_server.py` exposes:
- 13 tools
- Agent card **Resources** (URI scheme `agents://medicortex/{name}/card`)
- 3 workflow **Prompts**: `patient-full-review`, `drug-safety-check`, `medical-report-analysis`
- Transport: STDIO

## Config

All settings loaded from `.env` via `config.py` (Pydantic `BaseSettings`).

| Variable | Required | Notes |
|---|---|---|
| `OLLAMA_CLOUD_URL` | Yes | Ollama base URL, e.g. `http://homeserver:11434/v1` |
| `OLLAMA_CLOUD_MODEL` | Yes | Model name, e.g. `gemma4:e2b` |
| `OLLAMA_CLOUD_API_KEY` | No | Set to `ollama` (required by OpenAI-compat shim) |
| `GROQ_API_KEY` | No | Model-as-judge |
| `REDIS_URL` | No | Falls back to in-memory |
| `DATABASE_URL` | No | PostgreSQL |
| `MINIO_*` | No | Object storage |
| `MEDGEMMA_API_URL` | No | defaults to `http://localhost:8000/predict` |
| `ARANGODB_HOST` | No | Tailscale hostname |
| `ARANGODB_USERNAME` | No | |
| `ARANGODB_PASSWORD` | No | |
| `ARANGODB_DB_NAME` | No | |

All service hostnames use `homeserver` (Tailscale) except `MEDGEMMA_API_URL` which uses `localhost`. `ARANGO_URL` in `knowledge_core/medical_engine.py` is built from `settings.ARANGODB_HOST` — do not hardcode it.

## Infrastructure Resilience

| Service | Fallback |
|---|---|
| Redis | In-memory cache (idempotency + tool caching) |
| ArangoDB (homeserver via Tailscale) | Empty knowledge context; `_aql()` has 10s timeout |
| MedGemma (localhost) | Falls back to `gemma4:e2b` via homeserver Ollama |

## MCP Standards

- Tool `description` fields are prompt instructions — keep them precise and instructional.
- Read-only data access uses MCP Resources, not Tools.
- All tool inputs validated against JSON Schema before execution.
- Errors returned as structured text (never crash the server).
