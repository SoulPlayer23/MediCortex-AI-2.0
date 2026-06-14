# MediCortex AI 2.0

A self-hosted medical reasoning system that orchestrates specialized AI agents to assist healthcare professionals with clinical queries, differential diagnosis, drug interactions, patient record analysis, and medical imaging interpretation.

---

## Overview

MediCortex AI 2.0 is a privacy-first, HIPAA-aware backend that routes clinical questions through a pipeline of purpose-built agents. Each agent specializes in a specific domain — PubMed research, diagnosis, pharmacology, patient records, or medical imaging — and collaborates via a typed Agent-to-Agent (A2A) protocol. Patient identifiers are redacted at the edge using Microsoft Presidio and never exposed to any external LLM.

The system is fully self-hosted. All inference runs on local or homeserver hardware: Gemma 4 via Ollama for routing and aggregation, MedGemma for clinical synthesis, and Groq's Llama 3.3 70B as a model-as-judge reviewer. There is no dependency on OpenAI, Azure, or any paid cloud AI endpoint.

---

## Architecture

Requests enter the system as a LangGraph pipeline:

```
node_analyze_privacy
    → node_retrieve_knowledge
    → node_router
    → [pubmed | diagnosis | report_analyzer | patient | pharmacology]  (up to 3 parallel)
    → node_aggregator
    → node_reviewer
    → node_restore_privacy
    → END
```

Every stage is implemented as a node in `orchestrator.py`. Heavy singletons (graph, privacy manager, LLM clients) are initialized once in `lifespan()` to avoid triple-init under Uvicorn reload.

### Privacy / HIPAA

Presidio redacts 8 entity types on entry: `PERSON`, `PHONE_NUMBER`, `EMAIL_ADDRESS`, `DATE_TIME`, `LOCATION`, `US_SSN`, `URL`, `IP_ADDRESS`. These become `<PERSON_1>`, `<DATE_TIME_2>`, etc. The real-name mapping travels only inside `Envelope.payload` — never in any LLM prompt. Names are restored exactly once in `node_restore_privacy`.

For history re-injection, `redact_identifying_pii()` is used (not `redact_pii()`): it strips only identity fields, not `DATE_TIME` or `LOCATION`, which carry clinical meaning.

### LLM Stack

| Role | Model | Backend |
|---|---|---|
| Router / Aggregator / Extractor / Agent Planner | Gemma 4 31B | Homeserver Ollama (`OLLAMA_CLOUD_URL`) |
| Agent synthesis (primary) | MedGemma | `localhost:8000/predict` (RunPod or local GPU) |
| Agent synthesis (fallback) | Gemma 4 31B | Homeserver Ollama — used when MedGemma is offline or OOM |
| Model-as-Judge | Llama 3.3 70B Versatile | Groq API (evaluation only, never generation) |

`OLLAMA_FLASH_ATTENTION=0` must be set on the Ollama host — the 31B Dense model hangs on prompts over 3–4K tokens without it (upstream Ollama issue [#15350](https://github.com/ollama/ollama/issues/15350)).

---

## Agents

All agents extend `A2ABaseAgent` (`specialized_agents/base.py`) which implements the ReAct loop, 24-hour idempotency caching (Redis with in-memory fallback), and HIPAA-safe PII injection via `inspect.signature`.

| Registry Key | Specialization | Tools |
|---|---|---|
| `pubmed` | Research and literature | PubMed E-utilities, trusted medical web crawler |
| `diagnosis` | Differential diagnosis | Symptom analysis, diagnosis web crawler (UpToDate, Merck) |
| `report_analyzer` | PDF reports and medical imaging | PDF extraction, MedGemma vision (X-ray, MRI, CT, pathology) |
| `patient` | Patient record analysis | Secure record retrieval, vitals assessment, medication safety review, diagnosis patterns |
| `pharmacology` | Drug interactions and dosing | Drug interaction checker (severity graded), evidence-based recommendations |

Registry key, `A2ABaseAgent.name`, and `AgentCard.name` must all be identical — this is enforced by convention and checked at startup.

Web search uses DuckDuckGo with site-scoped queries against trusted domains. Google is not used (bot-detected).

---

## Data Layer

- **PostgreSQL** (asyncpg + SQLAlchemy): `chat_sessions`, `chat_messages`, `patients`. Agent ReAct steps are stored in a `thinking JSONB` column. Judge scores and model metadata in `message_metadata JSONB`.
- **MinIO**: Object storage for uploaded PDFs and medical images.
- **Redis**: Tool result cache (24-hour TTL) and idempotency key store. Degrades to in-memory if unavailable.
- **ArangoDB**: Medical knowledge graph queried during `node_retrieve_knowledge`. Accessed over Tailscale from homeserver. 10-second timeout; system completes with empty context if unreachable.
- **Patient data**: Seeded with 14,803 synthetic patients from Synthea CSV datasets (Apache 2.0 license). No real patient data is included in the repository.

---

## MCP Server

`tools/mcp_server.py` exposes the system as a Model Context Protocol server (stdio transport), allowing Claude Desktop or any MCP-compatible client to use MediCortex tools directly.

- 13 tools covering all agent capabilities
- Agent card resources at `agents://medicortex/{name}/card`
- 3 workflow prompts: `patient-full-review`, `drug-safety-check`, `medical-report-analysis`

---

## Frontend

React 19 + Vite SPA in `frontend/`. Connects to the backend over SSE for streaming responses and agent thinking. Key components:

| Component | File |
|---|---|
| Root layout, session state | `App.tsx` |
| Chat history sidebar | `Sidebar.tsx` |
| Message display, SSE streaming | `ChatArea.tsx` |
| Message rendering, thinking accordion | `MessageBubble.tsx` |
| Text / file / microphone input | `InputArea.tsx` |

---

## Deployment

The system runs as two Docker containers (orchestrator + Caddy reverse proxy) on a homeserver, exposed externally via Tailscale Funnel. See `docker-compose.prod.yml` and `Caddyfile`.

Infrastructure requirements for a self-hosted deployment:

- **Ollama server** with `gemma4:31b` or `gemma4:e2b` (4GB VRAM minimum for E2B; 31B requires more)
- **MedGemma** inference endpoint (RunPod serverless or a local GPU with CUDA)
- **PostgreSQL**, **MinIO**, **Redis** — all runnable via Docker
- **ArangoDB** for the knowledge graph (optional; system degrades gracefully without it)
- **Groq API key** for the model-as-judge reviewer (optional; reviewer fails open if absent)
- **Tailscale** for secure inter-service connectivity if backend services are on a separate host

All service addresses are configured in `.env` via `config.py` (Pydantic `BaseSettings`). No secrets are hardcoded.

---

## Running Locally

### Backend

```bash
# Install dependencies (use the repo .venv)
pip install -r requirements.txt

# Initialize the database schema
python -m database.init_db

# Build knowledge graph assets
python3 -m knowledge_core.build_fast_assets

# Start the API server (port 8001)
python orchestrator.py

# Optional: start the MCP server (stdio)
python tools/mcp_server.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev    # http://localhost:5173
```

### Environment

Minimum `.env` for local development:

```env
OLLAMA_CLOUD_URL=http://<your-ollama-host>:11434/v1
OLLAMA_CLOUD_API_KEY=ollama
OLLAMA_CLOUD_MODEL=gemma4:e2b

DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/medicortex
MINIO_URL=http://localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
REDIS_URL=redis://localhost:6379/0

# Optional
GROQ_API_KEY=your_groq_key_here
MEDGEMMA_API_URL=http://localhost:8000/predict
ARANGODB_HOST=your-arangodb-host
```

### Tests

```bash
source .venv/bin/activate

# All non-stress tests (run serially — parallel workers crash WSL)
.venv/bin/python3 -m pytest tests/unit/ tests/integration/ -v --tb=short -m "not stress"

# Single file
.venv/bin/python3 -m pytest tests/integration/test_reviewer_node.py -v --tb=short
```

---

## Try It

MediCortex AI 2.0 is self-hosted and not publicly accessible by default — the inference infrastructure (Gemma 4, MedGemma, ArangoDB knowledge graph) runs on private hardware via Tailscale.

If you want access to a live demo or want to try the system, contact me at **venkiteshsanand1920@gmail.com** and I can grant you access.

---

## License

This repository does not currently include a license file. Contact the author before using or adapting the code.
