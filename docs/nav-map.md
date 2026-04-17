# Navigation Map — Where to Make Changes

Use this file to find the right file(s) for any given change. Listed by UI feature or backend concern.

---

## Frontend UI Changes

| What you want to change | File(s) to edit |
|---|---|
| Overall page layout, sidebar toggle, active session state | `frontend/src/App.tsx` |
| Chat history list in sidebar | `frontend/src/components/Sidebar.tsx` → calls `GET /chats` |
| How messages are fetched or displayed | `frontend/src/components/ChatArea.tsx` → calls `GET /chats/{id}` and `POST /chat/stream` |
| SSE streaming display, smart scroll, ArrowDown button | `frontend/src/components/ChatArea.tsx` |
| Message rendering (Markdown, code blocks, thinking accordion) | `frontend/src/components/MessageBubble.tsx` |
| Streaming indicator (bouncing dots) | `frontend/src/components/MessageBubble.tsx` |
| Text input, file attachment button, microphone button | `frontend/src/components/InputArea.tsx` |
| Shared TypeScript types (Message, Session, Attachment…) | `frontend/src/types.ts` |
| Global CSS / Tailwind base styles | `frontend/src/index.css` |

---

## Backend API Endpoints

| Endpoint | File | Notes |
|---|---|---|
| `POST /chat` | `orchestrator.py` | Sync chat |
| `POST /chat/stream` | `orchestrator.py` | SSE streaming chat |
| `GET /chats` | `services/chat_service.py` | List sessions |
| `GET /chats/{id}` | `services/chat_service.py` | Fetch session messages |
| `POST /upload` | `orchestrator.py` | File upload → MinIO |
| `GET /.well-known/agent-cards` | `orchestrator.py` | Agent card manifest |

---

## LangGraph Pipeline Nodes

| Node | Where to edit | What it does |
|---|---|---|
| `node_analyze_privacy` | `orchestrator.py` | Presidio PII redaction |
| `node_retrieve_knowledge` | `orchestrator.py` | ArangoDB knowledge graph query |
| `node_router` | `orchestrator.py` | Agent selection (`route_decision`) |
| Agent nodes (pubmed/diagnosis/etc.) | `specialized_agents/*.py` + `agents.py` | Parallel agent execution |
| `node_aggregator` | `orchestrator.py` | GPT-4o-mini Markdown formatting |
| `node_reviewer` | `orchestrator.py` | Groq judge scoring + disclaimer |
| `node_restore_privacy` | `orchestrator.py` | Replace `<PERSON_N>` with real names |

---

## Agent Logic

| What to change | File(s) |
|---|---|
| PubMed search behavior | `specialized_agents/pubmed_agent.py`, `tools/pubmed_search_tools.py` |
| Diagnosis / symptom analysis | `specialized_agents/diagnosis_agent.py`, `tools/symptom_analysis_tools.py` |
| Report / image / PDF analysis | `specialized_agents/report_agent.py`, `tools/document_extraction_tools.py` |
| Patient record lookup | `specialized_agents/patient_agent.py`, `tools/patient_retriever_tools.py` |
| Drug interactions / recommendations | `specialized_agents/drug_agent.py`, `tools/drug_interaction_tools.py` |
| ReAct loop, idempotency, tool context injection | `specialized_agents/base.py` |
| A2A typed protocol (Envelope, AgentResponse, AgentCard) | `specialized_agents/protocols.py` |
| Agent registry (add/remove agents) | `agents.py` |

---

## Data / Storage

| What to change | File(s) |
|---|---|
| PostgreSQL schema | `database/schema.sql` (then `python -m database.init_db`) |
| DB access (sessions, messages) | `services/chat_service.py` |
| Patient DB migration / seed | `tools/migrate_db.py` |
| MinIO file upload/download | `services/minio_service.py` |
| Redis / in-memory caching | `utils/cache_utils.py` |

---

## LLM / Model Config

| What to change | File(s) |
|---|---|
| MedGemma URL or fallback behavior | `specialized_agents/medgemma_llm.py`, `config.py` (`MEDGEMMA_API_URL`) |
| Judge model, sample rate, token limit | `config.py` (`JUDGE_*`) |
| Router / aggregator model | `orchestrator.py` (look for `ChatOpenAI` instantiation in `lifespan`) |

---

## HIPAA / Privacy

| What to change | File(s) |
|---|---|
| PII entity types redacted | `orchestrator.py` → `node_analyze_privacy` (uses `PrivacyManager`) |
| Privacy manager logic | `services/privacy_service.py` (or similar; check import in `orchestrator.py`) |
| Tool context injection for new PII-needing tools | `specialized_agents/base.py` → `process()` method |

---

## MCP Server

| What to change | File(s) |
|---|---|
| Tool definitions, resources, prompts | `tools/mcp_server.py` |

---

## Config / Environment

| What to change | File(s) |
|---|---|
| Add/change env variables | `.env` + `config.py` |
| Infrastructure resilience settings | `config.py` |
