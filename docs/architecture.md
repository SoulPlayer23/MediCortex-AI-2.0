# Architecture — MediCortex AI 2.0

## Request Flow (LangGraph Pipeline)

Every `/chat/stream` or `/chat` request traverses this graph in `orchestrator.py`:

```
node_analyze_privacy      (Presidio redacts 8 PII entity types → placeholders; file_urls extracted)
    → node_retrieve_knowledge   (Gemma 4 E2B/31B extracts entities, queries knowledge graph; skipped if ArangoDB offline)
    → node_router               (Gemma 4 selects agents; always adds report_analyzer if file_urls present)
    → [pubmed | diagnosis | report_analyzer | patient | pharmacology]   (parallel or single, ≤3)
    → node_aggregator           (Gemma 4 formats Markdown)
    → node_reviewer             (Groq llama-3.3-70b-versatile scores 1–5; appends disclaimer if < 3)
    → node_restore_privacy      (replaces <PERSON_1> placeholders with real names)
    → END
```

## HIPAA Privacy

- Presidio redacts 8 entity types: `PERSON`, `PHONE_NUMBER`, `EMAIL_ADDRESS`, `DATE_TIME`, `LOCATION`, `US_SSN`, `URL`, `IP_ADDRESS`
- Placeholders are **never** sent to external LLMs
- `pii_mapping_json` travels only inside `Envelope.payload` — never injected into LLM prompt text
- `node_restore_privacy` is the only place real names are restored

**History Re-injection**: `redact_identifying_pii()` (not `redact_pii()`) strips only `PERSON`, `PHONE_NUMBER`, `EMAIL_ADDRESS`, `US_SSN` — **not** `DATE_TIME`/`LOCATION` (clinically meaningful). Full `redact_pii()` would leave unrestorable `<DATE_TIME_N>` placeholders in the final output.

## File Inputs

`/upload` stores files to MinIO and returns a presigned URL. Frontend sends `attachments: [{url, filename, type}]` in `ChatRequest`. Orchestrator extracts `file_urls` → injected into `report_analyzer` as `Files to analyze:\n<urls>`. `route_decision` automatically includes `report_analyzer` when `file_urls` is non-empty.

## SSE Streaming

`/chat/stream` uses a global `ACTIVE_STREAMS` dict keyed by `session_id`. Agents append to `live_thoughts` during the ReAct loop; the endpoint polls and yields `thought` events while the LangGraph task runs in the background.

## Singleton Initialization (lifespan pattern)

All heavy singletons (`MedicalReasoningEngine`, `PrivacyManager`, `ChatOpenAI`, `orchestrator_graph`) are created in `lifespan()` in `orchestrator.py` — **not** at module level. Module-level declarations are `None` placeholders assigned via `global`. This prevents triple-initialization with Uvicorn `reload=True`.

## Multi-Turn Routing Context

`node_router` receives a `routing_context` string (compact redacted summary of the last 3 turns) built by `_build_routing_context()` in `orchestrator.py`. `agents_used` is persisted in `message_metadata JSONB` and read back next turn.

**Routing rules:**
- Symptoms only → `['diagnosis']`
- Treatment options / medications for a disease → `['diagnosis', 'pharmacology']`
- Symptoms AND treatment → `['diagnosis', 'pharmacology']`
- Named drug (interaction/dosage/alternatives) → `['pharmacology']`
- Research/literature → `['pubmed']`
- Document/image attached → always includes `['report_analyzer']`

## LLM Stack

| Role | Model | Backend | Notes |
|---|---|---|---|
| Router / Aggregator / Extractor / Agent-Planner | Gemma 4 31B | Ollama Cloud (`OLLAMA_CLOUD_URL`) | Default. Warmup call fires at startup. |
| Router / Aggregator / Extractor / Agent-Planner | Gemma 4 E2B | Local Ollama (`http://<host>:11434/v1`) | Alternative for low-latency / offline use. Set `OLLAMA_CLOUD_URL` + `OLLAMA_CLOUD_MODEL=gemma4:e2b`. Requires `OLLAMA_FLASH_ATTENTION=0` on host. |
| Agent Synthesis (Phase 2) | MedGemma | `localhost:8000/predict` | Primary. Falls back to Gemma 4 Ollama Cloud if CUDA OOM or offline. |
| Judge / Reviewer | Groq llama-3.3-70b-versatile | Groq API | Evaluation only — never used for content generation. |

### Gemma 4 E2B — Local Setup (Ubuntu + GTX 1650 Super 4GB)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull E2B model (~7.2GB, ~2GB VRAM at Q4)
ollama pull gemma4:e2b

# Start server with Flash Attention disabled (prevents hang on prompts >3K tokens)
OLLAMA_FLASH_ATTENTION=0 OLLAMA_HOST=0.0.0.0 ollama serve

# Persistent systemd service
sudo systemctl edit ollama
# Add under [Service]:
#   Environment="OLLAMA_FLASH_ATTENTION=0"
#   Environment="OLLAMA_HOST=0.0.0.0"
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

Then in `.env`:
```env
OLLAMA_CLOUD_URL=http://<ubuntu-host-ip>:11434/v1
OLLAMA_CLOUD_API_KEY=ollama          # any non-empty string
OLLAMA_CLOUD_MODEL=gemma4:e2b
```

**Why E2B for local?** The 31B Dense model has a confirmed Flash Attention hang on prompts >3-4K tokens (GitHub [#15350](https://github.com/ollama/ollama/issues/15350)). E2B (2.3B active params, MoE-style) doesn't have this issue and fits entirely in 4GB VRAM. For high-quality synthesis, keep MedGemma as primary.

## Knowledge Core

`node_retrieve_knowledge` queries ArangoDB (homeserver via Tailscale VPN) through `MedicalReasoningEngine`. `_aql()` has a 10s timeout. `medical_engine` is set to `None` if unavailable — request completes with empty context.
