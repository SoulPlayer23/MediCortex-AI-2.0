# MediCortex AI 2.0

MediCortex is an advanced medical reasoning system designed to assist healthcare professionals by orchestrating specialized AI agents. It leverages a privacy-first architecture, HIPAA compliance, and a deterministic Agent-to-Agent (A2A) protocol.

---

## 🏗️ System Architecture

The system is built on a **Centralized Orchestration Architecture** with strict protocols:

### 1. Orchestration Layer (`orchestrator.py`)
-   **FastAPI & Structlog**: High-performance API with structured JSON logging (`config.py` managed).
-   **A2A Protocol**: Communicates with agents using structured `Envelope`s and `AgentCard`s (`specialized_agents/protocols.py`).
-   **Trace ID Propagation**: End-to-end `trace_id` bound to `structlog` context and passed through every `Envelope` (A2A §5.1).
-   **Agent Card Discovery**: `GET /.well-known/agent-cards` exposes all registered agent cards for A2A discovery.
-   **Privacy Manager**: Uses Microsoft Presidio to redact PII (PHI) before routing to agents. Real identifiers are restored only at the final `node_restore_privacy` step — never exposed to external LLMs (GPT).
-   **Router**: Intelligently routes user queries to specialized agents (capped at 3 concurrent agents per request via A2A §4.1 circuit breaker).
- **SSE Streaming**: `/chat/stream` endpoint streams agent thoughts and response tokens in real-time. Uses a global `ACTIVE_STREAMS` registry keyed by `session_id` to bypass LangGraph state deep-copy isolation, ensuring thoughts emitted from deep within the ReAct loop reach the client immediately.
- **Pydantic Aliasing**: Uses `message_metadata` alias in schemas to prevent collision with SQLAlchemy's internal `MetaData` registry when retrieving chat history.
- **Model-as-Judge Evaluator**: A `node_reviewer` (A2A §5.2) powered by Groq (`llama-3.3-70b-versatile`) that scores responses 1-5 to prevent hallucinations/danger.
- **Fail-Open Logic**: If evaluation fails (e.g., rate limits), the node fails open to ensure system availability.
- **Cost Optimization**: Truncates response to 500 tokens before evaluation to respect Groq free tier TPD (Tokens Per Day) limits.
- **Disclaimers**: Appends clinical disclaimers for low-quality responses (< 3/5), including the judge's score and specific rationale.
- **SSE Streaming**: `/chat/stream` endpoint streams agent thoughts and response tokens to the frontend in real-time via Server-Sent Events.

### 2. Specialized Agents (`specialized_agents/`)
All agents adhere to the **A2A Protocol**, taking an `Envelope` input and returning an `AgentResponse`. 
-   **Idempotency Cache**: Built-in 24-hour response caching (Redis-backed, memory fallback) using the Envelope's `idempotency_key` (A2A §4.2).
-   **LLM Fallback**: MedGemma-powered agents fall back to OpenAI **GPT-4o-mini** if the local inference server times out (60s) or returns an error.
-   **Robust ReAct Parsing**: Enhanced regex-based parsing in `base.py` to handle varied LLM output formats while strictly extracting `Thought`, `Action`, `Action Input`, and `Final Answer`.
-   Each agent publishes an `AgentCard` (name, input/output schema, capabilities, version).

-   **Report Agent** (`v2.0.0`): Extracts text from PDF reports and analyzes medical images (X-ray, MRI, CT) via MedGemma vision. Uses 3 specialized tools: document extraction → image extraction → clinical analysis.
-   **Diagnosis Agent**: Suggests differential diagnoses based on symptom analysis and trusted medical web crawling (UpToDate, Merck, etc.).
-   **Drug Agent** (`v2.0.0`): Checks interactions, contraindications, dosage, and recommendations via trusted pharmacology web search. Uses 2 specialized tools: interaction checker + recommendation/dosage/alternatives.
-   **PubMed Agent**: Searches NCBI PubMed for research papers and crawls trusted medical websites (Mayo Clinic, NIH, CDC, WHO, etc.) for clinical guidance.
-   **Patient Agent** (`v3.0.0`): HIPAA-compliant patient record retrieval and clinical analysis (MedGemma-powered). Uses 4 specialized tools: secure record retrieval, diagnosis pattern analysis, vitals assessment, and medication safety review — all on de-identified data. PII is resolved only internally by the retriever tool and never exposed to external LLMs.

### 3. Tool Layer (`tools/`) & MCP Server
The system exposes its capabilities via the **Model Context Protocol (MCP)**, allowing external clients (e.g., Claude Desktop) to use its tools directly.
-   **MCP Server**: `tools/mcp_server.py` — 13 tools + Agent Card Resources + 3 Standardized Prompts.
-   **MCP Resources**: Agent cards exposed via `agents://medicortex/{name}/card` URI scheme (MCP §3.1).
-   **MCP Prompts**: Exposes Agentic RAG workflows (`patient-full-review`, `drug-safety-check`, `medical-report-analysis`) directing external models to use tools correctly without hallucinating (MCP §3.2).
-   **Tool Caching**: External API tools (PubMed, Web Crawlers) use a `@redis_cache` decorator (`utils/cache_utils.py`) backed by Redis to cache tool execution results for 24 hours (TTL 86400), reducing latency and API costs.
-   **Tools** (grouped by agent):

    **PubMed & Web Research**
    -   `pubmed_search_tools.py`: NCBI PubMed E-utilities API search.
    -   `medical_webcrawler_tools.py`: General trusted medical site crawler.

    **Diagnosis**
    -   `symptom_analysis_tools.py`: Structured symptom parsing & knowledge core context integration.
    -   `diagnosis_webcrawler_tools.py`: Specialized crawler for diagnostic criteria (UpToDate, Merck, etc.).

    **Patient (HIPAA-compliant, de-identified)**
    -   `patient_retriever_tools.py`: PII-resolving patient record retriever. Returns structured JSON + Markdown. Re-redacts before output.
    -   `patient_history_analyzer_tools.py`: MedGemma diagnosis pattern analysis — comorbidities, risk factors, disease progression.
    -   `patient_vitals_tools.py`: Vital sign analysis — critical value flagging, condition-specific targets (e.g., diabetic BP goals), multi-visit trend detection.
    -   `patient_medication_review_tools.py`: Medication safety review — allergy cross-reference, polypharmacy detection (>5 meds), missing standard therapies, condition-contraindicated drugs.

    **Drug (Pharmacology)**
    -   `drug_interaction_tools.py`: Drug-drug interaction checker with severity grading (Major/Moderate/Minor).
    -   `drug_recommendation_tools.py`: Evidence-based drug recommendations, dosage guidelines, and alternative medications.

    **Report & Imaging**
    -   `document_extraction_tools.py`: PDF report → Markdown text extraction via pymupdf4llm.
    -   `image_extraction_tools.py`: Medical image analysis (X-ray, MRI, CT, pathology) via MedGemma vision (base64).
    -   `report_analysis_tools.py`: MedGemma clinical interpretation of extracted content — lab values, abnormalities, significance.

### 4. Frontend (`frontend/`)
-   **Vite + React** SPA at `http://localhost:5173`.
-   **Real-time streaming**: Consumes `/chat/stream` SSE endpoint for token-by-token response rendering.
-   **Agent Thinking UI**: Live accordion showing each agent's reasoning steps as they stream in, with persistence to the database.

### 5. Data Layer
-   **PostgreSQL**: Stores persistent chat history, session metadata, and agent thinking steps (`thinking` JSONB column on `chat_messages`). Schema managed via `database/schema.sql`.
-   **MinIO**: High-performance object storage for file uploads (PDFs, medical images).

---

## 🔐 PII / HIPAA Data Flow

The privacy pipeline ensures patient identifiers **never reach external LLMs** (GPT):

```
User Input → Presidio redact_pii() → <PERSON_1> + mapping stored in AgentState
    → GPT Router sees only <PERSON_1>
    → Patient Agent: retrieve_patient_records resolves PII internally for DB lookup, re-redacts output
    → All MedGemma analysis tools receive only de-identified data
    → GPT Aggregator formats <PERSON_1> text
    → node_restore_privacy() replaces <PERSON_1> → real name
    → Final response shown to user
```

---

## 🚀 Setup & Installation

### Prerequisites
-   Python 3.10+
-   Node.js & npm
-   PostgreSQL (Local or Docker)
-   MinIO Server (Local or Docker)
-   Redis (Local or Docker — Optional for Idempotency Cache, defaults to memory)
-   OpenAI API Key (GPT models)
-   Groq API Key (Model-as-Judge)

### 1. Environment Setup
Create a `.env` file in the root directory (validated by `config.py`):
```bash
OPENAI_API_KEY=your_key_here
GROQ_API_KEY=your_groq_key_here
JUDGE_SAMPLE_RATE=1.0  # Optional: 0.0-1.0 to sample evaluation frequency
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/medicortex
MINIO_URL=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
REDIS_URL=redis://localhost:6379/0
```

### 2. Backend Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Database Initialization
Apply the schema to PostgreSQL:
```bash
python -m database.init_db
```

### 4. Knowledge Core Assets
Build the optimized graph assets for the medical engine:
```bash
python3 -m knowledge_core.build_fast_assets
```

---

## 🏃‍♂️ Running the Application

### 1. Start the Backend Orchestrator
```bash
python orchestrator.py
```
-   **Server**: `http://0.0.0.0:8001`
-   **Docs**: `http://0.0.0.0:8001/docs`

### 2. Start the MCP Server (Optional)
```bash
python tools/mcp_server.py
```
(Configured via `stdin/stdout` for MCP clients)

### 3. Start the Frontend
```bash
cd frontend
npm install
npm run dev
```
-   **App**: `http://localhost:5173`

---

## 🏥 Verification & Health Check

**Run Health Check:**
```bash
python tests/health_check.py
```
**Expected Output:**
```text
[Database]
 ✅ PostgreSQL Connection Successful

[Storage]
 ✅ Upload Successful: http://...

[Orchestrator API]
 ✅ Status 200 OK | Active Agents: 5
```

---

## 📂 Project Structure

-   `orchestrator.py`: Main API server, LangGraph workflow, privacy manager, SSE streaming.
-   `config.py`: Centralized configuration management (Pydantic).
-   `specialized_agents/`:
    -   `base.py`: `A2ABaseAgent` — ReAct loop, tool dispatch, thinking step capture.
    -   `protocols.py`: Pydantic models — `AgentCard`, `Envelope`, `AgentResponse`.
    -   `report_agent.py`, `diagnosis_agent.py`, `drug_agent.py`, `pubmed_agent.py`, `patient_agent.py`
    -   `medgemma_llm.py`: MedGemma LLM wrapper.
-   `tools/`:
    -   `pubmed_search_tools.py`, `medical_webcrawler_tools.py`
    -   `symptom_analysis_tools.py`, `diagnosis_webcrawler_tools.py`
    -   `patient_retriever_tools.py`, `patient_history_analyzer_tools.py`, `patient_vitals_tools.py`, `patient_medication_review_tools.py`
    -   `drug_interaction_tools.py`, `drug_recommendation_tools.py`
    -   `document_extraction_tools.py`, `image_extraction_tools.py`, `report_analysis_tools.py`
    -   `mcp_server.py`: Model Context Protocol server with Resources, Tools, and STDIO transport.
-   `database/`:
    -   `models.py`: SQLAlchemy models (`ChatSession`, `ChatMessage` with `thinking` JSONB).
    -   `schema.sql`: Source of truth for database schema.
    -   `init_db.py`: Schema migration script.
-   `schemas/`: Pydantic models for API requests/responses.
-   `services/`: Business logic for Chat and Storage (MinIO).
-   `knowledge_core/`: Medical knowledge graph logic and fast asset builder.
-   `tests/`:
    -   `health_check.py`: Service health verification.
    -   `integration/test_reviewer_node.py`: Model-as-Judge logic and fallback verification.
-   `skills/`: Internal development standards — `A2A.md`, `MCP.md`.
