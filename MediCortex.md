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
-   **Privacy Manager**: Uses Microsoft Presidio to redact PII (PHI) before sending data to agents.
-   **Router**: Intelligently routes user queries to specialized agents (capped at 3 concurrent agents per request).

### 2. Specialized Agents (`specialized_agents/`)
All agents adhere to the **A2A Protocol**, taking an `Envelope` input and returning an `AgentResponse`.
-   **Report Agent**: Extracts text from PDF reports and analyzes medical images (X-ray, MRI, CT) via MedGemma vision. Provides structured clinical interpretation.
-   **Diagnosis Agent**: Suggests differential diagnoses based on symptom analysis and trusted medical web crawling.
-   **Drug Agent**: Checks interactions, contraindications, dosage, and recommendations via trusted pharmacology web search.
-   **PubMed Agent**: Searches NCBI PubMed for research papers and crawls trusted medical websites (Mayo Clinic, NIH, CDC, WHO, etc.) for clinical guidance.
-   **Patient Agent**: HIPAA-compliant patient record retrieval and clinical history analysis (MedGemma-powered).

### 3. Tool Layer (`tools/`) & MCP Server
The system exposes its capabilities via the **Model Context Protocol (MCP)**, allowing external clients (e.g., Claude Desktop) to use its tools directly.
-   **MCP Server**: `tools/mcp_server.py`
-   **MCP Resources**: Agent cards exposed via `agents://medicortex/{name}/card` URI scheme (MCP §3.1).
-   **Tools**:
    -   `pubmed_search_tools.py`: NCBI PubMed API keys.
    -   `medical_webcrawler_tools.py`: General trusted medical site crawler.
    -   `symptom_analysis_tools.py`: Structured symptom parsing & context integration.
    -   `diagnosis_webcrawler_tools.py`: Specialized crawler for diagnostic criteria (UpToDate, Merck, etc.).
    -   `patient_retriever_tools.py`: HIPAA-compliant patient record retriever with PII resolution.
    -   `patient_history_analyzer_tools.py`: MedGemma-powered clinical history analyzer.
    -   `drug_interaction_tools.py`: Drug-drug interaction checker (Drugs.com, RxList, etc.).
    -   `drug_recommendation_tools.py`: Drug recommendation, dosage, and alternatives.
    -   `document_extraction_tools.py`: PDF report → Markdown text extraction.
    -   `image_extraction_tools.py`: Medical image analysis via MedGemma vision.
    -   `report_analysis_tools.py`: Clinical interpretation of extracted content.
    -   Other domain-specific tools.

### 4. Data Layer
-   **PostgreSQL**: Stores persistent chat history and session metadata (Schema managed via `database/schema.sql`).
-   **MinIO**: High-performance object storage for file uploads.

---

## 🚀 Setup & Installation

### Prerequisites
-   Python 3.10+
-   Node.js & npm
-   PostgreSQL (Local or Docker)
-   MinIO Server (Local or Docker)
-   OpenAI API Key

### 1. Environment Setup
Create a `.env` file in the root directory (validated by `config.py`):
```bash
OPENAI_API_KEY=your_key_here
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/medicortex
MINIO_URL=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
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
To start the FastAPI server:
```bash
python orchestrator.py
```
-   **Server**: `http://0.0.0.0:8001`
-   **Docs**: `http://0.0.0.0:8001/docs`

### 2. Start the MCP Server (Optional)
To run the MCP server for external tool access:
```bash
python tools/mcp_server.py
```
(Configured via `stdin/stdout` for MCP clients)

### 3. Start the Frontend
In a separate terminal:
```bash
cd frontend
npm install
npm run dev
```
-   **App**: `http://localhost:5173`

---

## 🏥 Verification & Health Check

We provide a unified health check script to verify all dependent services.

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

-   `orchestrator.py`: Main API server and agent coordinator.
-   `config.py`: Centralized configuration management (Pydantic).
-   `specialized_agents/`:
    -   `base.py`: Base agent class enforcing A2A.
    -   `protocols.py`: Pydantic models for Agent Cards and Envelopes.
    -   `*_agent.py`: Individual agent logic.
-   `tools/`:
    -   `pubmed_search_tools.py`: NCBI PubMed E-utilities API search.
    -   `medical_webcrawler_tools.py`: Trusted medical website crawler.
    -   `diagnosis_tools.py`, `patient_retriever_tools.py`, `patient_history_analyzer_tools.py`
    -   `drug_interaction_tools.py`, `drug_recommendation_tools.py`
    -   `document_extraction_tools.py`, `image_extraction_tools.py`, `report_analysis_tools.py`
    -   `mcp_server.py`: Model Context Protocol server with Resources.
-   `database/`:
    -   `models.py`: SQLAlchemy models.
    -   `schema.sql`: Source of truth for database schema.
    -   `init_db.py`: Schema migration script.
-   `schemas/`: Pydantic models for API requests/responses.
-   `services/`: Business logic for Chat and Storage.
-   `knowledge_core/`: Medical knowledge graph logic.
