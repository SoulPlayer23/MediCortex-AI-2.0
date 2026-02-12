# MediCortex AI 2.0

MediCortex is an advanced medical reasoning system designed to assist healthcare professionals by orchestrating specialized AI agents. It leverages a privacy-first architecture to ensure HIPAA compliance and deterministic execution where possible.

---

## 🏗️ System Architecture

The system is built on a **Centralized Orchestration Architecture**:

1.  **Orchestration Layer (`orchestrator.py`)**:
    -   **Context Manager**: Injects conversation history (last 10 turns) into agent prompts for continuity.
    -   **Privacy Manager**: Uses Microsoft Presidio to redact PII (PHI) before sending data to agents.
    -   **Router**: Intelligently routes user queries (Text or Image) to the most appropriate specialized agent.
2.  **Specialized Agents** (`specialized_agents/`):
    -   **Report Agent**: Analyzes medical reports (PDF) and images (X-Ray, MRI, etc.) using PIL/OpenCV.
    -   **Diagnosis Agent**: Suggests differential diagnoses based on symptoms.
    -   **Drug Agent**: Checks for interactions and contraindications.
    -   **PubMed Agent**: Retrieving medical literature.
    -   **Patient Agent**: Retrieving secure patient records.
3.  **Knowledge Core** (`knowledge_core/`):
    -   Graph-based medical reasoning engine for verifying agent outputs.

### Data Layer
-   **PostgreSQL**: Stores persistent chat history and session metadata.
-   **MinIO**: High-performance object storage for file uploads (medical reports, images).

---

## 🚀 Setup & Installation

### Prerequisites
-   Python 3.10+
-   Node.js & npm
-   PostgreSQL (Local or Docker)
-   MinIO Server (Local or Docker)
-   OpenAI API Key

### 1. Environment Setup
Create a `.env` file in the root directory:
```bash
OPENAI_API_KEY=your_key_here
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/medicortex
MINIO_ENDPOINT=http://localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123
```

### 2. Backend Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Knowledge Core Assets
Build the optimized graph assets for the medical engine:
```bash
python3 -m knowledge_core.build_fast_assets
```

### 4. Frontend Dependencies
Navigate to the frontend directory and install dependencies:
```bash
cd frontend
npm install
```

---

## 🏃‍♂️ Running the Application

### 1. Start Services (Database & MinIO)
Ensure PostgreSQL and MinIO are running.

### 2. Start the Backend Orchestrator
To start the FastAPI server:
```bash
# From the root directory
python3 orchestrator.py
```
-   **Server**: `http://0.0.0.0:8001`
-   **Docs**: `http://0.0.0.0:8001/docs`

### 3. Start the Frontend
In a separate terminal:
```bash
cd frontend
npm run dev
```
-   **App**: `http://localhost:5173`

---

## 🏥 Verification & Health Check

We provide a unified health check script to verify all dependent services (Database, Storage, API) in one go.

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

-   `orchestrator.py`: Main entry point and API server.
-   `specialized_agents/`: Agent definitions, tools (including Image Analysis), and prompts.
-   `services/`:
    -   `chat_service.py`: Managing chat history in PostgreSQL.
    -   `minio_service.py`: Handling file uploads to MinIO.
-   `knowledge_core/`: Medical knowledge graph logic.
-   `frontend/`: React application.
-   `tests/`: Contains `health_check.py` for system verification.
