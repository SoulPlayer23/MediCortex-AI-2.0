# MediCortex AI 2.0

MediCortex is an advanced medical reasoning system designed to assist healthcare professionals by orchestrating specialized AI agents. It leverages a 3-layer architecture to ensure reliability, privacy compliance (HIPAA), and deterministic execution where possible.

---

## 🏗️ System Architecture

The system is built on a **3-Layer Architecture**:

1.  **Directive Layer**: Uses standard operating procedures (SOPs) defined in `directives/` to guide agent behavior.
2.  **Orchestration Layer (`orchestrator.py`)**:
    -   **LangGraph Workflow**: Manages the state and flow of information between agents.
    -   **Privacy Manager**: Uses Microsoft Presidio to redact PII (PHI) before sending data to LLMs and restores it before showing it to the user.
    -   **Router**: Intelligently routes user queries to the most appropriate specialized agent.
3.  **Execution Layer**:
    -   **Specialized Agents**: task-specific agents located in `specialized_agents/`.
    -   **Knowledge Core**: A graph-based medical reasoning engine in `knowledge_core/`.
    -   **Frontend**: A modern React application for user interaction.

### Core Components

-   **Orchestrator** (`orchestrator.py`): The brain of the system. It handles privacy, routing, and aggregating responses.
-   **Data Layer** (`database/`, `services/`):
    -   **PostgreSQL**: Stores persistent chat history and session metadata.
    -   **MinIO**: High-performance object storage for file uploads (medical reports, images).
-   **Knowledge Core** (`knowledge_core/`): Valid medical knowledge graph embeddings and reasoning logic.
-   **Specialized Agents** (`specialized_agents/`):
    -   `pubmed`: Searches medical literature.
    -   `diagnosis`: Suggests differential diagnoses based on symptoms.
    -   `report`: Analyzes medical reports and lab values.
    -   `patient`: Retrieves patient records (simulated/secure).
    -   `drug`: Checks for drug interactions and contraindications.

---

## 🚀 Setup & Installation

### Prerequisites
-   Python 3.10+
-   Node.js & npm
-   PostgreSQL (Local or Docker)
-   MinIO Server (Local or Docker)
-   OpenAI API Key (in `.env`)

### 1. Environment Setup
Create a `.env` file in the root directory:
```bash
OPENAI_API_KEY=your_key_here
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/medicortex
MINIO_ENDPOINT=http://localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
```
*Note: If your password contains special characters, ensure they are URL-encoded.*

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
*This will generate `vectors.npy` and `maps.pkl` in `knowledge_core/assets/`.*

### 4. Frontend Dependencies
Navigate to the frontend directory and install dependencies:
```bash
cd frontend
npm install
```

---

## 🏃‍♂️ Running the Application

### Start Services (Database & MinIO)
Ensure PostgreSQL and MinIO are running. If using Docker:
```bash
docker run -p 5432:5432 -e POSTGRES_PASSWORD=password postgres
docker run -p 9000:9000 -p 9001:9001 minio/minio server /data --console-address ":9001"
```

### Start the Backend Orchestrator
To start the FastAPI server:
```bash
# From the root directory
python3 orchestrator.py
```
-   **Server**: `http://0.0.0.0:8001`
-   **Health Check**: `http://0.0.0.0:8001/health`
-   **Docs**: `http://0.0.0.0:8001/docs`

### Start the Frontend
In a separate terminal, start the React development server:
```bash
cd frontend
npm run dev
```
-   **App**: Access via the URL provided in the terminal (usually `http://localhost:5173`).

---

## 📂 Project Structure

-   `orchestrator.py`: Main entry point for the backend.
-   `database/`: SQLAlchemy models and connection logic.
-   `services/`: Business logic for Chat and MinIO operations.
-   `specialized_agents/`: Contains agent definitions and tools.
-   `knowledge_core/`: Medical knowledge graph and reasoning logic.
-   `frontend/`: React application source code.
-   `AGENTS.md`: Detailed instructions for agent behavior and architecture principles.
