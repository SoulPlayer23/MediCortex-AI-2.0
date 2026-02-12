import os
import sys
import logging
from typing import Dict, TypedDict, List, Optional, Tuple, Annotated
import operator
import dotenv
import json
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from contextlib import asynccontextmanager

# Load environment variables first
dotenv.load_dotenv()

# Local Imports
from database.connection import engine, get_db
from database.models import Base, ChatSession, ChatMessage
from services.chat_service import chat_service
from services.minio_service import minio_service

# Setup Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger("Orchestrator")

# --- Third Party Imports ---
logger.info("Importing LangChain & Presidio dependencies...")
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# --- Local Imports ---
logger.info("Importing Local Engines...")
try:
    from knowledge_core.medical_engine import MedicalReasoningEngine
    logger.info("Initializing MedicalReasoningEngine...")
    medical_engine = MedicalReasoningEngine()
    logger.info("✅ MedicalReasoningEngine connected.")
except ImportError:
    logger.warning("Could not import MedicalKnowledgeEngine. Knowledge retrieval will fail.")
    medical_engine = None

from specialized_agents.agents import AGENT_REGISTRY

# ==========================================
# 🛡️ PRIVACY LAYER (HIPAA COMPLIANCE)
# ==========================================
class PrivacyManager:
    def __init__(self):
        logger.info("🛡️  Initializing HIPAA Privacy Layer (Presidio)...")
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        logger.info("   ✅ Presidio Engines Loaded.")

    def redact_pii(self, text: str) -> Tuple[str, Dict[str, str]]:
        if not text:
            return "", {}

        logger.debug(f"scanning text: {text[:20]}...")
        results = self.analyzer.analyze(
            text=text,
            entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "DATE_TIME"],
            language='en'
        )
        
        mapping = {}
        results = sorted(results, key=lambda x: x.start, reverse=True)
        redacted_text = text
        type_counts = {}

        for result in results:
            entity_type = result.entity_type
            start, end = result.start, result.end
            original_value = text[start:end]
            
            count = type_counts.get(entity_type, 0) + 1
            type_counts[entity_type] = count
            placeholder = f"<{entity_type}_{count}>"
            
            mapping[placeholder] = original_value
            redacted_text = redacted_text[:start] + placeholder + redacted_text[end:]
            
        logger.info(f"   🔒 Redacted {len(mapping)} entities.")
        return redacted_text, mapping

    def restore_privacy(self, text: str, mapping: Dict[str, str]) -> str:
        restored_text = text
        for placeholder, original_value in mapping.items():
            restored_text = restored_text.replace(placeholder, original_value)
        return restored_text

logger.info("Instantiating PrivacyManager Singleton...")
privacy_manager = PrivacyManager()

# ==========================================
# 🧠 AGENT STATE DEFINITION
# ==========================================
class AgentState(TypedDict):
    input: str
    redacted_input: str
    pii_mapping: Dict[str, str]
    context: List[str]
    history: List[str] # <--- Added history
    messages: List[BaseMessage]
    agent_outputs: Annotated[List[str], operator.add]
    final_output: str
    error: Optional[str]

# ==========================================
# 🛠️ TOOLS & LLM
# ==========================================
@tool
def consult_medical_knowledge(query: str) -> str:
    """Consults the structured medical knowledge graph."""
    logger.info(f"🛠️ [Tool] consult_medical_knowledge invoked: '{query}'")
    if not medical_engine:
        return "Knowledge Engine Offline."
    results = medical_engine.search_and_reason(query)
    formatted = [f"- {r['name']} ({r['relation']}, Hop: {r.get('hop', '?')})" for r in results]
    return "\n".join(formatted) if formatted else "No specific knowledge found in graph."

try:
    logger.info("Initializing OpenAI LLM Client (Router)...")
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0
    )
    logger.info("✅ OpenAI Client Ready.")
except Exception as e:
    logger.error(f"WARNING: OpenAI setup failed: {e}")
    llm = None

# ==========================================
# 🕸️ LANGGRAPH NODES
# ==========================================
def node_analyze_privacy(state: AgentState):
    logger.info("--- NODE: ANALYZE PRIVACY ---")
    redacted, mapping = privacy_manager.redact_pii(state['input'])
    return {
        "redacted_input": redacted,
        "pii_mapping": mapping,
        "messages": [HumanMessage(content=redacted)],
        "agent_outputs": [] 
    }

def node_retrieve_knowledge(state: AgentState):
    logger.info("--- NODE: RETRIEVE KNOWLEDGE (Smart Extraction) ---")
    user_query = state['redacted_input']
    
    # Smart Entity Extraction using Router LLM
    # We ask the LLM to extract keywords likely to be in the Knowledge Graph
    system_prompt = (
        "You are a medical entity extractor. "
        "Extract the SINGLE most important medical term (disease, symptom, or drug) to search in a knowledge graph. "
        "Return the result as a JSON object with a single key 'entity'. "
        "If multiple concepts exist, pick the most specific disease. "
        "If nothing relevant is found, return null."
        "\n\nExamples:\n"
        "User: 'Tell me about Ebola outbreaks' -> {\"entity\": \"Ebola\"}\n"
        "User: 'Symptoms of Heart Attack' -> {\"entity\": \"Heart Attack\"}\n"
        "User: 'Patient has high fever' -> {\"entity\": \"Fever\"}"
    )
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query)
        ]).content.strip()
        
        logger.info(f"🐛 Raw Extractor Output: {response}")
        
        # Parse JSON
        clean_response = response.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_response)
        if not data:
            search_term = None
        else:
            search_term = data.get("entity")
        
        if search_term and search_term.lower() != "none":
            logger.info(f"🔍 Extracted Search Term: '{search_term}'")
            graph_context = consult_medical_knowledge.invoke(search_term)
        else:
            logger.info("🔍 No specific medical entity found to search.")
            graph_context = "No specific medical knowledge concept found in query."
            
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        graph_context = "Error retrieving knowledge."

    return {
        "context": [graph_context],
        "messages": state["messages"] 
    }

def node_router(state: AgentState):
    logger.info("--- NODE: ROUTER ---")
    input_text = state['redacted_input']
    
    system_prompt = (
        "You are the MediCortex Orchestrator. Analyze the user request and select the BEST specialized agents "
        "to handle it. \n"
        "Available Agents:\n"
        "- 'pubmed': For research, literature, papers.\n"
        "- 'diagnosis': For symptoms, possibilities, differential diagnosis.\n"
        "- 'report': For lab results, medical reports, text analysis, and image analysis (X-Rays, MRIs, CT scans).\n"
        "- 'patient': For retrieving patient history or records.\n"
        "- 'drug': For medication interactions or contraindications.\n\n"
        "Return ONLY a JSON list of keys, e.g. ['pubmed', 'diagnosis']. If unsure, default to ['pubmed']."
    )
    
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=input_text)]
    
    try:
        response = llm.invoke(messages).content
        clean_response = response.replace("```json", "").replace("```", "").strip()
        clean_response = clean_response.replace("'", '"')
        routes = json.loads(clean_response)
        if not isinstance(routes, list):
            routes = ["pubmed"]
    except Exception as e:
        logger.error(f"Routing failed ({e}), defaulting to 'diagnosis'.")
        routes = ["diagnosis"]
        
    logger.info(f"➡️ Routing to: {routes}")
    return {"messages": [AIMessage(content=str(routes))]}

def make_agent_node(agent_key: str):
    def _node(state: AgentState):
        logger.info(f"--- NODE: AGENT [{agent_key.upper()}] ---")
        agent_executor = AGENT_REGISTRY.get(agent_key)
        if not agent_executor:
            return {"agent_outputs": [f"Error: Agent '{agent_key}' not found."]}
        
        context_str = "\\n".join(state.get("context", []))
        history_str = "\\n".join(state.get("history", [])) # <--- Format history
        
        # Inject History into Input
        enhanced_input = (
            f"Conversation History:\\n{history_str}\\n\\n"
            f"Current Request: {state['redacted_input']}\\n\\n"
            f"Context from Knowledge Core:\\n{context_str}"
        )
        
        try:
            result = agent_executor.invoke({"input": enhanced_input})
            output = result.get("output", "No output generated.")
            return {"agent_outputs": [f"## {agent_key.title()} Agent Response\\n{output}"]}
        except Exception as e:
            return {"agent_outputs": [f"## {agent_key.title()} Agent Error\\n{str(e)}"]}
    return _node

node_pubmed = make_agent_node("pubmed")
node_diagnosis = make_agent_node("diagnosis")
node_report = make_agent_node("report")
node_patient = make_agent_node("patient")
node_drug = make_agent_node("drug")

def node_aggregator(state: AgentState):
    logger.info("--- NODE: AGGREGATOR ---")
    raw_outputs = "\\n\\n".join(state["agent_outputs"])
    
    formatting_prompt = (
        "You are the MediCortex Interface. Format the following medical agent reports into "
        "a beautiful, human-readable Markdown response. \n"
        "Use bolding, italics, bullet points, and headers to make it easy to read. "
        "Do not change the factual content, just the presentation.\n\n"
        f"Raw Reports:\n{raw_outputs}"
    )
    
    try:
        formatted = llm.invoke([HumanMessage(content=formatting_prompt)]).content
    except Exception:
        formatted = raw_outputs 
        
    return {"final_output": formatted}

def node_restore_privacy(state: AgentState):
    logger.info("--- NODE: RESTORE PRIVACY ---")
    raw_output = state.get("final_output", "")
    mapping = state.get("pii_mapping", {})
    restored = privacy_manager.restore_privacy(raw_output, mapping)
    return {"final_output": restored}

# ==========================================
# 🚀 GRAPH CONSTRUCTION
# ==========================================
def route_decision(state: AgentState):
    last_msg = state["messages"][-1].content
    try:
        routes = eval(last_msg) 
        valid_routes = [r for r in routes if r in AGENT_REGISTRY]
        return valid_routes if valid_routes else ["diagnosis"]
    except:
        return ["diagnosis"] 

workflow = StateGraph(AgentState)
workflow.add_node("analyze_privacy", node_analyze_privacy)
workflow.add_node("retrieve_knowledge", node_retrieve_knowledge)
workflow.add_node("router", node_router)
workflow.add_node("pubmed", node_pubmed)
workflow.add_node("diagnosis", node_diagnosis)
workflow.add_node("report", node_report)
workflow.add_node("patient", node_patient)
workflow.add_node("drug", node_drug)
workflow.add_node("aggregator", node_aggregator)
workflow.add_node("restore_privacy", node_restore_privacy)

workflow.set_entry_point("analyze_privacy")
workflow.add_edge("analyze_privacy", "retrieve_knowledge")
workflow.add_edge("retrieve_knowledge", "router")

workflow.add_conditional_edges("router", route_decision, {k:k for k in AGENT_REGISTRY.keys()})

for agent_key in AGENT_REGISTRY.keys():
    workflow.add_edge(agent_key, "aggregator")

workflow.add_edge("aggregator", "restore_privacy")
workflow.add_edge("restore_privacy", END)

orchestrator_graph = workflow.compile()
logger.info("✅ Orchestrator Graph Compiled.")

# ==========================================
# 🌐 FASTAPI SERVER
# ==========================================
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting Orchestrator Server...")
    # Create DB Tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("✅ Database Tables Created.")
    
    # Ensure MinIO Bucket
    # await minio_service.ensure_bucket_exists() # TODO: Fix event warning
    
    yield
    # Shutdown
    logger.info("🛑 Shutting down...")

app = FastAPI(title="MediCortex Orchestrator", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For dev, allow all. In prod, specify frontend URL.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

@app.post("/chat")
async def chat_endpoint(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """
    Main chat endpoint for the Frontend.
    """
    try:
        logger.info(f"📨 Received request: {request.message[:50]}...")
        
        # 1. Create session if not provided
        session_id = request.session_id
        if not session_id:
            new_session = await chat_service.create_session(db)
            session_id = new_session.id
            
        # 2. Save User Message
        await chat_service.add_message(db, session_id, "user", request.message)
        
        # 3. Retrieve History for Context (Last 10 messages, excluding current)
        full_history = await chat_service.get_messages(db, session_id)
        # Exclude the very last message (which is the current user request we just added)
        past_turns = full_history[:-1] 
        # Format for Agent Context
        history_context = [f"{m.role.capitalize()}: {m.content}" for m in past_turns[-10:]]

        # 4. Invoke Orchestrator
        result = await orchestrator_graph.ainvoke({
            "input": request.message, 
            "messages": [],
            "history": history_context # <--- Pass history
        })
        response_text = result.get("final_output")
        
        # 5. Save AI Response
        await chat_service.add_message(db, session_id, "assistant", response_text)
        
        return {
            "response": response_text,
            "session_id": session_id
        }
    except Exception as e:
        logger.error(f"❌ Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chats")
async def get_chats(db: AsyncSession = Depends(get_db)):
    """Get all chat sessions"""
    return await chat_service.get_sessions(db)

@app.get("/chats/{session_id}")
async def get_chat_history(session_id: str, db: AsyncSession = Depends(get_db)):
    """Get messages for a specific session"""
    return await chat_service.get_messages(db, session_id)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload file to MinIO"""
    try:
        content = await file.read()
        url = await minio_service.upload_file(content, file.filename, file.content_type)
        if not url:
            raise HTTPException(status_code=500, detail="Upload failed")
        return {"url": url, "filename": file.filename}
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "online", "agents": list(AGENT_REGISTRY.keys())}

if __name__ == "__main__":
    # If run directly, start the server
    logger.info("🚀 Starting Orchestrator Server on port 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
