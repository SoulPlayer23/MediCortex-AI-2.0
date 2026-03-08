
import os
import sys
import random
import structlog
from typing import Dict, TypedDict, List, Optional, Tuple, Annotated
import operator
import json
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from contextlib import asynccontextmanager

# Local Imports
from config import settings
from schemas.models import (
    ChatRequest, ChatResponse, SessionResponse, MessageResponse, 
    UploadResponse, HealthResponse
)
from database.connection import engine, get_db
from database.models import Base, ChatSession, ChatMessage
from services.chat_service import chat_service
from services.minio_service import minio_service

# --- Third Party Imports ---
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# --- Setup Structlog ---
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer() if not settings.DEBUG else structlog.dev.ConsoleRenderer()
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)
logger = structlog.get_logger("Orchestrator")

# --- Local Imports (Late Import for Engines) ---
logger.info("Importing Local Engines...")
try:
    from knowledge_core.medical_engine import MedicalReasoningEngine
    logger.info("Initializing MedicalReasoningEngine...")
    medical_engine = MedicalReasoningEngine()
    logger.info("MedicalReasoningEngine connected", status="success")
except Exception as e:
    logger.warning("MedicalReasoningEngine unavailable, knowledge retrieval disabled", error=str(e))
    medical_engine = None

from specialized_agents.agents import AGENT_REGISTRY
from specialized_agents.protocols import Envelope, AgentResponse

# ==========================================
# 🛡️ PRIVACY LAYER (HIPAA COMPLIANCE)
# ==========================================
class PrivacyManager:
    def __init__(self):
        logger.info("Initializing HIPAA Privacy Layer (Presidio)")
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        logger.info("Presidio Engines Loaded", status="success")

    def redact_pii(self, text: str) -> Tuple[str, Dict[str, str]]:
        if not text:
            return "", {}

        # Covers the key HIPAA-relevant entity types Presidio supports
        results = self.analyzer.analyze(
            text=text,
            entities=[
                "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "DATE_TIME",
                "LOCATION", "US_SSN", "URL", "IP_ADDRESS",
            ],
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
            
        logger.info("Redacted entities", count=len(mapping))
        return redacted_text, mapping

    def restore_privacy(self, text: str, mapping: Dict[str, str]) -> str:
        restored_text = text
        for placeholder, original_value in mapping.items():
            restored_text = restored_text.replace(placeholder, original_value)
        return restored_text

logger.info("Instantiating PrivacyManager Singleton")
privacy_manager = PrivacyManager()

# ==========================================
# 🧠 AGENT STATE DEFINITION
# ==========================================
class AgentState(TypedDict):
    """
    State allowed to propagate through the graph.
    """
    input: str
    redacted_input: str
    pii_mapping: Dict[str, str]
    file_urls: List[str]
    context: List[str]
    history: List[str]
    messages: Annotated[List[BaseMessage], operator.add]
    agent_outputs: Annotated[List[str], operator.add]
    agent_thoughts: Annotated[List[str], operator.add] # New: Capture thinking steps
    final_output: str
    judge_score: Optional[int]            # A2A §5.2 — set by node_reviewer
    error: Optional[str]
    trace_id: Optional[str]
    session_id: Optional[str]

# ==========================================
# ⚡ SSE STREAMING SHARED STATE
# ==========================================
ACTIVE_STREAMS = {}

# ==========================================
# 🛠️ TOOLS & LLM
# ==========================================
@tool
def consult_medical_knowledge(query: str) -> str:
    """Consults the structured medical knowledge graph."""
    logger.info("consult_medical_knowledge invoked", query=query)
    if not medical_engine:
        return "Knowledge Engine Offline."
    results = medical_engine.search_and_reason(query)
    formatted = [f"- {r['name']} ({r['relation']}, Hop: {r.get('hop', '?')})" for r in results]
    return "\n".join(formatted) if formatted else "No specific knowledge found in graph."

try:
    logger.info("Initializing OpenAI LLM Client (Router)")
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        api_key=settings.OPENAI_API_KEY
    )
    logger.info("OpenAI Client Ready", status="success")
except Exception as e:
    logger.error("OpenAI setup failed", error=str(e))
    llm = None

# ==========================================
# 🕸️ LANGGRAPH NODES
# ==========================================
def node_analyze_privacy(state: AgentState):
    import uuid as _uuid
    trace_id = state.get("trace_id") or str(_uuid.uuid4())
    # A2A §5.1 — Bind trace_id to structured log context for full-chain tracing
    structlog.contextvars.bind_contextvars(trace_id=trace_id)
    logger.info("NODE: ANALYZE PRIVACY", trace_id=trace_id)
    redacted, mapping = privacy_manager.redact_pii(state['input'])
    return {
        "trace_id": trace_id,
        "redacted_input": redacted,
        "pii_mapping": mapping,
        "messages": [HumanMessage(content=redacted)],
        "agent_outputs": [] 
    }

def node_retrieve_knowledge(state: AgentState):
    logger.info("NODE: RETRIEVE KNOWLEDGE")
    user_query = state['redacted_input']
    
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
        
        # Parse JSON
        clean_response = response.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_response)
        if not data:
            search_term = None
        else:
            search_term = data.get("entity")
        
        if search_term and search_term.lower() != "none":
            logger.info("Extracted Search Term", term=search_term)
            raw_facts = consult_medical_knowledge.invoke(search_term)
            
            # ── Context Refinement (GPT-4o-mini) ──
            # We use GPT to "clean up" raw graph data into a coherent medical narrative
            # so that MedGemma receives structured, easy-to-parse context.
            refinement_prompt = (
                "You are a medical knowledge assistant. I will provide you with raw facts from a "
                "medical knowledge graph (concepts and their relations). "
                "Your task is to re-format these facts into a concise, structured narrative "
                "suitable for a clinical LLM to read. "
                "Focus on clarity and relationships. Do NOT add any information not present in the facts. "
                "Do NOT provide a diagnosis.\n\n"
                "Raw Facts:\n{facts}"
            )
            try:
                if "No specific knowledge found" not in raw_facts:
                    refined_response = llm.invoke([
                        SystemMessage(content=refinement_prompt.format(facts=raw_facts))
                    ]).content.strip()
                    graph_context = f"Structured Knowledge for '{search_term}':\n{refined_response}"
                    logger.info("Context Refined", term=search_term)
                else:
                    graph_context = raw_facts
            except Exception as ref_err:
                logger.warning("Context refinement failed, using raw facts", error=str(ref_err))
                graph_context = raw_facts
        else:
            logger.info("No specific medical entity found to search")
            graph_context = "No specific medical knowledge concept found in query."
            
    except Exception as e:
        logger.error("Entity extraction failed", error=str(e))
        graph_context = "Error retrieving knowledge."

    return {
        "context": [graph_context],
        "messages": state["messages"] 
    }

def node_router(state: AgentState):
    logger.info("NODE: ROUTER")
    input_text = state['redacted_input']
    context_str = "\n".join(state.get("context", []))
    
    system_prompt = (
        "You are the MediCortex Orchestrator. Your ONLY job is to select the best agent(s) to handle the user's query.\n\n"
        "═══ AGENT DECISION RULES ═══\n\n"
        "'diagnosis' — Use when the user describes symptoms, asks about a disease's symptoms, asks what disease they might have, "
        "asks for treatment options for a specific disease/condition, or asks for a differential diagnosis. "
        "EXAMPLES: 'symptoms of diabetes', 'what causes chest pain', 'treatment for hypertension', 'I have a headache and fever'.\n\n"
        "'drug' — Use ONLY when the user explicitly asks about a specific medication by name, asks about drug interactions, "
        "dosage, contraindications, or alternatives for a drug. "
        "EXAMPLES: 'interactions of Metformin and Lisinopril', 'dosage for Ibuprofen', 'alternatives to Atorvastatin'.\n\n"
        "'pubmed' — Use when the user asks for research papers, clinical studies, literature reviews, or recent evidence on a topic. "
        "EXAMPLES: 'latest research on Alzheimer's', 'clinical trials for immunotherapy', 'evidence for statins'.\n\n"
        "'report_analyzer' — Use ONLY when the user provides or references a document, lab result, PDF, image, X-ray, MRI, or CT scan. "
        "EXAMPLES: 'analyze my lab report', 'what does this X-ray show', 'interpret my HbA1c result'.\n\n"
        "'patient' — Use ONLY when the user asks about a specific named or identified patient's records, history, medications, or vitals. "
        "EXAMPLES: 'show me John's records', 'what medications is patient PT-10042 on'.\n\n"
        "═══ CRITICAL RULES ═══\n"
        "- A query about symptoms OR treatment of a disease → ['diagnosis'] only\n"
        "- A query about a named drug → ['drug'] only\n"
        "- A query about research/literature → ['pubmed'] only\n"
        "- Only combine agents when the query EXPLICITLY spans two domains (e.g., 'symptoms AND drug interactions for diabetes').\n"
        "- NEVER route to 'drug' for a pure symptoms/treatment query.\n"
        "- NEVER route to 'pubmed' unless research papers are explicitly requested.\n\n"
        "Return ONLY a JSON array of agent keys. Examples: ['diagnosis'] or ['drug'] or ['diagnosis', 'drug'].\n"
        "Do NOT add any explanation. Only output the JSON array."
    )
    
    user_message = f"User Query: {input_text}\n\nKnowledge Core Context (for your awareness): {context_str[:300]}"
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
    
    try:
        response = llm.invoke(messages).content
        clean_response = response.replace("```json", "").replace("```", "").strip()
        clean_response = clean_response.replace("'", '"')
        routes = json.loads(clean_response)
        if not isinstance(routes, list):
            routes = ["diagnosis"]
    except Exception as e:
        logger.error("Routing failed, defaulting to 'diagnosis'", error=str(e))
        routes = ["diagnosis"]
        
    logger.info("Routing to agents", routes=routes)
    return {"messages": [AIMessage(content=str(routes))]}

def make_agent_node(agent_key: str):
    def _node(state: AgentState, config: RunnableConfig):
        logger.info(f"NODE: AGENT [{agent_key.upper()}]")
        agent_executor = AGENT_REGISTRY.get(agent_key)
        if not agent_executor:
            return {"agent_outputs": [f"Error: Agent '{agent_key}' not found."]}
        
        context_str = "\n".join(state.get("context", []))
        history_str = "\n".join(state.get("history", []))
        
        enhanced_input = (
            f"Conversation History:\n{history_str}\n\n"
            f"Current Request: {state['redacted_input']}\n\n"
            f"Context from Knowledge Core:\n{context_str}"
        )

        # Inject file URLs for the report agent so its tools can download and analyze them
        if agent_key == "report_analyzer" and state.get("file_urls"):
            enhanced_input += "\n\nFiles to analyze:\n" + "\n".join(state["file_urls"])

        # A2A Protocol: Create Envelope with trace_id propagation (A2A §5.1)
        try:
            envelope = Envelope(
                trace_id=state.get("trace_id", ""),
                sender_id="orchestrator",
                receiver_id=agent_key,
                payload={"input": enhanced_input},
            )

            # HIPAA: Pass PII mapping to patient agent via payload only.
            # Never inject real PII values into enhanced_input — it would reach
            # the OpenAI fallback LLM if MedGemma is offline.
            if agent_key == "patient":
                import json as _json
                envelope.payload["pii_mapping_json"] = _json.dumps(state.get("pii_mapping", {}))

            # Bind live thoughts list so the agent can stream thoughts in real-time
            session_id_str = state.get("session_id", "default")
            live_thoughts = ACTIVE_STREAMS.get(session_id_str, [])
            envelope.payload["live_thoughts_queue"] = live_thoughts
            
            # Call Agent via Process
            response = agent_executor.process(envelope)
            
            # Capture thinking steps
            thoughts = response.thinking if response.thinking else []
            formatted_thoughts = [f"**[{agent_key.title()}]**: {t}" for t in thoughts]
            
            if response.error:
                logger.error(f"Agent {agent_key} returned error", error=response.error)
                return {
                    "agent_outputs": [f"## {agent_key.title()} Agent Error\n{response.error}"],
                    "agent_thoughts": formatted_thoughts
                }
            
            output = response.output if response.output else "No output generated."
            return {
                "agent_outputs": [f"## {agent_key.title()} Agent Response\n{output}"],
                "agent_thoughts": formatted_thoughts
            }
            
        except Exception as e:
            logger.error(f"Orchestrator failed to call agent {agent_key}", error=str(e))
            return {
                "agent_outputs": [f"## {agent_key.title()} Agent System Error\n{str(e)}"],
                "agent_thoughts": [f"**[{agent_key.title()}]**: System Error: {str(e)}"]
            }
    return _node

node_pubmed = make_agent_node("pubmed")
node_diagnosis = make_agent_node("diagnosis")
node_report_analyzer = make_agent_node("report_analyzer")
node_patient = make_agent_node("patient")
node_drug = make_agent_node("drug")

def node_aggregator(state: AgentState):
    logger.info("NODE: AGGREGATOR")
    raw_outputs = "\n\n".join(state["agent_outputs"])
    
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

def node_reviewer(state: AgentState):
    """
    A2A §5.2 — Model-as-Judge evaluation node.

    Scores the aggregated response 1–5 using Groq (llama-3.3-70b-versatile).
    If score < 3, appends a clinical disclaimer to protect the user.
    Respects JUDGE_SAMPLE_RATE, JUDGE_MAX_INPUT_TOKENS, and falls back to
    llama-3.1-8b-instant if the primary model hits rate limits.
    """
    logger.info("NODE: REVIEWER")

    # ── Sampling gate ─────────────────────────────────────────────────
    if not settings.JUDGE_ENABLED or random.random() > settings.JUDGE_SAMPLE_RATE:
        logger.info("reviewer_skipped", reason="disabled or sampled out")
        return {"judge_score": None}

    if not settings.GROQ_API_KEY:
        logger.warning("reviewer_skipped", reason="GROQ_API_KEY not set")
        return {"judge_score": None}

    # ── Truncate input to stay within TPD budget ──────────────────────
    response_text = state.get("final_output", "")
    # Rough token estimate: 1 token ≈ 4 chars
    max_chars = settings.JUDGE_MAX_INPUT_TOKENS * 4
    truncated = response_text[:max_chars]
    if len(response_text) > max_chars:
        truncated += "\n...[truncated for evaluation]"

    original_query = state.get("redacted_input", "")

    judge_prompt = f"""You are a clinical quality reviewer for a medical AI assistant.

Evaluate the following response on a scale of 1–5:
1 = Completely off-topic, dangerous, or hallucinates facts
2 = Partially relevant but contains significant errors or unsupported claims
3 = Relevant and mostly accurate, minor gaps acceptable
4 = Accurate, grounded in evidence, clearly addresses the query
5 = Excellent — accurate, complete, evidence-based, safe for clinical context

Criteria to check:
- Does the response address the user's query?
- Are all clinical claims grounded in tool outputs (no fabricated facts)?
- Does the response contain leaked PII placeholders (e.g. <PERSON_1>)?
- Is the response safe for a medical assistant context?

User Query: {original_query}

Response to evaluate:
{truncated}

Reply with ONLY a JSON object in this exact format, no other text:
{{"score": <1-5>, "reason": "<one sentence>", "confidence": "<0-100>%"}}"""

    def _call_groq(model_name: str) -> dict:
        judge_llm = ChatGroq(
            model=model_name,
            api_key=settings.GROQ_API_KEY,
            temperature=0,
            max_tokens=100,
        )
        result = judge_llm.invoke([HumanMessage(content=judge_prompt)]).content
        return json.loads(result.strip())

    # ── Call judge with fallback ──────────────────────────────────────
    judge_result = None
    for model in [settings.JUDGE_MODEL, settings.JUDGE_FALLBACK_MODEL]:
        try:
            judge_result = _call_groq(model)
            logger.info("reviewer_complete", model=model, score=judge_result.get("score"),
                        reason=judge_result.get("reason"))
            break
        except Exception as e:
            logger.warning("reviewer_model_failed", model=model, error=str(e))

    if judge_result is None:
        logger.error("reviewer_all_models_failed")
        return {"judge_score": None}

    score = int(judge_result.get("score", 3))
    reason = judge_result.get("reason", "")
    confidence = judge_result.get("confidence", "95%")

    # Store metadata on state so we can pick it up
    return_payload: dict = {"judge_score": score, "judge_reason": reason, "judge_confidence": confidence}

    # ── Append clinical disclaimer if quality is low ──────────────────
    current_output = state.get("final_output", "")
    if score < 3:
        disclaimer = (
            "\n\n---\n"
            "> ⚠️ **Clinical Disclaimer**: This response has been flagged by our quality "
            "reviewer for potential inaccuracies or incomplete information. "
            "Please consult a qualified healthcare professional before acting on this information. "
            f"*(Quality Score: {score}/5 — {reason})*"
        )
        logger.warning("reviewer_low_score_disclaimer_appended", score=score, reason=reason)
        return_payload["final_output"] = current_output + disclaimer

    return return_payload


def node_restore_privacy(state: AgentState):
    logger.info("NODE: RESTORE PRIVACY")
    raw_output = state.get("final_output", "")
    mapping = state.get("pii_mapping", {})
    restored = privacy_manager.restore_privacy(raw_output, mapping)
    return {"final_output": restored}

# ==========================================
# 🚀 GRAPH CONSTRUCTION
# ==========================================
# A2A §4.1 — Maximum agents per request (circuit breaker)
MAX_CONCURRENT_AGENTS = 3

def route_decision(state: AgentState):
    routes = []

    # Always route to report_analyzer if files were attached (A2A §4.1)
    if state.get("file_urls"):
        routes.append("report_analyzer")

    last_msg = state["messages"][-1].content
    try:
        llm_routes = json.loads(last_msg.replace("'", '"'))
        for r in llm_routes:
            if r in AGENT_REGISTRY and r not in routes:
                routes.append(r)
    except Exception:
        pass

    # A2A §4.1 — Circuit breaker: cap concurrent agent calls
    valid_routes = [r for r in routes if r in AGENT_REGISTRY][:MAX_CONCURRENT_AGENTS]
    return valid_routes or ["diagnosis"]

workflow = StateGraph(AgentState)
workflow.add_node("analyze_privacy", node_analyze_privacy)
workflow.add_node("retrieve_knowledge", node_retrieve_knowledge)
workflow.add_node("router", node_router)
workflow.add_node("pubmed", node_pubmed)
workflow.add_node("diagnosis", node_diagnosis)
workflow.add_node("report_analyzer", node_report_analyzer)
workflow.add_node("patient", node_patient)
workflow.add_node("drug", node_drug)
workflow.add_node("aggregator", node_aggregator)
workflow.add_node("reviewer", node_reviewer)       # A2A §5.2 — Model-as-Judge
workflow.add_node("restore_privacy", node_restore_privacy)

workflow.set_entry_point("analyze_privacy")
workflow.add_edge("analyze_privacy", "retrieve_knowledge")
workflow.add_edge("retrieve_knowledge", "router")

workflow.add_conditional_edges("router", route_decision, {k:k for k in AGENT_REGISTRY.keys()})

for agent_key in AGENT_REGISTRY.keys():
    workflow.add_edge(agent_key, "aggregator")

# A2A §5.2: aggregator → reviewer → restore_privacy
workflow.add_edge("aggregator", "reviewer")
workflow.add_edge("reviewer", "restore_privacy")
workflow.add_edge("restore_privacy", END)

orchestrator_graph = workflow.compile()
logger.info("Orchestrator Graph Compiled", status="success")

# ==========================================
# 🌐 FASTAPI SERVER
# ==========================================
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Orchestrator Server", app_name=settings.APP_NAME)
    
    # DB creation managed externally now
    logger.info("Database Schema Managed externally")
    
    yield
    # Shutdown
    logger.info("Shutting down")

app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """
    Streaming chat endpoint for Server-Sent Events (SSE).
    Sends 'thought' events for agent reasoning and 'response' event for final output.
    """
    async def event_generator():
        try:
            logger.info("Received streaming chat request", message_length=len(request.message))
            
            # 1. Create/Get Session
            session_id = request.session_id
            if not session_id:
                new_session = await chat_service.create_session(db)
                session_id = new_session.id
                yield f"data: {json.dumps({'type': 'session_id', 'content': str(session_id)})}\n\n"
            
            # 2. Extract file URLs from attachments (structured handoff to report agent)
            file_urls = [a["url"] for a in (request.attachments or []) if a.get("url")]

            # 3. Save User Message (with attachments for display)
            await chat_service.add_message(
                db, str(session_id), "user", request.message,
                attachments=request.attachments or [],
            )

            # 4. Retrieve History
            full_history = await chat_service.get_messages(db, str(session_id))
            past_turns = full_history[:-1]
            history_context = [f"{m.role.capitalize()}: {m.content}" for m in past_turns[-10:]]

            # 5. Stream Orchestrator Events
            agent_thoughts = []
            final_output = ""
            msg_metadata = {
                "llm_used": "MedGemma (via HF) / OpenAI Router",
                "judge_score": None,
                "judge_reason": None,
                "judge_confidence": None,
            }

            # Emit initial "thinking" state to show immediate activity
            yield f"data: {json.dumps({'type': 'thought', 'content': 'Querying Knowledge Core...'})}\n\n"

            import asyncio

            # Shared mutable state for the stream
            live_thoughts = []
            ACTIVE_STREAMS[str(session_id)] = live_thoughts

            final_output_container = {}

            async def run_graph():
                try:
                    result = await orchestrator_graph.ainvoke(
                        {
                            "input": request.message,
                            "messages": [],
                            "history": history_context,
                            "agent_thoughts": [],
                            "file_urls": file_urls,
                            "session_id": str(session_id),
                        }
                    )
                    final_output_container["result"] = result
                except Exception as e:
                    final_output_container["error"] = e
                    
            # Start graph execution in the background
            graph_task = asyncio.create_task(run_graph())
            
            last_thought_idx = 0
            # Poll for new thoughts while the graph is running
            while not graph_task.done():
                while last_thought_idx < len(live_thoughts):
                    thought = live_thoughts[last_thought_idx]
                    yield f"data: {json.dumps({'type': 'thought', 'content': thought})}\n\n"
                    if thought not in agent_thoughts:
                         agent_thoughts.append(thought)
                    last_thought_idx += 1
                await asyncio.sleep(0.1)
                
            # Process final output after graph completes
            if "error" in final_output_container:
                raise final_output_container["error"]
                
            graph_output = final_output_container.get("result", {})
            graph_final = graph_output.get("final_output")
            
            # Flush any remaining thoughts
            while last_thought_idx < len(live_thoughts):
                thought = live_thoughts[last_thought_idx]
                yield f"data: {json.dumps({'type': 'thought', 'content': thought})}\n\n"
                if thought not in agent_thoughts:
                    agent_thoughts.append(thought)
                last_thought_idx += 1
            
            if "judge_score" in graph_output:
                msg_metadata["judge_score"] = graph_output["judge_score"]
                msg_metadata["judge_reason"] = graph_output.get("judge_reason")
                msg_metadata["judge_confidence"] = graph_output.get("judge_confidence")
                yield f"data: {json.dumps({'type': 'metadata', 'content': msg_metadata})}\n\n"

            if graph_final:
                 final_output = graph_final
                 yield f"data: {json.dumps({'type': 'response', 'content': final_output})}\n\n"

            # 5. Save AI Response to DB (only once)
            if final_output:
                 await chat_service.add_message(db, str(session_id), "assistant", final_output, thinking=agent_thoughts, metadata=msg_metadata)
            
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error("Streaming error", error=str(e))
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        finally:
            if str(session_id) in ACTIVE_STREAMS:
                del ACTIVE_STREAMS[str(session_id)]

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """
    Legacy non-streaming chat endpoint.
    """
    try:
        logger.info("Received chat request", message_length=len(request.message))
        
        # 1. Create session if not provided
        session_id = request.session_id
        if not session_id:
            new_session = await chat_service.create_session(db)
            session_id = new_session.id
            
        # 2. Extract file URLs and save user message with attachments
        file_urls = [a["url"] for a in (request.attachments or []) if a.get("url")]
        await chat_service.add_message(
            db, str(session_id), "user", request.message,
            attachments=request.attachments or [],
        )

        # 3. Retrieve History for Context
        full_history = await chat_service.get_messages(db, str(session_id))
        past_turns = full_history[:-1]
        history_context = [f"{m.role.capitalize()}: {m.content}" for m in past_turns[-10:]]

        # 4. Invoke Orchestrator
        result = await orchestrator_graph.ainvoke({
            "input": request.message,
            "messages": [],
            "history": history_context,
            "agent_thoughts": [],
            "file_urls": file_urls,
        })
        response_text = result.get("final_output")
        agent_thinking = result.get("agent_thoughts", [])
        
        msg_metadata = {
            "llm_used": "MedGemma (via HF) / OpenAI Router",
            "judge_score": result.get("judge_score"),
            "judge_reason": result.get("judge_reason"),
            "judge_confidence": result.get("judge_confidence")
        }
        
        # 5. Save AI Response
        await chat_service.add_message(db, str(session_id), "assistant", response_text, thinking=agent_thinking, metadata=msg_metadata)
        
        return ChatResponse(
            response=response_text,
            session_id=session_id,
            thinking=agent_thinking,
            metadata=msg_metadata
        )
    except Exception as e:
        logger.error("Error processing request", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chats", response_model=List[SessionResponse])
async def get_chats(db: AsyncSession = Depends(get_db)):
    """Get all chat sessions"""
    return await chat_service.get_sessions(db)

@app.get("/chats/{session_id}", response_model=List[MessageResponse])
async def get_chat_history(session_id: str, db: AsyncSession = Depends(get_db)):
    """Get messages for a specific session"""
    return await chat_service.get_messages(db, session_id)

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload file to MinIO"""
    try:
        content = await file.read()
        url = await minio_service.upload_file(content, file.filename, file.content_type)
        if not url:
            raise HTTPException(status_code=500, detail="Upload failed")
        return UploadResponse(url=url, filename=file.filename)
    except Exception as e:
        logger.error("Upload error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="online", agents=list(AGENT_REGISTRY.keys()))

# ── A2A §1.1 — Agent Card Discovery Endpoint ────────────────────────
@app.get("/.well-known/agent-cards")
async def get_agent_cards():
    """Expose all registered agent cards for A2A discovery."""
    cards = {}
    for name, agent in AGENT_REGISTRY.items():
        card = agent.get_card()
        cards[name] = card.model_dump()
    return cards

@app.get("/.well-known/agent-cards/{agent_name}")
async def get_agent_card(agent_name: str):
    """Expose a specific agent's card for A2A discovery."""
    agent = AGENT_REGISTRY.get(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    return agent.get_card().model_dump()

if __name__ == "__main__":
    logger.info("Starting Orchestrator Server manually", port=8001)
    uvicorn.run(app, host="0.0.0.0", port=8001)
