
import os
import sys
import re
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
from langchain_ollama import ChatOllama
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
try:
    from knowledge_core.medical_engine import MedicalReasoningEngine
    _medical_engine_available = True
except Exception as _e:
    logger.warning("MedicalReasoningEngine module unavailable at import", error=str(_e))
    _medical_engine_available = False

# Singletons — set to None here; initialized once in lifespan() by the worker process.
# This prevents triple-initialization caused by reload=True (main process + reloader + worker).
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

    def redact_identifying_pii(self, text: str) -> str:
        """
        Narrow redaction for conversation history injected into LLM prompts.

        Only removes directly identifying entities (PERSON, PHONE_NUMBER,
        EMAIL_ADDRESS, US_SSN) — NOT DATE_TIME, LOCATION, URL, or IP_ADDRESS.
        Dates and locations are clinically meaningful in history context and
        stripping them breaks reasoning. Since the mapping is discarded
        (suppression only, no restoration needed) we must not create
        <DATE_TIME_N> placeholders that would leak into the final output.
        """
        if not text:
            return text

        results = self.analyzer.analyze(
            text=text,
            entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "US_SSN"],
            language="en",
        )
        results = sorted(results, key=lambda x: x.start, reverse=True)
        redacted_text = text
        type_counts: Dict[str, int] = {}

        for result in results:
            entity_type = result.entity_type
            start, end = result.start, result.end
            count = type_counts.get(entity_type, 0) + 1
            type_counts[entity_type] = count
            placeholder = f"<{entity_type}_{count}>"
            redacted_text = redacted_text[:start] + placeholder + redacted_text[end:]

        logger.info("Redacted history entities", count=sum(type_counts.values()))
        return redacted_text

    def restore_privacy(self, text: str, mapping: Dict[str, str]) -> str:
        restored_text = text
        for placeholder, original_value in mapping.items():
            restored_text = restored_text.replace(placeholder, original_value)
        return restored_text

privacy_manager = None

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
    # Compact redacted summary of recent turns for the router — built from DB
    # message_metadata.agents_used so the router can resolve follow-up pronouns.
    routing_context: Optional[str]
    messages: Annotated[List[BaseMessage], operator.add]
    agent_outputs: Annotated[List[str], operator.add]
    agent_thoughts: Annotated[List[str], operator.add]
    # Accumulates agent keys that ran this turn (e.g. ["patient", "pharmacology"])
    agents_used: Annotated[List[str], operator.add]
    # Accumulates {title, url} source dicts from tool observations across all agents
    agent_sources: Annotated[List[dict], operator.add]
    final_output: str
    judge_score: Optional[int]            # A2A §5.2 — set by node_reviewer
    judge_reason: Optional[str]
    judge_confidence: Optional[str]
    error: Optional[str]
    trace_id: Optional[str]
    session_id: Optional[str]
    # RAG-1 — multi-pass retrieval state
    retrieval_iteration: int              # 0 = initial pass, 1 = re-retrieval pass (max 1)
    retrieval_feedback: Annotated[List[dict], operator.add]  # [{agent, refined_query}] from low-context agents
    retrieval_ambiguous: bool             # True when no entities extracted from query
    clarification_question: Optional[str] # set by node_router when query is too vague to answer
    re_retrieval_skipped: bool            # True when re-retrieval KB also returned empty — agent re-run skipped

# ==========================================
# ⚡ SSE STREAMING SHARED STATE
# ==========================================
ACTIVE_STREAMS = {}

# ==========================================
# 🛠️ TOOLS & LLM
# ==========================================
_KB_EMPTY_SENTINEL = "No specific knowledge found in graph."


@tool
def consult_medical_knowledge(query: str) -> str:
    """Consults the structured medical knowledge graph."""
    logger.info("consult_medical_knowledge invoked", query=query)
    if not medical_engine:
        return "Knowledge Engine Offline."
    results = medical_engine.search_and_reason(query)
    formatted = [f"- {r['name']} ({r['relation']}, Hop: {r.get('hop', '?')})" for r in results]
    return "\n".join(formatted) if formatted else _KB_EMPTY_SENTINEL


def _kb_context_is_empty(context_sections: list) -> bool:
    """Return True if every section in *context_sections* is a KB placeholder (no real data)."""
    empty_markers = (_KB_EMPTY_SENTINEL, "Knowledge Engine Offline.", "No specific medical knowledge concept found")
    for section in context_sections:
        if not any(marker in section for marker in empty_markers):
            return False
    return True

llm = None
extractor_llm = None  # Fast model for entity extraction (reuses Gemma 4 llm instance)

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
        "agent_outputs": [],
        "agent_sources": [],
    }

def _refine_kb_context(term: str, raw_facts: str) -> str:
    """Refine raw KB graph facts for a single entity into a structured clinical narrative."""
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
            refined = llm.invoke([
                SystemMessage(content=refinement_prompt.format(facts=raw_facts))
            ]).content.strip()
            logger.info("Context Refined", term=term)
            return f"[KB: {term}]\n{refined}"
        return f"[KB: {term}]\n{raw_facts}"
    except Exception as ref_err:
        logger.warning("Context refinement failed, using raw facts", error=str(ref_err))
        return f"[KB: {term}]\n{raw_facts}"


def node_retrieve_knowledge(state: AgentState):
    """
    RAG-1 Part A — Multi-entity KB retrieval.

    Extracts ALL distinct medical entities from the query (not just one) and
    issues a separate KB lookup for each. Also detects:
    - Vague queries (0 entities → retrieval_ambiguous=True, triggers clarification)
    - Topic-shift follow-ups (e.g. "what about side effects?" after a diabetes query)
      — injects the prior-turn entity as an extra lookup.
    """
    logger.info("NODE: RETRIEVE KNOWLEDGE")
    user_query = state['redacted_input']

    system_prompt = (
        "You are a medical entity extractor. "
        "Extract ALL distinct medical entities (diseases, symptoms, drugs, procedures) from the user's query. "
        "Return a JSON array of entity strings. Return [] if none found.\n\n"
        "RULE: Generic anatomical terms alone (heart, back, stomach, head, chest, leg, arm) "
        "with no qualifying condition are NOT medical entities — return [].\n\n"
        "Examples:\n"
        "User: 'interactions between metformin and lisinopril' -> [\"metformin\", \"lisinopril\"]\n"
        "User: 'symptoms of Heart Attack' -> [\"Heart Attack\"]\n"
        "User: 'Patient has high fever and diabetes' -> [\"Fever\", \"Diabetes\"]\n"
        "User: 'my heart feels weird' -> []\n"
        "User: 'my back hurts' -> []\n"
        "User: 'I feel sick' -> []\n"
        "User: 'something feels wrong' -> []\n"
        "User: 'my stomach' -> []\n"
        "User: 'I don't feel well' -> []\n"
        "User: 'heart failure symptoms' -> [\"Heart Failure\"]\n"
        "User: 'back pain disorder treatment' -> [\"Back Pain Disorder\"]"
    )

    # Generic body parts with no qualifying condition — not actionable medical entities.
    # Used as a post-extraction filter so GPT-4o-mini over-extraction doesn't bypass
    # the clarification branch (BUG-2).
    _GENERIC_ANATOMY = frozenset({
        "heart", "back", "stomach", "head", "chest", "leg", "arm", "neck",
        "shoulder", "knee", "hip", "foot", "hand", "eye", "ear", "nose",
        "throat", "belly", "abdomen", "spine", "skin", "body",
    })

    entities = []
    retrieval_ambiguous = False
    try:
        _extr = extractor_llm or llm
        response = _extr.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query)
        ]).content.strip()
        clean_response = response.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean_response)
        if isinstance(parsed, list):
            entities = [e for e in parsed if e and str(e).lower() not in ("none", "null")]
    except Exception as e:
        logger.error("Multi-entity extraction failed", error=str(e))

    # Post-extraction filter: if every extracted entity is a single generic anatomical
    # term (no qualifier like "failure", "cancer", "pain disorder"), treat as ambiguous.
    if entities and all(e.strip().lower() in _GENERIC_ANATOMY for e in entities):
        logger.info("All extracted entities are generic anatomy — treating as ambiguous", entities=entities)
        entities = []

    # Topic-shift detection: if 0 entities extracted but routing_context has prior agents,
    # check whether the query implies a follow-up (vague references like "side effects",
    # "what about", "tell me more") and inject the most recent KB entity from history.
    routing_context = state.get("routing_context") or ""
    if not entities and routing_context:
        followup_indicators = ("side effect", "what about", "tell me more", "more about",
                               "also", "and what", "interactions", "dosage", "risk")
        query_lower = user_query.lower()
        if any(indicator in query_lower for indicator in followup_indicators):
            # Extract most recent entity from prior KB context if available
            prior_context = state.get("context", [])
            for ctx in reversed(prior_context):
                import re as _re
                m = _re.search(r'\[KB:\s*([^\]]+)\]', ctx)
                if m:
                    entities = [m.group(1).strip()]
                    logger.info("Topic-shift detected, injecting prior entity", entity=entities[0])
                    break

    if not entities:
        retrieval_ambiguous = True
        logger.info("No medical entities found — query is ambiguous")
        return {
            "context": ["No specific medical knowledge concept found in query."],
            "retrieval_ambiguous": True,
            "retrieval_iteration": state.get("retrieval_iteration", 0),
            "retrieval_feedback": [],
        }

    context_sections = []
    session_id = state.get("session_id")
    for term in entities:
        logger.info("KB lookup", term=term)
        if session_id and session_id in ACTIVE_STREAMS:
            ACTIVE_STREAMS[session_id].append(f"Querying Knowledge Core: **{term}**")
        raw_facts = consult_medical_knowledge.invoke(term)
        context_sections.append(_refine_kb_context(term, raw_facts))

    return {
        "context": context_sections,
        "retrieval_ambiguous": retrieval_ambiguous,
        "retrieval_iteration": state.get("retrieval_iteration", 0),
        "retrieval_feedback": [],
    }


def node_retrieve_knowledge_v2(state: AgentState):
    """
    RAG-1 Part B — Reactive re-retrieval using agent-supplied refined queries.

    Called when at least one agent flagged low_context=True. Uses the refined_query
    from the first flagging agent instead of extracting entities from the original query.
    Appends new context sections without replacing existing ones.
    """
    logger.info("NODE: RETRIEVE KNOWLEDGE V2 (re-retrieval)")
    feedback = state.get("retrieval_feedback", [])
    if not feedback:
        return {"retrieval_iteration": state.get("retrieval_iteration", 0) + 1}

    new_sections = []
    seen_terms = set()
    session_id = state.get("session_id")
    for fb in feedback:
        term = fb.get("refined_query")
        if not term or term in seen_terms:
            continue
        seen_terms.add(term)
        logger.info("Re-retrieval KB lookup", term=term, agent=fb.get("agent"))
        if session_id and session_id in ACTIVE_STREAMS:
            ACTIVE_STREAMS[session_id].append(f"Re-querying Knowledge Core: **{term}**")
        raw_facts = consult_medical_knowledge.invoke(term)
        new_sections.append(_refine_kb_context(term, raw_facts))

    # If every re-retrieved section is also a placeholder, the KB genuinely has
    # no data for this query.  Skipping the agent re-run avoids passing a
    # "No specific knowledge found" context string into MedGemma's synthesis
    # prompt — which is the root cause of the repetition loop (BUG-1).
    if _kb_context_is_empty(new_sections):
        logger.warning(
            "Re-retrieval KB lookup also returned empty — skipping agent re-run",
            terms=list(seen_terms),
        )
        return {
            "retrieval_iteration": state.get("retrieval_iteration", 0) + 1,
            "re_retrieval_skipped": True,
        }

    # Append new context (preserves original context sections)
    existing_context = state.get("context", [])
    return {
        "context": existing_context + new_sections,
        "retrieval_iteration": state.get("retrieval_iteration", 0) + 1,
        "re_retrieval_skipped": False,
    }

def node_router(state: AgentState):
    logger.info("NODE: ROUTER")
    input_text = state['redacted_input']
    context_str = "\n".join(state.get("context", []))
    routing_context = state.get("routing_context") or ""

    system_prompt = (
        "You are the MediCortex Orchestrator. Your ONLY job is to select the best agent(s) to handle the user's query.\n\n"
        "═══ AGENT DECISION RULES ═══\n\n"
        "'diagnosis' — Use when the user describes symptoms, asks about a disease's symptoms, asks what disease they might have, "
        "or asks for a differential diagnosis. "
        "EXAMPLES: 'symptoms of diabetes', 'what causes chest pain', 'I have a headache and fever', 'differential for chest tightness'.\n\n"
        "'pharmacology' — Use when the user asks about drug treatments, medications for a condition, drug interactions, "
        "dosage, contraindications, or alternatives for a drug. "
        "EXAMPLES: 'treatment options for hypertension', 'what medications are used for T2D', "
        "'interactions of Metformin and Lisinopril', 'dosage for Ibuprofen', 'alternatives to Atorvastatin'.\n\n"
        "'pubmed' — Use when the user asks for research papers, clinical studies, literature reviews, or recent evidence on a topic. "
        "EXAMPLES: 'latest research on Alzheimer's', 'clinical trials for immunotherapy', 'evidence for statins'.\n\n"
        "'report_analyzer' — Use ONLY when the user provides or references a document, lab result, PDF, image, X-ray, MRI, or CT scan. "
        "EXAMPLES: 'analyze my lab report', 'what does this X-ray show', 'interpret my HbA1c result'.\n\n"
        "'patient' — Use ONLY when the user asks about a specific named or identified patient's records, history, medications, or vitals. "
        "EXAMPLES: 'show me John's records', 'what medications is patient PT-10042 on'.\n\n"
        "═══ CRITICAL RULES ═══\n"
        "- A query about symptoms only → ['diagnosis'] only\n"
        "- A query about treatment options or medications for a disease → ['diagnosis', 'pharmacology']\n"
        "- A query about both symptoms AND treatment → ['diagnosis', 'pharmacology']\n"
        "- A query about a named drug (interaction/dosage/alternatives) with no disease context → ['pharmacology'] only\n"
        "- A query about research/literature → ['pubmed'] only\n"
        "- NEVER route to 'pubmed' unless research papers are explicitly requested.\n\n"
        "═══ FOLLOW-UP QUERY HANDLING ═══\n"
        "If the current query uses ambiguous pronouns or implicit references such as 'his', 'her', 'their', "
        "'the patient', 'these medications', 'the drug', 'the condition', 'it', 'they' — resolve the "
        "reference using the Recent Session Context provided below.\n"
        "Resolution rules:\n"
        "- Previous turn used [patient] AND follow-up asks about medications/interactions/drugs → ['pharmacology']\n"
        "- Previous turn used [patient] AND follow-up asks about symptoms/conditions/diagnosis → ['diagnosis']\n"
        "- Previous turn used [pharmacology] AND follow-up asks about the same drug(s) → ['pharmacology']\n"
        "- Previous turn used [pubmed] AND follow-up asks for more research → ['pubmed']\n"
        "- Previous turn used [diagnosis] AND follow-up asks about drugs for that condition → ['pharmacology']\n\n"
        "Return ONLY a JSON array of agent keys. Examples: ['diagnosis'] or ['pharmacology'] or ['diagnosis', 'pharmacology'].\n"
        "Do NOT add any explanation. Only output the JSON array."
    )

    user_message = (
        f"User Query: {input_text}\n\n"
        + (f"Recent Session Context (use this to resolve follow-up references):\n{routing_context}\n\n"
           if routing_context else "")
        + f"Knowledge Core Context (for your awareness): {context_str[:300]}"
    )
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

    # Clarification check: if the KB retrieval found no entities (query is vague/ambiguous)
    # AND the previous turn was NOT already a clarification, ask the user to elaborate.
    # We suppress a second clarification by checking routing_context for "AI asked for clarification".
    already_clarified = "AI asked for clarification" in routing_context
    if state.get("retrieval_ambiguous") and not already_clarified:
        clarification_prompt = (
            "The user's medical query is ambiguous — no specific medical entities could be identified. "
            "Generate ONE short, empathetic clarifying question to ask the user so you can give a "
            "more accurate answer. Maximum 30 words. Examples:\n"
            "- 'Could you describe the sensation in more detail — is it a sharp pain, pressure, or palpitations?'\n"
            "- 'Are you asking about a specific condition or would you like general information?'\n"
            "- 'Which medication or condition are you referring to?'\n"
            "Output only the question, no preamble."
        )
        try:
            clarification_q = llm.invoke([
                SystemMessage(content=clarification_prompt),
                HumanMessage(content=f"User query: {input_text}"),
            ]).content.strip()
            logger.info("Clarification question generated", question=clarification_q)
            return {
                "messages": [AIMessage(content='["__clarify__"]')],
                "clarification_question": clarification_q,
            }
        except Exception as e:
            logger.warning("Clarification generation failed, proceeding with routing", error=str(e))

    return {"messages": [AIMessage(content=str(routes))]}

def make_agent_node(agent_key: str):
    def _node(state: AgentState, config: RunnableConfig):
        logger.info(f"NODE: AGENT [{agent_key.upper()}]")
        agent_executor = AGENT_REGISTRY.get(agent_key)
        if not agent_executor:
            return {"agent_outputs": [f"Error: Agent '{agent_key}' not found."]}
        
        # Strip KB placeholder sections before passing context to the agent.
        # MedGemma's synthesis prompt must never contain "No specific knowledge found"
        # strings — they cause it to fill the Clinical Profile template with a default
        # sentence that then repeats in a loop (BUG-1 root cause).
        _empty_markers = (_KB_EMPTY_SENTINEL, "Knowledge Engine Offline.", "No specific medical knowledge concept found")
        meaningful_sections = [
            s for s in state.get("context", [])
            if not any(marker in s for marker in _empty_markers)
        ]
        context_str = "\n".join(meaningful_sections)
        raw_history_str = "\n".join(state.get("history", []))

        # HIPAA: Re-redact conversation history before injecting into any LLM prompt.
        # Prior turns are stored in the DB after node_restore_privacy (real names present).
        # Uses redact_identifying_pii() (PERSON/PHONE/EMAIL/SSN only) — NOT full redact_pii()
        # — to avoid creating <DATE_TIME_N> placeholders that would flow into MedGemma and
        # appear unrestored in the final output (dates are clinically necessary in history).
        if raw_history_str:
            history_str = privacy_manager.redact_identifying_pii(raw_history_str)
        else:
            history_str = raw_history_str

        enhanced_input = (
            f"Conversation History:\n{history_str}\n\n"
            f"Current Request: {state['redacted_input']}"
            + (f"\n\nContext from Knowledge Core:\n{context_str}" if context_str else "")
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

            # Diagnosis: pass knowledge core context so analyze_symptoms tool can
            # inject it into its internal MedGemma call via tool_context injection.
            if agent_key == "diagnosis":
                envelope.payload["knowledge_context"] = context_str

            # Bind live thoughts list so the agent can stream thoughts in real-time
            session_id_str = state.get("session_id", "default")
            live_thoughts = ACTIVE_STREAMS.get(session_id_str, [])
            envelope.payload["live_thoughts_queue"] = live_thoughts
            
            # Call Agent via Process
            response = agent_executor.process(envelope)
            
            # Capture thinking steps (already prefixed by emit_thought in base.py)
            thoughts = response.thinking if response.thinking else []

            if response.error:
                logger.error(f"Agent {agent_key} returned error", error=response.error)
                return {
                    "agent_outputs": [f"## {agent_key.title()} Agent Error\n{response.error}"],
                    "agent_thoughts": thoughts,
                    "agents_used": [agent_key],
                }

            output = response.output if response.output else "No output generated."
            result: dict = {
                "agent_outputs": [f"## {agent_key.title()} Agent Response\n{output}"],
                "agent_thoughts": thoughts,
                "agents_used": [agent_key],
                "agent_sources": response.sources if response.sources else [],
            }

            # RAG-1 Part B: propagate low-context signal for reactive re-retrieval
            if response.low_context:
                logger.info(f"Agent {agent_key} flagged low_context", refined_query=response.refined_query)
                result["retrieval_feedback"] = [{
                    "agent": agent_key,
                    "refined_query": response.refined_query,
                }]

            return result
            
        except Exception as e:
            logger.error(f"Orchestrator failed to call agent {agent_key}", error=str(e))
            return {
                "agent_outputs": [f"## {agent_key.title()} Agent System Error\n{str(e)}"],
                "agent_thoughts": [f"**[{agent_key.title()}]**: System Error: {str(e)}"],
                "agents_used": [agent_key],
            }
    return _node

node_pubmed = make_agent_node("pubmed")
node_diagnosis = make_agent_node("diagnosis")
node_report_analyzer = make_agent_node("report_analyzer")
node_patient = make_agent_node("patient")
node_pharmacology = make_agent_node("pharmacology")


def node_ask_clarification(state: AgentState):
    """
    User Clarification node — early-exit path for ambiguous queries.

    Emits the clarification question as the final_output and routes to
    restore_privacy → END, bypassing agents, aggregator, and reviewer.
    The question is a normal assistant message; the user replies in the next
    turn, and the pipeline picks up context from DB history naturally.
    """
    logger.info("NODE: ASK CLARIFICATION")
    question = state.get("clarification_question") or "Could you provide more details about your question?"
    return {
        "final_output": question,
        "agents_used": ["clarification"],
    }


def node_aggregator_with_reretrieval(state: AgentState):
    """
    RAG-1 Part B — Reactive re-retrieval wrapper around node_aggregator.

    If any agents flagged low_context=True and we haven't re-retrieved yet,
    perform an inline re-retrieval pass and re-run the flagging agents before
    aggregating. This avoids complex LangGraph fan-out rewiring.
    """
    logger.info("NODE: AGGREGATOR")

    # RAG-1 Part B: check for re-retrieval before aggregating
    feedback = state.get("retrieval_feedback", [])
    iteration = state.get("retrieval_iteration", 0)

    extra_outputs: list = []
    if feedback and iteration < 1:
        logger.info("Reactive re-retrieval triggered", feedback=feedback)
        reretrieval_result = node_retrieve_knowledge_v2(state)

        # If re-retrieval found no new KB data, skip the agent re-run entirely.
        # Passing an empty KB context string into MedGemma's synthesis prompt
        # is the root cause of the repetition loop (BUG-1): MedGemma fills its
        # Clinical Profile template with a default sentence and repeats it.
        if reretrieval_result.get("re_retrieval_skipped"):
            logger.info("Agent re-run skipped — re-retrieval returned no new KB data")
        else:
            new_context = reretrieval_result.get("context", state.get("context", []))

            # Strip placeholder sections so MedGemma never receives "No specific
            # knowledge found in graph." as part of the synthesis context.
            empty_markers = (_KB_EMPTY_SENTINEL, "Knowledge Engine Offline.", "No specific medical knowledge concept found")
            meaningful_context = [
                s for s in new_context
                if not any(marker in s for marker in empty_markers)
            ]

            re_run_keys = list({fb["agent"] for fb in feedback if fb.get("agent") in AGENT_REGISTRY})
            for agent_key in re_run_keys:
                agent_executor = AGENT_REGISTRY.get(agent_key)
                if not agent_executor:
                    continue
                context_str = "\n".join(meaningful_context)
                raw_history_str = "\n".join(state.get("history", []))
                history_str = privacy_manager.redact_identifying_pii(raw_history_str) if raw_history_str else ""
                enhanced_input = (
                    f"Conversation History:\n{history_str}\n\n"
                    f"Current Request: {state['redacted_input']}"
                    + (f"\n\nContext from Knowledge Core (enriched):\n{context_str}" if context_str else "")
                )
                envelope = Envelope(
                    trace_id=state.get("trace_id", ""),
                    sender_id="orchestrator",
                    receiver_id=agent_key,
                    payload={"input": enhanced_input},
                )
                if agent_key == "patient":
                    import json as _json
                    envelope.payload["pii_mapping_json"] = _json.dumps(state.get("pii_mapping", {}))
                if agent_key == "diagnosis":
                    envelope.payload["knowledge_context"] = context_str
                session_id_str = state.get("session_id", "default")
                envelope.payload["live_thoughts_queue"] = ACTIVE_STREAMS.get(session_id_str, [])
                try:
                    response = agent_executor.process(envelope)
                    if response.output and not response.error:
                        extra_outputs.append(
                            f"## {agent_key.title()} Agent Response (re-retrieved)\n{response.output}"
                        )
                        logger.info(f"Re-run agent {agent_key} succeeded after re-retrieval")
                except Exception as e:
                    logger.warning(f"Re-run agent {agent_key} failed after re-retrieval", error=str(e))

    # Proceed with standard aggregation (original outputs + any re-retrieved outputs)
    raw_outputs = "\n\n".join(state["agent_outputs"] + extra_outputs)

    formatting_prompt = (
        "You are the MediCortex Interface. Format the following medical agent reports into "
        "a beautiful, human-readable Markdown response.\n"
        "Use bolding, italics, bullet points, and headers to make it easy to read.\n"
        "DEDUPLICATION RULES (apply before formatting):\n"
        "1. If multiple agents recommend the same specialist referral or action, merge them into a single entry — do not repeat the same recommendation under different headings.\n"
        "2. If multiple source snippets convey the same fact (e.g. the same sentence from the same or different sources), keep only the first occurrence and drop all subsequent duplicates.\n"
        "3. Near-identical recommendations that differ only in minor wording should be consolidated into one.\n"
        "Do not change any factual content beyond deduplication.\n\n"
        "HEADING RULES:\n"
        "- Do NOT open the response with a generic heading like 'Medical Agent Reports', "
        "'Medical Agent Reports on [Topic]', 'Medical Agent Reports Summary', or any similar variation.\n"
        "- If a heading is needed, use a concise topic-specific heading (e.g. 'Hypertension: Symptoms & First-Line Treatment'). "
        "For shorter responses, omit the heading entirely.\n"
        "- The response must read as expert clinical guidance, not as an internal pipeline report.\n\n"
        "CITATION RULES (only apply if the raw reports contain URLs):\n"
        "- If the raw reports contain source URLs, add inline citation numbers like [1], [2] at the end of each sentence or claim that is supported by a source.\n"
        "- Collect all unique cited URLs and append a '## References' section at the very end of the response, formatted as a numbered markdown list: '1. [Title](url)'\n"
        "- Each number must correspond to exactly one unique URL. Do not assign the same number to two different URLs.\n"
        "- If the raw reports contain NO URLs at all, do NOT add a References section and do NOT add any inline citation numbers.\n\n"
        f"Raw Reports:\n{raw_outputs}"
    )

    try:
        formatted = llm.invoke([HumanMessage(content=formatting_prompt)]).content
    except Exception:
        formatted = raw_outputs

    return {"final_output": formatted}


def _parse_references(text: str) -> tuple[str, list[dict]]:
    """
    Split the aggregator's final_output into (body, sources).

    Looks for a '## References' section at the end of the response and extracts
    numbered markdown links from it. Returns the body without the References section
    and a list of {title, url} dicts. If no References section is present (i.e. the
    response had no source URLs), returns the original text and an empty list.
    """
    marker = "\n## References"
    idx = text.find(marker)
    if idx == -1:
        return text, []
    body = text[:idx].rstrip()
    ref_block = text[idx + len(marker):]
    sources = []
    for m in re.finditer(r'\d+\.\s+\[([^\]]+)\]\((https?://[^\s\)]+)\)', ref_block):
        sources.append({"title": m.group(1), "url": m.group(2)})
    # If the section existed but GPT produced no parseable links, return body only
    return body, sources


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
- Does the response contain leaked PII placeholders (e.g. <PERSON_1>)? NOTE: placeholders like <PERSON_1> are intentional de-identification tokens used by the privacy layer and are NOT a quality defect — do NOT penalise the score for their presence.
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
    last_msg = state["messages"][-1].content

    # Clarification early-exit: router set this when query was too vague
    try:
        parsed = json.loads(last_msg.replace("'", '"'))
        if isinstance(parsed, list) and parsed == ["__clarify__"]:
            return ["__clarify__"]
    except Exception:
        pass

    routes = []

    # Always route to report_analyzer if files were attached (A2A §4.1)
    if state.get("file_urls"):
        routes.append("report_analyzer")

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

orchestrator_graph = None

# ==========================================
# 🗺️ ROUTING CONTEXT HELPER
# ==========================================
def _build_routing_context(past_turns) -> str:
    """
    Build a compact, HIPAA-safe routing summary from the last 3 user/assistant
    message pairs so that node_router can resolve follow-up pronouns
    (e.g. "his medications", "the patient") without seeing raw PII.

    - User queries are run through redact_identifying_pii() — names stripped,
      dates/drugs/conditions preserved so the router has useful signal.
    - agents_used is read from message_metadata (populated by make_agent_node).

    Example output:
        User asked: What medications is <PERSON_1> currently taking?
        Routed to: [patient]
        User asked: Are there dangerous interactions between his medications?
        Routed to: [pharmacology]
    """
    lines = []
    for msg in past_turns[-6:]:   # up to 3 user/assistant pairs
        if msg.role == "user":
            redacted_q = privacy_manager.redact_identifying_pii(msg.content[:150])
            lines.append(f"User asked: {redacted_q}")
        elif msg.role == "assistant":
            meta = msg.message_metadata or {}
            if meta.get("is_clarification"):
                lines.append("AI asked for clarification")
            else:
                agents = meta.get("agents_used", [])
                agents_str = ", ".join(agents) if agents else "unknown"
                lines.append(f"Routed to: [{agents_str}]")
    return "\n".join(lines)

# ==========================================
# 🌐 FASTAPI SERVER
# ==========================================
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    global medical_engine, privacy_manager, llm, extractor_llm, orchestrator_graph

    # ── 1. MedicalReasoningEngine ──────────────────────────────────────
    logger.info("Importing Local Engines...")
    if _medical_engine_available:
        try:
            logger.info("Initializing MedicalReasoningEngine...")
            medical_engine = MedicalReasoningEngine()
            logger.info("MedicalReasoningEngine connected", status="success")
        except Exception as e:
            logger.warning("MedicalReasoningEngine unavailable, knowledge retrieval disabled", error=str(e))
            medical_engine = None

    # ── 2. HIPAA Privacy Layer ─────────────────────────────────────────
    logger.info("Instantiating PrivacyManager Singleton")
    privacy_manager = PrivacyManager()

    # ── 3. Gemma 4 via Ollama Cloud (Router / Aggregator / Extractor) ────
    # Replaces GPT-4o-mini everywhere. A warmup call is issued at startup
    # so the first user request is served from a hot model.
    try:
        _ollama_base = settings.OLLAMA_CLOUD_URL.removesuffix("/v1")
        logger.info("Initializing Gemma 4 LLM Client (ChatOllama)", model=settings.OLLAMA_CLOUD_MODEL, base_url=_ollama_base)
        llm = ChatOllama(
            model=settings.OLLAMA_CLOUD_MODEL,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            base_url=_ollama_base,
        )
        logger.info("Gemma 4 Client Ready — warming up model (first request may load model)")
        # Warmup call: fires asynchronously so startup doesn't block.
        # Runs as a background asyncio task — completes before first user message arrives
        # if startup took ≥ 5 minutes (typical Ollama Cloud cold-start for 31B).
        import asyncio as _asyncio
        from langchain_core.messages import HumanMessage as _WarmupMsg
        async def _warmup():
            try:
                await _asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: llm.invoke([_WarmupMsg(content="ping")])
                )
                logger.info("Gemma 4 warmup complete — model is hot")
            except Exception as _we:
                logger.warning("Gemma 4 warmup failed (model may still be loading)", error=str(_we))
        _asyncio.create_task(_warmup())
        logger.info("Gemma 4 LLM Ready (ChatOllama)", model=settings.OLLAMA_CLOUD_MODEL, base_url=_ollama_base, status="success")
    except Exception as e:
        logger.error("Gemma 4 Ollama Cloud setup failed", error=str(e))
        llm = None

    # extractor_llm reuses the same Gemma 4 instance for entity extraction
    extractor_llm = llm

    # ── 4. LangGraph Workflow ──────────────────────────────────────────
    workflow = StateGraph(AgentState)
    workflow.add_node("analyze_privacy", node_analyze_privacy)
    workflow.add_node("retrieve_knowledge", node_retrieve_knowledge)
    workflow.add_node("router", node_router)
    workflow.add_node("pubmed", node_pubmed)
    workflow.add_node("diagnosis", node_diagnosis)
    workflow.add_node("report_analyzer", node_report_analyzer)
    workflow.add_node("patient", node_patient)
    workflow.add_node("pharmacology", node_pharmacology)
    workflow.add_node("ask_clarification", node_ask_clarification)        # clarification early-exit
    workflow.add_node("aggregator", node_aggregator_with_reretrieval)     # RAG-1 Part B inline
    workflow.add_node("reviewer", node_reviewer)       # A2A §5.2 — Model-as-Judge
    workflow.add_node("restore_privacy", node_restore_privacy)
    workflow.set_entry_point("analyze_privacy")
    workflow.add_edge("analyze_privacy", "retrieve_knowledge")
    workflow.add_edge("retrieve_knowledge", "router")
    workflow.add_conditional_edges("router", route_decision, {
        **{k: k for k in AGENT_REGISTRY.keys()},
        "__clarify__": "ask_clarification",
    })
    for agent_key in AGENT_REGISTRY.keys():
        workflow.add_edge(agent_key, "aggregator")
    workflow.add_edge("ask_clarification", "restore_privacy")  # skip aggregator/reviewer
    workflow.add_edge("aggregator", "reviewer")
    workflow.add_edge("reviewer", "restore_privacy")
    workflow.add_edge("restore_privacy", END)
    orchestrator_graph = workflow.compile()
    logger.info("Orchestrator Graph Compiled", status="success")

    # ── Ready ──────────────────────────────────────────────────────────
    logger.info("Starting Orchestrator Server", app_name=settings.APP_NAME)
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

            # Build routing context from past turns so the router can resolve
            # follow-up references ("his medications", "the patient", etc.)
            routing_context = _build_routing_context(past_turns)

            # 5. Stream Orchestrator Events
            agent_thoughts = []
            final_output = ""
            msg_metadata = {
                "llm_used": "MedGemma (via HF) / Gemma 4 Router",
                "judge_score": None,
                "judge_reason": None,
                "judge_confidence": None,
                "agents_used": [],
            }



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
                            "routing_context": routing_context,
                            "agent_thoughts": [],
                            "agents_used": [],
                            "file_urls": file_urls,
                            "session_id": str(session_id),
                            # RAG-1 initial state
                            "retrieval_iteration": 0,
                            "retrieval_feedback": [],
                            "retrieval_ambiguous": False,
                            "clarification_question": None,
                            "re_retrieval_skipped": False,
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

            # Capture which agents ran so future turns can use this for routing
            msg_metadata["agents_used"] = list(dict.fromkeys(graph_output.get("agents_used", [])))

            # RAG-1 auditability fields
            msg_metadata["retrieval_iterations"] = graph_output.get("retrieval_iteration", 0)
            msg_metadata["retrieval_feedback"] = graph_output.get("retrieval_feedback", [])
            msg_metadata["retrieval_ambiguous"] = graph_output.get("retrieval_ambiguous", False)
            msg_metadata["is_clarification"] = bool(graph_output.get("clarification_question"))

            # Extract references section (only present when agents cited source URLs inline)
            # and merge with tool-observation URLs collected during agent ReAct loops.
            tool_sources: List[dict] = graph_output.get("agent_sources", [])
            if graph_final:
                clean_body, inline_sources = _parse_references(graph_final)
                final_output = clean_body
                # Merge inline citations + tool observation URLs, dedup by URL
                def _norm_url(u: str) -> str:
                    return u.rstrip("/").lower()

                all_sources = inline_sources[:]
                seen = {_norm_url(s["url"]) for s in inline_sources}
                for s in tool_sources:
                    if _norm_url(s["url"]) not in seen:
                        seen.add(_norm_url(s["url"]))
                        all_sources.append(s)
                if all_sources:
                    msg_metadata["sources"] = all_sources

            yield f"data: {json.dumps({'type': 'metadata', 'content': msg_metadata})}\n\n"

            if final_output:
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

        # Build routing context from past turns so the router can resolve follow-ups
        routing_context = _build_routing_context(past_turns)

        # 4. Invoke Orchestrator
        result = await orchestrator_graph.ainvoke({
            "input": request.message,
            "messages": [],
            "history": history_context,
            "routing_context": routing_context,
            "agent_thoughts": [],
            "agents_used": [],
            "file_urls": file_urls,
            "session_id": str(session_id),
            "retrieval_iteration": 0,
            "retrieval_feedback": [],
            "retrieval_ambiguous": False,
            "clarification_question": None,
            "re_retrieval_skipped": False,
        })
        response_text = result.get("final_output")
        agent_thinking = result.get("agent_thoughts", [])

        msg_metadata = {
            "llm_used": "MedGemma (via HF) / Gemma 4 Router",
            "judge_score": result.get("judge_score"),
            "judge_reason": result.get("judge_reason"),
            "judge_confidence": result.get("judge_confidence"),
            "agents_used": list(dict.fromkeys(result.get("agents_used", []))),
            "retrieval_iterations": result.get("retrieval_iteration", 0),
            "retrieval_feedback": result.get("retrieval_feedback", []),
            "retrieval_ambiguous": result.get("retrieval_ambiguous", False),
            "is_clarification": bool(result.get("clarification_question")),
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
    dev = "--dev" in sys.argv or settings.DEBUG
    logger.info("Starting Orchestrator Server manually", port=8001, reload=dev)
    uvicorn.run(
        "orchestrator:app",
        host="0.0.0.0",
        port=8001,
        reload=dev,
    )
