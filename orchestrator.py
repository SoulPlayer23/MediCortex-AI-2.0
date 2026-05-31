
import os
import sys
import re
import ast
import random
import asyncio
import structlog
from typing import Any, Dict, TypedDict, List, Optional, Tuple, Annotated
import operator
import json
import uvicorn
import requests
import httpx
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text as sa_text
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
    # OBS-1 — per-node latency map (ms); reducer merges dicts across all nodes
    node_timings: Annotated[Dict[str, float], lambda a, b: {**(a or {}), **(b or {})}]
    # OBS-1 — retrieval stats populated by node_retrieve_knowledge
    retrieval_stats: Optional[Dict]
    # OBS-1 — judge token usage from node_reviewer
    judge_token_usage: Optional[Dict]

# ==========================================
# ⚡ SSE STREAMING SHARED STATE
# ==========================================

class RedisThoughtQueue:
    """OPS-3: Multi-worker-safe thought queue backed by a Redis list.

    Exposes the same list-like interface (append / len / getitem) used by
    the SSE poller and KB nodes so no call-site changes are needed.
    Falls back to an in-process list when Redis is unavailable, preserving
    the existing single-worker behaviour.

    Redis key: streams:{session_id}  (TTL = 3600s)
    """

    _TTL = 3600

    def __init__(self, session_id: str, redis_client=None):
        self._session_id = session_id
        self._key = f"streams:{session_id}"
        self._redis = redis_client
        self._fallback: list = []

    # ── Write (called from agent thread-pool — sync is fine) ──────────
    def append(self, thought: str) -> None:
        if self._redis:
            try:
                self._redis.rpush(self._key, thought)
                self._redis.expire(self._key, self._TTL)
                return
            except Exception:
                pass
        self._fallback.append(thought)

    # ── Read (called from async SSE poller via len/index) ─────────────
    def __len__(self) -> int:
        if self._redis:
            try:
                return int(self._redis.llen(self._key) or 0)
            except Exception:
                pass
        return len(self._fallback)

    def __getitem__(self, idx: int) -> str:
        if self._redis:
            try:
                val = self._redis.lindex(self._key, idx)
                if val is not None:
                    return val
            except Exception:
                pass
        return self._fallback[idx]

    # ── Cleanup ───────────────────────────────────────────────────────
    def delete(self) -> None:
        if self._redis:
            try:
                self._redis.delete(self._key)
                return
            except Exception:
                pass
        self._fallback.clear()

    # ── Snapshot (for DB persistence) ─────────────────────────────────
    def snapshot(self) -> list:
        if self._redis:
            try:
                items = self._redis.lrange(self._key, 0, -1)
                if items is not None:
                    return list(items)
            except Exception:
                pass
        return list(self._fallback)


def _make_thought_queue(session_id: str) -> RedisThoughtQueue:
    """Create a RedisThoughtQueue, sharing the orchestrator's Redis connection."""
    try:
        if getattr(settings, "REDIS_URL", None):
            import redis as _redis_mod
            socket_timeout = getattr(settings, "REDIS_SOCKET_TIMEOUT", 2)
            rc = _redis_mod.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_timeout,
                retry_on_timeout=False,
                retry_on_error=[],
            )
            rc.ping()
            return RedisThoughtQueue(session_id, rc)
    except Exception:
        pass
    return RedisThoughtQueue(session_id, None)


# ACTIVE_STREAMS: maps session_id → RedisThoughtQueue (or plain list fallback).
# OPS-3: each queue is now multi-worker-safe via Redis when available.
ACTIVE_STREAMS: dict = {}

# OPS-8: track last real request time for activity-aware keepwarm.
_last_request_at: float = 0.0

# ==========================================
# 🛠️ TOOLS & LLM
# ==========================================
_KB_EMPTY_SENTINEL = "No specific knowledge found in graph."

import time as _time

def _model_label(llm_obj) -> str:
    """Return a short human-readable model name from any LangChain LLM object."""
    for attr in ("model", "model_name"):
        val = getattr(llm_obj, attr, None)
        if val:
            return str(val)
    return type(llm_obj).__name__

async def llm_ainvoke(llm_obj, messages, *, role: str = "") -> any:
    """Async LLM call with structured audit log: model name + round-trip time."""
    model = _model_label(llm_obj)
    t0 = _time.monotonic()
    result = await llm_obj.ainvoke(messages)
    rtt_ms = round((_time.monotonic() - t0) * 1000)
    logger.info("llm_call", model=model, role=role, rtt_ms=rtt_ms)
    return result

def llm_invoke(llm_obj, messages, *, role: str = "") -> any:
    """Sync LLM call with structured audit log: model name + round-trip time."""
    model = _model_label(llm_obj)
    t0 = _time.monotonic()
    result = llm_obj.invoke(messages)
    rtt_ms = round((_time.monotonic() - t0) * 1000)
    logger.info("llm_call", model=model, role=role, rtt_ms=rtt_ms)
    return result


@tool
async def consult_medical_knowledge(query: str) -> str:
    """Consults the structured medical knowledge graph."""
    logger.info("consult_medical_knowledge invoked", query=query)
    if not medical_engine:
        return "Knowledge Engine Offline."
    results = await medical_engine.search_and_reason(query)
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
extractor_llm = None  # Fast model for entity extraction (reuses Ollama llm instance)

# ==========================================
# 🕸️ LANGGRAPH NODES
# ==========================================
async def node_scope_guard(state: AgentState):
    """
    First node — rejects non-medical queries before anything else runs.

    Uses Groq llama-3.3-70b (same model as the reviewer judge) for a reliable
    1/0 classification. On out-of-scope, sets final_output so the conditional
    edge skips directly to restore_privacy → END without touching KB or agents.
    Falls back to in-scope if GROQ_API_KEY is not set or the call fails.
    """
    _t0_node = _time.monotonic()
    logger.info("NODE: SCOPE GUARD")
    query = state.get("input", "")

    if not settings.GROQ_API_KEY:
        logger.warning("Scope guard skipped — GROQ_API_KEY not set")
        return {}

    prompt = (
        "You are a medical AI scope filter. Reply with ONLY '1' (in scope) or '0' (out of scope). "
        "No explanation, no punctuation — a single digit.\n\n"
        "IN SCOPE: medicine, diseases, symptoms, drugs, pharmacology, anatomy, physiology, biology, "
        "lab results, medical procedures, patient care, mental health, genetics, nutrition related "
        "to health, public health, veterinary medicine.\n\n"
        "OUT OF SCOPE: cooking, travel, sports, politics, programming, history, entertainment, "
        "celebrity news, general trivia, math problems, creative writing, anything unrelated to "
        "health or biology.\n\n"
        "Reply: 1 or 0"
    )

    in_scope = True
    try:
        scope_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=settings.GROQ_API_KEY, temperature=0)
        raw = (await llm_ainvoke(scope_llm, [
            SystemMessage(content=prompt),
            HumanMessage(content=query),
        ], role="scope_guard")).content.strip()
        in_scope = not raw.startswith("0")
        logger.info("Scope guard result", raw=raw, in_scope=in_scope, query=query[:80])
    except Exception as e:
        logger.warning("Scope guard LLM call failed — defaulting to in-scope", error=str(e))

    logger.info("node_elapsed_ms", node="scope_guard", elapsed_ms=round((_time.monotonic() - _t0_node) * 1000))
    if not in_scope:
        return {
            "final_output": (
                "I'm MediCortex, a medical AI assistant. I can only help with medical, "
                "clinical, or biological questions. Please ask me something health-related."
            ),
            "agents_used": ["scope_guard"],
            "agent_outputs": [],
            "agent_thoughts": [],
            "agent_sources": [],
            "retrieval_ambiguous": False,
            "retrieval_iteration": state.get("retrieval_iteration", 0),
            "retrieval_feedback": [],
            "re_retrieval_skipped": False,
        }

    return {}


async def node_analyze_privacy(state: AgentState):
    _t0_node = _time.monotonic()
    import uuid as _uuid
    trace_id = state.get("trace_id") or str(_uuid.uuid4())
    structlog.contextvars.bind_contextvars(trace_id=trace_id)
    logger.info("NODE: ANALYZE PRIVACY", trace_id=trace_id)
    loop = asyncio.get_event_loop()
    redacted, mapping = await loop.run_in_executor(None, privacy_manager.redact_pii, state['input'])
    logger.info("node_elapsed_ms", node="analyze_privacy", elapsed_ms=round((_time.monotonic() - _t0_node) * 1000))
    return {
        "trace_id": trace_id,
        "redacted_input": redacted,
        "pii_mapping": mapping,
        "messages": [HumanMessage(content=redacted)],
        "agent_outputs": [],
        "agent_sources": [],
    }

async def _refine_kb_context(term: str, raw_facts: str) -> str:
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
            refined = (await llm_ainvoke(llm, [
                SystemMessage(content=refinement_prompt.format(facts=raw_facts))
            ], role="kb_refine")).content.strip()
            logger.info("Context Refined", term=term)
            return f"[KB: {term}]\n{refined}"
        return f"[KB: {term}]\n{raw_facts}"
    except Exception as ref_err:
        logger.warning("Context refinement failed, using raw facts", error=str(ref_err))
        return f"[KB: {term}]\n{raw_facts}"


async def node_retrieve_knowledge(state: AgentState):
    """
    RAG-1 Part A — Multi-entity KB retrieval.

    Extracts ALL distinct medical entities from the query (not just one) and
    issues a separate KB lookup for each. Also detects:
    - Vague queries (0 entities → retrieval_ambiguous=True, triggers clarification)
    - Topic-shift follow-ups (e.g. "what about side effects?" after a diabetes query)
      — injects the prior-turn entity as an extra lookup.

    All query expansion LLM calls and all ArangoDB lookups run in parallel via
    asyncio.gather, cutting retrieval from ~3–5s serial to ~400–600ms.
    """
    _t0_node = _time.monotonic()
    logger.info("NODE: RETRIEVE KNOWLEDGE")
    user_query = state['redacted_input']

    system_prompt = (
        "You are a medical entity extractor. "
        "Your ONLY job is to identify the medical CONDITIONS, DRUGS, or PROCEDURES being asked about — NOT to list their symptoms or effects.\n\n"
        "Return a JSON array of entity strings and NOTHING else. No explanation, no prose, no markdown.\n\n"
        "CRITICAL RULE: Extract the SUBJECT of the query, not the content.\n"
        "  'symptoms of diabetes' → [\"Diabetes\"]  (the subject is diabetes, not the symptoms)\n"
        "  'causes of hypertension' → [\"Hypertension\"]  (the subject is hypertension)\n"
        "  'what does metformin treat' → [\"Metformin\"]\n\n"
        "RULE: Generic anatomical terms alone (heart, back, stomach, head, chest, leg, arm) "
        "with no qualifying condition are NOT entities — return [].\n\n"
        "More examples:\n"
        "  'interactions between metformin and lisinopril' → [\"metformin\", \"lisinopril\"]\n"
        "  'symptoms of Heart Attack' → [\"Heart Attack\"]\n"
        "  'Patient has high fever and diabetes' → [\"Fever\", \"Diabetes\"]\n"
        "  'heart failure treatment options' → [\"Heart Failure\"]\n"
        "  'back pain disorder treatment' → [\"Back Pain Disorder\"]\n"
        "  'my heart feels weird' → []\n"
        "  'I feel sick' → []\n\n"
        "Output format — ONLY this, nothing else:\n"
        "[\"Entity1\", \"Entity2\"]"
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
        response = (await llm_ainvoke(_extr, [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query)
        ], role="entity_extract")).content.strip()
        clean_response = response.replace("```json", "").replace("```", "").strip()
        if not clean_response:
            logger.warning("Entity extractor returned empty response — treating as no entities")
            parsed = []
        else:
            try:
                parsed = json.loads(clean_response)
            except json.JSONDecodeError:
                # model sometimes wraps the array in prose — extract the first [...] block
                import re as _re_json
                m = _re_json.search(r'\[.*?\]', clean_response, _re_json.DOTALL)
                parsed = json.loads(m.group()) if m else []
                if parsed:
                    logger.warning("Entity extractor returned prose — recovered JSON array via regex")
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
                               "also", "and what", "interactions", "dosage", "risk",
                               "medication", "medications", "treatment")
        query_lower = user_query.lower()
        if any(indicator in query_lower for indicator in followup_indicators):
            import re as _re
            for line in reversed(routing_context.splitlines()):
                if line.startswith("User asked:"):
                    query_text = line[len("User asked:"):].strip()
                    _STOP = {"what", "are", "the", "a", "an", "is", "tell", "me",
                             "about", "my", "i", "do", "does", "how", "for", "of",
                             "in", "on", "m", "s", "t"}
                    _PRESIDIO = {"PERSON", "LOCATION", "DATE_TIME", "NRP", "ORG",
                                 "PHONE_NUMBER", "EMAIL_ADDRESS", "IP_ADDRESS"}
                    words = [w for w in _re.findall(r"[a-zA-Z]+", query_text)
                             if w not in _PRESIDIO and w.lower() not in _STOP]
                    if words:
                        entities = [" ".join(words[:3])]
                        logger.info("Topic-shift detected, injecting entity from routing_context",
                                    entity=entities[0])
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

    # Query expansion: single batched LLM call for all entities at once.
    # Returns a JSON object {entity: [synonym, ...]} — one round-trip regardless of entity count.
    _MAX_EXPANDED_TERMS = 10
    _extr = extractor_llm or llm

    async def _expand_entities_batch(entity_list: list[str]) -> dict[str, list[str]]:
        prompt = (
            "You are a medical terminology expert.\n"
            "Return a SINGLE JSON object where each key is one of the given entities and its value is "
            "an array of exactly 3 synonyms/abbreviations/related terms from a medical knowledge graph.\n"
            "CRITICAL: Output ONE JSON object only — not multiple objects, not prose, not code fences.\n\n"
            "Output format (ONLY this):\n"
            "{\"Entity1\": [\"syn1\", \"syn2\", \"syn3\"], \"Entity2\": [\"syn1\", \"syn2\", \"syn3\"]}\n\n"
            "Example for [\"metformin\", \"heart failure\"]:\n"
            "{\"metformin\": [\"biguanide\", \"Glucophage\", \"oral hypoglycemic\"], "
            "\"heart failure\": [\"CHF\", \"cardiac failure\", \"cardiomyopathy\"]}\n\n"
            f"Entities: {json.dumps(entity_list)}"
        )
        try:
            raw = (await llm_ainvoke(_extr, [HumanMessage(content=prompt)], role="entity_expand")).content.strip()
            clean = raw.replace("```json", "").replace("```", "").strip()
            try:
                parsed = json.loads(clean)
            except json.JSONDecodeError:
                # model sometimes emits one JSON object per line — merge them
                import re as _re_exp
                merged: dict = {}
                for m in _re_exp.finditer(r'\{[^{}]+\}', clean, _re_exp.DOTALL):
                    try:
                        merged.update(json.loads(m.group()))
                    except json.JSONDecodeError:
                        pass
                parsed = merged if merged else {}
                if parsed:
                    logger.warning("Expansion returned split objects — merged via regex")
            if isinstance(parsed, dict):
                return parsed
        except Exception as exp_err:
            logger.warning("Batched entity expansion failed", error=str(exp_err))
        return {}

    expansion_map = await _expand_entities_batch(entities)
    expansion_results = [
        [e] + [s for s in expansion_map.get(e, []) if isinstance(s, str)]
        for e in entities
    ]

    seen_for_expansion: set[str] = set()
    expanded_terms: list[str] = []
    for terms in expansion_results:
        for t in terms:
            if t.lower() not in seen_for_expansion:
                seen_for_expansion.add(t.lower())
                expanded_terms.append(t)
                if len(expanded_terms) >= _MAX_EXPANDED_TERMS:
                    break
        if len(expanded_terms) >= _MAX_EXPANDED_TERMS:
            break

    logger.info("KB query expansion", original=entities, expanded=expanded_terms)

    session_id = state.get("session_id")
    if session_id and session_id in ACTIVE_STREAMS:
        ACTIVE_STREAMS[session_id].append(
            f"Querying Knowledge Core: **{', '.join(entities)}** (+{len(expanded_terms) - len(entities)} expanded terms)"
        )

    async def _lookup_and_refine(term: str) -> str | None:
        logger.info("KB lookup", term=term)
        raw_facts = await consult_medical_knowledge.ainvoke(term)
        section = await _refine_kb_context(term, raw_facts)
        return section if not _kb_context_is_empty([section]) else None

    # Fire all KB lookups + refinements in parallel
    sections = await asyncio.gather(*[_lookup_and_refine(t) for t in expanded_terms])
    context_sections = [s for s in sections if s is not None]

    if not context_sections:
        context_sections = [f"[KB: {', '.join(entities)}]\n{_KB_EMPTY_SENTINEL}"]

    _elapsed_retrieve = round((_time.monotonic() - _t0_node) * 1000)
    logger.info("node_elapsed_ms", node="retrieve_knowledge", elapsed_ms=_elapsed_retrieve)
    return {
        "context": context_sections,
        "retrieval_ambiguous": retrieval_ambiguous,
        "retrieval_iteration": state.get("retrieval_iteration", 0),
        "retrieval_feedback": [],
        "retrieval_stats": {
            "entities_extracted": entities,
            "raw_terms_count": len(expanded_terms),
            "kb_available": bool(medical_engine),
        },
        "node_timings": {"retrieve_knowledge": _elapsed_retrieve},
    }


async def node_retrieve_knowledge_v2(state: AgentState):
    """
    RAG-1 Part B — Reactive re-retrieval using agent-supplied refined queries.

    Called when at least one agent flagged low_context=True. Uses the refined_query
    from the first flagging agent instead of extracting entities from the original query.
    Appends new context sections without replacing existing ones.

    All expansion LLM calls and KB lookups run in parallel via asyncio.gather.
    """
    logger.info("NODE: RETRIEVE KNOWLEDGE V2 (re-retrieval)")
    feedback = state.get("retrieval_feedback", [])
    if not feedback:
        return {"retrieval_iteration": state.get("retrieval_iteration", 0) + 1}

    seen_terms: set[str] = set()
    session_id = state.get("session_id")

    _MAX_RERETRIEVAL_TERMS = 10
    _extr = extractor_llm or llm

    # Deduplicate seed terms from agent feedback
    seed_terms: list[str] = []
    for fb in feedback:
        term = fb.get("refined_query")
        if term and term.lower() not in seen_terms:
            seen_terms.add(term.lower())
            seed_terms.append(term)

    async def _expand_terms_batch(term_list: list[str]) -> dict[str, list[str]]:
        prompt = (
            "You are a medical terminology expert. Given a list of clinical entities, return a JSON object "
            "where each key is an entity and the value is an array of 3 synonyms, abbreviations, or related "
            "terms that a medical knowledge graph might store separately. "
            "Return ONLY valid JSON, no prose, no code fences. Do not repeat the input term in its own array.\n\n"
            f"Entities: {json.dumps(term_list)}"
        )
        try:
            raw = (await llm_ainvoke(_extr, [HumanMessage(content=prompt)], role="entity_expand")).content.strip()
            parsed = json.loads(raw.replace("```json", "").replace("```", "").strip())
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return {}

    expansion_map = await _expand_terms_batch(seed_terms)
    expansion_results = [
        [t] + [s for s in expansion_map.get(t, []) if isinstance(s, str)]
        for t in seed_terms
    ]

    expanded_terms: list[str] = []
    seen_expanded: set[str] = set()
    for terms in expansion_results:
        for t in terms:
            if t.lower() not in seen_expanded:
                seen_expanded.add(t.lower())
                expanded_terms.append(t)
                if len(expanded_terms) >= _MAX_RERETRIEVAL_TERMS:
                    break
        if len(expanded_terms) >= _MAX_RERETRIEVAL_TERMS:
            break

    if not expanded_terms:
        logger.warning("Re-retrieval: no seed terms from agent feedback — skipping")
        return {"retrieval_iteration": state.get("retrieval_iteration", 0) + 1}

    if session_id and session_id in ACTIVE_STREAMS:
        ACTIVE_STREAMS[session_id].append(
            f"Re-querying Knowledge Core: **{', '.join(expanded_terms[:3])}**{'...' if len(expanded_terms) > 3 else ''}"
        )

    async def _lookup_and_refine(term: str) -> str | None:
        logger.info("Re-retrieval KB lookup", term=term)
        raw_facts = await consult_medical_knowledge.ainvoke(term)
        section = await _refine_kb_context(term, raw_facts)
        return section if not _kb_context_is_empty([section]) else None

    # Fire all KB lookups + refinements in parallel
    sections = await asyncio.gather(*[_lookup_and_refine(t) for t in expanded_terms])
    new_sections = [s for s in sections if s is not None]

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

async def node_router(state: AgentState):
    _t0_node = _time.monotonic()
    logger.info("NODE: ROUTER")
    input_text = state['redacted_input']
    context_str = "\n".join(state.get("context", []))
    routing_context = state.get("routing_context") or ""

    system_prompt = (
        "You are the MediCortex Orchestrator. Your ONLY job is to decide which specialist agents to call.\n\n"
        "AGENTS — use ONLY these keys, never invent new ones:\n"
        "- \"pubmed\"          → research papers, clinical trials, evidence-based guidelines\n"
        "- \"diagnosis\"       → symptoms, differential diagnosis, pathophysiology, clinical assessment\n"
        "- \"report_analyzer\" → lab results, imaging, ECG, pathology reports, uploaded files\n"
        "- \"patient\"         → specific named/identified patient's history, records, vitals\n"
        "- \"pharmacology\"    → drugs, dosing, interactions, side effects, contraindications, mechanisms\n\n"
        "DECISION PRINCIPLE — use the minimum agents needed to fully answer the query:\n"
        "- A query that touches only ONE domain → single agent.\n"
        "- A query that genuinely spans multiple domains → multiple agents (max 3).\n"
        "- Ask: 'Would a complete answer require knowledge from more than one specialist?' "
        "If yes, include all relevant agents. If no, use one.\n\n"
        "HARD RULES:\n"
        "1. Return ONLY a valid JSON array of agent keys — no prose, no explanation.\n"
        "2. Never route to 'pubmed' unless the user explicitly asks for research, trials, or guidelines.\n"
        "3. Always include 'report_analyzer' when files or images are attached.\n\n"
        "FEW-SHOT EXAMPLES (study the reasoning pattern):\n"
        "\"Side effects of metoprolol\" → [\"pharmacology\"]\n"
        "  # Pure drug question — one agent is enough.\n\n"
        "\"Patient has chest pain and sweating\" → [\"diagnosis\"]\n"
        "  # Pure symptom/assessment question — one agent is enough.\n\n"
        "\"What drugs treat hypertension and what are their side effects?\" → [\"diagnosis\", \"pharmacology\"]\n"
        "  # Needs clinical context of the condition (diagnosis) AND drug details (pharmacology).\n\n"
        "\"Is metformin safe for someone with CKD?\" → [\"pharmacology\", \"pubmed\"]\n"
        "  # Drug safety in a specific condition context + guideline evidence requested implicitly.\n\n"
        "\"Interpret this CBC — WBC 14k, Hgb 8.2\" → [\"report_analyzer\"]\n"
        "  # Uploaded/pasted lab report — report_analyzer handles this alone.\n\n"
        "\"45yo with fever and cough — what's the diagnosis and what should I prescribe?\" → [\"diagnosis\", \"pharmacology\"]\n"
        "  # Spans two domains: differential (diagnosis) + treatment selection (pharmacology).\n\n"
        "\"John's last visit records and his current beta blocker dose\" → [\"patient\", \"pharmacology\"]\n"
        "  # Named patient records (patient) + drug info (pharmacology).\n\n"
        "\"Latest RCTs on SGLT2 inhibitors in heart failure\" → [\"pubmed\"]\n"
        "  # Explicit research/trial request — pubmed only.\n\n"
        "\"What is sepsis and how is it managed in ICU?\" → [\"diagnosis\", \"pharmacology\"]\n"
        "  # Condition overview (diagnosis) + management/treatment (pharmacology).\n\n"
        "FOLLOW-UP RESOLUTION — when the query uses pronouns (his/her/their/the patient/the drug/it), "
        "resolve the referent using the Recent Session Context and apply the same decision principle:\n"
        "- Prior [patient] turn + asks about drugs → [\"pharmacology\"]\n"
        "- Prior [diagnosis] turn + asks about treatment → [\"pharmacology\"] or [\"diagnosis\", \"pharmacology\"]\n"
        "- Prior [pharmacology] turn + asks about same drug → [\"pharmacology\"]\n\n"
        "Return ONLY the JSON array."
    )

    file_urls = state.get("file_urls") or []
    file_note = (
        f"Attached Files: {len(file_urls)} file(s) uploaded by the user. "
        f"Route to 'report_analyzer' — the user is asking about these files.\n\n"
        if file_urls else ""
    )
    user_message = (
        f"{file_note}"
        f"User Query: {input_text}\n\n"
        + (f"Recent Session Context (use this to resolve follow-up references):\n{routing_context}\n\n"
           if routing_context else "")
        + f"Knowledge Core Context (for your awareness): {context_str[:300]}"
    )
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
    
    try:
        response = (await llm_ainvoke(llm, messages, role="router")).content
        clean_response = response.replace("```json", "").replace("```", "").strip()
        clean_response = clean_response.replace("'", '"')
        try:
            routes = json.loads(clean_response)
        except json.JSONDecodeError:
            import re as _re_router
            m = _re_router.search(r'\[.*?\]', clean_response, _re_router.DOTALL)
            routes = json.loads(m.group()) if m else ["diagnosis"]
            if m:
                logger.warning("Router returned prose — recovered JSON array via regex")
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
    # Suppress clarification whenever the router confidently identified a specific agent.
    # Entity extraction can fail (empty LLM response) while routing still succeeds — in that case
    # retrieval_ambiguous=True but the query is not actually ambiguous.
    _KNOWN_AGENTS = {"report_analyzer", "pharmacology", "diagnosis", "patient", "pubmed"}
    specific_route_found = bool(set(routes) & _KNOWN_AGENTS)
    if state.get("retrieval_ambiguous") and not already_clarified and not specific_route_found:
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
            clarification_q = (await llm_ainvoke(llm, [
                SystemMessage(content=clarification_prompt),
                HumanMessage(content=f"User query: {input_text}"),
            ], role="clarification")).content.strip()
            logger.info("Clarification question generated", question=clarification_q)
            return {
                "messages": [AIMessage(content='["__clarify__"]')],
                "clarification_question": clarification_q,
            }
        except Exception as e:
            logger.warning("Clarification generation failed, proceeding with routing", error=str(e))

    logger.info("node_elapsed_ms", node="router", elapsed_ms=round((_time.monotonic() - _t0_node) * 1000))
    return {"messages": [AIMessage(content=str(routes))]}

def make_agent_node(agent_key: str):
    async def _node(state: AgentState, config: RunnableConfig):
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
        # OPS-6: idempotency key derived from content so retries of the same
        # message in the same session hit the cache instead of always missing.
        import hashlib as _hashlib
        _idem_src = f"{state.get('session_id','')}{agent_key}{enhanced_input}"
        _idem_key = _hashlib.sha256(_idem_src.encode()).hexdigest()
        try:
            envelope = Envelope(
                trace_id=state.get("trace_id", ""),
                idempotency_key=_idem_key,
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
            
            # OPS-2: offload blocking agent.process() to thread pool so the
            # event loop stays free for concurrent requests.
            _t0_agent = _time.monotonic()
            response = await asyncio.get_event_loop().run_in_executor(
                None, agent_executor.process, envelope
            )
            logger.info("agent_call", agent=agent_key, rtt_ms=round((_time.monotonic() - _t0_agent) * 1000))
            
            # Capture thinking steps (already prefixed by emit_thought in base.py)
            thoughts = response.thinking if response.thinking else []

            if response.error:
                logger.error(f"Agent {agent_key} returned error", error=response.error)
                return {
                    "agent_outputs": [response.error],
                    "agent_thoughts": thoughts,
                    "agents_used": [agent_key],
                }

            output = response.output if response.output else "No output generated."
            result: dict = {
                "agent_outputs": [output],
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
                "agent_outputs": [f"Something went wrong while processing your request. Please try again."],
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


async def node_aggregator_with_reretrieval(state: AgentState):
    """
    RAG-1 Part B — Reactive re-retrieval wrapper around node_aggregator.

    If any agents flagged low_context=True and we haven't re-retrieved yet,
    perform an inline re-retrieval pass and re-run the flagging agents before
    aggregating. This avoids complex LangGraph fan-out rewiring.
    """
    _t0_node = _time.monotonic()
    logger.info("NODE: AGGREGATOR")

    # RAG-1 Part B: check for re-retrieval before aggregating
    feedback = state.get("retrieval_feedback", [])
    iteration = state.get("retrieval_iteration", 0)

    extra_outputs: list = []
    re_retrieval_ran = False
    if feedback and iteration < 1:
        logger.info("Reactive re-retrieval triggered", feedback=feedback)
        re_retrieval_ran = True
        reretrieval_result = await node_retrieve_knowledge_v2(state)

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
            context_str = "\n".join(meaningful_context)
            raw_history_str = "\n".join(state.get("history", []))
            history_str = privacy_manager.redact_identifying_pii(raw_history_str) if raw_history_str else ""
            session_id_str = state.get("session_id", "default")

            async def _rerun_agent(agent_key: str) -> str | None:
                import hashlib as _hashlib
                import json as _json
                agent_executor = AGENT_REGISTRY.get(agent_key)
                if not agent_executor:
                    return None
                enhanced_input = (
                    f"Conversation History:\n{history_str}\n\n"
                    f"Current Request: {state['redacted_input']}"
                    + (f"\n\nContext from Knowledge Core (enriched):\n{context_str}" if context_str else "")
                )
                _idem_key = _hashlib.sha256(
                    f"{state.get('session_id','')}{agent_key}{enhanced_input}".encode()
                ).hexdigest()
                envelope = Envelope(
                    trace_id=state.get("trace_id", ""),
                    idempotency_key=_idem_key,
                    sender_id="orchestrator",
                    receiver_id=agent_key,
                    payload={"input": enhanced_input},
                )
                if agent_key == "patient":
                    envelope.payload["pii_mapping_json"] = _json.dumps(state.get("pii_mapping", {}))
                if agent_key == "diagnosis":
                    envelope.payload["knowledge_context"] = context_str
                envelope.payload["live_thoughts_queue"] = ACTIVE_STREAMS.get(session_id_str, [])
                try:
                    response = await asyncio.get_event_loop().run_in_executor(
                        None, agent_executor.process, envelope
                    )
                    if response.output and not response.error:
                        logger.info(f"Re-run agent {agent_key} succeeded after re-retrieval")
                        return response.output
                except Exception as e:
                    logger.warning(f"Re-run agent {agent_key} failed after re-retrieval", error=str(e))
                return None

            rerun_results = await asyncio.gather(*[_rerun_agent(k) for k in re_run_keys])
            extra_outputs.extend(r for r in rerun_results if r is not None)

    # Proceed with standard aggregation (original outputs + any re-retrieved outputs)
    raw_outputs = "\n\n".join(state["agent_outputs"] + extra_outputs)

    formatting_prompt = (
        "You are the MediCortex Interface. Format the following medical agent reports into "
        "a beautiful, human-readable Markdown response.\n"
        "Use bolding, italics, bullet points, and headers to make it easy to read.\n\n"
        "OPENING RULES:\n"
        "- Begin DIRECTLY with the clinical content. Do NOT open with any acknowledgment phrase such as "
        "'Okay', 'Sure', 'Here is', 'Here\\'s', 'Of course', 'Certainly', 'Below is', or any similar filler.\n"
        "- Do NOT open with a declaration like 'Okay, here\\'s the Markdown response' or "
        "'I\\'ve formatted the response as requested' — just write the response.\n\n"
        "HEADING RULES:\n"
        "- Do NOT add any heading that names an agent (e.g. 'Pharmacology', 'Diagnosis Agent', etc.).\n"
        "- Do NOT open the response with a generic heading like 'Medical Agent Reports', "
        "'Medical Agent Reports on [Topic]', 'Comprehensive Clinical Guidance', 'Medical Agent Reports Summary', or any similar variation.\n"
        "- If a heading is needed, use a concise topic-specific heading (e.g. 'Hypertension: Symptoms & First-Line Treatment'). "
        "For shorter responses, omit the heading entirely.\n"
        "- The response must read as expert clinical guidance, not as an internal pipeline report.\n\n"
        "DEDUPLICATION RULES (apply before formatting):\n"
        "1. If multiple agents recommend the same specialist referral or action, merge them into a single entry — do not repeat the same recommendation under different headings.\n"
        "2. If multiple source snippets convey the same fact, keep only the first occurrence and drop all subsequent duplicates.\n"
        "3. Near-identical recommendations that differ only in minor wording should be consolidated into one.\n"
        "Do not change any factual content beyond deduplication.\n\n"
        "CITATION RULES (only apply if the raw reports contain URLs):\n"
        "- If the raw reports contain source URLs, add inline citation numbers like [1], [2] at the end of each sentence or claim that is supported by a source.\n"
        "- Collect all unique cited URLs and append a '## References' section at the very end of the response, formatted as a numbered markdown list: '1. [Title](url)'\n"
        "- Each number must correspond to exactly one unique URL. Do not assign the same number to two different URLs.\n"
        "- If the raw reports contain NO URLs at all, do NOT add a References section and do NOT add any inline citation numbers.\n\n"
        f"Raw Reports:\n{raw_outputs}"
    )

    try:
        formatted = (await llm_ainvoke(llm, [HumanMessage(content=formatting_prompt)], role="aggregator_format")).content
    except Exception:
        formatted = raw_outputs

    # BUG-6: propagate re-retrieval iteration count back through LangGraph state
    # so the streaming endpoint can persist it in message_metadata.retrieval_iterations.
    out: Dict[str, Any] = {"final_output": formatted}
    if re_retrieval_ran:
        out["retrieval_iteration"] = iteration + 1
    logger.info("node_elapsed_ms", node="aggregator", elapsed_ms=round((_time.monotonic() - _t0_node) * 1000))
    return out


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


async def node_reviewer(state: AgentState):
    """
    A2A §5.2 — Model-as-Judge evaluation node.

    Scores the aggregated response 1–5 using Groq (llama-3.3-70b-versatile).
    If score < 3, appends a clinical disclaimer to protect the user.
    Respects JUDGE_SAMPLE_RATE, JUDGE_MAX_INPUT_TOKENS, and falls back to
    llama-3.1-8b-instant if the primary model hits rate limits.
    """
    _t0_node = _time.monotonic()
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
    history = state.get("history", [])
    # Truncate each history entry to 300 chars so a long prior assistant response
    # (e.g. a 2000-word SGLT2 essay) doesn't drown out the current query in Groq's context.
    truncated_history = [h[:300] + ("…" if len(h) > 300 else "") for h in history[-6:]]
    history_str = "\n".join(truncated_history) if truncated_history else ""
    logger.info("reviewer_query_check", original_query=original_query, history_turns=len(history))

    conversation_block = (
        f"Conversation History (last turns):\n{history_str}\n\n" if history_str else ""
    )

    agents_used = state.get("agents_used", [])
    agents_str = ", ".join(agents_used) if agents_used else "unknown"

    # ── Include document analyzer output when files were attached ─────
    # The judge needs access to what was actually extracted from the document
    # to verify that the final response accurately reflects the report content.
    file_urls = state.get("file_urls") or []
    agent_outputs: list = state.get("agent_outputs") or []
    doc_context_block = ""
    if file_urls and agent_outputs:
        doc_outputs = [
            o for o in agent_outputs
            if "report_analyzer" in o.lower() or "## full document analysis" in o.lower()
        ]
        if doc_outputs:
            # Budget: reserve up to 40% of token budget for document context
            doc_budget = (max_chars * 4) // 10  # chars
            doc_combined = "\n\n---\n\n".join(doc_outputs)[:doc_budget]
            doc_context_block = (
                f"Document Analysis (extracted from attached file):\n{doc_combined}\n\n"
            )

    judge_prompt = f"""You are a clinical quality reviewer for a medical AI assistant.

CURRENT USER QUERY (your evaluation target — ignore all other queries):
>>> {original_query} <<<

Evaluate the "Response to evaluate" below on a scale of 1–5:
1 = Completely off-topic, dangerous, or hallucinates facts
2 = Partially relevant but contains significant errors or unsupported claims
3 = Relevant and mostly accurate, minor gaps acceptable
4 = Accurate, grounded in evidence, clearly addresses the query
5 = Excellent — accurate, complete, evidence-based, safe for clinical context

IMPORTANT — what to evaluate:
- Evaluate ONLY whether the "Response to evaluate" addresses the CURRENT USER QUERY shown above.
- The "Conversation History" below is for continuity context ONLY — do NOT evaluate against any earlier query.

Criteria to check:
- Does the response directly address the CURRENT USER QUERY (not a prior query)?
- Are all clinical claims grounded in tool outputs (no fabricated facts)?
- Does the response contain leaked PII placeholders (e.g. <PERSON_1>)? NOTE: placeholders like <PERSON_1> are intentional de-identification tokens used by the privacy layer and are NOT a quality defect — do NOT penalise the score for their presence.
- Is the response safe for a medical assistant context?
- Does the response maintain continuity with prior turns in the conversation?
- When a document was attached: does the response accurately reflect the extracted document content without omitting key findings?
- Agents that generated this response: {agents_str}

{conversation_block}{doc_context_block}Response to evaluate:
{truncated}

Reply with ONLY a JSON object in this exact format, no other text:
{{"score": <1-5>, "reason": "<one sentence referencing the query above>", "confidence": "<0-100>%"}}"""

    async def _call_groq(model_name: str) -> tuple[dict, dict]:
        judge_llm = ChatGroq(
            model=model_name,
            api_key=settings.GROQ_API_KEY,
            temperature=0,
            max_tokens=100,
        )
        raw = await llm_ainvoke(judge_llm, [HumanMessage(content=judge_prompt)], role="reviewer")
        usage = {}
        if hasattr(raw, "usage_metadata") and raw.usage_metadata:
            usage = {
                "prompt_tokens": raw.usage_metadata.get("input_tokens", 0),
                "completion_tokens": raw.usage_metadata.get("output_tokens", 0),
            }
        return json.loads(raw.content.strip()), usage

    # ── Call judge with fallback ──────────────────────────────────────
    judge_result = None
    judge_usage: dict = {}
    for model in [settings.JUDGE_MODEL, settings.JUDGE_FALLBACK_MODEL]:
        try:
            judge_result, judge_usage = await _call_groq(model)
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
    _elapsed_reviewer = round((_time.monotonic() - _t0_node) * 1000)
    return_payload: dict = {
        "judge_score": score,
        "judge_reason": reason,
        "judge_confidence": confidence,
        "judge_token_usage": judge_usage,
        "node_timings": {"reviewer": _elapsed_reviewer},
    }

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

    logger.info("node_elapsed_ms", node="reviewer", elapsed_ms=round((_time.monotonic() - _t0_node) * 1000))
    return return_payload


async def node_restore_privacy(state: AgentState):
    _t0_node = _time.monotonic()
    logger.info("NODE: RESTORE PRIVACY")
    raw_output = state.get("final_output", "")
    mapping = state.get("pii_mapping", {})
    loop = asyncio.get_event_loop()
    restored = await loop.run_in_executor(None, privacy_manager.restore_privacy, raw_output, mapping)
    logger.info("node_elapsed_ms", node="restore_privacy", elapsed_ms=round((_time.monotonic() - _t0_node) * 1000))
    return {"final_output": restored}

# ==========================================
# 🚀 GRAPH CONSTRUCTION
# ==========================================
# A2A §4.1 — Maximum agents per request (circuit breaker)
MAX_CONCURRENT_AGENTS = 3

def _parse_route_list(raw: str) -> Optional[list]:
    """BUG-7 — Tolerant parser for the router's route output.

    The router LLM may emit single-quoted Python-style lists, JSON, or
    surround the list with extra prose. Try, in order:
      1. ast.literal_eval (handles both "[...]"/'[...]' and Python list syntax)
      2. json.loads after a naive ' → " swap
      3. Regex fallback that pulls the first [...] substring out and retries.
    Returns the parsed list or None if nothing usable is found.
    """
    if not isinstance(raw, str):
        return None
    candidate = raw.strip()
    # Strip code fences the model occasionally adds.
    candidate = candidate.replace("```json", "").replace("```", "").strip()

    for attempt in (candidate, candidate.replace("'", '"')):
        try:
            parsed = ast.literal_eval(attempt)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

    m = re.search(r"\[[^\[\]]*\]", candidate)
    if m:
        inner = m.group(0)
        try:
            parsed = ast.literal_eval(inner)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    return None


def route_decision(state: AgentState):
    last_msg = state["messages"][-1].content

    # Clarification early-exit: detect the sentinel robustly. We require BOTH
    # the substring AND a parseable list with exactly the sentinel — substring
    # alone could appear inside an unrelated explanation the model emitted.
    if "__clarify__" in last_msg:
        parsed = _parse_route_list(last_msg)
        if parsed == ["__clarify__"]:
            return ["__clarify__"]

    routes = []

    # Always route to report_analyzer if files were attached (A2A §4.1)
    if state.get("file_urls"):
        routes.append("report_analyzer")

    llm_routes = _parse_route_list(last_msg) or []
    for r in llm_routes:
        if r in AGENT_REGISTRY and r not in routes:
            routes.append(r)

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

# OPS-5: optional rate limiting via slowapi. Imported defensively so the
# module still works in test environments that don't have slowapi installed.
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    limiter = Limiter(key_func=get_remote_address, enabled=settings.RATELIMIT_ENABLED)
    _SLOWAPI_AVAILABLE = True
except Exception:
    limiter = None
    RateLimitExceeded = Exception  # type: ignore[assignment,misc]
    _SLOWAPI_AVAILABLE = False


# DEPLOY-2: On-demand MedGemma warmup — fired as a fire-and-forget task at the
# start of each chat request so the RunPod worker warms up in parallel while the
# KB retrieval pipeline runs (entity extraction + ArangoDB, ~3–10s), giving
# MedGemma time to be ready before agent synthesis begins.
async def _fire_medgemma_warmup() -> None:
    if not settings.MEDGEMMA_KEEPWARM_URL:
        return
    run_url = settings.MEDGEMMA_KEEPWARM_URL.replace("/runsync", "/run")
    headers = {"Content-Type": "application/json"}
    if settings.RUNPOD_API_KEY:
        headers["Authorization"] = f"Bearer {settings.RUNPOD_API_KEY}"
    warmup_payload = {"input": {"prompt": "ping", "max_tokens": 8}}
    try:
        async with httpx.AsyncClient() as client:
            await client.post(run_url, json=warmup_payload, headers=headers, timeout=10.0)
        logger.info("MedGemma on-demand warmup ping ok")
    except Exception as e:
        logger.warning("MedGemma on-demand warmup ping failed", error=str(e))


async def _keepwarm_loop() -> None:
    """OPS-8: activity-aware RunPod keepwarm.

    Pings every 8s while within 5 min of the last real request; otherwise
    sleeps 300s before rechecking. Avoids burning RunPod credits during idle
    periods while keeping the worker hot immediately after user activity.
    """
    import time as _time_mod
    while True:
        if not settings.MEDGEMMA_KEEPWARM_URL:
            await asyncio.sleep(300)
            continue
        idle_secs = _time_mod.time() - _last_request_at
        if idle_secs < 300:
            await _fire_medgemma_warmup()
            logger.debug("keepwarm: worker hot", idle_secs=round(idle_secs))
            await asyncio.sleep(8)
        else:
            logger.debug("keepwarm: idle, skipping")
            await asyncio.sleep(300)


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

    # ── 3. Ollama Cloud LLM (Router / Aggregator / Extractor) ───────────
    # A warmup call is issued at startup so the first user request is served
    # from a hot model.
    try:
        _ollama_base = settings.OLLAMA_CLOUD_URL.removesuffix("/v1")
        logger.info("Initializing Ollama LLM Client (ChatOllama)", model=settings.OLLAMA_CLOUD_MODEL, base_url=_ollama_base)
        llm = ChatOllama(
            model=settings.OLLAMA_CLOUD_MODEL,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            base_url=_ollama_base,
            think=False,  # router/aggregator/extractor don't need chain-of-thought
        )
        logger.info("Ollama Client Ready — warming up model (first request may load model)")
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
                logger.info("Ollama warmup complete — model is hot")
            except Exception as _we:
                logger.warning("Ollama warmup failed (model may still be loading)", error=str(_we))
        _asyncio.create_task(_warmup())
        logger.info("Ollama LLM Ready (ChatOllama)", model=settings.OLLAMA_CLOUD_MODEL, base_url=_ollama_base, status="success")
    except Exception as e:
        logger.error("Ollama Cloud setup failed", error=str(e))
        llm = None

    # extractor_llm reuses the same Ollama instance for entity extraction
    extractor_llm = llm

    # ── 4. LangGraph Workflow ──────────────────────────────────────────
    workflow = StateGraph(AgentState)
    workflow.add_node("scope_guard", node_scope_guard)
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
    workflow.set_entry_point("scope_guard")
    workflow.add_conditional_edges("scope_guard", lambda s: "restore_privacy" if s.get("final_output") else "analyze_privacy", {
        "restore_privacy": "restore_privacy",
        "analyze_privacy": "analyze_privacy",
    })
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

    # ── OPS-3: log thought-streaming backend in use ───────────────────
    web_concurrency_env = os.environ.get("WEB_CONCURRENCY", str(settings.WEB_CONCURRENCY))
    _redis_available = False
    try:
        if getattr(settings, "REDIS_URL", None):
            import redis as _r
            _st = getattr(settings, "REDIS_SOCKET_TIMEOUT", 2)
            _rc = _r.from_url(settings.REDIS_URL, socket_timeout=_st,
                              socket_connect_timeout=_st,
                              retry_on_timeout=False, retry_on_error=[])
            _rc.ping()
            _redis_available = True
    except Exception:
        pass

    if _redis_available:
        logger.info("OPS-3: thought streaming via Redis — multi-worker safe")
    else:
        # OPS-3: without Redis-backed ACTIVE_STREAMS, running >1 worker drops SSE
        # thoughts from sibling processes. Refuse to start in that unsafe config.
        if int(web_concurrency_env) > 1:
            raise RuntimeError(
                f"OPS-3: WEB_CONCURRENCY={web_concurrency_env} requires Redis for "
                "SSE thought-streaming. Start Redis and set REDIS_URL, or set "
                "WEB_CONCURRENCY=1."
            )
        logger.warning(
            "OPS-3: Redis unavailable — thought streaming is process-local (single worker only).",
            web_concurrency=web_concurrency_env,
        )

    # ── OPS-8: start activity-aware keepwarm background loop ──────────
    _asyncio.create_task(_keepwarm_loop())

    # ── Ready ──────────────────────────────────────────────────────────
    logger.info("Starting Orchestrator Server", app_name=settings.APP_NAME)
    logger.info("Database Schema Managed externally")

    yield

    logger.info("Shutting down")

app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

# OBS-1: observability dashboard endpoints
from routes.dashboard import router as dashboard_router
app.include_router(dashboard_router, prefix="/api")

# OPS-5: register slowapi limiter + 429 handler if available.
if _SLOWAPI_AVAILABLE and limiter is not None:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# DEPLOY-3: explicit allowlist replaces wildcard. The SettingsValidator in
# config.py rejects "*" when DEBUG=False, so this is safe to leave dynamic.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _ratelimit(rule: str):
    """No-op decorator when slowapi is unavailable; otherwise applies the rate."""
    if _SLOWAPI_AVAILABLE and limiter is not None:
        return limiter.limit(rule)
    def _identity(fn):
        return fn
    return _identity


@app.post("/chat/stream")
@_ratelimit(settings.RATELIMIT_CHAT_STREAM)
async def chat_stream_endpoint(request: Request, body: ChatRequest, db: AsyncSession = Depends(get_db)):
    """
    Streaming chat endpoint for Server-Sent Events (SSE).
    Sends 'thought' events for agent reasoning and 'response' event for final output.
    """
    # OPS-8: mark request time for activity-aware keepwarm
    global _last_request_at
    import time as _time_req
    _last_request_at = _time_req.time()

    # BUG-5: track whether the LangGraph result was fully computed and not yet
    # persisted, so the finally block can still save the assistant message
    # even if the SSE generator was cancelled by a client disconnect.
    persistence_state: Dict[str, Any] = {
        "session_id": None,
        "saved": False,
        "final_output": "",
        "agent_thoughts": [],
        "msg_metadata": None,
    }

    # DEPLOY-2: warm MedGemma in parallel while KB retrieval pipeline runs
    if settings.MEDGEMMA_KEEPWARM_URL:
        asyncio.create_task(_fire_medgemma_warmup())

    async def event_generator():
        session_id = None
        try:
            logger.info("Received streaming chat request", message_length=len(body.message))

            # 1. Resolve session — create if not provided or if the given UUID doesn't exist
            session_id = body.session_id
            if not session_id:
                new_session = await chat_service.create_session(db)
                session_id = new_session.id
                yield f"data: {json.dumps({'type': 'session_id', 'content': str(session_id)})}\n\n"
            else:
                existing = await chat_service.get_session(db, str(session_id))
                if not existing:
                    new_session = await chat_service.create_session(db)
                    session_id = new_session.id
                    yield f"data: {json.dumps({'type': 'session_id', 'content': str(session_id)})}\n\n"

            persistence_state["session_id"] = session_id
            
            # 2. Extract file URLs from attachments (structured handoff to report agent)
            file_urls = [a["url"] for a in (body.attachments or []) if a.get("url")]

            # 3. Save User Message (with attachments for display)
            await chat_service.add_message(
                db, str(session_id), "user", body.message,
                attachments=body.attachments or [],
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
            request_started_at = _time.monotonic()
            msg_metadata = {
                "llm_used": "MedGemma (via HF) / gemma4:31b-cloud Router",
                "judge_score": None,
                "judge_reason": None,
                "judge_confidence": None,
                "agents_used": [],
                "node_timings": {},
                "request_elapsed_ms": None,
                "retrieval": None,
                "token_usage": None,
            }



            import asyncio

            # OPS-3: Redis-backed thought queue (falls back to in-process list)
            live_thoughts = _make_thought_queue(str(session_id))
            ACTIVE_STREAMS[str(session_id)] = live_thoughts

            final_output_container = {}

            async def run_graph():
                try:
                    result = await orchestrator_graph.ainvoke(
                        {
                            "input": body.message,
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
                            "node_timings": {},
                            "pii_mapping": {},
                            "redacted_input": "",
                            "context": [],
                            "final_output": "",
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

            # OBS-1 observability fields
            msg_metadata["node_timings"] = graph_output.get("node_timings") or {}
            msg_metadata["request_elapsed_ms"] = round((_time.monotonic() - request_started_at) * 1000)
            if graph_output.get("retrieval_stats"):
                msg_metadata["retrieval"] = graph_output["retrieval_stats"]
            if graph_output.get("judge_token_usage"):
                msg_metadata["token_usage"] = graph_output["judge_token_usage"]

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

            # BUG-5: stage the final result for finally-block persistence FIRST,
            # so even if the client disconnects between this point and [DONE],
            # the message is saved on cleanup.
            persistence_state["final_output"] = final_output
            persistence_state["agent_thoughts"] = agent_thoughts
            persistence_state["msg_metadata"] = msg_metadata

            yield f"data: {json.dumps({'type': 'metadata', 'content': msg_metadata})}\n\n"

            if final_output:
                yield f"data: {json.dumps({'type': 'response', 'content': final_output})}\n\n"

            # 5. Save AI Response to DB (only once). Mark saved so the finally
            # block does not double-write on a clean exit.
            if final_output:
                await chat_service.add_message(
                    db, str(session_id), "assistant", final_output,
                    thinking=agent_thoughts, metadata=msg_metadata,
                )
                persistence_state["saved"] = True

            yield "data: [DONE]\n\n"

        except (asyncio.CancelledError, GeneratorExit):
            # BUG-5: client disconnected mid-stream. Re-raise after finally
            # so FastAPI cleans up properly.
            logger.info("client disconnected — persisting completed graph result if available", session_id=str(session_id) if session_id else None)
            raise
        except Exception as e:
            logger.error("Streaming error", error=str(e))
            try:
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            except Exception:
                pass
        finally:
            sid = session_id or persistence_state.get("session_id")
            if sid and str(sid) in ACTIVE_STREAMS:
                ACTIVE_STREAMS[str(sid)].delete()
                del ACTIVE_STREAMS[str(sid)]

            # BUG-5: persist the assistant message even on disconnect, as long
            # as the LangGraph pipeline finished and we haven't saved yet.
            if (
                sid
                and not persistence_state["saved"]
                and persistence_state["final_output"]
            ):
                try:
                    await chat_service.add_message(
                        db,
                        str(sid),
                        "assistant",
                        persistence_state["final_output"],
                        thinking=persistence_state["agent_thoughts"],
                        metadata=persistence_state["msg_metadata"] or {},
                    )
                    persistence_state["saved"] = True
                    logger.info("post-disconnect save completed", session_id=str(sid))
                except Exception as save_err:
                    logger.error(
                        "post-disconnect DB save failed",
                        session_id=str(sid),
                        error=str(save_err),
                    )

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/chat", response_model=ChatResponse)
@_ratelimit(settings.RATELIMIT_CHAT)
async def chat_endpoint(request: Request, body: ChatRequest, db: AsyncSession = Depends(get_db)):
    """
    Legacy non-streaming chat endpoint.
    """
    # OPS-8: mark request time for activity-aware keepwarm
    global _last_request_at
    import time as _time_req
    _last_request_at = _time_req.time()

    # DEPLOY-2: warm MedGemma in parallel while KB retrieval pipeline runs
    if settings.MEDGEMMA_KEEPWARM_URL:
        asyncio.create_task(_fire_medgemma_warmup())

    try:
        logger.info("Received chat request", message_length=len(body.message))

        # 1. Resolve session — create if not provided or if the given UUID doesn't exist
        session_id = body.session_id
        if not session_id:
            new_session = await chat_service.create_session(db)
            session_id = new_session.id
        else:
            existing = await chat_service.get_session(db, str(session_id))
            if not existing:
                new_session = await chat_service.create_session(db)
                session_id = new_session.id

        # 2. Extract file URLs and save user message with attachments
        file_urls = [a["url"] for a in (body.attachments or []) if a.get("url")]
        await chat_service.add_message(
            db, str(session_id), "user", body.message,
            attachments=body.attachments or [],
        )

        # 3. Retrieve History for Context
        full_history = await chat_service.get_messages(db, str(session_id))
        past_turns = full_history[:-1]
        history_context = [f"{m.role.capitalize()}: {m.content}" for m in past_turns[-10:]]

        # Build routing context from past turns so the router can resolve follow-ups
        routing_context = _build_routing_context(past_turns)

        # 4. Invoke Orchestrator
        result = await orchestrator_graph.ainvoke({
            "input": body.message,
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
            "node_timings": {},
            "pii_mapping": {},
            "redacted_input": "",
            "context": [],
            "final_output": "",
        })
        response_text = result.get("final_output")
        agent_thinking = result.get("agent_thoughts", [])

        msg_metadata = {
            "llm_used": "MedGemma (via HF) / gemma4:31b-cloud Router",
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
@_ratelimit(settings.RATELIMIT_UPLOAD)
async def upload_file(request: Request, file: UploadFile = File(...)):
    """Upload file to MinIO with SEC-1 size cap."""
    _ = request
    _ALLOWED_MIME_PREFIXES = ("image/", "application/pdf", "text/plain", "text/csv")
    ct = (file.content_type or "").lower()
    if not any(ct.startswith(p) for p in _ALLOWED_MIME_PREFIXES):
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {ct}")
    try:
        # SEC-1: read up to MAX_UPLOAD_BYTES + 1; if we exceed the cap, reject.
        cap = settings.MAX_UPLOAD_BYTES
        content = await file.read(cap + 1)
        if len(content) > cap:
            raise HTTPException(
                status_code=413,
                detail=f"File too large (max {cap // (1024 * 1024)} MB)",
            )
        url = await minio_service.upload_file(content, file.filename, file.content_type)
        if not url:
            raise HTTPException(status_code=500, detail="Upload failed")
        return UploadResponse(url=url, filename=file.filename, content_type=file.content_type or "")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Upload error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ── OBS-2: liveness + readiness probes ──────────────────────────────
async def _check_postgres() -> bool:
    try:
        async with engine.connect() as conn:
            await conn.execute(sa_text("SELECT 1"))
        return True
    except Exception as e:
        logger.warning("readyz: postgres check failed", error=str(e))
        return False


def _check_ollama() -> bool:
    try:
        base = settings.OLLAMA_CLOUD_URL.removesuffix("/v1")
        resp = requests.get(f"{base}/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception as e:
        logger.warning("readyz: ollama check failed", error=str(e))
        return False


def _check_medgemma() -> bool:
    if not settings.MEDGEMMA_API_URL:
        return True  # not configured = not required
    try:
        # Probe the matching health endpoint without forcing a generation.
        if _looks_like_runpod_url(settings.MEDGEMMA_API_URL):
            # RunPod endpoints don't expose /health on /runsync; skip and trust keepwarm.
            return True
        probe = settings.MEDGEMMA_API_URL.replace("/predict", "/health")
        headers = {}
        if settings.RUNPOD_API_KEY:
            headers["Authorization"] = f"Bearer {settings.RUNPOD_API_KEY}"
        resp = requests.get(probe, headers=headers, timeout=5)
        return resp.status_code < 500
    except Exception as e:
        logger.warning("readyz: medgemma check failed", error=str(e))
        return False


def _looks_like_runpod_url(url: str) -> bool:
    return "runpod.ai" in url.lower()


def _check_redis() -> bool:
    try:
        import redis as _redis
        c = _redis.from_url(
            settings.REDIS_URL,
            socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
            socket_connect_timeout=settings.REDIS_SOCKET_TIMEOUT,
        )
        return bool(c.ping())
    except Exception as e:
        logger.warning("readyz: redis check failed", error=str(e))
        return False


@app.get("/livez")
async def livez():
    """Liveness probe — does the process respond? Used by uptime monitoring."""
    return {"status": "alive"}


@app.get("/readyz")
async def readyz():
    """Readiness probe — are all critical dependencies reachable?

    Returns 200 when every gate is green; 503 with a per-component breakdown
    otherwise. Suitable for monitoring (Better Stack, Uptime Kuma) and for
    RunPod / load-balancer readiness gating.
    """
    loop = asyncio.get_event_loop()
    pg_ok = await _check_postgres()
    ollama_ok = await loop.run_in_executor(None, _check_ollama)
    medgemma_ok = await loop.run_in_executor(None, _check_medgemma)
    redis_ok = await loop.run_in_executor(None, _check_redis)

    components = {
        "postgres": pg_ok,
        "ollama": ollama_ok,
        "medgemma": medgemma_ok,
        "redis": redis_ok,
    }
    healthy = all(components.values())
    payload = {"status": "ready" if healthy else "degraded", "components": components}
    return JSONResponse(content=payload, status_code=200 if healthy else 503)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Legacy alias — prefer /livez / /readyz for new monitoring."""
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
