import inspect
import json
import logging
import re
import time as _time
import redis
import structlog
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama

from .medgemma_llm import MedGemmaLLM
from .protocols import AgentCard, Envelope, AgentResponse
from config import settings

logger = structlog.get_logger("SpecializedAgents")


def _llm_invoke_audit(llm_obj, messages, *, agent: str, role: str) -> any:
    """Sync LLM call with structured audit log: model name, agent, role, RTT."""
    model = getattr(llm_obj, "model", None) or getattr(llm_obj, "model_name", None) or type(llm_obj).__name__
    t0 = _time.monotonic()
    result = llm_obj.invoke(messages)
    rtt_ms = round((_time.monotonic() - t0) * 1000)
    logger.info("llm_call", model=model, agent=agent, role=role, rtt_ms=rtt_ms)
    return result

# MedGemma — used exclusively for clinical synthesis (Phase 2).
# Tool orchestration is handled by gemma4:31b-cloud (Phase 1).
llm = MedGemmaLLM()

# Tools whose observations may contain patient PHI (names embedded in PDFs/images).
# Web crawl tools are deliberately excluded — they return public literature only.
_PHI_PRODUCING_TOOLS = {"extract_document_text", "extract_image_findings"}

# HIPAA identifiers to redact from extraction tool observations.
# DATE_TIME and LOCATION excluded: clinically meaningful and no restoration needed.
_PHI_ENTITIES = ["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "US_SSN", "US_PASSPORT"]

# Singleton Presidio analyzer — instantiated once at import, not per tool call.
_phi_analyzer = None

def _get_phi_analyzer():
    global _phi_analyzer
    if _phi_analyzer is None:
        try:
            import logging as _logging
            _logging.getLogger("presidio-analyzer").setLevel(_logging.ERROR)
            from presidio_analyzer import AnalyzerEngine
            _phi_analyzer = AnalyzerEngine()
        except Exception as e:
            logger.warning("phi_analyzer_init_failed", error=str(e))
    return _phi_analyzer


def _redact_observation(observation: str, existing_mapping: Dict[str, str]) -> Tuple[str, Dict[str, str]]:
    """
    Run Presidio over a raw tool observation and return (redacted_text, new_mappings).

    Uses the existing pii_mapping to derive placeholder offsets so that new
    placeholders never collide with ones already known to node_restore_privacy.
    new_mappings contains ONLY the entries added by this call — callers merge
    them into the full mapping themselves.
    """
    if not observation or len(observation) < 5:
        return observation, {}

    analyzer = _get_phi_analyzer()
    if analyzer is None:
        return observation, {}

    # Count existing placeholders per type to derive the starting index offset
    # so new placeholders never collide with ones already in the state mapping.
    type_offsets: Dict[str, int] = {}
    for placeholder in existing_mapping:
        m = re.match(r"<([A-Z_]+)_(\d+)>", placeholder)
        if m:
            etype, idx = m.group(1), int(m.group(2))
            type_offsets[etype] = max(type_offsets.get(etype, 0), idx)

    try:
        results = analyzer.analyze(text=observation, entities=_PHI_ENTITIES, language="en")
    except Exception as e:
        logger.warning("phi_redact_observation_failed", error=str(e))
        return observation, {}

    if not results:
        return observation, {}

    results = sorted(results, key=lambda r: r.start, reverse=True)
    redacted = observation
    new_mappings: Dict[str, str] = {}
    type_counts: Dict[str, int] = dict(type_offsets)

    for result in results:
        etype = result.entity_type
        start, end = result.start, result.end
        original = observation[start:end]

        # Skip if this exact value is already in the mapping (deduplicate).
        if original in existing_mapping.values() or original in new_mappings.values():
            existing_placeholder = next(
                (k for k, v in {**existing_mapping, **new_mappings}.items() if v == original),
                None,
            )
            if existing_placeholder:
                redacted = redacted[:start] + existing_placeholder + redacted[end:]
                continue

        count = type_counts.get(etype, 0) + 1
        type_counts[etype] = count
        placeholder = f"<{etype}_{count}>"
        new_mappings[placeholder] = original
        redacted = redacted[:start] + placeholder + redacted[end:]

    if new_mappings:
        logger.info("phi_redacted_observation", new_entities=len(new_mappings))

    return redacted, new_mappings


class A2ABaseAgent:
    """
    Base Agent implementing the A2A Protocol with a two-phase execution model:

      Phase 1 — Tool-calling planner (gemma4:31b-cloud):
        Decides which tools to call and in what order. Uses bind_tools() and
        the Ollama tool-calling API. Emits tool thoughts in real-time for SSE
        streaming.

      Phase 2 — MedGemma (synthesizer):
        Receives the original query + all gathered tool results.
        Called exactly once to produce the final clinical response.

    This cleanly separates tool orchestration (FunctionGemma's speciality) from
    medical knowledge synthesis (MedGemma's strength), and eliminates the
    token waste of making a 4B medical model reason about tool selection.
    """

    def __init__(
        self,
        name: str,
        llm: MedGemmaLLM,
        tools: List[BaseTool],
        system_prompt: str,
        card: AgentCard,
        max_iterations: int = 3,
        skip_medgemma_synthesis: bool = False,
    ):
        self.name = name
        self.llm = llm              # MedGemma — synthesis only
        self.tools = {t.name: t for t in tools}
        self.system_prompt = system_prompt
        self.card = card
        self.max_iterations = max_iterations
        self.skip_medgemma_synthesis = skip_medgemma_synthesis

        # Idempotency cache — Redis with in-memory fallback. OPS-7: bounded
        # socket timeouts so a slow/down Redis cannot block agent registry
        # construction (this runs at orchestrator import time).
        self._response_cache: Dict[str, AgentResponse] = {}
        self._redis_cache = None
        try:
            if getattr(settings, "REDIS_URL", None):
                socket_timeout = getattr(settings, "REDIS_SOCKET_TIMEOUT", 2)
                self._redis_cache = redis.from_url(
                    settings.REDIS_URL,
                    decode_responses=True,
                    socket_timeout=socket_timeout,
                    socket_connect_timeout=socket_timeout,
                    retry_on_timeout=False,
                    retry_on_error=[],
                )
                self._redis_cache.ping()
                logger.info(f"[{self.name}] Connected to Redis cache.")
        except Exception as e:
            logger.warning(f"[{self.name}] Redis unavailable, using in-memory cache: {e}")
            self._redis_cache = None

        # OPS-4: cache the planner instance + tool-bound runnable. Re-building
        # ChatOllama and bind_tools() on every request is wasteful at the
        # 10-user / 5-agent scale. Tools are static per agent class.
        self._planner_cached = None
        self._planner_with_tools_cached = None

    # ── Public API ────────────────────────────────────────────────────────────

    def get_card(self) -> AgentCard:
        return self.card

    def process(self, envelope: Envelope) -> AgentResponse:
        """Main A2A entry point. Validates the envelope, checks idempotency cache,
        then runs the two-phase plan-and-synthesize pipeline."""
        logger.info(f"[{self.name}] Envelope {envelope.idempotency_key} from {envelope.sender_id}")

        user_input = envelope.payload.get("input", "")
        if not user_input:
            return AgentResponse(
                envelope_id=envelope.idempotency_key,
                output=None,
                error="Validation Error: 'input' field missing in payload.",
            )

        # ── Idempotency cache check ───────────────────────────────────────────
        cache_key = f"medicortex:idempotency:{envelope.idempotency_key}"

        if self._redis_cache:
            try:
                cached_json = self._redis_cache.get(cache_key)
                if cached_json:
                    logger.info(f"[{self.name}] Cache HIT (Redis)")
                    try:
                        return AgentResponse.model_validate_json(cached_json)
                    except AttributeError:
                        return AgentResponse.parse_raw(cached_json)
            except Exception as e:
                logger.warning(f"[{self.name}] Redis read failed: {e}")

        if envelope.idempotency_key in self._response_cache:
            logger.info(f"[{self.name}] Cache HIT (in-memory)")
            return self._response_cache[envelope.idempotency_key]

        logger.info(f"[{self.name}] Cache MISS — executing pipeline.")

        # ── Execute ───────────────────────────────────────────────────────────
        try:
            live_thoughts_queue = envelope.payload.get("live_thoughts_queue")

            # tool_context carries sensitive data that must never appear in any
            # LLM prompt. _call_tool() injects matching keys at call time via
            # inspect.signature, keeping PII out of both gemma4:31b-cloud and MedGemma.
            tool_context: Dict[str, Any] = {}
            if pii_json := envelope.payload.get("pii_mapping_json"):
                tool_context["pii_mapping_json"] = pii_json
            if kc := envelope.payload.get("knowledge_context"):
                tool_context["knowledge_context"] = kc

            output, thinking_steps, sources, low_context, refined_query = self._plan_and_synthesize(
                user_input, live_thoughts_queue, tool_context
            )

            # Compute any new PII mappings discovered during tool observations
            # so the orchestrator can merge them into state for node_restore_privacy.
            pii_extension: Dict[str, str] = {}
            if tool_context.get("pii_mapping_json"):
                try:
                    full_mapping = json.loads(tool_context["pii_mapping_json"])
                    original_mapping = json.loads(envelope.payload.get("pii_mapping_json") or "{}")
                    pii_extension = {k: v for k, v in full_mapping.items() if k not in original_mapping}
                except Exception:
                    pass

            response = AgentResponse(
                envelope_id=envelope.idempotency_key,
                output=output,
                thinking=thinking_steps,
                sources=sources,
                low_context=low_context,
                refined_query=refined_query,
                pii_mapping_extension=pii_extension,
            )

            # Cache write
            if self._redis_cache:
                try:
                    try:
                        resp_json = response.model_dump_json()
                    except AttributeError:
                        resp_json = response.json()
                    self._redis_cache.setex(cache_key, 86400, resp_json)
                except Exception as e:
                    logger.warning(f"[{self.name}] Redis write failed: {e}")
            else:
                self._response_cache[envelope.idempotency_key] = response

            return response

        except Exception as e:
            logger.error(f"[{self.name}] Pipeline error: {e}")
            return AgentResponse(
                envelope_id=envelope.idempotency_key,
                output=None,
                error=str(e),
            )

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy compatibility shim — wraps process() with a dummy Envelope."""
        env = Envelope(
            sender_id="legacy_invoke",
            receiver_id=self.name,
            payload=inputs,
        )
        resp = self.process(env)
        if resp.error:
            raise Exception(resp.error)
        return {"output": resp.output, "thinking": resp.thinking}

    # ── Private pipeline ──────────────────────────────────────────────────────

    def _plan_and_synthesize(
        self,
        user_input: str,
        live_thoughts_queue: Optional[list] = None,
        tool_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, List[str], List[dict], bool, Optional[str]]:
        """
        Two-phase pipeline:
          Phase 1 — gemma4:31b-cloud gathers tool data (emits thoughts in real-time).
          Phase 2 — MedGemma synthesizes the final clinical response (single call).

        Returns (final_answer, thinking_steps, sources, low_context, refined_query).
        low_context and refined_query support RAG-1 Part B reactive re-retrieval.
        """
        thinking_steps: List[str] = []
        tool_context = tool_context or {}

        def emit_thought(t: str):
            thinking_steps.append(t)
            if live_thoughts_queue is not None:
                live_thoughts_queue.append(t)

        # Phase 1 — tool gathering
        low_context = False
        refined_query = None
        try:
            tool_results, sources, low_context, refined_query = self._gather_tool_results(
                user_input, emit_thought, tool_context
            )
        except Exception as e:
            logger.error(f"[{self.name}] Tool gathering failed: {e}")
            tool_results, sources = [], []
            low_context = True
            emit_thought(
                f"**[{self.name.title()}]**: Tool gathering failed — "
                f"synthesizing from medical knowledge only."
            )

        # Phase 2 — MedGemma synthesis
        emit_thought(f"**[{self.name.title()}]**: Synthesizing clinical response…")
        try:
            final_answer = self._synthesize(user_input, tool_results)
        except Exception as e:
            logger.error(f"[{self.name}] Synthesis failed: {e}")
            final_answer = f"Error generating clinical response: {e}"

        return final_answer, thinking_steps, sources, low_context, refined_query

    # Matches any http/https URL, stopping at whitespace, closing brackets, or quotes
    _URL_RE = re.compile(r'https?://[^\s\)\]"\',<>]+')

    def _extract_sources_from_observation(self, observation: str) -> List[dict]:
        """
        Extract {title, url} pairs from a single tool observation string.

        Pass 1 — structured format used by all web crawl tools:
            ### N. <Page Title>
            - **Source:** <domain>
            - **URL:** <url>
        Backtracks up to 10 lines from each URL line to find its titled heading,
        so the Sources accordion shows "Type 2 diabetes - Mayo Clinic" instead of
        the raw URL.

        Pass 2 — bare URL fallback for any URLs not captured by Pass 1
        (e.g. URLs embedded inline in non-structured observations).
        """
        lines = observation.splitlines()
        sources: List[dict] = []
        seen: set = set()

        # Pass 1: structured "### N. Title" → "- **URL:** url"
        for i, line in enumerate(lines):
            m = re.match(r'\s*-\s+\*\*URL:\*\*\s+(https?://\S+)', line)
            if not m:
                continue
            url = m.group(1)
            if url in seen:
                continue
            seen.add(url)
            title = url  # fallback
            for j in range(i - 1, max(i - 10, -1), -1):
                hm = re.match(r'###\s+\d+\.\s+(.+)', lines[j])
                if hm:
                    title = hm.group(1).strip()
                    break
            sources.append({"title": title, "url": url})

        # Pass 2: bare URL fallback for anything not caught above
        for url in self._URL_RE.findall(observation):
            if url not in seen:
                seen.add(url)
                sources.append({"title": url, "url": url})

        return sources

    def _gather_tool_results(
        self,
        user_input: str,
        emit_thought,
        tool_context: Dict[str, Any],
    ) -> Tuple[List[Tuple[str, str]], List[dict], bool, Optional[str]]:
        """
        Phase 1: gemma4:31b-cloud decides which tools to call and in what order.

        Uses LangChain bind_tools() so tool schemas are generated automatically.
        Loops up to self.max_iterations times to allow multi-tool pipelines
        (e.g. patient agent: retrieve → history → vitals → medications).

        tool_context keys (pii_mapping_json, knowledge_context) are injected
        at call time by _call_tool() and are never visible to gemma4:31b-cloud.

        Returns (tool_results, sources, low_context, refined_query) where:
        - low_context is True when tool observations are empty or thin (<200 chars total)
          OR when the planner emits a CONTEXT_INSUFFICIENT:<term> sentinel.
        - refined_query is a more specific KB search term suggested by the planner,
          or None if context was sufficient.
        """
        # OPS-4: build planner + bind_tools once per agent instance.
        if self._planner_with_tools_cached is None:
            self._planner_cached = ChatOllama(
                model=getattr(settings, "OLLAMA_TOOL_MODEL", "gemma4:31b-cloud"),
                temperature=0.0,
                base_url=settings.OLLAMA_CLOUD_URL.removesuffix("/v1"),
                timeout=getattr(settings, "OLLAMA_TIMEOUT_SECONDS", 60),
                think=False,  # tool-call planning is structured — thinking adds no value
            )
            self._planner_with_tools_cached = self._planner_cached.bind_tools(
                list(self.tools.values())
            )
        planner_with_tools = self._planner_with_tools_cached

        messages = [
            SystemMessage(content=(
                f"You are a strict function-calling dispatcher for the {self.name} medical agent.\n"
                f"RULES — read carefully and follow without exception:\n"
                f"1. You MUST call one or more of the provided tools to gather data. No exceptions.\n"
                f"2. NEVER write prose, explanations, or a final answer. Output ONLY tool calls.\n"
                f"3. Call tools one at a time. After each tool result, decide whether more calls are needed.\n"
                f"4. Stop calling tools only when you have gathered enough data to fully answer the query.\n"
                f"5. For ANY query — including explanations, mechanisms, or pathophysiology — "
                f"you MUST call a tool. Extract the key medical topic and pass it as the search term. "
                f"Examples: 'how does sepsis cause shock' → search_diagnosis(condition='sepsis'); "
                f"'beta blocker mechanism' → lookup_drug_dosage(drug_name='metoprolol'). "
                f"Refusing to call a tool is NEVER allowed.\n"
                f"6. Only if ALL tools have been exhausted and data is still missing, output EXACTLY:\n"
                f"   CONTEXT_INSUFFICIENT:<2-5 word medical search term>\n"
                f"   No other text. No explanation.\n"
                f"Available tools: {', '.join(self.tools.keys())}"
            )),
            HumanMessage(content=user_input),
        ]

        results: List[Tuple[str, str]] = []
        seen_urls: set = set()
        sources: List[dict] = []

        for _ in range(self.max_iterations):
            response = _llm_invoke_audit(planner_with_tools, messages, agent=self.name, role="planner")
            messages.append(response)

            if not response.tool_calls:
                # gemma4:31b-cloud decided no more tools needed
                break

            for tc in response.tool_calls:
                tool_name = tc["name"]
                args = tc["args"]

                # Surface the tool call as a thought for the UI accordion
                emit_thought(self._describe_tool_call(tool_name, args))
                logger.info(f"[{self.name}] Tool call: {tool_name}({args})")

                try:
                    observation = self._call_tool(tool_name, args, tool_context)
                except Exception as tool_err:
                    logger.warning(f"[{self.name}] Tool {tool_name} failed: {tool_err}")
                    observation = f"[Tool error — skipped]"

                results.append((tool_name, observation))

                # Extract titled sources from this observation
                for src in self._extract_sources_from_observation(observation):
                    if src["url"] not in seen_urls:
                        seen_urls.add(src["url"])
                        sources.append(src)

                emit_thought(self._summarize_observation(tool_name, observation))

                messages.append(ToolMessage(content=observation, tool_call_id=tc["id"]))

        # RAG-1 Part B: check for CONTEXT_INSUFFICIENT sentinel in the final planner message
        low_context = False
        refined_query = None
        last_planner_msg = messages[-1] if messages else None
        if hasattr(last_planner_msg, "content") and isinstance(last_planner_msg.content, str):
            content = last_planner_msg.content.strip()
            if content.startswith("CONTEXT_INSUFFICIENT:"):
                low_context = True
                refined_query = content[len("CONTEXT_INSUFFICIENT:"):].strip() or None

        # Auto-detect near-zero results even without explicit sentinel.
        # Threshold kept low (50 chars) to avoid false positives when the web
        # tools return a small-but-valid response — the explicit CONTEXT_INSUFFICIENT
        # sentinel is the primary signal; this only catches truly empty tool loops.
        if not low_context and results:
            total_obs_len = sum(len(obs) for _, obs in results)
            if total_obs_len < 50:
                low_context = True

        if not low_context and not results:
            low_context = True

        return results, sources, low_context, refined_query

    def _describe_tool_call(self, tool_name: str, args: dict) -> str:
        """Return a human-readable thought describing what a tool is about to do."""
        arg_val = str(next(iter(args.values()), "")).strip()[:100] if args else ""
        _DESCRIPTIONS = {
            "crawl_diagnosis_articles":       lambda v: f'Searching clinical resources for "{v}"',
            "crawl_medical_articles":         lambda v: f'Searching medical literature for "{v}"',
            "search_pubmed":                  lambda v: f'Searching PubMed for "{v}"',
            "recommend_drugs":                lambda v: f'Looking up drug recommendations for "{v}"',
            "crawl_drug_interactions":        lambda v: f'Checking drug interactions for "{v}"',
            "check_drug_interactions":        lambda v: f'Checking interactions for "{v}"',
            "analyze_symptoms":               lambda v: f'Analyzing symptoms: "{v}"',
            "extract_document_text":          lambda v: "Extracting and reading document content",
            "langextract_structured_extract": lambda v: "Parsing structured data from document",
            "analyze_report":                 lambda v: "Analyzing the medical report",
            "extract_image_findings":         lambda v: "Analyzing medical image for findings",
            "get_patient_records":            lambda v: f'Retrieving records for "{v}"',
            "retrieve_patient_records":       lambda v: f'Retrieving patient records',
            "analyze_patient_history":        lambda v: "Reviewing patient history",
            "analyze_patient_vitals":         lambda v: "Reviewing patient vitals",
            "review_patient_medications":     lambda v: "Reviewing patient medications",
        }
        describe = _DESCRIPTIONS.get(tool_name)
        if describe:
            label = describe(arg_val)
        else:
            # Humanise any unlisted tool: underscores → spaces, capitalised
            human_name = tool_name.replace("_", " ").capitalize()
            label = f'{human_name}: "{arg_val}"' if arg_val else human_name
        return f"**[{self.name.title()}]**: {label}"

    # Prefixes that indicate a tool returned an error — suppress from user view.
    _ERROR_PREFIXES = ("error:", "warning:", "[tool error", "[structuring failed", "[langextract")

    def _summarize_observation(self, tool_name: str, observation: str) -> str:
        """One-sentence LLM summary of a tool observation for the thinking accordion.

        Error observations are suppressed — internal failures must not surface
        as user-visible thinking steps. A generic retry message is shown instead.
        Falls back to a plain snippet if the planner isn't ready.
        """
        # Suppress tool errors — never expose internal failure messages to the user.
        obs_lower = observation.lower().lstrip()
        if any(obs_lower.startswith(p) for p in self._ERROR_PREFIXES):
            return f"**[{self.name.title()}]**: Gathering additional data…"

        planner = self._planner_cached
        if planner is None or len(observation) < 60:
            snippet = observation[:200] + "…" if len(observation) > 200 else observation
            return f"**[{self.name.title()}]**: {snippet}"
        try:
            prompt = (
                f"Tool called: {tool_name}\n"
                f"Tool output (truncated to 800 chars):\n{observation[:800]}\n\n"
                "You are a medical AI reasoning through a query. Write ONE short internal "
                "thought (max 25 words) reflecting what you just learned from this tool output "
                "and how it helps you answer the question. "
                "Write in first person, present tense, as if thinking aloud. "
                "Do NOT mention errors, failures, or tool names. No preamble, no quotes."
            )
            msg = planner.invoke([HumanMessage(content=prompt)])
            summary = (msg.content or "").strip().splitlines()[0][:220]
            return f"**[{self.name.title()}]**: {summary}"
        except Exception:
            snippet = observation[:200] + "…" if len(observation) > 200 else observation
            return f"**[{self.name.title()}]**: {snippet}"

    def _call_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        tool_context: Dict[str, Any],
    ) -> str:
        """
        Execute a tool, transparently injecting tool_context keys that match
        the tool's function signature (e.g. pii_mapping_json for patient tools).
        Neither gemma4:31b-cloud nor MedGemma ever sees these injected values.
        """
        tool = self.tools.get(tool_name)
        if not tool:
            return f"Error: Tool '{tool_name}' not found. Available: {list(self.tools)}"

        try:
            merged_args = dict(args)
            if tool_context:
                func = getattr(tool, "func", tool)
                try:
                    sig_params = list(inspect.signature(func).parameters.keys())
                    injectable = {k: v for k, v in tool_context.items() if k in sig_params}
                except (ValueError, TypeError):
                    injectable = {}
                merged_args.update(injectable)

            observation = str(tool.invoke(merged_args))

            # HIPAA: only PHI-producing tools (document/image extraction) need
            # observation redaction. Web crawl tools return public literature
            # with no patient identifiers, so we skip Presidio there to avoid
            # false positives and unnecessary latency (~50-100ms per call).
            if tool_name in _PHI_PRODUCING_TOOLS:
                existing_mapping: Dict[str, str] = {}
                if tool_context.get("pii_mapping_json"):
                    try:
                        existing_mapping = json.loads(tool_context["pii_mapping_json"])
                    except Exception:
                        pass

                observation, new_mappings = _redact_observation(observation, existing_mapping)

                if new_mappings:
                    merged_mapping = {**existing_mapping, **new_mappings}
                    tool_context["pii_mapping_json"] = json.dumps(merged_mapping)

            return observation

        except Exception as e:
            logger.warning(f"[{self.name}] Tool '{tool_name}' error: {e}")
            return f"Error calling {tool_name}: {str(e)}"

    def _synthesize(
        self,
        user_input: str,
        tool_results: List[Tuple[str, str]],
    ) -> str:
        """
        Two-phase synthesis:

        Phase 2a — MedGemma: clinical analysis of gathered tool data.
          Receives raw tool observations and applies medical domain expertise to
          interpret findings, flag abnormalities, and reason clinically.

        Phase 2b — Gemma4:31b-cloud: human-friendly consolidation.
          Always runs after MedGemma. Receives MedGemma's clinical analysis and
          formats it into a clear, well-structured response for the user.
          This separation keeps medical reasoning with MedGemma while ensuring
          the final output is always coherent and readable.
        """
        # If every tool result is an error string, skip MedGemma and return the
        # error directly so the user gets a clean message instead of hallucinated output.
        if tool_results:
            all_errors = all(
                obs.strip().startswith("Error:") or obs.strip().startswith("Warning:")
                for _, obs in tool_results
            )
            if all_errors:
                return tool_results[0][1]  # return the first error message verbatim

        if tool_results:
            gathered = "\n\n".join(
                f"[{name} results]\n{obs}" for name, obs in tool_results
            )
            medgemma_prompt = (
                f"{self.system_prompt}\n\n"
                f"User Query: {user_input}\n\n"
                f"Gathered Data:\n{gathered}\n\n"
                f"Provide your clinical analysis of the gathered data above:"
            )
        else:
            medgemma_prompt = (
                f"{self.system_prompt}\n\n"
                f"User Query: {user_input}\n\n"
                f"Provide your clinical analysis:"
            )

        if self.skip_medgemma_synthesis:
            logger.info("medgemma_synthesis_skipped", agent=self.name, reason="tools already perform clinical synthesis")
            clinical_analysis = ""
        else:
            logger.info("medgemma_request", agent=self.name, prompt_chars=len(medgemma_prompt), prompt_preview=medgemma_prompt[:300])
            t0_synth = _time.monotonic()
            clinical_analysis = self.llm.invoke(medgemma_prompt)
            synth_rtt_ms = round((_time.monotonic() - t0_synth) * 1000)
            logger.info("medgemma_response", agent=self.name, rtt_ms=synth_rtt_ms, response_chars=len(clinical_analysis), response_preview=clinical_analysis[:500])

            if self._is_looping(clinical_analysis):
                logger.warning(
                    f"[{self.name}] MedGemma degenerate output — skipping to gemma4:31b-cloud consolidation",
                    looping_output=clinical_analysis[:200],
                )
                clinical_analysis = ""
            else:
                logger.info("llm_call", model="medgemma", agent=self.name, role="clinical_analysis", rtt_ms=synth_rtt_ms, chars=len(clinical_analysis))

        # Phase 2b — Gemma4 always consolidates into a human-friendly final response.
        from langchain_core.messages import HumanMessage as _HumanMessage
        consolidator = ChatOllama(
            model=settings.OLLAMA_CLOUD_MODEL,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            base_url=settings.OLLAMA_CLOUD_URL.removesuffix("/v1"),
        )
        if clinical_analysis:
            consolidation_prompt = (
                f"You are a medical communication specialist. A medical AI has produced the following "
                f"clinical analysis in response to this query: \"{user_input}\"\n\n"
                f"Clinical Analysis:\n{clinical_analysis}\n\n"
                f"Consolidate this into a clear, well-structured, human-friendly response. "
                f"Preserve all clinical facts, values, and recommendations. "
                f"Use markdown formatting with headers and bullet points where appropriate."
            )
        else:
            # MedGemma produced nothing useful — consolidate directly from tool data
            gathered_fallback = "\n\n".join(f"[{n}]\n{o}" for n, o in tool_results) if tool_results else ""
            consolidation_prompt = (
                f"You are a medical AI assistant. Answer this query using the gathered data below.\n\n"
                f"Query: {user_input}\n\n"
                f"Gathered Data:\n{gathered_fallback}\n\n"
                f"Provide a clear, well-structured clinical response."
            )
        t0_cons = _time.monotonic()
        output = consolidator.invoke([_HumanMessage(content=consolidation_prompt)]).content
        cons_rtt_ms = round((_time.monotonic() - t0_cons) * 1000)
        logger.info("llm_call", model=settings.OLLAMA_CLOUD_MODEL, agent=self.name, role="consolidation", rtt_ms=cons_rtt_ms, chars=len(output))
        return output

    @staticmethod
    def _is_looping(text: str, max_repeats: int = 3) -> bool:
        """Return True if output is degenerate: sentence loops OR backtick/whitespace garbage."""
        stripped = text.strip()
        if not stripped:
            return True

        # Degenerate: >60% of non-whitespace chars are backticks (e.g. 3072 chars of ```)
        non_ws = stripped.replace(" ", "").replace("\n", "")
        if non_ws and non_ws.count("`") / len(non_ws) > 0.6:
            return True

        # Degenerate: >80% of lines are empty or just punctuation/backticks
        lines = stripped.splitlines()
        if len(lines) > 10:
            junk_lines = sum(1 for l in lines if not l.strip() or set(l.strip()) <= {"`", " ", "#", "-"})
            if junk_lines / len(lines) > 0.8:
                return True

        # Token-level repetition: catch comma/space-delimited loops (e.g. Russian gibberish,
        # repeated short tokens with no sentence endings). Split on comma+space or newline.
        tokens = [t.strip().lower() for t in re.split(r'[,\n]+', stripped) if t.strip()]
        if len(tokens) >= 20:
            unique_ratio = len(set(tokens)) / len(tokens)
            if unique_ratio < 0.15:  # >85% of tokens are repeats
                return True

        # Sentence-level repetition
        sentences = re.split(r'(?<=[.!?])\s+', stripped)
        if len(sentences) < max_repeats + 1:
            return False
        counts: Dict[str, int] = {}
        for s in sentences:
            normalized = s.strip().lower()
            if not normalized:
                continue
            counts[normalized] = counts.get(normalized, 0) + 1
            if counts[normalized] > max_repeats:
                return True
        return False
