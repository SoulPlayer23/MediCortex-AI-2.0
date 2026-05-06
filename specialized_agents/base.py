import inspect
import logging
import re
import redis
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama

from .medgemma_llm import MedGemmaLLM
from .protocols import AgentCard, Envelope, AgentResponse
from config import settings

logger = logging.getLogger("SpecializedAgents")

# MedGemma — used exclusively for clinical synthesis (Phase 2).
# Tool orchestration is handled by Gemma3:1b (Phase 1).
llm = MedGemmaLLM()


class A2ABaseAgent:
    """
    Base Agent implementing the A2A Protocol with a two-phase execution model:

      Phase 1 — FunctionGemma 270M (planner):
        Decides which tools to call and in what order. Uses bind_tools() —
        FunctionGemma is fine-tuned exclusively for function calling (gemma3:1b
        does not support the Ollama tool-calling API). Emits tool thoughts in
        real-time for SSE streaming.

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
    ):
        self.name = name
        self.llm = llm              # MedGemma — synthesis only
        self.tools = {t.name: t for t in tools}
        self.system_prompt = system_prompt
        self.card = card
        self.max_iterations = max_iterations

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
            # inspect.signature, keeping PII out of both Gemma3:1b and MedGemma.
            tool_context: Dict[str, Any] = {}
            if pii_json := envelope.payload.get("pii_mapping_json"):
                tool_context["pii_mapping_json"] = pii_json
            if kc := envelope.payload.get("knowledge_context"):
                tool_context["knowledge_context"] = kc

            output, thinking_steps, sources, low_context, refined_query = self._plan_and_synthesize(
                user_input, live_thoughts_queue, tool_context
            )

            response = AgentResponse(
                envelope_id=envelope.idempotency_key,
                output=output,
                thinking=thinking_steps,
                sources=sources,
                low_context=low_context,
                refined_query=refined_query,
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
          Phase 1 — Gemma3:1b gathers tool data (emits thoughts in real-time).
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
        Phase 1: Gemma3:1b decides which tools to call and in what order.

        Uses LangChain bind_tools() so tool schemas are generated automatically.
        Loops up to self.max_iterations times to allow multi-tool pipelines
        (e.g. patient agent: retrieve → history → vitals → medications).

        tool_context keys (pii_mapping_json, knowledge_context) are injected
        at call time by _call_tool() and are never visible to Gemma3:1b.

        Returns (tool_results, sources, low_context, refined_query) where:
        - low_context is True when tool observations are empty or thin (<200 chars total)
          OR when the planner emits a CONTEXT_INSUFFICIENT:<term> sentinel.
        - refined_query is a more specific KB search term suggested by the planner,
          or None if context was sufficient.
        """
        # OPS-4 / BUG-9: build planner + bind_tools once per agent instance.
        # FunctionGemma (270M) is used exclusively here — it is fine-tuned for
        # function calling and supports the Ollama tool API. gemma3:1b does not.
        if self._planner_with_tools_cached is None:
            self._planner_cached = ChatOllama(
                model=getattr(settings, "OLLAMA_TOOL_MODEL", "functiongemma"),
                temperature=0.0,  # FunctionGemma performs best deterministically
                base_url=settings.OLLAMA_CLOUD_URL.removesuffix("/v1"),
                timeout=getattr(settings, "OLLAMA_TIMEOUT_SECONDS", 60),
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
            response = planner_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                # Gemma3:1b decided no more tools needed
                break

            for tc in response.tool_calls:
                tool_name = tc["name"]
                args = tc["args"]

                # Surface the tool call as a thought for the UI accordion
                arg_summary = next(iter(args.values()), "") if args else ""
                emit_thought(
                    f"**[{self.name.title()}]**: Calling `{tool_name}` "
                    f"with `{str(arg_summary)[:120]}`"
                )
                logger.info(f"[{self.name}] Tool call: {tool_name}({args})")

                observation = self._call_tool(tool_name, args, tool_context)
                results.append((tool_name, observation))

                # Extract titled sources from this observation
                for src in self._extract_sources_from_observation(observation):
                    if src["url"] not in seen_urls:
                        seen_urls.add(src["url"])
                        sources.append(src)

                snippet = observation[:300] + "…" if len(observation) > 300 else observation
                emit_thought(f"**Observation** (`{tool_name}`): {snippet}")

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

    def _call_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        tool_context: Dict[str, Any],
    ) -> str:
        """
        Execute a tool, transparently injecting tool_context keys that match
        the tool's function signature (e.g. pii_mapping_json for patient tools).
        Neither Gemma3:1b nor MedGemma ever sees these injected values.
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

            return str(tool.invoke(merged_args))

        except Exception as e:
            logger.warning(f"[{self.name}] Tool '{tool_name}' error: {e}")
            return f"Error calling {tool_name}: {str(e)}"

    def _synthesize(
        self,
        user_input: str,
        tool_results: List[Tuple[str, str]],
    ) -> str:
        """
        Phase 2: MedGemma is called exactly once with the original query and all
        gathered tool data. It produces the final clinical response without any
        awareness of tool orchestration — it only sees medical content.

        After synthesis, a repetition guard checks whether any sentence appears
        more than 3 times. If so, MedGemma has entered a loop — the output is
        discarded and Gemma3:1b synthesizes instead.
        """
        if tool_results:
            gathered = "\n\n".join(
                f"[{name} results]\n{obs}" for name, obs in tool_results
            )
            prompt = (
                f"{self.system_prompt}\n\n"
                f"User Query: {user_input}\n\n"
                f"Gathered Data:\n{gathered}\n\n"
                f"Using the gathered data above, provide your complete clinical response:"
            )
        else:
            # No tools were called — answer from medical knowledge alone
            prompt = (
                f"{self.system_prompt}\n\n"
                f"User Query: {user_input}\n\n"
                f"Provide your clinical response:"
            )

        output = self.llm.invoke(prompt)

        # Repetition guard: MedGemma sometimes loops a single sentence when it
        # receives a prompt it cannot ground (e.g. empty KB context). Detect and
        # fall back to Gemma3:1b rather than returning garbage to the user.
        if self._is_looping(output):
            logger.warning(
                f"[{self.name}] MedGemma loop detected — falling back to Gemma3:1b"
            )
            try:
                from langchain_core.messages import HumanMessage as _HumanMessage
                fallback = ChatOllama(
                    model=settings.OLLAMA_CLOUD_MODEL,
                    temperature=1.0,
                    top_p=0.95,
                    top_k=64,
                    base_url=settings.OLLAMA_CLOUD_URL.removesuffix("/v1"),
                )
                output = fallback.invoke([_HumanMessage(content=prompt)]).content
                logger.info(f"[{self.name}] synthesis complete", model="gemma3_fallback", chars=len(output))
            except Exception as e:
                logger.error(f"[{self.name}] Gemma3:1b fallback also failed: {e}")
        else:
            logger.info(f"[{self.name}] synthesis complete", model="medgemma", chars=len(output))
        return output

    @staticmethod
    def _is_looping(text: str, max_repeats: int = 3) -> bool:
        """Return True if any sentence in *text* appears more than *max_repeats* times."""
        # Split on sentence-ending punctuation followed by whitespace or end-of-string
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
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
