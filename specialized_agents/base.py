import inspect
import logging
import redis
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from .medgemma_llm import MedGemmaLLM
from .protocols import AgentCard, Envelope, AgentResponse
from config import settings

logger = logging.getLogger("SpecializedAgents")

# MedGemma — used exclusively for clinical synthesis (Phase 2).
# Tool orchestration is handled by GPT-4o-mini (Phase 1).
llm = MedGemmaLLM()


class A2ABaseAgent:
    """
    Base Agent implementing the A2A Protocol with a two-phase execution model:

      Phase 1 — GPT-4o-mini (planner):
        Decides which tools to call, calls them, collects observations.
        Emits tool thoughts in real-time for SSE streaming.

      Phase 2 — MedGemma (synthesizer):
        Receives the original query + all gathered tool results.
        Called exactly once to produce the final clinical response.

    This cleanly separates agentic orchestration (GPT-4o-mini's strength) from
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

        # Idempotency cache — Redis with in-memory fallback
        self._response_cache: Dict[str, AgentResponse] = {}
        self._redis_cache = None
        try:
            if getattr(settings, "REDIS_URL", None):
                self._redis_cache = redis.from_url(settings.REDIS_URL, decode_responses=True)
                self._redis_cache.ping()
                logger.info(f"[{self.name}] Connected to Redis cache.")
        except Exception as e:
            logger.warning(f"[{self.name}] Redis unavailable, using in-memory cache: {e}")
            self._redis_cache = None

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
            # inspect.signature, keeping PII out of both GPT-4o-mini and MedGemma.
            tool_context: Dict[str, Any] = {}
            if pii_json := envelope.payload.get("pii_mapping_json"):
                tool_context["pii_mapping_json"] = pii_json
            if kc := envelope.payload.get("knowledge_context"):
                tool_context["knowledge_context"] = kc

            output, thinking_steps = self._plan_and_synthesize(
                user_input, live_thoughts_queue, tool_context
            )

            response = AgentResponse(
                envelope_id=envelope.idempotency_key,
                output=output,
                thinking=thinking_steps,
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
    ) -> Tuple[str, List[str]]:
        """
        Two-phase pipeline:
          Phase 1 — GPT-4o-mini gathers tool data (emits thoughts in real-time).
          Phase 2 — MedGemma synthesizes the final clinical response (single call).
        """
        thinking_steps: List[str] = []
        tool_context = tool_context or {}

        def emit_thought(t: str):
            thinking_steps.append(t)
            if live_thoughts_queue is not None:
                live_thoughts_queue.append(t)

        # Phase 1 — tool gathering
        try:
            tool_results = self._gather_tool_results(user_input, emit_thought, tool_context)
        except Exception as e:
            logger.error(f"[{self.name}] Tool gathering failed: {e}")
            tool_results = []
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

        return final_answer, thinking_steps

    def _gather_tool_results(
        self,
        user_input: str,
        emit_thought,
        tool_context: Dict[str, Any],
    ) -> List[Tuple[str, str]]:
        """
        Phase 1: GPT-4o-mini decides which tools to call and in what order.

        Uses LangChain bind_tools() so tool schemas are generated automatically.
        Loops up to self.max_iterations times to allow multi-tool pipelines
        (e.g. patient agent: retrieve → history → vitals → medications).

        tool_context keys (pii_mapping_json, knowledge_context) are injected
        at call time by _call_tool() and are never visible to GPT-4o-mini.
        """
        planner = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=settings.OPENAI_API_KEY,
        )
        planner_with_tools = planner.bind_tools(list(self.tools.values()))

        messages = [
            SystemMessage(content=(
                f"You are a data-gathering orchestrator for the {self.name} medical agent. "
                f"Your ONLY job is to call the available tools to collect all information "
                f"needed to answer the user's medical query. "
                f"Do NOT generate a final answer or explain your reasoning — "
                f"only call tools. Stop when you have gathered sufficient data."
            )),
            HumanMessage(content=user_input),
        ]

        results: List[Tuple[str, str]] = []

        for _ in range(self.max_iterations):
            response = planner_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                # GPT-4o-mini decided no more tools needed
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

                snippet = observation[:300] + "…" if len(observation) > 300 else observation
                emit_thought(f"**Observation** (`{tool_name}`): {snippet}")

                messages.append(ToolMessage(content=observation, tool_call_id=tc["id"]))

        return results

    def _call_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        tool_context: Dict[str, Any],
    ) -> str:
        """
        Execute a tool, transparently injecting tool_context keys that match
        the tool's function signature (e.g. pii_mapping_json for patient tools).
        Neither GPT-4o-mini nor MedGemma ever sees these injected values.
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

        return self.llm.invoke(prompt)
