import inspect
import logging
import re
import uuid
import json
import redis
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.tools import BaseTool
from langchain_core.callbacks import BaseCallbackHandler

from .medgemma_llm import MedGemmaLLM
from .protocols import AgentCard, Envelope, AgentResponse
from config import settings

logger = logging.getLogger("SpecializedAgents")
llm = MedGemmaLLM()

class AgentStreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler to intercept LLM tokens and route them to a streaming queue/list."""
    def __init__(self, callback_list: list):
        self.callback_list = callback_list

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        # If we just want thoughts, token-by-token isn't strictly necessary for the thought accordion,
        # but we can capture it here if we want to stream the raw ReAct text. We'll skip it for now and
        # rely on the regex parser to emit complete thoughts.
        pass

class A2ABaseAgent:
    """
    Base Agent implementing the A2A Protocol.
    """
    def __init__(self, name: str, llm: MedGemmaLLM, tools: List[BaseTool], system_prompt: str, card: AgentCard, max_iterations: int = 3):
        self.name = name
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.system_prompt = system_prompt
        self.card = card
        self.max_iterations = max_iterations
        
        # Idempotency Cache: In-memory fallback
        self._response_cache: Dict[str, AgentResponse] = {}
        # Optional Redis Backing
        self._redis_cache = None
        try:
            if getattr(settings, "REDIS_URL", None):
                self._redis_cache = redis.from_url(settings.REDIS_URL, decode_responses=True)
                self._redis_cache.ping() # test connection
                logger.info(f"[{self.name}] Connected to Redis Cache.")
        except Exception as e:
            logger.warning(f"[{self.name}] Redis connection failed, falling back to in-memory cache: {e}")
            self._redis_cache = None

        # Build prompt template
        tool_desc = "\n".join([f"{t.name}: {t.description}" for t in tools])
        tool_names = ", ".join([f"[{t.name}]" for t in tools])
        
        self.template = f"""You are the {name}.
{system_prompt}

IMPORTANT: You may receive a 'Conversation History' in your input. USE IT to maintain context (e.g., patient age, previous diagnosis). 
Do not ask for information that has already been provided in the history.

TOOLS:
------
You have access to the following tools:

{tool_desc}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of {tool_names}
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

New input: {{input}}
{{agent_scratchpad}}
"""

    def get_card(self) -> AgentCard:
        return self.card

    def process(self, envelope: Envelope) -> AgentResponse:
        """
        Main entry point for A2A communication.
        1. Validates Envelope
        2. Validates Payload against Schema
        3. Executes Logic
        4. Returns AgentResponse
        """
        logger.info(f"[{self.name}] Received Envelope: {envelope.idempotency_key} from {envelope.sender_id}")
        
        # Schema Validation
        # simplified check: ensure keys match
        # in prod, use logic to validate payload against self.card.input_schema
        
        user_input = envelope.payload.get("input", "")
        if not user_input:
             return AgentResponse(
                envelope_id=envelope.idempotency_key,
                output=None,
                error="Validation Error: 'input' field missing in payload."
            )

        # ── Idempotency Cache Check ──
        cache_key = f"medicortex:idempotency:{envelope.idempotency_key}"
        
        # 1. Check Redis
        if self._redis_cache:
            try:
                cached_json = self._redis_cache.get(cache_key)
                if cached_json:
                    logger.info(f"[{self.name}] Cache HIT (Redis) for {envelope.idempotency_key}")
                    import pydantic
                    try: # Support pydantic v1 parse_raw and v2 model_validate_json
                        return AgentResponse.model_validate_json(cached_json)
                    except AttributeError:
                        return AgentResponse.parse_raw(cached_json)
            except Exception as e:
                logger.warning(f"[{self.name}] Failed to retrieve from Redis: {e}")
        
        # 2. Check In-Memory (Fallback)
        if envelope.idempotency_key in self._response_cache:
            logger.info(f"[{self.name}] Cache HIT (In-Memory) for {envelope.idempotency_key}")
            return self._response_cache[envelope.idempotency_key]
        
        logger.info(f"[{self.name}] Cache MISS for {envelope.idempotency_key}. Executing logic...")
        
        # ── Execution ──
        try:
            live_thoughts_queue = envelope.payload.get("live_thoughts_queue")

            # Build tool_context from payload fields that tools may need.
            # pii_mapping_json is injected here so patient tools can resolve
            # redacted placeholders without real names ever appearing in the LLM prompt.
            tool_context: Dict[str, Any] = {}
            if pii_json := envelope.payload.get("pii_mapping_json"):
                tool_context["pii_mapping_json"] = pii_json

            output, thinking_steps = self._execute_rect_loop(user_input, live_thoughts_queue, tool_context)
            
            response = AgentResponse(
                envelope_id=envelope.idempotency_key,
                output=output,
                thinking=thinking_steps
            )
            
            # ── Cache Write ──
            if self._redis_cache:
                try:
                    try:
                        resp_json = response.model_dump_json()
                    except AttributeError:
                        resp_json = response.json()
                    # Cache for 24 hours
                    self._redis_cache.setex(cache_key, 86400, resp_json)
                except Exception as e:
                    logger.warning(f"[{self.name}] Failed to save to Redis: {e}")
            else:
                self._response_cache[envelope.idempotency_key] = response
                
            return response
            
        except Exception as e:
            logger.error(f"[{self.name}] Error processing request: {e}")
            return AgentResponse(
                envelope_id=envelope.idempotency_key,
                output=None,
                error=str(e)
            )

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Legacy invoke method for compatibility, wrappers around process()
        """
        # Create a dummy envelope if called directly
        env = Envelope(
            sender_id="legacy_invoke",
            receiver_id=self.name,
            payload=inputs
        )
        resp = self.process(env)
        if resp.error:
            raise Exception(resp.error)
        return {"output": resp.output, "thinking": resp.thinking}

    def _execute_rect_loop(
        self,
        user_input: str,
        live_thoughts_queue: Optional[list] = None,
        tool_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, List[str]]:
        scratchpad = ""
        thinking_steps: List[str] = []
        tool_context = tool_context or {}
        
        def emit_thought(t: str):
            thinking_steps.append(t)
            if live_thoughts_queue is not None:
                live_thoughts_queue.append(t)

        for i in range(self.max_iterations):
            # Construct Prompt
            prompt = self.template.format(input=user_input, agent_scratchpad=scratchpad)
            
            # Force the model to start with Thought if it's the first iteration or scratchpad is empty
            if not scratchpad.strip():
                 prompt += "\nFormat your first string starting strictly with 'Thought:' or 'Final Answer:'"
            
            # Call LLM
            step_msg = f"Thinking (Step {i+1})..."
            logger.info(f"[{self.name}] {step_msg}")
            
            response = self.llm.invoke(prompt)
            
            # Clean response of potential markdown code blocks
            clean_resp = response.replace("```json", "").replace("```", "").strip()

            # Robust ReAct parsing
            thought_text = ""
            action_text = ""
            action_input_text = ""
            final_answer_text = ""

            # Try to extract Thought
            thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|\nFinal Answer:|$)", clean_resp, re.DOTALL | re.IGNORECASE)
            if thought_match:
                thought_text = thought_match.group(1).strip()
            elif "Thought:" in clean_resp and "Action:" not in clean_resp and "Final Answer:" not in clean_resp:
                # The whole response might just be a thought
                thought_text = clean_resp.replace("Thought:", "").strip()

            # Try to extract Action & Input
            action_match = re.search(r"Action:\s*([^\n]+)", clean_resp, re.IGNORECASE)
            input_match = re.search(r"Action Input:\s*(.*?)(?=\nObservation:|\nThought:|\nFinal Answer:|$)", clean_resp, re.DOTALL | re.IGNORECASE)
            
            if action_match:
                action_text = action_match.group(1).strip()
            if input_match:
                action_input_text = input_match.group(1).strip()

            # Try to extract Final Answer
            final_answer_match = re.search(r"Final Answer:\s*(.+)", clean_resp, re.DOTALL | re.IGNORECASE)
            if final_answer_match:
                final_answer_text = final_answer_match.group(1).strip()

            # Emit thought
            if thought_text:
                emit_thought(f"**[{self.name.title()}]**: {thought_text}")
            elif clean_resp:
                # If no formal thought block was found but there is text, emit a chunk of it as a thought
                snippet = clean_resp[:300] + "..." if len(clean_resp) > 300 else clean_resp
                emit_thought(f"**[{self.name.title()}]**: {snippet}")

            if final_answer_text:
                return final_answer_text, thinking_steps
            
            if action_text and action_input_text:
                action = action_text
                action_input = action_input_text
                
                # Check for brackets removal if model adds them
                action = action.strip("[]")
                
                tool = self.tools.get(action)
                observation_msg = ""
                if tool:
                    logger.info(f"[{self.name}] Calling Tool: {action} with '{action_input}'")
                    action_thought = f"**Action**: Calling tool `{action}` with input `{action_input}`"
                    emit_thought(action_thought)
                    try:
                        # Inject tool_context keys (e.g. pii_mapping_json) transparently
                        # so the LLM never needs to pass them explicitly in Action Input.
                        if tool_context:
                            try:
                                func = getattr(tool, "func", tool)
                                params = list(inspect.signature(func).parameters.keys())
                                injectable = {k: v for k, v in tool_context.items() if k in params[1:]}
                            except (ValueError, TypeError):
                                injectable = {}
                            if injectable:
                                first_param = params[0] if params else "__arg1"
                                observation = tool.invoke({first_param: action_input, **injectable})
                            else:
                                observation = tool.invoke(action_input)
                        else:
                            observation = tool.invoke(action_input)
                        observation_msg = f"Tool Output: {str(observation)[:200]}..."
                    except Exception as e:
                        observation = f"Error: {str(e)}"
                        observation_msg = f"Tool Error: {str(e)}"
                else:
                    observation = f"Error: Tool '{action}' not found."
                    observation_msg = f"Error: Tool '{action}' not found."
                
                obs_thought = f"**Observation**: {observation_msg}"
                emit_thought(obs_thought)

                # Update scratchpad
                scratchpad += f"{response}\nObservation: {observation}\n"
                continue
            
            # Fallback
            if "Thought:" not in response and "Action:" not in response and "Final Answer:" not in response:
                 # If the model just outputs dialogue, treat it as a final answer to prevent infinite looping
                 return response.strip(), thinking_steps
            
            # If stuck parsing Action
            scratchpad += f"{response}\nObservation: Could not parse Action or Final Answer. Please stick EXACTLY to the requested format (Thought: -> Action: -> Action Input: OR Final Answer:).\n"

        return "Agent reached max iterations without final answer.", thinking_steps
