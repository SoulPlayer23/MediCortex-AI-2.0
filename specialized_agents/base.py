import logging
import re
from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool
from .medgemma_llm import MedGemmaLLM

# Setup Logger
logger = logging.getLogger("SpecializedAgents")

# Initialize Shared LLM
llm = MedGemmaLLM()

import logging
import re
import uuid
import datetime
from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool
from .medgemma_llm import MedGemmaLLM
from .protocols import AgentCard, Envelope, AgentResponse

# Setup Logger
logger = logging.getLogger("SpecializedAgents")

# Initialize Shared LLM
llm = MedGemmaLLM()

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

        try:
            output = self._execute_rect_loop(user_input)
            return AgentResponse(
                envelope_id=envelope.idempotency_key,
                output=output
            )
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
        return {"output": resp.output}

    def _execute_rect_loop(self, user_input: str) -> str:
        scratchpad = ""
        
        for i in range(self.max_iterations):
            # Construct Prompt
            prompt = self.template.format(input=user_input, agent_scratchpad=scratchpad)
            
            # Call LLM
            logger.info(f"[{self.name}] Thinking (Step {i+1})...")
            response = self.llm.invoke(prompt)
            
            # Parse Response
            action_match = re.search(r"Action:\s*(.+)", response)
            input_match = re.search(r"Action Input:\s*(.+)", response)
            final_answer_match = re.search(r"Final Answer:\s*(.+)", response, re.DOTALL)
            
            if final_answer_match:
                return final_answer_match.group(1).strip()
            
            if action_match and input_match:
                action = action_match.group(1).strip()
                action_input = input_match.group(1).strip()
                
                # Check for brackets removal if model adds them
                action = action.strip("[]")
                
                tool = self.tools.get(action)
                if tool:
                    logger.info(f"[{self.name}] Calling Tool: {action} with '{action_input}'")
                    try:
                        observation = tool.invoke(action_input)
                    except Exception as e:
                        observation = f"Error: {str(e)}"
                else:
                    observation = f"Error: Tool '{action}' not found."
                
                # Update scratchpad
                scratchpad += f"{response}\nObservation: {observation}\n"
                continue
            
            # Fallback
            if "Thought:" not in response and "Action:" not in response:
                 return response.strip()
            
            # If stuck
            scratchpad += f"{response}\nObservation: Could not parse action. Review format.\n"

        return "Agent reached max iterations without final answer."
