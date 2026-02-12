import logging
import re
from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool
from .medgemma_llm import MedGemmaLLM
from .tools import (
    search_pubmed,
    consult_medical_guidelines,
    parse_lab_values,
    search_patient_records,
    check_drug_interactions
)

# Setup Logger
logger = logging.getLogger("SpecializedAgents")

# Initialize Shared LLM
llm = MedGemmaLLM()

class SimpleReactAgent:
    """
    A lightweight ReAct agent implementation for models without native tool support.
    """
    def __init__(self, name: str, llm: MedGemmaLLM, tools: List[BaseTool], system_prompt: str, max_iterations: int = 3):
        self.name = name
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.system_prompt = system_prompt
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

    def invoke(self, inputs: Dict[str, str]) -> Dict[str, str]:
        user_input = inputs.get("input", "")
        scratchpad = ""
        
        for i in range(self.max_iterations):
            # Construct Prompt
            prompt = self.template.format(input=user_input, agent_scratchpad=scratchpad)
            
            # Call LLM
            logger.info(f"[{self.name}] Thinking (Step {i+1})...")
            response = self.llm.invoke(prompt)
            
            # Parse Response
            # Look for Action and Action Input
            action_match = re.search(r"Action:\s*(.+)", response)
            input_match = re.search(r"Action Input:\s*(.+)", response)
            final_answer_match = re.search(r"Final Answer:\s*(.+)", response, re.DOTALL)
            
            if final_answer_match:
                return {"output": final_answer_match.group(1).strip()}
            
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
            
            # If no action and no final answer, treat as final answer or retry
            # For simplicity, if we get raw text without formatting, assume it's the answer
            if "Thought:" not in response and "Action:" not in response:
                 return {"output": response.strip()}
            
            # If stuck, break
            scratchpad += f"{response}\nObservation: Could not parse action. Review format.\n"

        return {"output": "Agent reached max iterations without final answer."}

# --- Agent Definitions ---

# 1. PubMed Retriever Agent
pubmed_agent = SimpleReactAgent(
    name="PubMed Retriever Agent",
    llm=llm,
    tools=[search_pubmed],
    system_prompt="Your goal is to find relevant medical literature. Query PubMed for the user's specific medical topic."
)

# 2. Diagnosis Agent
diagnosis_agent = SimpleReactAgent(
    name="Diagnosis Agent",
    llm=llm,
    tools=[consult_medical_guidelines],
    system_prompt="Your goal is to suggest potential diagnoses based on symptoms. ALWAYS reference medical guidelines."
)

# 3. Report Analyzer Agent
report_agent = SimpleReactAgent(
    name="Report Analyzer Agent",
    llm=llm,
    tools=[parse_lab_values],
    system_prompt="Your goal is to extract and interpret structured data from unstructured medical reports (PDFs) or medical images (X-Rays, MRIs). For images, analyze the metadata and visual features."
)

# 4. Patient Retriever Agent
patient_agent = SimpleReactAgent(
    name="Patient Retriever Agent",
    llm=llm,
    tools=[search_patient_records],
    system_prompt="Your goal is to retrieve patient history and demographics securely."
)

# 5. Drug Interaction Agent
drug_agent = SimpleReactAgent(
    name="Drug Interaction Agent",
    llm=llm,
    tools=[check_drug_interactions],
    system_prompt=(
        "Your goal is to identify dangerous drug interactions and contraindications. "
        "Check the 'Conversation History' for patient conditions (`Prostate Cancer`, `Diabetes`) or previously mentioned drugs. "
        "If a specific medication list is provided, check for interactions between them or with the patient's condition. "
        "If NO medications are listed in the user query, look for a condition in the history and provide 'Standard Contraindications' for that condition. "
        "Do NOT ask for clarification unless absolutely necessary."
    )
)

AGENT_REGISTRY = {
    "pubmed": pubmed_agent,
    "diagnosis": diagnosis_agent,
    "report": report_agent,
    "patient": patient_agent,
    "drug": drug_agent
}
