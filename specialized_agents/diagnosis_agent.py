from .base import A2ABaseAgent, llm
from .protocols import AgentCard
from tools.diagnosis_tools import consult_medical_guidelines

# Define Agent Card
diagnosis_card = AgentCard(
    name="diagnosis",
    description="Specialized in analyzing symptoms and suggesting potential diagnoses based on medical guidelines.",
    input_schema={
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "Symptoms or patient presentation"}
        },
        "required": ["input"]
    },
    output_schema={
        "type": "object",
        "properties": {
            "output": {"type": "string", "description": "Differential diagnosis and considerations"}
        }
    },
    capabilities=["diagnosis", "symptom-analysis"]
)

# Instantiate Agent
diagnosis_agent = A2ABaseAgent(
    name="diagnosis",
    llm=llm,
    tools=[consult_medical_guidelines],
    system_prompt="Your goal is to suggest potential diagnoses based on symptoms. ALWAYS reference medical guidelines.",
    card=diagnosis_card
)
