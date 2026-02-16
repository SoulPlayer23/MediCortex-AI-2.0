from .base import A2ABaseAgent, llm
from .protocols import AgentCard
from tools.patient_tools import search_patient_records

# Define Agent Card
patient_card = AgentCard(
    name="patient_retriever",
    description="Specialized in retrieving patient history, demographics, and electronic health records securetly.",
    input_schema={
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "Patient Name or ID"}
        },
        "required": ["input"]
    },
    output_schema={
        "type": "object",
        "properties": {
            "output": {"type": "string", "description": "Patient record details"}
        }
    },
    capabilities=["ehr-search", "patient-history"]
)

# Instantiate Agent
patient_agent = A2ABaseAgent(
    name="patient_retriever",
    llm=llm,
    tools=[search_patient_records],
    system_prompt="Your goal is to retrieve patient history and demographics securely.",
    card=patient_card
)
