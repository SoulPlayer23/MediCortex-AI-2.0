from .base import A2ABaseAgent, llm
from .protocols import AgentCard
from tools.drug_tools import check_drug_interactions

# Define Agent Card
drug_card = AgentCard(
    name="drug_interaction",
    description="Specialized in identifying dangerous drug interactions, contraindications, and side effects.",
    input_schema={
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "Medication list or patient condition"}
        },
        "required": ["input"]
    },
    output_schema={
        "type": "object",
        "properties": {
            "output": {"type": "string", "description": "Interaction warnings and safety analysis"}
        }
    },
    capabilities=["drug-safety", "interaction-check"]
)

# Instantiate Agent
drug_agent = A2ABaseAgent(
    name="drug_interaction",
    llm=llm,
    tools=[check_drug_interactions],
    system_prompt=(
        "Your goal is to identify dangerous drug interactions and contraindications. "
        "Check the 'Conversation History' for patient conditions (`Prostate Cancer`, `Diabetes`) or previously mentioned drugs. "
        "If a specific medication list is provided, check for interactions between them or with the patient's condition. "
        "If NO medications are listed in the user query, look for a condition in the history and provide 'Standard Contraindications' for that condition. "
        "Do NOT ask for clarification unless absolutely necessary."
    ),
    card=drug_card
)
