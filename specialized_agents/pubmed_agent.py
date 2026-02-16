from .base import A2ABaseAgent, llm
from .protocols import AgentCard
from tools.pubmed_tools import search_pubmed

# Define Agent Card
pubmed_card = AgentCard(
    name="pubmed",
    description="Specialized in retrieving medical literature, research papers, and clinical studies from PubMed.",
    input_schema={
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "The research query or topic"}
        },
        "required": ["input"]
    },
    output_schema={
        "type": "object",
        "properties": {
            "output": {"type": "string", "description": "Summary of research findings"}
        }
    },
    capabilities=["research", "literature-review", "medical-papers"]
)

# Instantiate Agent
pubmed_agent = A2ABaseAgent(
    name="pubmed",
    llm=llm,
    tools=[search_pubmed],
    system_prompt="Your goal is to find relevant medical literature. Query PubMed for the user's specific medical topic. Always cite sources.",
    card=pubmed_card
)
