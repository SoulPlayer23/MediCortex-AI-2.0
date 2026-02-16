from .base import A2ABaseAgent, llm
from .protocols import AgentCard
from tools.report_tools import parse_lab_values

# Define Agent Card
report_card = AgentCard(
    name="report_analyzer",
    description="Specialized in extracting and interpreting structured data from unstructured medical reports (PDFs) or medical images.",
    input_schema={
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "Text content or file path of the report"}
        },
        "required": ["input"]
    },
    output_schema={
        "type": "object",
        "properties": {
            "output": {"type": "string", "description": "Structured extraction of lab values and findings"}
        }
    },
    capabilities=["ocr", "report-analysis", "lab-values"]
)

# Instantiate Agent
report_agent = A2ABaseAgent(
    name="report_analyzer",
    llm=llm,
    tools=[parse_lab_values],
    system_prompt="Your goal is to extract and interpret structured data from unstructured medical reports (PDFs) or medical images (X-Rays, MRIs). For images, analyze the metadata and visual features.",
    card=report_card
)
