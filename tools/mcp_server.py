import asyncio
import logging
import sys
import os

# Add project root to sys.path to ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server

# Import tools
from tools.pubmed_search_tools import search_pubmed
from tools.medical_webcrawler_tools import crawl_medical_articles
from tools.diagnosis_tools import consult_medical_guidelines
from tools.report_tools import parse_lab_values
from tools.patient_tools import search_patient_records
from tools.drug_tools import check_drug_interactions

# Import agent registry for MCP Resources (MCP §3.1)
from specialized_agents.agents import AGENT_REGISTRY

# Setup Logging
# We use stderr so it doesn't interfere with stdout (which is used for MCP protocol)
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("mcp_server")

server = Server("medicortex-mcp")

# ── MCP Resources: Agent Card Discovery (MCP §3.1) ──────────────────
@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """Expose agent cards as readable MCP Resources."""
    resources = []
    for name, agent in AGENT_REGISTRY.items():
        card = agent.get_card()
        resources.append(
            types.Resource(
                uri=f"agents://medicortex/{name}/card",
                name=f"{card.name} Agent Card",
                description=card.description,
                mimeType="application/json",
            )
        )
    return resources

@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Read a specific agent card resource by URI."""
    import json as _json
    # Parse URI: agents://medicortex/{agent_name}/card
    parts = uri.replace("agents://medicortex/", "").split("/")
    if len(parts) < 1:
        raise ValueError(f"Invalid resource URI: {uri}")
    agent_name = parts[0]
    agent = AGENT_REGISTRY.get(agent_name)
    if not agent:
        raise ValueError(f"Agent '{agent_name}' not found")
    return _json.dumps(agent.get_card().model_dump(), indent=2)

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools."""
    return [
        types.Tool(
            name="search_pubmed",
            description="Search the NCBI PubMed database for medical research papers. Returns paper titles, authors, journal, year, abstract excerpt, DOI, and a direct PubMed URL. Use for evidence-based research and clinical literature.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Focused medical topic or research question"},
                    "max_results": {"type": "integer", "description": "Number of papers to return (1-20, default 5)", "default": 5}
                },
                "required": ["query"]
            },
        ),
        types.Tool(
            name="crawl_medical_articles",
            description="Search the web for medical articles from trusted healthcare sites (Mayo Clinic, NIH, CDC, WHO, WebMD, Medscape, Cleveland Clinic, BMJ, NEJM, etc.). Returns article titles, source domains, URLs, summaries, and content excerpts.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Medical topic to search for"},
                    "max_results": {"type": "integer", "description": "Number of articles to return (1-10, default 5)", "default": 5}
                },
                "required": ["query"]
            },
        ),
        types.Tool(
            name="consult_medical_guidelines",
            description="Consult medical guidelines for symptoms and diagnosis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symptom": {"type": "string", "description": "Symptom or condition description"}
                },
                "required": ["symptom"]
            },
        ),
        types.Tool(
            name="parse_lab_values",
            description="Extract structured lab values from medical report text.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Report text or content"}
                },
                "required": ["text"]
            },
        ),
        types.Tool(
            name="search_patient_records",
            description="Search for patient health records.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Patient Name or ID"}
                },
                "required": ["query"]
            },
        ),
        types.Tool(
            name="check_drug_interactions",
            description="Check for drug interactions and contraindications.",
            inputSchema={
                "type": "object",
                "properties": {
                    "medication_list": {"type": "string", "description": "List of medications or conditions"}
                },
                "required": ["medication_list"]
            },
        ),
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool execution requests."""
    if not arguments:
        raise ValueError("Missing arguments")

    logger.info(f"Executing tool: {name}")

    try:
        if name == "search_pubmed":
            query = arguments.get("query")
            max_results = arguments.get("max_results", 5)
            result = search_pubmed.invoke({"query": query, "max_results": max_results})
            return [types.TextContent(type="text", text=str(result))]

        elif name == "crawl_medical_articles":
            query = arguments.get("query")
            max_results = arguments.get("max_results", 5)
            result = crawl_medical_articles.invoke({"query": query, "max_results": max_results})
            return [types.TextContent(type="text", text=str(result))]

        elif name == "consult_medical_guidelines":
            symptom = arguments.get("symptom")
            result = consult_medical_guidelines.invoke(symptom)
            return [types.TextContent(type="text", text=str(result))]

        elif name == "parse_lab_values":
            text = arguments.get("text")
            result = parse_lab_values.invoke(text)
            return [types.TextContent(type="text", text=str(result))]

        elif name == "search_patient_records":
            query = arguments.get("query")
            result = search_patient_records.invoke(query)
            return [types.TextContent(type="text", text=str(result))]

        elif name == "check_drug_interactions":
            meds = arguments.get("medication_list")
            result = check_drug_interactions.invoke(meds)
            return [types.TextContent(type="text", text=str(result))]

        else:
            raise ValueError(f"Unknown tool: {name}")

    except Exception as e:
        logger.error(f"Error executing tool {name}: {e}")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]

async def main():
    # Run the server using stdin/stdout streams
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="medicortex-mcp",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
