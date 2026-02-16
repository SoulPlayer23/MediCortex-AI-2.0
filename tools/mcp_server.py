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
from tools.pubmed_tools import search_pubmed
from tools.diagnosis_tools import consult_medical_guidelines
from tools.report_tools import parse_lab_values
from tools.patient_tools import search_patient_records
from tools.drug_tools import check_drug_interactions

# Setup Logging
# We use stderr so it doesn't interfere with stdout (which is used for MCP protocol)
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("mcp_server")

server = Server("medicortex-mcp")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools."""
    return [
        types.Tool(
            name="search_pubmed",
            description="Search PubMed for medical literature and research papers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Medical topic or query"}
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
            # LangChain tools .invoke can take dict or string.
            result = search_pubmed.invoke(query)
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
