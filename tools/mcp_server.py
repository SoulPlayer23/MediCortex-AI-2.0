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
from tools.symptom_analysis_tools import analyze_symptoms
from tools.diagnosis_webcrawler_tools import crawl_diagnosis_articles
from tools.document_extraction_tools import extract_document_text
from tools.image_extraction_tools import extract_image_findings
from tools.report_analysis_tools import analyze_report
from tools.patient_retriever_tools import retrieve_patient_records
from tools.patient_history_analyzer_tools import analyze_patient_history
from tools.drug_interaction_tools import check_drug_interactions
from tools.drug_recommendation_tools import recommend_drugs

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
            name="analyze_symptoms",
            description="Analyze symptoms and combine with knowledge core context to create a clinical profile.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "User's symptom description"},
                    "knowledge_context": {"type": "string", "description": "Context from knowledge core"}
                },
                "required": ["query"]
            },
        ),
        types.Tool(
            name="crawl_diagnosis_articles",
            description="Search trusted medical sites for differential diagnosis and diagnostic criteria.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Medical topic or symptom set to research"},
                    "max_results": {"type": "integer", "description": "Number of articles to return (1-10, default 5)", "default": 5}
                },
                "required": ["query"]
            },
        ),
        types.Tool(
            name="extract_document_text",
            description="Download a PDF report from a URL and extract its text content as Markdown.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_url": {"type": "string", "description": "HTTP/HTTPS URL to PDF file"}
                },
                "required": ["file_url"]
            },
        ),
        types.Tool(
            name="extract_image_findings",
            description="Download a medical image and analyze it using MedGemma vision (X-rays, MRIs, CT scans).",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_url": {"type": "string", "description": "HTTP/HTTPS URL to image file"},
                    "clinical_context": {"type": "string", "description": "Optional clinical context"}
                },
                "required": ["file_url"]
            },
        ),
        types.Tool(
            name="analyze_report",
            description="Analyze extracted report content to identify lab values, abnormalities, and clinical significance.",
            inputSchema={
                "type": "object",
                "properties": {
                    "extracted_content": {"type": "string", "description": "Text from document or image extraction"},
                    "report_type": {"type": "string", "enum": ["lab_report", "discharge_summary", "imaging", "general"], "default": "general"}
                },
                "required": ["extracted_content"]
            },
        ),
        types.Tool(
            name="retrieve_patient_records",
            description="Retrieve patient records from the database using a redacted patient identifier. HIPAA-compliant: resolves PII placeholders internally and returns de-identified data.",
            inputSchema={
                "type": "object",
                "properties": {
                    "redacted_identifier": {"type": "string", "description": "Patient name placeholder (e.g., '<PERSON_1>') or patient ID"},
                    "pii_mapping_json": {"type": "string", "description": "JSON-encoded PII mapping dict", "default": "{}"}
                },
                "required": ["redacted_identifier"]
            },
        ),
        types.Tool(
            name="analyze_patient_history",
            description="Analyze a de-identified patient record to identify clinical patterns, risk factors, drug interactions, and care recommendations using MedGemma.",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_record": {"type": "string", "description": "De-identified patient record text"}
                },
                "required": ["patient_record"]
            },
        ),
        types.Tool(
            name="check_drug_interactions",
            description="Check for drug-drug interactions, contraindications, and adverse effects using trusted pharmacology sources.",
            inputSchema={
                "type": "object",
                "properties": {
                    "medications": {"type": "string", "description": "Comma-separated list of drugs"},
                    "patient_conditions": {"type": "string", "description": "Optional patient conditions"}
                },
                "required": ["medications"]
            },
        ),
        types.Tool(
            name="recommend_drugs",
            description="Get drug recommendations, dosage guidelines, or alternatives for a condition.",
            inputSchema={
                "type": "object",
                "properties": {
                    "condition": {"type": "string", "description": "Medical condition or drug name"},
                    "query_type": {"type": "string", "enum": ["recommendation", "dosage", "alternatives"], "default": "recommendation"},
                    "patient_info": {"type": "string", "description": "Optional patient context (e.g., age, renal function)"}
                },
                "required": ["condition"]
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

        elif name == "analyze_symptoms":
            query = arguments.get("query")
            knowledge_context = arguments.get("knowledge_context", "")
            result = analyze_symptoms.invoke({"query": query, "knowledge_context": knowledge_context})
            return [types.TextContent(type="text", text=str(result))]

        elif name == "crawl_diagnosis_articles":
            query = arguments.get("query")
            max_results = arguments.get("max_results", 5)
            result = crawl_diagnosis_articles.invoke({"query": query, "max_results": max_results})
            return [types.TextContent(type="text", text=str(result))]

        elif name == "extract_document_text":
            file_url = arguments.get("file_url")
            result = extract_document_text.invoke({"file_url": file_url})
            return [types.TextContent(type="text", text=str(result))]

        elif name == "extract_image_findings":
            file_url = arguments.get("file_url")
            context = arguments.get("clinical_context", "")
            result = extract_image_findings.invoke({"file_url": file_url, "clinical_context": context})
            return [types.TextContent(type="text", text=str(result))]

        elif name == "analyze_report":
            content = arguments.get("extracted_content")
            r_type = arguments.get("report_type", "general")
            result = analyze_report.invoke({"extracted_content": content, "report_type": r_type})
            return [types.TextContent(type="text", text=str(result))]

        elif name == "retrieve_patient_records":
            redacted_identifier = arguments.get("redacted_identifier")
            pii_mapping_json = arguments.get("pii_mapping_json", "{}")
            result = retrieve_patient_records.invoke({"redacted_identifier": redacted_identifier, "pii_mapping_json": pii_mapping_json})
            return [types.TextContent(type="text", text=str(result))]

        elif name == "analyze_patient_history":
            patient_record = arguments.get("patient_record")
            result = analyze_patient_history.invoke({"patient_record": patient_record})
            return [types.TextContent(type="text", text=str(result))]

        elif name == "check_drug_interactions":
            meds = arguments.get("medications")
            conditions = arguments.get("patient_conditions", "")
            result = check_drug_interactions.invoke({"medications": meds, "patient_conditions": conditions})
            return [types.TextContent(type="text", text=str(result))]

        elif name == "recommend_drugs":
            condition = arguments.get("condition")
            q_type = arguments.get("query_type", "recommendation")
            p_info = arguments.get("patient_info", "")
            result = recommend_drugs.invoke({"condition": condition, "query_type": q_type, "patient_info": p_info})
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
