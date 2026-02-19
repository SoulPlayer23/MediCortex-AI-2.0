"""
Report Analysis Tool — Uses MedGemma to clinically interpret extracted
report content (from documents or images).

Compliant with:
  - FastAPI (structlog, no print())
  - MCP (prompt-engineered description, error-as-information)
  - A2A (typed schemas via LangChain @tool)
"""

import structlog
from langchain_core.tools import tool
from specialized_agents.medgemma_llm import MedGemmaLLM

logger = structlog.get_logger("ReportAnalysisTool")

# Initialize LLM for tool use
llm = MedGemmaLLM()


@tool
def analyze_report(extracted_content: str, report_type: str = "general") -> str:
    """Analyze extracted report content to identify lab values, abnormalities,
    and clinical significance using MedGemma.

    This tool takes the TEXT output from `extract_document_text` or the
    FINDINGS from `extract_image_findings` and performs structured clinical
    interpretation.

    Args:
        extracted_content: The text extracted from a document or image analysis findings.
        report_type: Type hint — "lab_report", "discharge_summary", "imaging", or "general".

    Returns:
        Structured clinical analysis with identified values, abnormalities, and recommendations.
    """
    logger.info("report_analysis_start", report_type=report_type)

    if not extracted_content or len(extracted_content.strip()) < 10:
        return "Error: No content provided for analysis. Please extract the document or image first."

    # Build type-specific prompt
    type_guidance = {
        "lab_report": (
            "Focus on:\n"
            "- Identifying specific lab tests and their numeric values\n"
            "- Flagging values outside normal reference ranges (high/low)\n"
            "- Noting critical values that need immediate attention\n"
            "- Grouping related tests (CBC, BMP, LFT, etc.)"
        ),
        "discharge_summary": (
            "Focus on:\n"
            "- Admission diagnosis and final diagnosis\n"
            "- Key procedures performed\n"
            "- Discharge medications and instructions\n"
            "- Follow-up requirements"
        ),
        "imaging": (
            "Focus on:\n"
            "- Describing the imaging modality and body region\n"
            "- Identifying normal vs. abnormal findings\n"
            "- Comparing with typical presentation patterns\n"
            "- Suggesting differential diagnoses if applicable"
        ),
        "general": (
            "Focus on:\n"
            "- Extracting all clinically relevant data points\n"
            "- Identifying any abnormal findings\n"
            "- Summarizing the overall clinical picture"
        ),
    }

    guidance = type_guidance.get(report_type, type_guidance["general"])

    system_prompt = (
        "You are an expert medical report analyst. Analyze the following extracted report "
        "content and provide a structured clinical interpretation.\n\n"
        f"Report Type: {report_type}\n\n"
        f"{guidance}\n\n"
        "Output as structured Markdown with these sections:\n"
        "1. **Report Summary** — What type of report and key overview\n"
        "2. **Key Findings** — Important values, measurements, or observations\n"
        "3. **Abnormalities** — Any values or findings outside normal ranges\n"
        "4. **Clinical Significance** — What these findings may indicate\n"
        "5. **Recommendations** — Suggested follow-up actions\n\n"
        "CRITICAL: Base your analysis ONLY on the provided content. Do NOT fabricate values."
    )

    try:
        prompt = f"{system_prompt}\n\n--- REPORT CONTENT ---\n{extracted_content}"
        response = llm.invoke(prompt)

        logger.info("report_analysis_complete")
        return response

    except Exception as e:
        logger.error("report_analysis_failed", error=str(e))
        return f"Error analyzing report with MedGemma: {str(e)}"
