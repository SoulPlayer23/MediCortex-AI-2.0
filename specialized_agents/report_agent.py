from .base import A2ABaseAgent, llm
from .protocols import AgentCard
from tools.document_extraction_tools import extract_document_text
from tools.image_extraction_tools import extract_image_findings
from tools.report_analysis_tools import analyze_report

# ── Agent Card (A2A §1.1) ───────────────────────────────────────────
report_card = AgentCard(
    name="report_analyzer",
    description=(
        "Medical Report & Image Analysis Agent. Extracts text from PDF reports "
        "(lab results, discharge summaries) and analyzes medical images (X-rays, "
        "MRIs, CT scans) using MedGemma vision. Provides structured clinical "
        "interpretation with identified abnormalities and recommendations."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": "File URL (PDF or image) or raw report text"
            }
        },
        "required": ["input"]
    },
    output_schema={
        "type": "object",
        "properties": {
            "output": {
                "type": "string",
                "description": (
                    "Structured clinical analysis with key findings, "
                    "abnormalities, and recommendations."
                )
            }
        }
    },
    version="2.0.0",
    capabilities=[
        "pdf-extraction",
        "medical-image-analysis",
        "lab-value-interpretation",
        "report-analysis",
        "ocr",
        "vision"
    ]
)

# ── System Prompt ────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are the Medical Report & Image Analysis Agent for MediCortex.

YOUR MISSION: Using the extracted document and image data, provide a structured
clinical interpretation of the medical report or scan.

═══ OUTPUT FORMAT ═══

Structure your response as:
1. **Report Summary** — Report type and high-level overview.
2. **Key Findings** — Important values, measurements, or observations.
3. **Abnormalities** — Any values or findings outside normal ranges (flag with ⚠️).
4. **Clinical Significance** — What these findings may indicate clinically.
5. **Recommendations** — Suggested follow-up actions or specialist referrals.

CRITICAL:
- NEVER fabricate lab values or imaging findings. Only interpret what was extracted.
- If no data was successfully extracted, say so clearly and suggest the user re-upload
  the file or provide the report text directly.
- Always specify whether findings came from a PDF extraction or image analysis.
"""

# ── Agent Instance ───────────────────────────────────────────────────
report_agent = A2ABaseAgent(
    name="report_analyzer",
    llm=llm,
    tools=[extract_document_text, extract_image_findings, analyze_report],
    system_prompt=_SYSTEM_PROMPT,
    card=report_card,
    max_iterations=5,
)
