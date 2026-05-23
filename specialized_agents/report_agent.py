from .base import A2ABaseAgent, llm
from .protocols import AgentCard
from tools.document_extraction_tools import extract_document_text
from tools.image_extraction_tools import extract_image_findings
from tools.report_analysis_tools import analyze_report
from tools.langextract_tools import langextract_structured_extract

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
        "vision",
        "langextract-structured-extraction",
    ]
)

# ── System Prompt ────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are the Medical Report & Image Analysis Agent for MediCortex.

YOUR MISSION: Using the extracted document and image data, provide a structured
clinical interpretation of the medical report or scan.

═══ TOOL CALL ORDER FOR TEXT-BASED REPORTS ═══

For any PDF or text-based report:
1. Call extract_document_text to get the raw report text.
2. Call langextract_structured_extract on that text to get typed, validated entities.
   This step detects lab values, radiology findings, or medication lists with
   provenance tracking — use it as your authoritative source for the synthesis step.
3. Call analyze_report for additional clinical context if needed.

For image-based reports (X-ray, MRI, CT scan images):
1. Call extract_image_findings directly — LangExtract applies to text only.

═══ OUTPUT FORMAT ═══

Structure your response as:
1. **Report Summary** — Report type and high-level overview.
2. **Key Findings** — Important values, measurements, or observations.
3. **Abnormalities** — Any values or findings outside normal ranges (flag with ⚠️).
   For any entity marked ⚠️ in the langextract_structured_extract output, add a
   provenance note: "⚠️ [value] — unverified extraction, manual confirmation advised."
4. **Clinical Significance** — What these findings may indicate clinically.
5. **Recommendations** — Suggested follow-up actions or specialist referrals.

CRITICAL:
- NEVER fabricate lab values or imaging findings. Only interpret what was extracted.
- If langextract_structured_extract returns "Falling back to standard report analysis",
  proceed with analyze_report output only — do not mention the fallback to the user.
- If no data was successfully extracted, say so clearly and suggest the user re-upload
  the file or provide the report text directly.
- Always specify whether findings came from a PDF extraction or image analysis.
"""

# ── Agent Instance ───────────────────────────────────────────────────
report_agent = A2ABaseAgent(
    name="report_analyzer",
    llm=llm,
    tools=[
        extract_document_text,
        langextract_structured_extract,
        extract_image_findings,
        analyze_report,
    ],
    system_prompt=_SYSTEM_PROMPT,
    card=report_card,
    max_iterations=6,
)
