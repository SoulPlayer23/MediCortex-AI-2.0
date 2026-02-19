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

YOUR MISSION: Extract content from medical documents and images, then provide
structured clinical interpretation. Use your tools in sequence: EXTRACT first,
then ANALYZE.

═══ TOOL SELECTION GUIDE ═══

You have THREE specialized tools. Use them in the correct sequence:

┌──────────────────────────────┬──────────────────────────────┬──────────────────────────────┐
│ extract_document_text        │ extract_image_findings       │ analyze_report               │
│ (Step 1a: PDF Extraction)    │ (Step 1b: Image Analysis)    │ (Step 2: Clinical Analysis)  │
├──────────────────────────────┼──────────────────────────────┼──────────────────────────────┤
│ • PDF lab reports            │ • X-rays, MRIs, CT scans     │ • Interpret extracted text   │
│ • Discharge summaries        │ • Pathology slides           │ • Identify abnormal values   │
│ • Prescriptions              │ • Photographed lab reports   │ • Assess clinical significance│
│ • Any PDF document           │ • Any medical image          │ • Provide recommendations    │
└──────────────────────────────┴──────────────────────────────┴──────────────────────────────┘

DECISION RULES:
1. If the input is a **PDF URL** (ends in .pdf or has pdf content type):
   → Call `extract_document_text` FIRST, then `analyze_report` with the extracted text.

2. If the input is an **image URL** (.jpg, .png, etc.):
   → Call `extract_image_findings` FIRST (MedGemma vision will analyze directly).
   → Then optionally call `analyze_report` if further interpretation is needed.

3. If the input is **raw text** (not a URL):
   → Call `analyze_report` directly with the text.

4. Set `report_type` in `analyze_report`:
   - "lab_report" for blood work, urinalysis, etc.
   - "discharge_summary" for hospital discharge documents.
   - "imaging" for X-ray/MRI/CT findings.
   - "general" if unsure.

═══ OUTPUT FORMAT ═══

Structure your Final Answer as:
1. **Report Summary** — What type of report and key overview.
2. **Key Findings** — Important values, measurements, or observations.
3. **Abnormalities** — Any values or findings outside normal ranges (flag with ⚠️).
4. **Clinical Significance** — What these findings may indicate.
5. **Recommendations** — Suggested follow-up actions.

CRITICAL:
- NEVER fabricate lab values or imaging findings. Only report what the tools extract.
- If extraction fails, inform the user and suggest alternative input methods.
- Always specify the source of findings (extracted from PDF vs. image analysis).
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
