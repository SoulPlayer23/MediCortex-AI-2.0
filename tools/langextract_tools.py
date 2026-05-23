"""
REP-1 — LangExtract structured pre-extraction for report_analyzer.

Runs LangExtract with a local MedGemma/Gemma4 Ollama provider to extract
typed, schema-validated, hallucination-flagged clinical entities from
text-based medical reports before the ReAct synthesis loop runs.

Supports three document types:
  - lab_report      → LabReportExtraction (CBC, metabolic panel, etc.)
  - radiology       → RadiologyExtraction (X-ray, MRI, CT findings)
  - discharge       → DischargeSummaryExtraction (meds, diagnoses, procedures)
"""

import re
import structlog
import langextract as lx
from langchain_core.tools import tool
from langextract.providers.ollama import OllamaLanguageModel
from langextract.core.data import AlignmentStatus

from config import settings
from tools.langextract_examples import LAB_EXAMPLES, RADIOLOGY_EXAMPLES, DISCHARGE_EXAMPLES

logger = structlog.get_logger("LangExtractTools")

# ── Prompt descriptions per schema ───────────────────────────────────────────

_LAB_PROMPT = (
    "Extract all laboratory test results. For each result extract: "
    "lab_test_name, value (numeric only), unit, reference_range, "
    "flag (H=High, L=Low, Critical, or empty if normal), and specimen_type."
)

_RADIOLOGY_PROMPT = (
    "Extract all imaging findings from this radiology report. For each finding extract: "
    "finding (description), anatomic_location, laterality (left/right/bilateral), "
    "severity (mild/moderate/significant/critical), and impression_line "
    "(the corresponding sentence from the IMPRESSION section)."
)

_DISCHARGE_PROMPT = (
    "Extract all medications, diagnoses, and procedures from this discharge summary. "
    "For each medication extract: medication_name, dosage, route, frequency, duration, "
    "indication, diagnosis (primary condition), and procedure (if applicable)."
)

# ── Document type heuristics ─────────────────────────────────────────────────

_LAB_KEYWORDS = re.compile(
    r"\b(cbc|wbc|rbc|hemoglobin|hematocrit|platelet|sodium|potassium|creatinine|"
    r"glucose|bun|alt|ast|bilirubin|tsh|hba1c|troponin|egfr|metabolic panel|"
    r"k/ul|mg/dl|meq/l|iu/l|ref\s*[\d.]+)\b",
    re.IGNORECASE,
)
_RADIOLOGY_KEYWORDS = re.compile(
    r"\b(x-ray|xray|ct scan|mri|ultrasound|impression|finding|opacity|effusion|"
    r"consolidation|lesion|mass|nodule|fracture|radiograph|imaging|radiolog)\b",
    re.IGNORECASE,
)


def _detect_doc_type(text: str) -> str:
    lab_hits = len(_LAB_KEYWORDS.findall(text))
    rad_hits = len(_RADIOLOGY_KEYWORDS.findall(text))
    if lab_hits >= 3:
        return "lab_report"
    if rad_hits >= 2:
        return "radiology"
    return "discharge"


def _build_model() -> OllamaLanguageModel:
    """Build an OllamaLanguageModel pointing at the homeserver Ollama instance."""
    base_url = settings.OLLAMA_CLOUD_URL.removesuffix("/v1")
    model_id = getattr(settings, "OLLAMA_CLOUD_MODEL", "gemma4:31b-cloud")
    return OllamaLanguageModel(
        model_id=model_id,
        model_url=base_url,
        timeout=getattr(settings, "OLLAMA_TIMEOUT_SECONDS", 120),
    )


def _run_extraction(text: str, doc_type: str) -> dict:
    """Run LangExtract for the given doc_type. Returns structured result dict."""
    if doc_type == "lab_report":
        prompt = _LAB_PROMPT
        examples = LAB_EXAMPLES
        extraction_class = "lab_result"
    elif doc_type == "radiology":
        prompt = _RADIOLOGY_PROMPT
        examples = RADIOLOGY_EXAMPLES
        extraction_class = "radiology_finding"
    else:
        prompt = _DISCHARGE_PROMPT
        examples = DISCHARGE_EXAMPLES
        extraction_class = "medication"

    model = _build_model()

    result = lx.extract(
        text_or_documents=text,
        model=model,
        prompt_description=prompt,
        examples=examples,
        extraction_passes=2,
    )

    entities = []
    grounded = []
    ungrounded = []

    for doc_result in result:
        for extraction in doc_result.extractions:
            entity = {
                "class": extraction.extraction_class or extraction_class,
                "text": extraction.extraction_text,
                "attributes": extraction.attributes or {},
                "char_interval": (
                    [extraction.char_interval.start, extraction.char_interval.end]
                    if extraction.char_interval else None
                ),
                "alignment_status": (
                    extraction.alignment_status.value
                    if extraction.alignment_status else None
                ),
            }
            entities.append(entity)
            grounded_statuses = {
                AlignmentStatus.MATCH_EXACT,
                AlignmentStatus.MATCH_GREATER,
                AlignmentStatus.MATCH_LESSER,
                AlignmentStatus.MATCH_FUZZY,
            }
            if (extraction.char_interval is not None and
                    extraction.alignment_status in grounded_statuses):
                grounded.append(entity)
            else:
                ungrounded.append(entity)

    return {
        "doc_type": doc_type,
        "entities": entities,
        "grounded": grounded,
        "ungrounded": ungrounded,
        "extraction_passes": 2,
        "model_used": getattr(settings, "OLLAMA_CLOUD_MODEL", "gemma4:31b-cloud"),
    }


@tool
def langextract_structured_extract(report_text: str, doc_type: str = "auto") -> str:
    """Extract structured clinical entities from a medical report text using LangExtract.

    Use this tool FIRST on any text-based medical report before synthesis.
    It returns typed, schema-validated entities with hallucination flags.

    Args:
        report_text: The full text of the medical report (from extract_document_text).
        doc_type: Document type — "lab_report", "radiology", "discharge", or "auto"
          (default). "auto" detects type from content heuristics.

    Returns:
        Structured JSON summary with grounded entities (verified against source text)
        and ungrounded entities (hallucination suspects, mark with ⚠️ in output).
    """
    if not report_text or not report_text.strip():
        return "Error: No report text provided to extract from."

    detected_type = _detect_doc_type(report_text) if doc_type == "auto" else doc_type
    logger.info("langextract_start", doc_type=detected_type, text_chars=len(report_text))

    try:
        result = _run_extraction(report_text, detected_type)
    except Exception as e:
        logger.warning("langextract_parse_error", event="langextract_parse_error", error=str(e))
        return (
            f"LangExtract extraction failed ({type(e).__name__}: {e}). "
            "Falling back to standard report analysis."
        )

    grounded_count = len(result["grounded"])
    ungrounded_count = len(result["ungrounded"])
    total = len(result["entities"])

    logger.info(
        "langextract_complete",
        doc_type=detected_type,
        total=total,
        grounded=grounded_count,
        ungrounded=ungrounded_count,
    )

    if total == 0:
        return (
            f"LangExtract found no structured entities in this {detected_type} "
            "document. Proceed with standard text analysis."
        )

    lines = [
        f"## Structured Extraction ({detected_type.replace('_', ' ').title()})",
        f"Model: {result['model_used']} | Passes: {result['extraction_passes']} | "
        f"Entities: {total} ({grounded_count} grounded, {ungrounded_count} unverified)",
        "",
    ]

    if result["grounded"]:
        lines.append("### Verified Entities (char-grounded in source)")
        for e in result["grounded"]:
            attrs = "; ".join(f"{k}: {v}" for k, v in e["attributes"].items() if v)
            lines.append(f"- **{e['text']}** — {attrs}")
        lines.append("")

    if result["ungrounded"]:
        lines.append("### ⚠️ Unverified Entities (not grounded — verify manually)")
        for e in result["ungrounded"]:
            attrs = "; ".join(f"{k}: {v}" for k, v in e["attributes"].items() if v)
            lines.append(f"- ⚠️ **{e['text']}** — {attrs}")
        lines.append("")

    return "\n".join(lines)
