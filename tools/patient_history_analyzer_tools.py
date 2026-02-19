"""
Patient History Analyzer Tool — Uses MedGemma to analyze de-identified
patient history including diagnoses, medications, and clinical patterns.

Compliant with:
  - FastAPI (structlog, no print())
  - MCP (prompt-engineered description, error-as-information)
  - A2A (typed schemas via LangChain @tool)
"""

import structlog
from langchain_core.tools import tool
from specialized_agents.medgemma_llm import MedGemmaLLM

logger = structlog.get_logger("PatientHistoryAnalyzerTool")

# Initialize LLM for tool use
llm = MedGemmaLLM()


@tool
def analyze_patient_history(patient_record: str) -> str:
    """Analyze a de-identified patient record to identify clinical patterns,
    risk factors, drug interactions, and care recommendations.

    This tool uses the MedGemma medical LLM to perform clinical reasoning on
    the patient's history. It does NOT have access to PII — it works only with
    de-identified data (names replaced with placeholders like '<PERSON_1>').

    Use this tool AFTER retrieving patient records to gain clinical insights.

    Args:
        patient_record: De-identified patient record text containing diagnoses,
                        medications, vitals, and demographics.

    Returns:
        A structured clinical analysis including risk assessment, medication
        review, and care recommendations.
    """
    logger.info("patient_history_analysis_start")

    if not patient_record or patient_record.strip() == "":
        return "Error: No patient record provided for analysis."

    system_prompt = (
        "You are an expert clinical analyst. Analyze the following de-identified patient record "
        "and provide a structured clinical summary.\n\n"
        "Output ONLY a markdown-formatted Clinical Analysis with the following sections:\n"
        "1. **Patient Overview** — Brief summary of demographics and current health status.\n"
        "2. **Diagnosis Summary** — Review of active conditions and their clinical significance.\n"
        "3. **Medication Review** — Assessment of current medications, potential interactions, "
        "and adherence considerations.\n"
        "4. **Risk Factors** — Identified risk factors based on diagnoses, vitals, and history.\n"
        "5. **Care Recommendations** — Suggested follow-ups, screenings, or lifestyle modifications.\n\n"
        "CRITICAL RULES:\n"
        "- Do NOT attempt to identify the patient. Use only placeholders present in the record.\n"
        "- Base your analysis ONLY on the data provided. Do not fabricate information.\n"
        "- Flag any concerning patterns (e.g., drug interactions, worsening trends)."
    )

    try:
        prompt = f"{system_prompt}\n\n--- PATIENT RECORD ---\n{patient_record}"
        response = llm.invoke(prompt)

        logger.info("patient_history_analysis_complete")
        return response

    except Exception as e:
        logger.error("patient_history_analysis_failed", error=str(e))
        return f"Error analyzing patient history with MedGemma: {str(e)}"
