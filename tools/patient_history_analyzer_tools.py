"""
Patient History Analyzer Tool — Uses MedGemma to analyze de-identified
patient clinical history, focusing on diagnosis patterns and risk factors.

This tool works ONLY on de-identified data. It never receives PII.

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
    """Analyze a de-identified patient record to identify diagnosis patterns,
    comorbidity relationships, disease progression risks, and risk factors.

    This tool focuses SPECIFICALLY on the patient's diagnosis history — it does NOT
    review medications or vitals (use the dedicated tools for those).

    Use this tool AFTER retrieving patient records to understand the clinical
    significance of the patient's condition profile and identify risk factors
    based on their diagnosis history.

    The tool uses MedGemma for clinical reasoning on de-identified data only.
    It does NOT have access to PII — names appear as placeholders like '<PERSON_1>'.

    Args:
        patient_record: De-identified patient record text containing diagnoses,
                        demographics, and clinical history.

    Returns:
        A structured clinical analysis focused on diagnosis patterns,
        comorbidity assessment, and risk factor identification.
    """
    logger.info("patient_history_analysis_start")

    if not patient_record or patient_record.strip() == "":
        return "Error: No patient record provided for analysis."

    system_prompt = (
        "You are an expert clinical analyst specializing in diagnosis pattern analysis. "
        "Analyze the following de-identified patient record and provide a structured "
        "clinical summary focused ONLY on diagnoses and risk factors.\n\n"
        "Output ONLY a markdown-formatted Clinical Analysis with the following sections:\n"
        "1. **Patient Overview** — Brief summary of demographics and current health status.\n"
        "2. **Diagnosis Summary** — Review of active conditions, their clinical significance, "
        "and how long they've been present.\n"
        "3. **Comorbidity Analysis** — How the patient's conditions interact with and "
        "exacerbate each other (e.g., diabetes + hypertension = increased cardiovascular risk).\n"
        "4. **Risk Factors** — Identified risk factors based on age, sex, diagnoses, "
        "and condition combinations. Include disease progression risks.\n"
        "5. **Recommended Screenings** — Condition-specific screenings or follow-ups "
        "based on the diagnosis profile.\n\n"
        "CRITICAL RULES:\n"
        "- Do NOT attempt to identify the patient. Use only placeholders present in the record.\n"
        "- Base your analysis ONLY on the data provided. Do not fabricate information.\n"
        "- Do NOT review medications or vitals — those are handled by separate tools.\n"
        "- Focus on the RELATIONSHIPS between diagnoses and their compounded risks."
    )

    try:
        prompt = f"{system_prompt}\n\n--- PATIENT RECORD ---\n{patient_record}"
        response = llm.invoke(prompt)

        logger.info("patient_history_analysis_complete")
        return response

    except Exception as e:
        logger.error("patient_history_analysis_failed", error=str(e))
        return f"Error analyzing patient history with MedGemma: {str(e)}"
