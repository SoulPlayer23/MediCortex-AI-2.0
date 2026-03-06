"""
Patient Vitals Analysis Tool — Analyzes vital signs from de-identified
patient records, flags critical values, and compares against
condition-specific targets.

This tool works ONLY on de-identified data. It never receives PII.

Compliant with:
  - FastAPI (structlog, no print())
  - MCP (prompt-engineered description, error-as-information)
  - A2A (typed schemas via LangChain @tool)
"""

import structlog
from langchain_core.tools import tool
from specialized_agents.medgemma_llm import MedGemmaLLM

logger = structlog.get_logger("PatientVitalsTool")

# Initialize LLM for tool use
llm = MedGemmaLLM()


@tool
def analyze_patient_vitals(patient_record: str) -> str:
    """Analyze vital signs from a de-identified patient record to identify
    critical values, trends across visits, and condition-specific target deviations.

    This tool focuses SPECIFICALLY on vital sign interpretation — it does NOT
    review medications or diagnoses in depth (use the dedicated tools for those).

    Capabilities:
    - Flags vitals outside normal ranges with severity (⚠️ Warning / 🔴 Critical)
    - Compares against condition-specific targets (e.g., diabetic patients have
      stricter BP targets of <130/80 mmHg per ADA guidelines)
    - Detects trends across multiple visits (improving, stable, or worsening)
    - Identifies BMI category and weight change trends

    Use this tool AFTER retrieving patient records when the query involves
    vital signs, health trends, or physiological measurements.

    The tool uses MedGemma for clinical reasoning on de-identified data only.
    It does NOT have access to PII — names appear as placeholders like '<PERSON_1>'.

    Args:
        patient_record: De-identified patient record text containing vitals data,
                        vitals history, and diagnoses (for condition-specific targets).

    Returns:
        A structured vitals analysis with flagged abnormalities, trend detection,
        and condition-specific target comparisons.
    """
    logger.info("patient_vitals_analysis_start")

    if not patient_record or patient_record.strip() == "":
        return "Error: No patient record provided for vitals analysis."

    system_prompt = (
        "You are an expert clinical analyst specializing in vital sign interpretation. "
        "Analyze the following de-identified patient record and provide a structured "
        "vitals assessment.\n\n"
        "Output ONLY a markdown-formatted Vitals Analysis with the following sections:\n"
        "1. **Current Vitals Summary** — Latest recorded vital signs with normal range comparison.\n"
        "2. **Flagged Abnormalities** — Any values outside normal ranges, marked with:\n"
        "   - ⚠️ WARNING: Values slightly outside normal (e.g., BP 130-139/85-89)\n"
        "   - 🔴 CRITICAL: Values significantly abnormal (e.g., BP ≥140/90, SpO2 <92%)\n"
        "3. **Condition-Specific Targets** — Compare vitals against condition-specific "
        "guidelines when applicable:\n"
        "   - Diabetic patients: BP target <130/80 mmHg (ADA), HbA1c <7%\n"
        "   - Heart disease patients: HR target 60-80 bpm, BP <130/80\n"
        "   - CKD patients: BP target <130/80, monitor SpO2 closely\n"
        "   - COPD/Asthma patients: SpO2 target ≥95%\n"
        "4. **Trend Analysis** — If vitals history is available, analyze trends:\n"
        "   - 📈 Improving, 📊 Stable, or 📉 Worsening\n"
        "   - Note weight changes and BMI trajectory\n"
        "5. **Clinical Recommendations** — Specific follow-up actions based on findings.\n\n"
        "STANDARD NORMAL RANGES FOR REFERENCE:\n"
        "- Blood Pressure: <120/80 (normal), 120-129/<80 (elevated), 130-139/80-89 (Stage 1 HTN), ≥140/90 (Stage 2 HTN)\n"
        "- Heart Rate: 60-100 bpm\n"
        "- Temperature: 97.8-99.1°F (36.5-37.3°C)\n"
        "- Respiratory Rate: 12-20 breaths/min\n"
        "- SpO2: ≥95% (normal), 90-94% (concerning), <90% (critical)\n"
        "- BMI: <18.5 (underweight), 18.5-24.9 (normal), 25-29.9 (overweight), ≥30 (obese)\n\n"
        "CRITICAL RULES:\n"
        "- Do NOT attempt to identify the patient. Use only placeholders present in the record.\n"
        "- Base your analysis ONLY on the data provided. Do not fabricate vital values.\n"
        "- Always note the date of the most recent vitals reading.\n"
        "- If vitals history has multiple entries, ALWAYS provide trend analysis."
    )

    try:
        prompt = f"{system_prompt}\n\n--- PATIENT RECORD ---\n{patient_record}"
        response = llm.invoke(prompt)

        logger.info("patient_vitals_analysis_complete")
        return response

    except Exception as e:
        logger.error("patient_vitals_analysis_failed", error=str(e))
        return f"Error analyzing patient vitals with MedGemma: {str(e)}"
