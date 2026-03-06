"""
Patient Medication Review Tool — Cross-references the patient's active
medications against their diagnoses, allergies, and vitals to identify
safety concerns and therapy gaps.

This tool works ONLY on de-identified data. It never receives PII.

Compliant with:
  - FastAPI (structlog, no print())
  - MCP (prompt-engineered description, error-as-information)
  - A2A (typed schemas via LangChain @tool)
"""

import structlog
from langchain_core.tools import tool
from specialized_agents.medgemma_llm import MedGemmaLLM

logger = structlog.get_logger("PatientMedicationReviewTool")

# Initialize LLM for tool use
llm = MedGemmaLLM()


@tool
def review_patient_medications(patient_record: str) -> str:
    """Review a de-identified patient's medication list against their diagnoses,
    allergies, and clinical profile to identify safety concerns and therapy gaps.

    This tool focuses SPECIFICALLY on medication safety and completeness — it does
    NOT perform diagnosis analysis or vitals assessment (use dedicated tools for those).

    Capabilities:
    - Allergy cross-reference: flags medications that may conflict with documented allergies
      (e.g., Penicillin allergy + Amoxicillin = flagged)
    - Polypharmacy assessment: flags patients on >5 concurrent medications
    - Missing standard therapies: identifies guideline-recommended medications not
      currently prescribed (e.g., ACE inhibitor for diabetic patient with no renal contraindication)
    - Condition-contraindicated medications: flags drugs inappropriate for the patient's
      conditions (e.g., NSAIDs with CKD)
    - Duplicate therapy class detection: identifies overlapping medication classes

    Use this tool AFTER retrieving patient records when the query involves medication
    review, drug safety, or therapy optimization.

    The tool uses MedGemma for clinical reasoning on de-identified data only.
    It does NOT have access to PII — names appear as placeholders like '<PERSON_1>'.

    Args:
        patient_record: De-identified patient record text containing medications,
                        diagnoses, and allergies.

    Returns:
        A structured medication review with safety flags, therapy gaps,
        and optimization recommendations.
    """
    logger.info("patient_medication_review_start")

    if not patient_record or patient_record.strip() == "":
        return "Error: No patient record provided for medication review."

    system_prompt = (
        "You are an expert clinical pharmacist performing a comprehensive medication review. "
        "Analyze the following de-identified patient record and provide a structured "
        "medication safety and completeness assessment.\n\n"
        "Output ONLY a markdown-formatted Medication Review with the following sections:\n"
        "1. **Current Medication Summary** — List all active medications with dosages and frequencies.\n"
        "2. **Allergy Safety Check** — Cross-reference each medication against documented allergies:\n"
        "   - 🔴 ALERT: Direct allergy conflict (e.g., prescribed a drug the patient is allergic to)\n"
        "   - ⚠️ CAUTION: Drug class cross-reactivity risk (e.g., Penicillin allergy → Cephalosporin caution)\n"
        "   - ✅ SAFE: No allergy conflicts detected\n"
        "3. **Polypharmacy Assessment** — If the patient is on >5 medications, assess:\n"
        "   - Necessity of each medication\n"
        "   - Risk of adverse drug-drug interactions\n"
        "   - Opportunities to simplify the regimen\n"
        "4. **Condition-Medication Alignment** — For each active diagnosis, check:\n"
        "   - Are guideline-recommended therapies prescribed?\n"
        "   - Are any medications contraindicated given the patient's conditions?\n"
        "   - Examples: NSAID use with CKD, Metformin with severe renal impairment\n"
        "5. **Therapy Gaps** — Identify any standard-of-care medications that are MISSING:\n"
        "   - Diabetic patients: Metformin, statin, ACE/ARB\n"
        "   - CAD patients: Aspirin, beta-blocker, statin, ACE/ARB\n"
        "   - Hypertension patients: First-line antihypertensive\n"
        "   - CKD patients: ACE/ARB for renal protection\n"
        "6. **Recommendations** — Specific, actionable suggestions for medication optimization.\n\n"
        "CRITICAL RULES:\n"
        "- Do NOT attempt to identify the patient. Use only placeholders present in the record.\n"
        "- Base your analysis ONLY on the data provided. Do not fabricate medications.\n"
        "- Always state severity of concerns clearly.\n"
        "- Note that you are providing clinical decision SUPPORT — final decisions are made by the physician.\n"
        "- Include a disclaimer that medication changes require physician review."
    )

    try:
        prompt = f"{system_prompt}\n\n--- PATIENT RECORD ---\n{patient_record}"
        response = llm.invoke(prompt)

        logger.info("patient_medication_review_complete")
        return response

    except Exception as e:
        logger.error("patient_medication_review_failed", error=str(e))
        return f"Error reviewing patient medications with MedGemma: {str(e)}"
