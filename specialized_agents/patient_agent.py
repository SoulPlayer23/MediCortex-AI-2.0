from .base import A2ABaseAgent, llm
from .protocols import AgentCard
from tools.patient_retriever_tools import retrieve_patient_records
from tools.patient_history_analyzer_tools import analyze_patient_history
from tools.patient_vitals_tools import analyze_patient_vitals
from tools.patient_medication_review_tools import review_patient_medications

# ── Agent Card (A2A §1.1) ───────────────────────────────────────────
patient_card = AgentCard(
    name="patient",
    description=(
        "HIPAA-compliant Patient Records & Clinical Analysis Agent. Securely retrieves "
        "patient demographics, diagnoses, medications, and vitals using de-identified "
        "records. Provides specialized analysis via 4 focused tools: record retrieval, "
        "diagnosis pattern analysis, vital sign assessment, and medication safety review — "
        "all without exposing PII."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": "Patient name placeholder or patient ID"
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
                    "De-identified patient record with clinical analysis, "
                    "vitals assessment, medication review, and care recommendations."
                )
            }
        }
    },
    version="3.0.0",
    capabilities=[
        "patient-record-retrieval",
        "clinical-history-analysis",
        "vitals-assessment",
        "medication-safety-review",
        "risk-assessment",
        "polypharmacy-detection",
        "hipaa-compliant"
    ]
)

# ── System Prompt ────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are the Patient Records & Clinical Analysis Agent for MediCortex.

YOUR MISSION: Using the gathered patient data, provide a comprehensive clinical
summary while maintaining strict HIPAA compliance. You must NEVER output raw PII —
use only the redacted placeholders (e.g., <PERSON_1>) that appear in the data.

═══ OUTPUT FORMAT ═══

Structure your response as:
1. **Patient Demographics** — Age, sex, blood type, allergies.
2. **Clinical History** — Active diagnoses and their status.
3. **Vitals Assessment** — Current vitals with flagged abnormalities and trends.
4. **Medication Review** — Safety checks, therapy gaps, polypharmacy flags.
5. **Clinical Analysis** — Risk factors and comorbidity insights.
6. **Recommendations** — Suggested follow-up actions.

CRITICAL HIPAA GUARDRAILS:
- NEVER output the patient's real name. ALWAYS use placeholders like <PERSON_1>.
- If the gathered data contains "No patient records found", respond only with
  "No patient records found for <PERSON_1>." — do not hallucinate any clinical data.
- All data provided to you is already de-identified. Keep it that way in your response.
"""

# ── Agent Instance ───────────────────────────────────────────────────
patient_agent = A2ABaseAgent(
    name="patient",
    llm=llm,
    tools=[retrieve_patient_records, analyze_patient_history, analyze_patient_vitals, review_patient_medications],
    system_prompt=_SYSTEM_PROMPT,
    card=patient_card,
    max_iterations=5,  # retrieve + up to 3 analysis tools + 1 buffer
)
