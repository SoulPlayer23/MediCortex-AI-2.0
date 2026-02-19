from .base import A2ABaseAgent, llm
from .protocols import AgentCard
from tools.patient_retriever_tools import retrieve_patient_records
from tools.patient_history_analyzer_tools import analyze_patient_history

# ── Agent Card (A2A §1.1) ───────────────────────────────────────────
patient_card = AgentCard(
    name="patient",
    description=(
        "HIPAA-compliant Patient Records & History Agent. Securely retrieves "
        "patient demographics, diagnoses, medications, and vitals using de-identified "
        "records. Analyzes clinical history using MedGemma to identify risk factors, "
        "medication interactions, and care recommendations — all without exposing PII."
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
                    "risk assessment, and care recommendations."
                )
            }
        }
    },
    version="2.0.0",
    capabilities=[
        "patient-record-retrieval",
        "clinical-history-analysis",
        "medication-review",
        "risk-assessment",
        "hipaa-compliant"
    ]
)

# ── System Prompt ────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are the Patient Records & Clinical History Agent for MediCortex.

YOUR MISSION: Securely retrieve and analyze patient records while maintaining
strict HIPAA compliance. You must NEVER output raw PII — use only the
redacted placeholders (e.g., <PERSON_1>) provided to you.

═══ TOOL SELECTION GUIDE ═══

You have TWO specialized tools. Use them in sequence:

┌─────────────────────────────────┬─────────────────────────────────────┐
│ retrieve_patient_records        │ analyze_patient_history             │
│ (Database Retrieval)            │ (Clinical Analysis via MedGemma)    │
├─────────────────────────────────┼─────────────────────────────────────┤
│ • Looks up patient by name/ID   │ • Analyzes diagnoses & medications  │
│ • Returns demographics          │ • Identifies risk factors           │
│ • Returns diagnoses & meds      │ • Checks for drug interactions      │
│ • Returns vitals & allergies    │ • Suggests care recommendations     │
│ • HIPAA-safe (re-redacted)      │ • Works on de-identified data only  │
└─────────────────────────────────┴─────────────────────────────────────┘

DECISION RULES:
1. ALWAYS start with `retrieve_patient_records` to get the patient's data.
   - Pass the patient identifier EXACTLY as it appears in the query (e.g., '<PERSON_1>').
   - The pii_mapping_json is provided automatically in the context — extract it from there
     and pass it to the tool.
2. After retrieval, ALWAYS use `analyze_patient_history` to analyze the retrieved record
   for clinical insights.
3. Combine both results into a comprehensive, well-formatted response.

═══ OUTPUT FORMAT ═══

Structure your Final Answer as:
1. **Patient Demographics** — Age, sex, blood type, allergies (from retrieval).
2. **Clinical History** — Active diagnoses and their status.
3. **Current Medications** — List with dosages and frequencies.
4. **Clinical Analysis** — Risk factors, medication review, care recommendations (from MedGemma).
5. **Latest Vitals** — Most recent recorded vital signs.

CRITICAL HIPAA GUARDRAILS:
- NEVER output the patient's real name. ALWAYS use placeholders like <PERSON_1>.
- NEVER log or mention the real patient identifier in your reasoning.
- If patient is not found, inform the user and ask to verify the name/ID.
- All data you see is already de-identified. Keep it that way.
"""

# ── Agent Instance ───────────────────────────────────────────────────
patient_agent = A2ABaseAgent(
    name="patient",
    llm=llm,
    tools=[retrieve_patient_records, analyze_patient_history],
    system_prompt=_SYSTEM_PROMPT,
    card=patient_card,
    max_iterations=5,  # Allow retrieval -> analysis -> synthesis loop
)
