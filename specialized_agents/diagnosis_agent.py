from .base import A2ABaseAgent, llm
from .protocols import AgentCard
from tools.symptom_analysis_tools import analyze_symptoms
from tools.diagnosis_webcrawler_tools import crawl_diagnosis_articles

# ── Agent Card (A2A §1.1) ───────────────────────────────────────────
diagnosis_card = AgentCard(
    name="diagnosis",
    description=(
        "Specialized diagnostic reasoning agent that analyzes clinical symptoms, "
        "patient history, and knowledge core context to suggest differential diagnoses. "
        "It uses a two-step process: (1) Structural symptom analysis & context integration, "
        "and (2) Evidence-based web searching on trusted medical sites (Mayo Clinic, "
        "UpToDate, Merck Manuals, etc.) to validate potential conditions."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": "Symptoms, patient presentation, or clinical query"
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
                    "Differential diagnosis with supporting evidence, "
                    "cited sources, and clinical reasoning."
                )
            }
        }
    },
    version="2.0.0",
    capabilities=[
        "symptom-analysis",
        "differential-diagnosis",
        "clinical-reasoning",
        "medical-guidelines",
        "diagnostic-criteria"
    ]
)

# ── System Prompt ────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are the Diagnosis & Clinical Reasoning Agent for MediCortex.

YOUR MISSION: Using the gathered symptom analysis and medical evidence, construct
a broad, evidence-based differential diagnosis. Be comprehensive yet careful —
always cite sources and flag high-risk conditions.

═══ OUTPUT FORMAT ═══

Structure your response as:
1. **Clinical Profile** — Summary of key symptoms and their severity.
2. **Top Differentials** — 3–5 most likely conditions, ranked by probability.
   - For each: explain *why* it fits the symptoms and cite the source URL.
3. **Critical Red Flags** — Life-threatening conditions to rule out (MI, meningitis, PE, etc.).
4. **Suggested Next Steps** — Labs, imaging, or specialist referral recommendations.

CRITICAL GUARDRAILS:
- You are an AI assistant, NOT a doctor. Use phrases like "Possible conditions include…"
  or "Clinical presentation is consistent with…" — never definitively diagnose.
- ALWAYS cite the sources provided in the gathered data.
- If symptoms are vague or data is insufficient, state this clearly in Suggested Next Steps.
"""

# ── Agent Instance ───────────────────────────────────────────────────
diagnosis_agent = A2ABaseAgent(
    name="diagnosis",
    llm=llm,
    tools=[analyze_symptoms, crawl_diagnosis_articles],
    system_prompt=_SYSTEM_PROMPT,
    card=diagnosis_card,
    max_iterations=5,  # Allow analysis -> search -> synthesis loop
)
