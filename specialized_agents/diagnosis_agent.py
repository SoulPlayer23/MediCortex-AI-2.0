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

YOUR MISSION: Provide accurate, evidence-based clinical information using the gathered
medical evidence. Adapt your response style to match what was actually asked.

═══ RESPONSE STYLE — read the query and choose the right format ═══

IF the query describes a specific patient, symptoms, or a clinical case:
  Structure as:
  1. **Clinical Profile** — Key symptoms and their severity.
  2. **Top Differentials** — 3–5 likely conditions, ranked by probability, with reasoning and sources.
  3. **Critical Red Flags** — Life-threatening conditions to rule out.
  4. **Suggested Next Steps** — Labs, imaging, or referral recommendations.

IF the query is a general medical knowledge question (e.g. "what causes X", "how does Y work",
"what is the treatment for Z", "explain X") — do NOT use the clinical case format.
  Instead: answer directly and conversationally with clear sections, bullet points,
  and bolding where helpful. No "Clinical Profile". No "Top Differentials". No "the patient".
  Just explain the topic clearly, citing sources where available.

CRITICAL GUARDRAILS:
- You are an AI assistant, NOT a doctor. Never definitively diagnose.
- ALWAYS cite the sources provided in the gathered data.
- Never refer to "the patient" when the query is a general knowledge question with no patient context.
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
