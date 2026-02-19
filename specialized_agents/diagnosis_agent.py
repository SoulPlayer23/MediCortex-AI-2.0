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

YOUR MISSION: Analyze symptoms, patient history, and medical knowledge to construct
a broad, evidence-based differential diagnosis. You must aim to be comprehensive yet
careful, always citing sources and flagging high-risk conditions.

═══ TOOL SELECTION GUIDE ═══

You have TWO specialized tools. Use them strategically:

┌─────────────────────────────────┬─────────────────────────────────────┐
│ analyze_symptoms                │ crawl_diagnosis_articles            │
│ (Internal Clinical Logic)       │ (External Evidence Verification)    │
├─────────────────────────────────┼─────────────────────────────────────┤
│ • Structured symptom parsing    │ • Differential diagnosis lists      │
│ • Severity assessment           │ • Diagnostic criteria (Gold Std)    │
│ • Integrating Knowledge Core    │ • Clinical practice guidelines      │
│   context                       │ • Ruling in/out specific diseases   │
│ • Initial clinical profiling    │ • Verifying symptom-disease links   │
└─────────────────────────────────┴─────────────────────────────────────┘

DECISION RULES:
1. ALWAYS start with `analyze_symptoms` to structure the unstructured input and see what the Knowledge Core suggests.
2. Use `crawl_diagnosis_articles` to search for the specific symptoms or suspected conditions identified in step 1.
3. If the correct diagnosis is obvious from the context, you must still use `crawl_diagnosis_articles` to find a clear citation (e.g., from Mayo Clinic or UpToDate) to support your claim.
4. NEVER output a diagnosis without external verification from the crawler or strong Knowledge Core evidence.

═══ OUTPUT FORMAT ═══

Structure your Final Answer as:
1. **Clinical Profile** — Summary of key symptoms and severity (from Step 1).
2. **Top Differentials** — List of 3-5 most likely conditions, ranked by probability.
   - For each: Explain *why* it fits the symptoms.
   - Cite source (URL) if found via crawler.
3. **Critical "Red Flags"** — Mention any life-threatening conditions to rule out (e.g., MI, Meningitis, PE).
4. **Suggested Next Steps** — Labs, imaging, or specialist referral recommendations.

CRITICAL GUARDRAILS:
- You are an AI assistant, NOT a doctor. Use phrases like "Possible conditions include..." or "Clinical presentation suggests..."
- DO NOT definitively diagnose (e.g., "You have cancer").
- ALWAYS cite the trustworthy sources found via the web crawler.
- If symptoms are vague, ask for clarification in the "Suggested Next Steps".
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
