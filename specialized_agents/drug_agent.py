from .base import A2ABaseAgent, llm
from .protocols import AgentCard
from tools.drug_interaction_tools import check_drug_interactions
from tools.drug_recommendation_tools import recommend_drugs

# ── Agent Card (A2A §1.1) ───────────────────────────────────────────
drug_card = AgentCard(
    name="pharmacology",
    description=(
        "Specialized Pharmacology Agent. Checks drug-drug interactions, identifying "
        "contraindications and adverse effects using trusted medical sources. Provides "
        "evidence-based drug recommendations, dosage guidelines, and alternative "
        "medications for specific conditions."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": "Medication list, condition, or pharmacology query"
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
                    "Interaction analysis, drug recommendation, dosage info, or "
                    "clarification request."
                )
            }
        }
    },
    version="2.0.0",
    capabilities=[
        "drug-interactions",
        "contraindications",
        "dosage-guidelines",
        "drug-recommendations",
        "alternative-medicine"
    ]
)

# ── System Prompt ────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are the Pharmacology & Drug Safety Agent for MediCortex.

YOUR MISSION: Using the gathered pharmacology data, provide accurate, evidence-based
drug information. NEVER hallucinate drug interactions, dosages, or recommendations —
only report what the gathered data contains.

═══ OUTPUT FORMAT ═══

Structure your response as:
1. **Summary** — Direct answer to the user's question.
2. **Evidence/Analysis** — Detailed findings with source citations (Drugs.com, FDA, etc.).
3. **Safety Warnings** — Any major/moderate interactions or contraindications.
4. **Disclaimer** — "Consult a healthcare professional before making any medication changes."

INTERACTION SEVERITY RATINGS:
- **Major**: Life-threatening or permanent damage risk.
- **Moderate**: Significant worsening of condition or side effects.
- **Minor**: Limited clinical significance.

If no interaction data was found, state explicitly:
"No documented interactions found between [Drug A] and [Drug B] in standard databases."

If the query is ambiguous (no specific drug or condition mentioned), state:
"⚠️ CLARIFICATION NEEDED: [your specific question]"
"""

# ── Agent Instance ───────────────────────────────────────────────────
drug_agent = A2ABaseAgent(
    name="pharmacology",
    llm=llm,
    tools=[check_drug_interactions, recommend_drugs],
    system_prompt=_SYSTEM_PROMPT,
    card=drug_card,
    max_iterations=5,
)
