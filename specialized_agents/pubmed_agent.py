from .base import A2ABaseAgent, llm
from .protocols import AgentCard
from tools.pubmed_search_tools import search_pubmed
from tools.medical_webcrawler_tools import crawl_medical_articles

# ── Agent Card (A2A §1.1) ───────────────────────────────────────────
pubmed_card = AgentCard(
    name="pubmed",
    description=(
        "Specialized medical research agent with two capabilities: "
        "(1) Searching the NCBI PubMed database for peer-reviewed research papers, "
        "clinical trials, systematic reviews, and meta-analyses. "
        "(2) Crawling trusted medical websites (Mayo Clinic, NIH, CDC, WHO, WebMD, "
        "Medscape, Cleveland Clinic, BMJ, NEJM, The Lancet, etc.) for clinical "
        "guidance, patient education, and community discussions."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": "A medical research query, clinical question, or health topic"
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
                    "Synthesized findings combining peer-reviewed research and "
                    "trusted medical web sources, with cited URLs"
                )
            }
        }
    },
    version="2.0.0",
    capabilities=[
        "pubmed-search",
        "literature-review",
        "medical-papers",
        "clinical-trials",
        "web-crawling",
        "medical-articles",
        "evidence-synthesis",
    ]
)

# ── System Prompt ────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are the PubMed & Medical Research Agent for MediCortex.

YOUR MISSION: Synthesize the gathered research data to give a comprehensive,
evidence-based answer to the user's medical query.

═══ OUTPUT FORMAT ═══

Structure your response as:
1. **Key Findings** — The most important takeaway in 1-2 sentences.
2. **Research Evidence** — Findings from peer-reviewed sources, citing paper titles and URLs.
3. **Clinical Guidance** — Findings from medical websites, citing source and URL.
4. **Summary** — A concise synthesis tying all sources together.

CRITICAL RULES:
- ALWAYS cite sources with their URLs. Never fabricate citations.
- Clearly distinguish between peer-reviewed evidence and clinical website content.
- If conflicting information is found, flag the discrepancy explicitly.
- Prefer recent publications (last 5 years) when possible.
- State limitations of the evidence when applicable.
- ONLY output clinical findings — never describe your reasoning process.
"""

# ── Agent Instance ───────────────────────────────────────────────────
pubmed_agent = A2ABaseAgent(
    name="pubmed",
    llm=llm,
    tools=[search_pubmed, crawl_medical_articles],
    system_prompt=_SYSTEM_PROMPT,
    card=pubmed_card,
    max_iterations=5,  # Allow more iterations since agent has 2 tools to orchestrate
)

