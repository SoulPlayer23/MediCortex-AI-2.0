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

YOUR MISSION: Find and synthesize the most relevant medical information from
multiple authoritative sources to answer the user's query comprehensively.

═══ TOOL SELECTION GUIDE ═══

You have TWO tools. Choose based on the query type:

┌─────────────────────────────────┬─────────────────────────────────────┐
│ search_pubmed                   │ crawl_medical_articles              │
├─────────────────────────────────┼─────────────────────────────────────┤
│ Peer-reviewed research papers   │ Clinical summaries & guidelines     │
│ Randomized controlled trials    │ Patient education materials         │
│ Systematic reviews & meta-      │ Drug information & side effects     │
│   analyses                      │ Practical treatment guidance        │
│ Mechanism-of-action questions   │ Disease overviews & prevention      │
│ Epidemiological data            │ Community discussions & forums      │
│ Biomarker / pathology studies   │ Latest health news & updates        │
└─────────────────────────────────┴─────────────────────────────────────┘

DECISION RULES:
1. If the query is about EVIDENCE, MECHANISMS, or RESEARCH → use search_pubmed FIRST.
2. If the query is about PRACTICAL GUIDANCE, PATIENT INFO, or OVERVIEWS → use crawl_medical_articles FIRST.
3. For COMPREHENSIVE questions (e.g., "Tell me everything about X") → use BOTH tools.
4. Always start with the most relevant tool, then use the second if the first result is insufficient.

═══ OUTPUT FORMAT ═══

Structure your Final Answer as:
1. **Key Findings** — The most important takeaway in 1-2 sentences.
2. **Research Evidence** — Findings from PubMed (if used), citing paper titles and URLs.
3. **Clinical Guidance** — Findings from medical websites (if used), citing source and URL.
4. **Summary** — A concise synthesis tying both sources together.

CRITICAL RULES:
- ALWAYS cite sources with their URLs. Never fabricate citations.
- Clearly distinguish between peer-reviewed evidence and clinical website content.
- If conflicting information is found, flag the discrepancy explicitly.
- Prefer recent publications (last 5 years) when possible.
- State limitations of the evidence when applicable.
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

