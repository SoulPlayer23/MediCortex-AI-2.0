"""
Drug Interaction Tool — Searches trusted pharmacology websites for drug-drug
interactions, contraindications, and adverse effects.

Compliant with:
  - FastAPI (async httpx, structlog, no print())
  - MCP (prompt-engineered description, error-as-information)
  - A2A (typed schemas via LangChain @tool)
"""

import re
from urllib.parse import urlparse

import httpx
import structlog
from bs4 import BeautifulSoup
from ddgs import DDGS
from langchain_core.tools import tool
from utils.cache_utils import redis_cache

logger = structlog.get_logger("DrugInteractionTool")

# ── Trusted Pharmacology Domains ─────────────────────────────────────
TRUSTED_PHARMA_DOMAINS = [
    "drugs.com",
    "rxlist.com",
    "medscape.com",
    "mayoclinic.org",
    "webmd.com",
    "nih.gov",
    "fda.gov",
    "drugbank.com",
    "medlineplus.gov",
    "goodrx.com",
]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _sanitize_query(query: str) -> str:
    """Sanitize user input to prevent injection in search queries."""
    return re.sub(r"[;|&`$\\\"']", "", query)[:500].strip()


def _is_trusted(url: str) -> bool:
    """Return True if the URL belongs to a trusted pharmacology domain."""
    domain = urlparse(url).netloc.replace("www.", "")
    return any(domain == d or domain.endswith("." + d) for d in TRUSTED_PHARMA_DOMAINS)


def _search_ddg(query: str, max_results: int = 6) -> list[dict]:
    """
    Search DuckDuckGo and return structured results (title, url, snippet).
    Filters to trusted pharmacology domains. Falls back to all domains if
    the filtered set is empty (DDG may not surface trusted results for every query).
    """
    try:
        with DDGS() as ddgs:
            site_filter = " OR ".join(f"site:{d}" for d in TRUSTED_PHARMA_DOMAINS)
            raw = list(ddgs.text(f"{query} ({site_filter})", max_results=max_results))

        trusted = [
            {"title": r["title"], "url": r["href"], "snippet": r["body"][:300],
             "domain": urlparse(r["href"]).netloc.replace("www.", "")}
            for r in raw if _is_trusted(r["href"])
        ]

        # Fallback: no trusted results — use all DDG results for the base query
        if not trusted:
            logger.warning("drug_interaction_no_trusted_results", query=query)
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=4))
            trusted = [
                {"title": r["title"], "url": r["href"], "snippet": r["body"][:300],
                 "domain": urlparse(r["href"]).netloc.replace("www.", "")}
                for r in raw
            ]

        return trusted[:max_results]

    except Exception as e:
        logger.error("ddg_search_error", error=str(e))
        return []


def _extract_interaction_content(html: str, max_chars: int = 1500) -> str:
    """Extract content focused on interactions and warnings."""
    soup = BeautifulSoup(html, "lxml")

    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    target = (
        soup.find("div", id=re.compile(r"(interaction|warning|contraindication)", re.I))
        or soup.find("section", id=re.compile(r"(interaction|warning|contraindication)", re.I))
        or soup.find("div", class_=re.compile(r"(interaction_list|monograph-content)", re.I))
        or soup.find("article")
        or soup.body
    )

    if not target:
        return ""

    text = re.sub(r"\s+", " ", target.get_text(separator=" ", strip=True))
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def _format_results(results: list[dict], query: str) -> str:
    """Format crawled results into readable markdown."""
    if not results:
        return f"No documented interactions found for '{query}' in trusted databases."

    lines = [f"## Drug Interaction Check: *{query}*\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"### {i}. {r['title']}")
        lines.append(f"- **Source:** {r['domain']}")
        lines.append(f"- **URL:** {r['url']}")
        if r.get("snippet"):
            lines.append(f"- **Summary:** {r['snippet']}")
        if r.get("content"):
            lines.append(f"- **Excerpt:** {r['content'][:500]}")
        lines.append("")

    return "\n".join(lines)


@tool
@redis_cache(ttl=86400, prefix="medicortex:drug_interaction")
def check_drug_interactions(medications: str, patient_conditions: str = "") -> str:
    """Check for drug-drug interactions, contraindications, and adverse effects.

    Searches trusted pharmacology sources (Drugs.com, RxList, Medscape, etc.)
    for interactions between listed medications or between medications and
    patient conditions.

    Args:
        medications: Comma-separated list of drugs (e.g. "Metformin, Lisinopril").
        patient_conditions: Optional list of patient conditions (e.g. "Kidney disease").

    Returns:
        Structured findings on interactions, severity (Major/Moderate/Minor),
        and contraindications.
    """
    logger.info("drug_interaction_check_start", meds=medications, conditions=patient_conditions)

    meds = _sanitize_query(medications)
    conditions = _sanitize_query(patient_conditions)

    if not meds:
        return "Error: No medications provided. Please list specific drugs to check."

    query_parts = [meds, "interactions"]
    if conditions:
        query_parts.append(f"and {conditions} contraindications")
    base_query = " ".join(query_parts)

    results = _search_ddg(base_query, max_results=4)

    if not results:
        return f"No documented interactions found for '{base_query}' in trusted databases."

    # Fetch page content for top 2 results
    try:
        with httpx.Client(timeout=12.0, headers=_HEADERS, follow_redirects=True) as client:
            for r in results[:2]:
                try:
                    page_resp = client.get(r["url"], timeout=8.0)
                    r["content"] = _extract_interaction_content(page_resp.text)
                except Exception:
                    r["content"] = ""
    except Exception as e:
        logger.warning("drug_interaction_page_fetch_failed", error=str(e))

    logger.info("drug_interaction_check_complete", results=len(results))
    return _format_results(results, base_query)
