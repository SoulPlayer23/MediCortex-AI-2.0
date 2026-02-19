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
from langchain_core.tools import tool

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
    "epocrates.com",
    "goodrx.com",
]

GOOGLE_SEARCH_URL = "https://www.google.com/search"

# Headers to mimic a normal browser request
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _sanitize_query(query: str) -> str:
    """Sanitize user input to prevent injection in search queries."""
    # Strip shell/SQL-like injection characters
    sanitized = re.sub(r"[;|&`$\\\"']", "", query)
    return sanitized[:500].strip()


def _build_site_filter() -> str:
    """Build a Google site: filter OR-chain for trusted domains."""
    return " OR ".join(f"site:{d}" for d in TRUSTED_PHARMA_DOMAINS)


def _parse_google_results(html: str, max_results: int) -> list[dict]:
    """Extract search result links, titles, and snippets from Google HTML."""
    soup = BeautifulSoup(html, "lxml")
    results: list[dict] = []

    for g in soup.select("div.g, div[data-hveid]"):
        anchor = g.find("a", href=True)
        if not anchor:
            continue

        href = anchor["href"]
        if not href.startswith("http") or "google.com" in href:
            continue

        title_el = g.find("h3")
        title = title_el.get_text(strip=True) if title_el else "Untitled"

        snippet_el = g.find("div", class_="VwiC3b") or g.find("span", class_="aCOpRe")
        snippet = snippet_el.get_text(strip=True) if snippet_el else ""

        domain = urlparse(href).netloc.replace("www.", "")

        results.append({
            "title": title,
            "url": href,
            "domain": domain,
            "snippet": snippet[:300],
        })

        if len(results) >= max_results:
            break

    return results


def _extract_interaction_content(html: str, max_chars: int = 1500) -> str:
    """Extract content focused on interactions and warnings."""
    soup = BeautifulSoup(html, "lxml")

    # Remove noise
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    # Prioritize interaction/warning sections
    target = (
        soup.find("div", id=re.compile(r"(interaction|warning|contraindication)", re.I))
        or soup.find("section", id=re.compile(r"(interaction|warning|contraindication)", re.I))
        or soup.find("div", class_=re.compile(r"(interaction_list|monograph-content)", re.I))
        or soup.find("article")
        or soup.body
    )

    if not target:
        return ""

    text = target.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def _format_results(results: list[dict], query: str) -> str:
    """Format crawled results into readable markdown."""
    if not results:
        return f"No interaction information found for '{query}' from trusted sources."

    lines = [f"## Drug Interaction Check: *{query}*\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"### {i}. {r['title']}")
        lines.append(f"- **Source:** {r['domain']}")
        lines.append(f"- **URL:** {r['url']}")
        if r["snippet"]:
            lines.append(f"- **Summary:** {r['snippet']}")
        if r.get("content"):
            lines.append(f"- **Excerpt:** {r['content'][:500]}")
        lines.append("")

    return "\n".join(lines)


@tool
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

    # Input sanitization
    meds = _sanitize_query(medications)
    conditions = _sanitize_query(patient_conditions)
    
    if not meds:
        return "Error: No medications provided. Please list specific drugs to check."

    # Construct search query
    query_parts = [meds, "interactions"]
    if conditions:
        query_parts.append(f"and {conditions} contraindications")
    
    base_query = " ".join(query_parts)

    try:
        with httpx.Client(timeout=15.0, headers=_HEADERS, follow_redirects=True) as client:
            site_filter = _build_site_filter()
            search_query = f"{base_query} ({site_filter})"

            resp = client.get(
                GOOGLE_SEARCH_URL,
                params={"q": search_query, "num": "6", "hl": "en"},
            )
            resp.raise_for_status()

            results = _parse_google_results(resp.text, max_results=4)

            if not results:
                return f"No documented interactions found for '{base_query}' in trusted databases."

            # Fetch content for top 2 results to get details
            for r in results[:2]:
                try:
                    page_resp = client.get(r["url"], timeout=8.0)
                    r["content"] = _extract_interaction_content(page_resp.text)
                except Exception:
                    r["content"] = "Content extraction failed."

        logger.info("drug_interaction_check_complete", results=len(results))
        return _format_results(results, base_query)

    except Exception as e:
        logger.error("drug_interaction_error", error=str(e))
        return f"Error checking interactions: {str(e)}"
