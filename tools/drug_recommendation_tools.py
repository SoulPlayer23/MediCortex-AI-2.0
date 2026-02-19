"""
Drug Recommendation Tool — searches for drug recommendations, dosage guidelines,
and alternative medications from trusted pharmacology sources.

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

logger = structlog.get_logger("DrugRecommendationTool")

# reuse standard pharma domains + guidelines
TRUSTED_GUIDELINE_DOMAINS = [
    "drugs.com",
    "mayoclinic.org",
    "webmd.com",
    "aafp.org",
    "nice.org.uk",  # good for guidelines
    "diabetes.org", # ADA
    "heart.org",    # AHA
    "uptodate.com",
]

GOOGLE_SEARCH_URL = "https://www.google.com/search"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _sanitize_query(query: str) -> str:
    return re.sub(r"[;|&`$\\\"']", "", query)[:500].strip()


def _build_site_filter() -> str:
    return " OR ".join(f"site:{d}" for d in TRUSTED_GUIDELINE_DOMAINS)


def _parse_google_results(html: str, max_results: int) -> list[dict]:
    # Simplified parsing (same as standard pattern)
    soup = BeautifulSoup(html, "lxml")
    results = []
    for g in soup.select("div.g, div[data-hveid]"):
        anchor = g.find("a", href=True)
        if not anchor: continue
        href = anchor["href"]
        if not href.startswith("http") or "google.com" in href: continue
        
        title = g.find("h3").get_text(strip=True) if g.find("h3") else "Untitled"
        snippet_el = g.find("div", class_="VwiC3b") or g.find("span", class_="aCOpRe")
        snippet = snippet_el.get_text(strip=True) if snippet_el else ""
        
        results.append({
            "title": title,
            "url": href,
            "domain": urlparse(href).netloc.replace("www.", ""),
            "snippet": snippet[:300]
        })
        if len(results) >= max_results: break
    return results


def _extract_text_content(html: str, max_chars: int = 1500) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(["script", "style", "nav", "footer", "form"]):
        tag.decompose()
    
    # Try finding dosage/treatment sections
    target = (
        soup.find("div", id=re.compile(r"(dosage|treatment|uses)", re.I))
        or soup.find("section", id=re.compile(r"(dosage|treatment|uses)", re.I))
        or soup.find("article")
        or soup.body
    )
    if not target: return ""
    
    text = target.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text)[:max_chars] + "..."


@tool
def recommend_drugs(condition: str, query_type: str = "recommendation", patient_info: str = "") -> str:
    """Retrieve drug recommendations, dosage usage, or alternative medications.
    
    Args:
        condition: The medical condition (e.g., "Type 2 Diabetes") or drug name (e.g. "Lisinopril").
        query_type: One of "recommendation" (default), "dosage", or "alternatives".
          - "recommendation": Best first-line drugs for a condition.
          - "dosage": Standard dosage guidelines for a drug/condition.
          - "alternatives": Alternative drugs with similar mechanism or indication.
        patient_info: Optional patient context (e.g. "elderly", "kidney failure").
    
    Returns:
        Structured summary of findings from trusted medical guidelines.
    """
    logger.info("drug_recommendation_start", condition=condition, type=query_type)

    cond = _sanitize_query(condition)
    info = _sanitize_query(patient_info)
    
    if not cond:
        return "Error: No condition or drug specified."

    # Construct specific search queries based on intent
    if query_type == "dosage":
        search_query = f"{cond} dosage guidelines {info}"
    elif query_type == "alternatives":
        search_query = f"{cond} alternatives substitutes {info}"
    else: # recommendation
        search_query = f"{cond} treatment guidelines first-line drugs {info}"

    try:
        with httpx.Client(timeout=15.0, headers=_HEADERS, follow_redirects=True) as client:
            site_filter = _build_site_filter()
            final_query = f"{search_query} ({site_filter})"

            resp = client.get(
                GOOGLE_SEARCH_URL,
                params={"q": final_query, "num": "5", "hl": "en"},
            )
            resp.raise_for_status()

            results = _parse_google_results(resp.text, max_results=4)

            if not results:
                return f"No results found for '{search_query}'."

            # Fetch content for top result
            if results:
                try:
                    page_resp = client.get(results[0]["url"], timeout=8.0)
                    results[0]["content"] = _extract_text_content(page_resp.text)
                except:
                    pass

        # Format output
        findings = [f"## Drug Query: {query_type.title()} for *{cond}*\n"]
        for r in results:
            findings.append(f"### {r['title']}")
            findings.append(f"- **Source:** {r['domain']} | [Link]({r['url']})")
            if r.get("content"):
                findings.append(f"- **Excerpt:** {r['content'][:400]}")
            elif r["snippet"]:
                findings.append(f"- **Summary:** {r['snippet']}")
            findings.append("")
        
        return "\n".join(findings)

    except Exception as e:
        logger.error("drug_recommendation_error", error=str(e))
        return f"Error retrieving drug info: {str(e)}"
