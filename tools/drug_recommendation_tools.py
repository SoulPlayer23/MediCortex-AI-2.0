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
from ddgs import DDGS
from langchain_core.tools import tool
from utils.cache_utils import redis_cache

logger = structlog.get_logger("DrugRecommendationTool")

TRUSTED_GUIDELINE_DOMAINS = [
    "drugs.com",
    "mayoclinic.org",
    "webmd.com",
    "aafp.org",
    "nice.org.uk",
    "diabetes.org",
    "heart.org",
    "uptodate.com",
    "medlineplus.gov",
    "nih.gov",
]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _sanitize_query(query: str) -> str:
    return re.sub(r"[;|&`$\\\"']", "", query)[:500].strip()


def _is_trusted(url: str) -> bool:
    domain = urlparse(url).netloc.replace("www.", "")
    return any(domain == d or domain.endswith("." + d) for d in TRUSTED_GUIDELINE_DOMAINS)


def _search_ddg(query: str, max_results: int = 5) -> list[dict]:
    """
    Search DuckDuckGo and return structured results filtered to trusted guideline
    domains. Falls back to unfiltered results if no trusted hits are found.
    """
    try:
        site_filter = " OR ".join(f"site:{d}" for d in TRUSTED_GUIDELINE_DOMAINS)
        with DDGS() as ddgs:
            raw = list(ddgs.text(f"{query} ({site_filter})", max_results=max_results))

        trusted = [
            {"title": r["title"], "url": r["href"], "snippet": r["body"][:300],
             "domain": urlparse(r["href"]).netloc.replace("www.", "")}
            for r in raw if _is_trusted(r["href"])
        ]

        if not trusted:
            logger.warning("drug_recommendation_no_trusted_results", query=query)
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


def _extract_text_content(html: str, max_chars: int = 1500) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(["script", "style", "nav", "footer", "form"]):
        tag.decompose()

    target = (
        soup.find("div", id=re.compile(r"(dosage|treatment|uses)", re.I))
        or soup.find("section", id=re.compile(r"(dosage|treatment|uses)", re.I))
        or soup.find("article")
        or soup.body
    )
    if not target:
        return ""

    text = re.sub(r"\s+", " ", target.get_text(separator=" ", strip=True))
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


@tool
@redis_cache(ttl=86400, prefix="medicortex:drug_recommendation")
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

    if query_type == "dosage":
        search_query = f"{cond} dosage guidelines {info}".strip()
    elif query_type == "alternatives":
        search_query = f"{cond} alternative medications {info}".strip()
    else:
        search_query = f"{cond} first-line treatment drugs guidelines {info}".strip()

    results = _search_ddg(search_query, max_results=4)

    if not results:
        return f"No results found for '{search_query}' in trusted databases."

    # Fetch page content for top result
    try:
        with httpx.Client(timeout=12.0, headers=_HEADERS, follow_redirects=True) as client:
            try:
                page_resp = client.get(results[0]["url"], timeout=8.0)
                results[0]["content"] = _extract_text_content(page_resp.text)
            except Exception:
                results[0]["content"] = ""
    except Exception as e:
        logger.warning("drug_recommendation_page_fetch_failed", error=str(e))

    findings = [f"## Drug Query: {query_type.title()} for *{cond}*\n"]
    for r in results:
        findings.append(f"### {r['title']}")
        findings.append(f"- **Source:** {r['domain']} | [Link]({r['url']})")
        if r.get("content"):
            findings.append(f"- **Excerpt:** {r['content'][:400]}")
        elif r.get("snippet"):
            findings.append(f"- **Summary:** {r['snippet']}")
        findings.append("")

    logger.info("drug_recommendation_complete", results=len(results))
    return "\n".join(findings)
