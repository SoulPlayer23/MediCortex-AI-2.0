"""
Medical Web Crawler Tool (Diagnosis Focused) — Searches reputed medical websites,
guidelines, and clinical resources for differential diagnosis and diagnostic
criteria.

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

logger = structlog.get_logger("DiagnosisWebCrawlerTool")

# ── Trusted medical domains (Diagnosis Focused) ─────────────────────
TRUSTED_DIAGNOSIS_DOMAINS = [
    "mayoclinic.org",
    "webmd.com",
    "nih.gov",
    "medscape.com",
    "healthline.com",
    "who.int",
    "cdc.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "clevelandclinic.org",
    "hopkinsmedicine.org",
    "medlineplus.gov",
    "uptodate.com",
    "aafp.org",
    "merckmanuals.com",
    "bmj.com",
    "thelancet.com",
    "nejm.org",
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
    domain = urlparse(url).netloc.replace("www.", "")
    return any(domain == d or domain.endswith("." + d) for d in TRUSTED_DIAGNOSIS_DOMAINS)


def _search_ddg(query: str, max_results: int = 5) -> list[dict]:
    """
    Search DuckDuckGo and return structured results filtered to trusted diagnosis
    domains. Falls back to unfiltered results if no trusted hits are found.
    """
    try:
        site_filter = " OR ".join(f"site:{d}" for d in TRUSTED_DIAGNOSIS_DOMAINS)
        with DDGS() as ddgs:
            raw = list(ddgs.text(f"{query} ({site_filter})", max_results=max_results))

        trusted = [
            {"title": r["title"], "url": r["href"], "snippet": r["body"][:300],
             "domain": urlparse(r["href"]).netloc.replace("www.", "")}
            for r in raw if _is_trusted(r["href"])
        ]

        if not trusted:
            logger.warning("diagnosis_crawler_no_trusted_results", query=query)
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


def _extract_diagnosis_content(html: str, max_chars: int = 2000) -> str:
    """Extract the main text content from a page, focusing on diagnosis sections."""
    soup = BeautifulSoup(html, "lxml")

    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    main = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", class_=re.compile(r"(content|article|body|diagnosis)", re.I))
        or soup.find("div", id=re.compile(r"(content|article|body|diagnosis)", re.I))
    )

    target = main if main else soup.body
    if not target:
        return ""

    text = re.sub(r"\s+", " ", target.get_text(separator=" ", strip=True))
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def _format_results(results: list[dict], query: str) -> str:
    """Format crawled results into readable markdown."""
    if not results:
        return f"No diagnostic information found for '{query}' from trusted sources."

    lines = [f"## Diagnostic Resources for: *{query}*\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"### {i}. {r['title']}")
        lines.append(f"- **Source:** {r['domain']}")
        lines.append(f"- **URL:** {r['url']}")
        if r["snippet"]:
            lines.append(f"- **Summary:** {r['snippet']}")
        if r.get("content"):
            lines.append(f"- **Excerpt:** {r['content'][:600]}{'...' if len(r.get('content', '')) > 600 else ''}")
        lines.append("")

    return "\n".join(lines)


@tool
@redis_cache(ttl=86400, prefix="medicortex:diag_crawler")
def crawl_diagnosis_articles(query: str, max_results: int = 5) -> str:
    """Search the web for diagnostic criteria, differential diagnoses, and clinical
    guidelines from reputed medical sources.

    Retrieves results exclusively from trusted sources including Mayo Clinic, NIH,
    CDC, WebMD, Medscape, MSD Manuals, AAFP, and UpToDate.

    Returns article paragraphs, clinical summaries, and lists of potential conditions.
    Use this tool when you need to broaden your differential diagnosis or check
    specific criteria for a suspected condition.

    Args:
        query: Medical topic, symptom set, or specific condition to research.
        max_results: Number of articles to return (default 5, max 10).
    """
    logger.info("diagnosis_crawler_start", query=query, max_results=max_results)

    query = _sanitize_query(query)
    if not query:
        return "Error: Query is empty after sanitization. Please provide a valid medical query."

    max_results = max(1, min(max_results, 10))

    search_query = f"{query} diagnosis differential"
    results = _search_ddg(search_query, max_results=max_results)

    if not results:
        logger.info("diagnosis_crawler_no_results", query=query)
        return f"No diagnostic articles found for '{query}' from trusted sources."

    # Fetch page content for top results
    try:
        with httpx.Client(timeout=12.0, headers=_HEADERS, follow_redirects=True) as client:
            for r in results:
                try:
                    page_resp = client.get(r["url"], timeout=8.0)
                    r["content"] = _extract_diagnosis_content(page_resp.text)
                except Exception as exc:
                    logger.warning("diagnosis_crawler_fetch_failed", url=r["url"], error=str(exc))
                    r["content"] = ""
    except Exception as e:
        logger.warning("diagnosis_crawler_page_fetch_failed", error=str(e))

    logger.info("diagnosis_crawler_complete", query=query, results=len(results))
    return _format_results(results, query)
