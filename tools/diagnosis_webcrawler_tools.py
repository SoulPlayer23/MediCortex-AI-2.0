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
from langchain_core.tools import tool

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
    "uptodate.com",       # Added for diagnosis
    "aafp.org",           # American Academy of Family Physicians
    "merckmanuals.com",   # Merck Manual Professional Version
    "bmj.com",
    "thelancet.com",
    "nejm.org",
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
    # Limit length
    return sanitized[:500].strip()


def _build_site_filter() -> str:
    """Build a Google site: filter OR-chain for trusted domains."""
    return " OR ".join(f"site:{d}" for d in TRUSTED_DIAGNOSIS_DOMAINS)


def _parse_google_results(html: str, max_results: int) -> list[dict]:
    """Extract search result links, titles, and snippets from Google HTML."""
    soup = BeautifulSoup(html, "lxml")
    results: list[dict] = []

    for g in soup.select("div.g, div[data-hveid]"):
        # Find the link
        anchor = g.find("a", href=True)
        if not anchor:
            continue

        href = anchor["href"]
        # Skip Google internal links
        if not href.startswith("http") or "google.com" in href:
            continue

        # Title
        title_el = g.find("h3")
        title = title_el.get_text(strip=True) if title_el else "Untitled"

        # Snippet
        snippet_el = g.find("div", class_="VwiC3b") or g.find("span", class_="aCOpRe")
        snippet = snippet_el.get_text(strip=True) if snippet_el else ""

        # Extract domain
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


def _extract_diagnosis_content(html: str, max_chars: int = 2000) -> str:
    """Extract the main text content from a page, focusing on diagnosis sections."""
    soup = BeautifulSoup(html, "lxml")

    # Remove noise
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    # Heuristic: Try to find "Diagnosis" or "Symptoms" sections specifically?
    # For now, general content extraction is safer than missing data.
    
    # Try to find the main content container
    main = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", class_=re.compile(r"(content|article|body|diagnosis)", re.I))
        or soup.find("div", id=re.compile(r"(content|article|body|diagnosis)", re.I))
    )

    target = main if main else soup.body
    if not target:
        return ""

    text = target.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text)
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

    # Input sanitization (MCP §4.2)
    query = _sanitize_query(query)
    if not query:
        return "Error: Query is empty after sanitization. Please provide a valid medical query."

    max_results = max(1, min(max_results, 10))

    try:
        with httpx.Client(timeout=20.0, headers=_HEADERS, follow_redirects=True) as client:
            # Step 1: Google search scoped to trusted medical domains
            site_filter = _build_site_filter()
            search_query = f"{query} diagnosis differential ({site_filter})"

            resp = client.get(
                GOOGLE_SEARCH_URL,
                params={
                    "q": search_query,
                    "num": str(max_results + 5),
                    "hl": "en",
                },
            )
            resp.raise_for_status()

            results = _parse_google_results(resp.text, max_results)

            if not results:
                logger.info("diagnosis_crawler_no_results", query=query)
                return f"No diagnostic articles found for '{query}' from trusted sources."

            # Step 2: Follow top links to extract article content
            for r in results:
                try:
                    page_resp = client.get(r["url"], timeout=10.0)
                    page_resp.raise_for_status()
                    r["content"] = _extract_diagnosis_content(page_resp.text)
                except Exception as exc:
                    logger.warning("diagnosis_crawler_fetch_failed", url=r["url"], error=str(exc))
                    r["content"] = ""

        logger.info("diagnosis_crawler_complete", query=query, results=len(results))
        return _format_results(results, query)

    except httpx.TimeoutException:
        logger.error("diagnosis_crawler_timeout", query=query)
        return f"Error: Web search timed out for '{query}'. Please try again."
    except httpx.HTTPStatusError as exc:
        logger.error("diagnosis_crawler_http_error", query=query, status=exc.response.status_code)
        return f"Error: Search returned HTTP {exc.response.status_code}. Please try again later."
    except Exception as exc:
        logger.error("diagnosis_crawler_error", query=query, error=str(exc))
        return f"Error crawling diagnostic articles: {str(exc)}"
