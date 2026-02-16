"""
PubMed Search Tool — Queries the NCBI PubMed E-utilities API
to retrieve real medical research papers with metadata.

Compliant with:
  - FastAPI (async httpx, structlog, no print())
  - MCP (prompt-engineered description, error-as-information)
  - A2A (typed schemas via LangChain @tool)
"""

import xml.etree.ElementTree as ET

import httpx
import structlog
from langchain_core.tools import tool

logger = structlog.get_logger("PubMedSearchTool")

# ── NCBI E-utilities endpoints ──────────────────────────────────────
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_ARTICLE_BASE = "https://pubmed.ncbi.nlm.nih.gov"


def _parse_ids(xml_text: str) -> list[str]:
    """Extract PubMed IDs from an esearch XML response."""
    root = ET.fromstring(xml_text)
    return [id_el.text for id_el in root.findall(".//IdList/Id") if id_el.text]


def _parse_articles(xml_text: str) -> list[dict]:
    """Extract structured article data from an efetch XML response."""
    root = ET.fromstring(xml_text)
    articles: list[dict] = []

    for article_el in root.findall(".//PubmedArticle"):
        medline = article_el.find(".//MedlineCitation")
        if medline is None:
            continue

        pmid_el = medline.find("PMID")
        pmid = pmid_el.text if pmid_el is not None else "N/A"

        art = medline.find("Article")
        if art is None:
            continue

        # Title
        title_el = art.find("ArticleTitle")
        title = title_el.text if title_el is not None else "No title"

        # Journal
        journal_el = art.find(".//Journal/Title")
        journal = journal_el.text if journal_el is not None else "Unknown journal"

        # Year
        year_el = art.find(".//Journal/JournalIssue/PubDate/Year")
        year = year_el.text if year_el is not None else ""

        # Authors (first 3 + et al.)
        authors_raw: list[str] = []
        for author in art.findall(".//AuthorList/Author"):
            last = author.find("LastName")
            fore = author.find("ForeName")
            if last is not None and fore is not None:
                authors_raw.append(f"{last.text} {fore.text}")
        if len(authors_raw) > 3:
            authors_str = ", ".join(authors_raw[:3]) + " et al."
        else:
            authors_str = ", ".join(authors_raw) if authors_raw else "Unknown authors"

        # Abstract
        abstract_parts: list[str] = []
        for abs_text in art.findall(".//Abstract/AbstractText"):
            if abs_text.text:
                abstract_parts.append(abs_text.text)
        abstract = " ".join(abstract_parts) if abstract_parts else "No abstract available."

        # DOI
        doi = ""
        for eid in art.findall(".//ELocationID"):
            if eid.get("EIdType") == "doi" and eid.text:
                doi = eid.text
                break

        # URL
        url = f"{PUBMED_ARTICLE_BASE}/{pmid}/"

        articles.append({
            "pmid": pmid,
            "title": title,
            "authors": authors_str,
            "journal": journal,
            "year": year,
            "abstract": abstract[:500] + ("..." if len(abstract) > 500 else ""),
            "doi": doi,
            "url": url,
        })

    return articles


def _format_results(articles: list[dict], query: str) -> str:
    """Format parsed articles into a readable markdown string."""
    if not articles:
        return f"No PubMed results found for '{query}'."

    lines = [f"## PubMed Results for: *{query}*\n"]
    for i, a in enumerate(articles, 1):
        lines.append(f"### {i}. {a['title']}")
        lines.append(f"- **Authors:** {a['authors']}")
        lines.append(f"- **Journal:** {a['journal']} ({a['year']})")
        if a["doi"]:
            lines.append(f"- **DOI:** `{a['doi']}`")
        lines.append(f"- **PubMed URL:** {a['url']}")
        lines.append(f"- **Abstract:** {a['abstract']}")
        lines.append("")

    return "\n".join(lines)


@tool
def search_pubmed(query: str, max_results: int = 5) -> str:
    """Search the NCBI PubMed database for medical research papers matching the
    query. Returns structured results including paper title, authors, journal,
    year, abstract excerpt, DOI, and a direct PubMed URL for each article.
    Use this tool when the user asks about medical research, clinical trials,
    evidence-based treatments, or recent scientific findings. The 'query'
    should be a focused medical topic or research question. 'max_results'
    controls how many papers are returned (default 5, max 20)."""

    logger.info("pubmed_search_start", query=query, max_results=max_results)

    # Clamp max_results
    max_results = max(1, min(max_results, 20))

    try:
        with httpx.Client(timeout=15.0) as client:
            # Step 1: esearch — find matching PubMed IDs
            search_resp = client.get(
                ESEARCH_URL,
                params={
                    "db": "pubmed",
                    "term": query,
                    "retmax": str(max_results),
                    "sort": "relevance",
                    "retmode": "xml",
                },
            )
            search_resp.raise_for_status()
            pmids = _parse_ids(search_resp.text)

            if not pmids:
                logger.info("pubmed_search_no_results", query=query)
                return f"No PubMed results found for '{query}'."

            # Step 2: efetch — get article metadata
            fetch_resp = client.get(
                EFETCH_URL,
                params={
                    "db": "pubmed",
                    "id": ",".join(pmids),
                    "retmode": "xml",
                    "rettype": "abstract",
                },
            )
            fetch_resp.raise_for_status()
            articles = _parse_articles(fetch_resp.text)

        logger.info("pubmed_search_complete", query=query, results=len(articles))
        return _format_results(articles, query)

    except httpx.TimeoutException:
        logger.error("pubmed_search_timeout", query=query)
        return f"Error: PubMed search timed out for '{query}'. Please try again."
    except httpx.HTTPStatusError as exc:
        logger.error("pubmed_search_http_error", query=query, status=exc.response.status_code)
        return f"Error: PubMed returned HTTP {exc.response.status_code}. Please try again later."
    except Exception as exc:
        logger.error("pubmed_search_error", query=query, error=str(exc))
        return f"Error searching PubMed: {str(exc)}"
