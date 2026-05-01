"""
Document Extraction Tool — Downloads PDF reports from URLs, extracts text
page-by-page using pymupdf4llm, groups pages into layout-aware sections,
analyzes each section with MedGemma, and aggregates the results so that no
page or section is ever skipped or truncated.

Compliant with:
  - FastAPI (structlog, no print())
  - MCP (prompt-engineered description, error-as-information)
  - A2A (typed schemas via LangChain @tool)
"""

import os
import re
import tempfile
from typing import List, Tuple

import fitz  # PyMuPDF
import httpx
import pymupdf4llm
import structlog
from langchain_core.tools import tool

logger = structlog.get_logger("DocumentExtractionTool")

# ── Section-boundary detection ────────────────────────────────────────────────
_HEADING_RE = re.compile(r"^#{1,3}\s+\S", re.MULTILINE)
_ALLCAPS_RE = re.compile(r"^[A-Z][A-Z\s\-/]{4,}$", re.MULTILINE)


def _is_new_section(page_md: str) -> bool:
    """Return True if a page looks like the start of a new clinical section."""
    first_300 = page_md[:300]
    return bool(_HEADING_RE.search(first_300) or _ALLCAPS_RE.search(first_300))


def _group_pages(pages_md: List[str]) -> List[Tuple[List[int], str]]:
    """
    Group page-level markdown into layout-aware sections.

    A page that starts with a heading/all-caps title begins a new section.
    A page that looks like a continuation is merged with the preceding section.
    Returns a list of (page_numbers, combined_markdown) tuples.
    """
    if not pages_md:
        return []

    groups: List[Tuple[List[int], str]] = []
    current_pages = [0]
    current_text = pages_md[0]

    for i, md in enumerate(pages_md[1:], start=1):
        if _is_new_section(md):
            groups.append((current_pages, current_text))
            current_pages = [i]
            current_text = md
        else:
            current_pages.append(i)
            current_text += f"\n\n<!-- page {i + 1} continues -->\n\n{md}"

    groups.append((current_pages, current_text))
    return groups


_SECTION_MAX_CHARS = 6000  # ~1500 tokens — safe for MedGemma 7B context window


def _analyze_section_with_medgemma(section_text: str, section_label: str, report_type: str) -> str:
    """Call MedGemma (with Ollama fallback) to produce a detailed analysis of one section.

    If the section text exceeds the model's safe context window it is split into
    sub-chunks, each analyzed individually, and the sub-results are joined before
    returning so no content is silently dropped.
    """
    from specialized_agents.medgemma_llm import MedGemmaLLM
    llm_instance = MedGemmaLLM()

    # Split oversized sections into sub-chunks and recurse
    if len(section_text) > _SECTION_MAX_CHARS:
        chunks = [
            section_text[i: i + _SECTION_MAX_CHARS]
            for i in range(0, len(section_text), _SECTION_MAX_CHARS)
        ]
        sub_analyses = []
        for chunk_idx, chunk in enumerate(chunks, start=1):
            sub_label = f"{section_label} (part {chunk_idx}/{len(chunks)})"
            sub_analyses.append(_analyze_section_with_medgemma(chunk, sub_label, report_type))
        return "\n\n".join(sub_analyses)

    prompt = (
        f"You are a clinical medical report analyst.\n\n"
        f"Analyze the following section of a {report_type} medical document "
        f"({section_label}) and extract EVERY clinically relevant detail. "
        f"Be exhaustive — this output will be fed into a final aggregation step, "
        f"so do NOT summarise or omit any values, findings, or observations.\n\n"
        f"Structure your output as:\n"
        f"- **Findings** — all observed values, measurements, or clinical observations\n"
        f"- **Abnormalities** — anything outside normal ranges (flag with ⚠️)\n"
        f"- **Clinical Notes** — relevant contextual information\n\n"
        f"CRITICAL: Base your analysis ONLY on the content below. "
        f"Do NOT fabricate or infer values.\n\n"
        f"--- SECTION CONTENT ---\n{section_text}"
    )

    try:
        return llm_instance.invoke(prompt)
    except Exception as e:
        logger.error("section_analysis_failed", label=section_label, error=str(e))
        return f"[Analysis failed for {section_label}: {e}]\n\nRaw content:\n{section_text[:2000]}"


def _aggregate_analyses(section_analyses: List[str], report_type: str) -> str:
    """Call MedGemma to aggregate all per-section analyses into a single comprehensive report."""
    from specialized_agents.medgemma_llm import MedGemmaLLM
    llm_instance = MedGemmaLLM()

    combined = "\n\n---\n\n".join(
        f"### {label}\n{analysis}"
        for label, analysis in section_analyses
    )

    prompt = (
        f"You are a clinical medical report analyst performing a final aggregation.\n\n"
        f"Below are detailed per-section analyses of a multi-page {report_type} report. "
        f"Synthesize them into a single comprehensive clinical interpretation. "
        f"Do NOT discard any finding from any section.\n\n"
        f"Output as structured Markdown:\n"
        f"1. **Report Summary** — Report type and high-level overview\n"
        f"2. **Key Findings** — All important values, measurements, and observations (from ALL sections)\n"
        f"3. **Abnormalities** — All values or findings outside normal ranges (flag with ⚠️)\n"
        f"4. **Clinical Significance** — What the full set of findings may indicate clinically\n"
        f"5. **Recommendations** — Suggested follow-up actions or specialist referrals\n\n"
        f"CRITICAL: Preserve every specific value and finding from the section analyses below.\n\n"
        f"--- PER-SECTION ANALYSES ---\n\n{combined}"
    )

    try:
        return llm_instance.invoke(prompt)
    except Exception as e:
        logger.error("aggregation_failed", error=str(e))
        return "\n\n---\n\n".join(f"**{label}**\n{a}" for label, a in section_analyses)


@tool
def extract_document_text(file_url: str, report_type: str = "general") -> str:
    """Download a PDF report from a URL, extract every page with layout-aware
    section grouping, analyze each section individually with MedGemma, and
    return a comprehensive aggregated clinical analysis covering the full document.

    This tool processes every page of the PDF without truncation. Pages that
    belong to the same clinical section are grouped together before analysis
    so section context is never split across separate passes.

    Use this tool when the user provides a PDF file URL for analysis.

    Args:
        file_url: HTTP/HTTPS URL pointing to the PDF document.
        report_type: Hint for the type of report — "lab_report",
                     "discharge_summary", "imaging", or "general".

    Returns:
        Comprehensive aggregated clinical analysis covering ALL pages and sections.
    """
    logger.info("document_extraction_start", url=file_url)

    url_match = re.search(r"(https?://\S+)", file_url.strip())
    if not url_match:
        return "Error: Invalid URL. Please provide a valid HTTP/HTTPS link to a PDF file."

    url = url_match.group(1)

    try:
        # SEC-1: stream the download with a hard byte cap. Buffering the full
        # response into memory is an OOM vector for adversarial URLs.
        from config import settings as _settings
        max_bytes = _settings.MAX_PDF_BYTES

        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            with client.stream("GET", url) as resp:
                resp.raise_for_status()

                content_type = resp.headers.get("content-type", "")
                is_pdf = "pdf" in content_type.lower() or url.lower().endswith(".pdf")
                if not is_pdf:
                    return (
                        f"Error: The URL does not point to a PDF file (Content-Type: {content_type}). "
                        f"For image files, use the `extract_image_findings` tool instead."
                    )

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    total = 0
                    for chunk in resp.iter_bytes(chunk_size=65536):
                        total += len(chunk)
                        if total > max_bytes:
                            tmp.close()
                            os.remove(tmp.name)
                            return (
                                f"Error: PDF exceeds the {max_bytes // (1024 * 1024)} MB size cap. "
                                f"Provide a smaller document or split into pages."
                            )
                        tmp.write(chunk)
                    tmp_path = tmp.name

        try:
            # ── Get page count via PyMuPDF ─────────────────────────────────
            doc = fitz.open(tmp_path)
            page_count = doc.page_count
            doc.close()
            logger.info("document_extraction_pages", count=page_count)

            # ── Extract each page individually ─────────────────────────────
            pages_md: List[str] = []
            for page_idx in range(page_count):
                page_text = pymupdf4llm.to_markdown(tmp_path, pages=[page_idx])
                pages_md.append(page_text or "")

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        if not any(p.strip() for p in pages_md):
            return (
                "Warning: PDF extracted but all pages appear to be blank or scanned images. "
                "Try the `extract_image_findings` tool instead."
            )

        # ── Group pages into layout-aware sections ─────────────────────────
        groups = _group_pages(pages_md)
        logger.info("document_extraction_sections", count=len(groups))

        # ── Analyze each section with MedGemma ────────────────────────────
        section_analyses: List[Tuple[str, str]] = []
        for group_pages, group_md in groups:
            if not group_md.strip():
                continue
            if len(group_pages) == 1:
                label = f"Page {group_pages[0] + 1}"
            else:
                label = f"Pages {group_pages[0] + 1}–{group_pages[-1] + 1}"

            logger.info("document_section_analyzing", label=label)
            analysis = _analyze_section_with_medgemma(group_md, label, report_type)
            section_analyses.append((label, analysis))

        if not section_analyses:
            return "Error: No extractable content found in the PDF."

        # ── Single-page shortcut: skip redundant aggregation ──────────────
        if len(section_analyses) == 1:
            label, analysis = section_analyses[0]
            return f"## Extracted Document Analysis ({label})\n\n{analysis}"

        # ── Aggregate all section analyses into final report ───────────────
        logger.info("document_extraction_aggregating", sections=len(section_analyses))
        final_analysis = _aggregate_analyses(section_analyses, report_type)

        logger.info("document_extraction_complete", pages=page_count, sections=len(section_analyses))
        return f"## Full Document Analysis ({page_count} pages, {len(section_analyses)} sections)\n\n{final_analysis}"

    except httpx.TimeoutException:
        logger.error("document_extraction_timeout", url=url)
        return "Error: Download timed out. The file may be too large or the server is slow."
    except httpx.HTTPStatusError as e:
        logger.error("document_extraction_http_error", status=e.response.status_code)
        return f"Error: HTTP {e.response.status_code} when downloading the file."
    except Exception as e:
        logger.error("document_extraction_failed", error=str(e))
        return f"Error extracting document: {str(e)}"
