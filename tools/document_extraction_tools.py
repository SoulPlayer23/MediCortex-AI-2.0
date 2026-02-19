"""
Document Extraction Tool — Downloads PDF reports from URLs and converts
them to structured Markdown text using pymupdf4llm.

Compliant with:
  - FastAPI (structlog, no print())
  - MCP (prompt-engineered description, error-as-information)
  - A2A (typed schemas via LangChain @tool)
"""

import os
import re
import tempfile

import httpx
import pymupdf4llm
import structlog
from langchain_core.tools import tool

logger = structlog.get_logger("DocumentExtractionTool")


@tool
def extract_document_text(file_url: str) -> str:
    """Download a PDF report from a URL and extract its text content as Markdown.

    Converts PDF documents (lab reports, discharge summaries, prescriptions)
    into structured Markdown text for downstream analysis. Supports PDFs
    hosted on MinIO, cloud storage, or any accessible HTTP URL.

    Use this tool when the user provides a PDF file URL for analysis.

    Args:
        file_url: HTTP/HTTPS URL pointing to the PDF document.

    Returns:
        Extracted text content in Markdown format, truncated to 3000 chars.
    """
    logger.info("document_extraction_start", url=file_url)

    # Validate URL
    url_match = re.search(r"(https?://\S+)", file_url.strip())
    if not url_match:
        return "Error: Invalid URL. Please provide a valid HTTP/HTTPS link to a PDF file."

    url = url_match.group(1)

    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()

            # Determine file type
            content_type = resp.headers.get("content-type", "")
            is_pdf = (
                "pdf" in content_type.lower()
                or url.lower().endswith(".pdf")
            )

            if not is_pdf:
                return (
                    f"Error: The URL does not point to a PDF file (Content-Type: {content_type}). "
                    f"For image files, use the `extract_image_findings` tool instead."
                )

            # Save to temp file and convert
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(resp.content)
                tmp_path = tmp.name

            logger.info("document_extraction_converting", path=tmp_path)
            md_text = pymupdf4llm.to_markdown(tmp_path)

            # Cleanup
            os.remove(tmp_path)

            if not md_text or len(md_text.strip()) < 10:
                return "Warning: PDF extracted but contains very little text. It may be a scanned document — try the image extraction tool instead."

            # Truncate for safety
            truncated = md_text[:3000]
            if len(md_text) > 3000:
                truncated += "\n\n... [Content truncated. Full document contained more text.]"

            logger.info("document_extraction_complete", chars=len(md_text))
            return f"## Extracted Document Content\n\n{truncated}"

    except httpx.TimeoutException:
        logger.error("document_extraction_timeout", url=url)
        return "Error: Download timed out. The file may be too large or the server is slow."
    except httpx.HTTPStatusError as e:
        logger.error("document_extraction_http_error", status=e.response.status_code)
        return f"Error: HTTP {e.response.status_code} when downloading the file."
    except Exception as e:
        logger.error("document_extraction_failed", error=str(e))
        return f"Error extracting document: {str(e)}"
