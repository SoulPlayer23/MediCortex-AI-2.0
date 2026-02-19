"""
Image Extraction Tool — Downloads medical images from URLs and sends them
to MedGemma's vision API for analysis (X-rays, MRIs, CT scans, lab photos).

Compliant with:
  - FastAPI (structlog, no print())
  - MCP (prompt-engineered description, error-as-information)
  - A2A (typed schemas via LangChain @tool)
"""

import base64
import re

import httpx
import requests
import structlog
from langchain_core.tools import tool

logger = structlog.get_logger("ImageExtractionTool")

# MedGemma API endpoint (same as medgemma_llm.py)
MEDGEMMA_API_URL = "http://100.107.2.102:8000/predict"

# Supported image extensions
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")


@tool
def extract_image_findings(file_url: str, clinical_context: str = "") -> str:
    """Download a medical image from a URL and analyze it using MedGemma vision.

    Sends the image to MedGemma's multimodal endpoint for visual analysis.
    Supports X-rays, MRIs, CT scans, pathology slides, and photographed
    lab reports. Returns the model's clinical findings.

    Use this tool when the user provides an image URL for analysis.

    Args:
        file_url: HTTP/HTTPS URL pointing to the image file (JPG, PNG, etc.).
        clinical_context: Optional clinical context (e.g., "chest X-ray, 45yo male with cough").

    Returns:
        MedGemma's visual analysis findings.
    """
    logger.info("image_extraction_start", url=file_url)

    # Validate URL
    url_match = re.search(r"(https?://\S+)", file_url.strip())
    if not url_match:
        return "Error: Invalid URL. Please provide a valid HTTP/HTTPS link to an image."

    url = url_match.group(1)

    try:
        # Download the image
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            is_image = (
                "image" in content_type.lower()
                or url.lower().endswith(_IMAGE_EXTENSIONS)
            )

            if not is_image:
                return (
                    f"Error: URL does not point to an image (Content-Type: {content_type}). "
                    f"For PDF documents, use the `extract_document_text` tool instead."
                )

            # Encode as base64
            image_b64 = base64.b64encode(resp.content).decode("utf-8")

        # Build MedGemma vision prompt
        prompt = "Analyze this medical image. "
        if clinical_context:
            prompt += f"Clinical context: {clinical_context}. "
        prompt += (
            "Provide:\n"
            "1. **Image Type** — What kind of medical image this is (X-ray, MRI, CT, lab report photo, etc.)\n"
            "2. **Key Findings** — Observable features, abnormalities, or normal structures\n"
            "3. **Clinical Significance** — What the findings may indicate\n"
            "4. **Recommendations** — Suggested follow-up if applicable\n\n"
            "If this is a photographed lab report, extract the text values instead."
        )

        # Send to MedGemma vision API
        logger.info("image_extraction_sending_to_medgemma")
        payload = {
            "prompt": prompt,
            "image_base64": image_b64,
            "max_tokens": 512
        }

        response = requests.post(MEDGEMMA_API_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        findings = result.get("response", "")

        if not findings:
            return "Warning: MedGemma returned empty analysis. The image may not be a recognizable medical image."

        logger.info("image_extraction_complete")
        return f"## Medical Image Analysis\n\n{findings}"

    except httpx.TimeoutException:
        logger.error("image_extraction_download_timeout")
        return "Error: Image download timed out."
    except requests.exceptions.Timeout:
        logger.error("image_extraction_medgemma_timeout")
        return "Error: MedGemma analysis timed out. The model may be under heavy load."
    except requests.exceptions.ConnectionError:
        logger.error("image_extraction_medgemma_offline")
        return "Error: MedGemma vision service is unavailable. Please try again later."
    except Exception as e:
        logger.error("image_extraction_failed", error=str(e))
        return f"Error analyzing image: {str(e)}"
