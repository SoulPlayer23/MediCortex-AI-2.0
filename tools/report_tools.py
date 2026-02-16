from langchain_core.tools import tool
import logging
import tempfile
import httpx
import pymupdf4llm
import os
import re

logger = logging.getLogger("SpecializedTools")

@tool
def parse_lab_values(report_content: str) -> str:
    """
    Extracts and interprets lab values from raw report text or a file URL.
    Supported formats: PDF, JPG, PNG (via URL).
    """
    logger.info(f"📋 [Report] Parsing lab values...")
    
    # Check if input looks like a URL
    url = None
    if report_content.startswith("http") or "http" in report_content:
        # Simple extraction of http... until end of string or whitespace
        url_match = re.search(r"(https?://\S+)", report_content)
        if url_match:
            url = url_match.group(1)
    
    text_to_analyze = report_content
    
    if url:
        logger.info(f"   📥 Downloading report from: {url}")
        try:
            # Download file
            with httpx.Client() as client:
                resp = client.get(url)
                resp.raise_for_status()
                
                # Save to temp file
                # Determine extension from url or content-type
                ext = ".pdf"
                if url.lower().endswith((".jpg", ".jpeg")):
                     ext = ".jpg"
                elif url.lower().endswith(".png"):
                     ext = ".png"
                     
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(resp.content)
                    tmp_path = tmp.name
                    
            if ext == ".pdf":
                logger.info(f"   📄 Converting PDF to Markdown...")
                md_text = pymupdf4llm.to_markdown(tmp_path)
                text_to_analyze = md_text
            else:
                 # Image Handling
                 logger.info(f"   🖼️ Analyzing Image...")
                 from PIL import Image, ImageStat
                 try:
                     import pytesseract
                 except ImportError:
                     pytesseract = None
                     
                 analysis_report = []
                 try:
                     with Image.open(tmp_path) as img:
                         analysis_report.append(f"Start Image Analysis for {os.path.basename(url)}")
                         analysis_report.append(f"- Format: {img.format}")
                         analysis_report.append(f"- Size: {img.size}")
                         analysis_report.append(f"- Mode: {img.mode}")
                         
                         # Check if it looks like a document (heuristic: high contrast)
                         gray = img.convert("L")
                         stat = ImageStat.Stat(gray)
                         analysis_report.append(f"- Mean Brightness: {stat.mean}")
                         
                         if pytesseract:
                             try:
                                 # Limit OCR to avoid hanging on massive images
                                 text = pytesseract.image_to_string(gray, timeout=10) 
                                 if len(text.strip()) > 10:
                                     analysis_report.append(f"\n[OCR Text Extraction]:\n{text[:2000]}")
                                 else:
                                     analysis_report.append("\n[OCR]: No significant text detected.")
                             except Exception as e:
                                 analysis_report.append(f"\n[OCR Error]: {e}")
                         else:
                             analysis_report.append("\n[OCR]: Skipped (pytesseract not installed).")
                             
                     text_to_analyze = "\n".join(analysis_report)
                 except Exception as e:
                     text_to_analyze = f"Failed to analyze image: {e}"
            
            # Cleanup
            os.remove(tmp_path)
            
        except Exception as e:
            logger.error(f"   ❌ Failed to process file URL: {e}")
            return f"Error processing file: {e}. Please provide raw text instead."

    # For now, we just return the raw text (or extracted text) 
    # In a real scenario, we would use an LLM or regex here to structure it.
    # Since the agent will see this output, returning the Markdown is perfect.
    return f"Extracted Report Content:\n{text_to_analyze[:2000]}..." # Truncate for safety if huge
