"""
Tests for report tools: document extraction, image extraction, and report analysis.
"""
import pytest
from unittest.mock import patch, MagicMock


class TestExtractDocumentText:
    """Tests for extract_document_text tool."""

    def test_invalid_url(self):
        from tools.document_extraction_tools import extract_document_text
        result = extract_document_text.invoke({"file_url": "not-a-url"})
        assert "Error" in result

    def test_non_pdf_url(self):
        from tools.document_extraction_tools import extract_document_text
        mock_resp = MagicMock()
        mock_resp.headers = {"content-type": "image/jpeg"}
        mock_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(return_value=mock_resp)

        with patch("tools.document_extraction_tools.httpx.Client", return_value=mock_client):
            result = extract_document_text.invoke({"file_url": "https://example.com/image.jpg"})
        assert "not point to a PDF" in result or "image" in result.lower()

    def test_pdf_extraction(self):
        from tools.document_extraction_tools import extract_document_text
        mock_resp = MagicMock()
        mock_resp.headers = {"content-type": "application/pdf"}
        mock_resp.content = b"%PDF-1.4 fake content"
        mock_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(return_value=mock_resp)

        with patch("tools.document_extraction_tools.httpx.Client", return_value=mock_client):
            with patch("tools.document_extraction_tools.pymupdf4llm.to_markdown", return_value="# Lab Report\nHemoglobin: 14.2 g/dL"):
                result = extract_document_text.invoke({"file_url": "https://example.com/report.pdf"})

        assert "Extracted Document" in result
        assert "Hemoglobin" in result

    def test_corrupt_pdf_cleans_up_temp_file(self):
        """Verify temp file is deleted even when pymupdf4llm raises on a corrupt PDF."""
        import os
        from tools.document_extraction_tools import extract_document_text

        mock_resp = MagicMock()
        mock_resp.headers = {"content-type": "application/pdf"}
        mock_resp.content = b"%PDF-corrupt"
        mock_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(return_value=mock_resp)

        created_paths = []
        real_to_markdown = None

        def tracking_to_markdown(path):
            created_paths.append(path)
            raise RuntimeError("Corrupt PDF")

        with patch("tools.document_extraction_tools.httpx.Client", return_value=mock_client):
            with patch("tools.document_extraction_tools.pymupdf4llm.to_markdown", side_effect=tracking_to_markdown):
                result = extract_document_text.invoke({"file_url": "https://example.com/corrupt.pdf"})

        assert "Error" in result
        for path in created_paths:
            assert not os.path.exists(path), f"Temp file was not cleaned up: {path}"

    def test_timeout(self):
        from tools.document_extraction_tools import extract_document_text
        import httpx as _httpx

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(side_effect=_httpx.TimeoutException("timeout"))

        with patch("tools.document_extraction_tools.httpx.Client", return_value=mock_client):
            result = extract_document_text.invoke({"file_url": "https://example.com/slow.pdf"})
        assert "timed out" in result.lower()


class TestExtractImageFindings:
    """Tests for extract_image_findings tool."""

    def test_invalid_url(self):
        from tools.image_extraction_tools import extract_image_findings
        result = extract_image_findings.invoke({"file_url": "not-a-url", "clinical_context": ""})
        assert "Error" in result

    def test_non_image_url(self):
        from tools.image_extraction_tools import extract_image_findings
        mock_resp = MagicMock()
        mock_resp.headers = {"content-type": "application/pdf"}
        mock_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(return_value=mock_resp)

        with patch("tools.image_extraction_tools.httpx.Client", return_value=mock_client):
            result = extract_image_findings.invoke({
                "file_url": "https://example.com/report.pdf",
                "clinical_context": ""
            })
        assert "not point to an image" in result or "PDF" in result

    def test_image_analysis_success(self):
        from tools.image_extraction_tools import extract_image_findings

        # Mock image download
        mock_resp = MagicMock()
        mock_resp.headers = {"content-type": "image/jpeg"}
        mock_resp.content = b"\xff\xd8\xff\xe0"  # JPEG magic bytes
        mock_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(return_value=mock_resp)

        # Mock MedGemma vision response
        mock_api = MagicMock()
        mock_api.json.return_value = {"response": "Normal chest X-ray. No abnormalities detected."}
        mock_api.raise_for_status = MagicMock()

        with patch("tools.image_extraction_tools.httpx.Client", return_value=mock_client):
            with patch("tools.image_extraction_tools.requests.post", return_value=mock_api):
                result = extract_image_findings.invoke({
                    "file_url": "https://example.com/xray.jpg",
                    "clinical_context": "chest X-ray"
                })

        assert "Medical Image Analysis" in result
        assert "Normal chest X-ray" in result

    def test_medgemma_offline(self):
        from tools.image_extraction_tools import extract_image_findings
        import requests

        mock_resp = MagicMock()
        mock_resp.headers = {"content-type": "image/png"}
        mock_resp.content = b"\x89PNG"
        mock_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(return_value=mock_resp)

        with patch("tools.image_extraction_tools.httpx.Client", return_value=mock_client):
            with patch("tools.image_extraction_tools.requests.post",
                       side_effect=requests.exceptions.ConnectionError("offline")):
                result = extract_image_findings.invoke({
                    "file_url": "https://example.com/mri.png",
                    "clinical_context": ""
                })

        assert "unavailable" in result.lower() or "Error" in result


class TestAnalyzeReport:
    """Tests for analyze_report tool."""

    def test_lab_report_analysis(self):
        from tools.report_analysis_tools import analyze_report
        with patch("tools.report_analysis_tools.llm") as mock_llm:
            mock_llm.invoke.return_value = "**Report Summary**: CBC panel\n**Abnormalities**: Low hemoglobin"
            result = analyze_report.invoke({
                "extracted_content": "Hemoglobin: 10.2 g/dL (L), WBC: 7.5",
                "report_type": "lab_report"
            })
        assert isinstance(result, str)

    def test_empty_content(self):
        from tools.report_analysis_tools import analyze_report
        result = analyze_report.invoke({
            "extracted_content": "",
            "report_type": "general"
        })
        assert "Error" in result

    def test_imaging_report_type(self):
        from tools.report_analysis_tools import analyze_report
        with patch("tools.report_analysis_tools.llm") as mock_llm:
            mock_llm.invoke.return_value = "CT findings: normal anatomy"
            result = analyze_report.invoke({
                "extracted_content": "CT scan shows normal lung fields.",
                "report_type": "imaging"
            })
        assert isinstance(result, str)
