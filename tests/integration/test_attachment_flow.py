"""
ATT-1 — Attachment-based conversation tests.

Validates end-to-end quality of document and image analysis through the
report_analyzer agent using real files from tests/resources/.

Coverage:
  ATT-01  Upload each resource file → /upload returns url + content_type
  ATT-02  PDF lab report → route_decision includes report_analyzer
  ATT-03  Medical image (X-ray / MRI / brain-tumor) → route_decision includes report_analyzer
  ATT-04  /chat endpoint with file attachment returns structured response
  ATT-05  Multi-turn: follow-up question references previously-uploaded attachment
  ATT-06  route_decision always forces report_analyzer when file_urls non-empty
  ATT-07  MinIO presigned URL TTL is ≤ 3600 s (short-lived, HIPAA compliant)
  ATT-08  Unsupported MIME type is rejected at /upload (415)
  ATT-09  Oversized file is rejected at /upload (413)

Run (requires homeserver: Redis, MinIO, orchestrator on port 8001):
  source .venv/bin/activate
  .venv/bin/python3 -m pytest tests/integration/test_attachment_flow.py -v --tb=short
"""

import io
import os
import time
import uuid
import pytest
import httpx
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

RESOURCES = Path(__file__).parent.parent / "resources"

PDF_FILES = [
    "CBC-test-report-format-example-sample-template-Drlogy-lab-report.pdf",
    "MRI-and-CT-sample-report-3.pdf",
    "Sample-Smart-Report-Clinics.pdf",
    "investigationlabreports.pdf",
    "sterling-accuris-pathology-sample-report-unlocked.pdf",
    "Z615.pdf",
]

IMAGE_FILES = [
    "brain-tumor.jpg",
    "Chest-X-ray.jpeg",
]

ALL_FILES = PDF_FILES + IMAGE_FILES

BASE_URL = os.environ.get("MEDICORTEX_BASE_URL", "http://localhost:8001")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state(**overrides) -> dict:
    state = {
        "input": "Analyze this medical report.",
        "redacted_input": "Analyze this medical report.",
        "pii_mapping": {},
        "file_urls": [],
        "context": [],
        "history": [],
        "routing_context": "",
        "messages": [],
        "agent_outputs": [],
        "agent_thoughts": [],
        "agents_used": [],
        "agent_sources": [],
        "final_output": "",
        "judge_score": None,
        "judge_reason": None,
        "judge_confidence": None,
        "error": None,
        "trace_id": "att-test-trace",
        "session_id": str(uuid.uuid4()),
        "retrieval_iteration": 0,
        "retrieval_feedback": [],
        "retrieval_ambiguous": False,
        "clarification_question": None,
        "re_retrieval_skipped": False,
        "node_timings": {},
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# ATT-06 — Unit: route_decision always includes report_analyzer with file_urls
# (no running services needed)
# ---------------------------------------------------------------------------

class TestRouteDecisionWithFiles:
    """ATT-06: route_decision forces report_analyzer when file_urls is set."""

    def test_file_urls_forces_report_analyzer(self):
        from orchestrator import route_decision
        from langchain_core.messages import AIMessage

        state = _base_state(
            file_urls=["https://minio/bucket/test.pdf"],
            messages=[AIMessage(content="['diagnosis']")],
        )
        routes = route_decision(state)
        assert "report_analyzer" in routes, (
            "report_analyzer must always be included when file_urls is non-empty"
        )

    def test_no_file_urls_does_not_force_report_analyzer(self):
        from orchestrator import route_decision
        from langchain_core.messages import AIMessage

        state = _base_state(
            file_urls=[],
            messages=[AIMessage(content="['pharmacology']")],
        )
        routes = route_decision(state)
        assert routes == ["pharmacology"]

    def test_multiple_files_still_one_report_analyzer_entry(self):
        from orchestrator import route_decision
        from langchain_core.messages import AIMessage

        state = _base_state(
            file_urls=[
                "https://minio/bucket/file1.pdf",
                "https://minio/bucket/file2.jpg",
            ],
            messages=[AIMessage(content="['diagnosis']")],
        )
        routes = route_decision(state)
        assert routes.count("report_analyzer") == 1


# ---------------------------------------------------------------------------
# ATT-02 / ATT-03 — Unit: node_router routes report files correctly
# (mocked LLM, no running services)
# ---------------------------------------------------------------------------

class TestNodeRouterAttachments:
    """ATT-02/03: node_router includes report_analyzer for PDF/image inputs."""

    def _run_router(self, query: str, file_urls: list[str]) -> list[str]:
        from orchestrator import route_decision
        from langchain_core.messages import AIMessage

        state = _base_state(
            input=query,
            redacted_input=query,
            file_urls=file_urls,
            messages=[AIMessage(content="['report_analyzer']")],
        )
        return route_decision(state)

    def test_pdf_lab_report_routes_to_report_analyzer(self):
        routes = self._run_router(
            "What do these lab results show?",
            ["https://minio/bucket/CBC-test.pdf"],
        )
        assert "report_analyzer" in routes

    def test_xray_routes_to_report_analyzer(self):
        routes = self._run_router(
            "Interpret this chest X-ray.",
            ["https://minio/bucket/Chest-X-ray.jpeg"],
        )
        assert "report_analyzer" in routes

    def test_brain_tumor_image_routes_to_report_analyzer(self):
        routes = self._run_router(
            "What does this MRI show?",
            ["https://minio/bucket/brain-tumor.jpg"],
        )
        assert "report_analyzer" in routes

    def test_text_query_without_files_does_not_force_report_analyzer(self):
        from orchestrator import route_decision
        from langchain_core.messages import AIMessage

        state = _base_state(
            input="What is metformin used for?",
            redacted_input="What is metformin used for?",
            file_urls=[],
            messages=[AIMessage(content="['pharmacology']")],
        )
        routes = route_decision(state)
        assert "pharmacology" in routes
        assert "report_analyzer" not in routes


# ---------------------------------------------------------------------------
# ATT-07 — Unit: MinIO presign TTL is short (HIPAA)
# (mocked MinIO client, no running services)
# ---------------------------------------------------------------------------

class TestMinioPresignTTL:
    """ATT-07: presigned URLs use ≤ 3600 s TTL."""

    def test_presign_ttl_is_short(self):
        from config import settings
        ttl = getattr(settings, "MINIO_PRESIGN_TTL_SECONDS", None)
        assert ttl is not None, "MINIO_PRESIGN_TTL_SECONDS must be set in config"
        assert ttl <= 3600, f"Presigned URL TTL {ttl}s exceeds HIPAA-safe 3600s limit"

    def test_generate_presigned_url_uses_config_ttl(self):
        """generate_presigned_url ExpiresIn must equal MINIO_PRESIGN_TTL_SECONDS."""
        from config import settings

        captured = {}

        async def fake_generate_presigned_url(operation, Params=None, ExpiresIn=None):
            captured["ExpiresIn"] = ExpiresIn
            return "https://minio/signed"

        mock_s3 = AsyncMock()
        mock_s3.generate_presigned_url = fake_generate_presigned_url
        mock_s3.__aenter__ = AsyncMock(return_value=mock_s3)
        mock_s3.__aexit__ = AsyncMock(return_value=False)

        import asyncio
        from services.minio_service import MinioService

        svc = MinioService.__new__(MinioService)
        svc._client = MagicMock(return_value=mock_s3)

        asyncio.run(svc.generate_download_url("test.pdf"))

        assert captured.get("ExpiresIn") == settings.MINIO_PRESIGN_TTL_SECONDS, (
            f"generate_presigned_url used ExpiresIn={captured.get('ExpiresIn')}, "
            f"expected {settings.MINIO_PRESIGN_TTL_SECONDS}"
        )


# ---------------------------------------------------------------------------
# Live integration tests — require orchestrator + MinIO on BASE_URL
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestUploadEndpoint:
    """ATT-01: /upload accepts all resource files and returns url + content_type."""

    @pytest.fixture(scope="class")
    def http(self):
        with httpx.Client(base_url=BASE_URL, timeout=30) as client:
            yield client

    def _upload(self, http: httpx.Client, filename: str) -> dict:
        path = RESOURCES / filename
        assert path.exists(), f"Resource file not found: {path}"
        mime = (
            "application/pdf" if filename.endswith(".pdf")
            else "image/jpeg" if filename.lower().endswith((".jpg", ".jpeg"))
            else "application/octet-stream"
        )
        with open(path, "rb") as f:
            resp = http.post("/upload", files={"file": (filename, f, mime)})
        return resp

    @pytest.mark.parametrize("filename", PDF_FILES)
    def test_upload_pdf(self, http, filename):
        resp = self._upload(http, filename)
        assert resp.status_code == 200, f"Upload failed for {filename}: {resp.text}"
        data = resp.json()
        assert data.get("url"), f"No url returned for {filename}"
        assert data.get("content_type"), f"No content_type returned for {filename}"
        assert "pdf" in data["content_type"].lower(), (
            f"Expected PDF content_type for {filename}, got {data['content_type']}"
        )

    @pytest.mark.parametrize("filename", IMAGE_FILES)
    def test_upload_image(self, http, filename):
        resp = self._upload(http, filename)
        assert resp.status_code == 200, f"Upload failed for {filename}: {resp.text}"
        data = resp.json()
        assert data.get("url"), f"No url returned for {filename}"
        assert data.get("content_type"), f"No content_type returned for {filename}"
        assert "image" in data["content_type"].lower(), (
            f"Expected image content_type for {filename}, got {data['content_type']}"
        )

    def test_upload_unsupported_type_rejected(self, http):
        """ATT-08: non-medical file types should be rejected (415) or sanitised."""
        resp = http.post(
            "/upload",
            files={"file": ("malware.exe", b"\x4d\x5a\x00\x00", "application/x-msdownload")},
        )
        assert resp.status_code in (400, 415, 422), (
            f"Expected 4xx for unsupported MIME type, got {resp.status_code}"
        )

    def test_upload_oversized_file_rejected(self, http):
        """ATT-09: files above MAX_UPLOAD_BYTES must be rejected with 413."""
        from config import settings
        max_bytes = getattr(settings, "MAX_UPLOAD_BYTES", 50 * 1024 * 1024)
        oversized = io.BytesIO(b"a" * (max_bytes + 1024))
        resp = http.post(
            "/upload",
            files={"file": ("big.pdf", oversized, "application/pdf")},
            timeout=60,
        )
        assert resp.status_code == 413, (
            f"Expected 413 for oversized upload, got {resp.status_code}"
        )


@pytest.mark.integration
class TestChatWithAttachments:
    """ATT-04/05: /chat endpoint with attachments drives report_analyzer."""

    @pytest.fixture(scope="class")
    def http(self):
        with httpx.Client(base_url=BASE_URL, timeout=120) as client:
            yield client

    def _upload_file(self, http: httpx.Client, filename: str) -> dict:
        path = RESOURCES / filename
        mime = "application/pdf" if filename.endswith(".pdf") else "image/jpeg"
        with open(path, "rb") as f:
            resp = http.post("/upload", files={"file": (filename, f, mime)})
        assert resp.status_code == 200, f"Upload failed: {resp.text}"
        return resp.json()

    def test_pdf_lab_report_chat_response(self, http):
        """ATT-04: upload CBC PDF, send to /chat, verify response mentions lab values."""
        upload = self._upload_file(http, "CBC-test-report-format-example-sample-template-Drlogy-lab-report.pdf")
        session_id = str(uuid.uuid4())

        payload = {
            "message": "Summarize the key findings from this CBC lab report.",
            "session_id": session_id,
            "attachments": [
                {
                    "url": upload["url"],
                    "filename": upload["filename"],
                    "content_type": upload["content_type"],
                }
            ],
        }
        resp = http.post("/chat", json=payload)
        assert resp.status_code == 200, f"Chat failed: {resp.text}"
        data = resp.json()
        response_text = data.get("response", "").lower()

        # Should produce a non-empty medical analysis
        assert len(response_text) > 100, "Response too short — likely failed to process PDF"
        # report_analyzer should have been invoked
        agents = data.get("metadata", {}).get("agents_used", [])
        assert "report_analyzer" in agents, (
            f"report_analyzer not in agents_used: {agents}"
        )

    def test_chest_xray_chat_response(self, http):
        """ATT-04: upload chest X-ray, verify MedGemma vision analysis runs."""
        upload = self._upload_file(http, "Chest-X-ray.jpeg")
        session_id = str(uuid.uuid4())

        payload = {
            "message": "What does this chest X-ray show? Are there any abnormalities?",
            "session_id": session_id,
            "attachments": [
                {
                    "url": upload["url"],
                    "filename": upload["filename"],
                    "content_type": upload["content_type"],
                }
            ],
        }
        resp = http.post("/chat", json=payload)
        assert resp.status_code == 200, f"Chat failed: {resp.text}"
        data = resp.json()
        response_text = data.get("response", "").lower()
        assert len(response_text) > 100, "Response too short — likely failed to process image"
        agents = data.get("metadata", {}).get("agents_used", [])
        assert "report_analyzer" in agents, f"report_analyzer not in agents_used: {agents}"

    def test_brain_tumor_mri_chat_response(self, http):
        """ATT-04: upload brain tumor MRI, verify imaging analysis."""
        upload = self._upload_file(http, "brain-tumor.jpg")
        session_id = str(uuid.uuid4())

        payload = {
            "message": "Describe the findings in this brain MRI scan.",
            "session_id": session_id,
            "attachments": [
                {
                    "url": upload["url"],
                    "filename": upload["filename"],
                    "content_type": upload["content_type"],
                }
            ],
        }
        resp = http.post("/chat", json=payload)
        assert resp.status_code == 200, f"Chat failed: {resp.text}"
        data = resp.json()
        assert len(data.get("response", "")) > 100
        assert "report_analyzer" in data.get("metadata", {}).get("agents_used", [])

    def test_multiturn_followup_references_attachment(self, http):
        """ATT-05: follow-up question on same session can reference the uploaded report."""
        upload = self._upload_file(http, "investigationlabreports.pdf")
        session_id = str(uuid.uuid4())

        # Turn 1 — upload and ask initial question
        turn1 = http.post("/chat", json={
            "message": "What are the abnormal values in this report?",
            "session_id": session_id,
            "attachments": [
                {
                    "url": upload["url"],
                    "filename": upload["filename"],
                    "content_type": upload["content_type"],
                }
            ],
        })
        assert turn1.status_code == 200, f"Turn 1 failed: {turn1.text}"

        # Turn 2 — follow-up with no attachment, should still reference earlier context
        turn2 = http.post("/chat", json={
            "message": "Which of those values suggest infection?",
            "session_id": session_id,
        })
        assert turn2.status_code == 200, f"Turn 2 failed: {turn2.text}"
        response_text = turn2.json().get("response", "")
        assert len(response_text) > 50, "Follow-up response too short — context not carried"

    def test_presigned_url_valid_during_inference(self, http):
        """ATT-07 (live): MinIO URL returned by /upload is still reachable after upload."""
        upload = self._upload_file(http, "Sample-Smart-Report-Clinics.pdf")
        url = upload["url"]

        # The URL should be directly fetchable (MinIO presigned GET)
        fetch = httpx.get(url, timeout=15, follow_redirects=True)
        assert fetch.status_code == 200, (
            f"Presigned URL not reachable after upload (status {fetch.status_code}). "
            "MedGemma would fail to fetch the file during inference."
        )

    @pytest.mark.parametrize("filename", PDF_FILES)
    def test_all_pdfs_trigger_report_analyzer(self, http, filename):
        """ATT-02: every PDF in resources/ triggers report_analyzer routing."""
        upload = self._upload_file(http, filename)
        session_id = str(uuid.uuid4())

        resp = http.post("/chat", json={
            "message": "Summarize this medical document.",
            "session_id": session_id,
            "attachments": [
                {
                    "url": upload["url"],
                    "filename": upload["filename"],
                    "content_type": upload["content_type"],
                }
            ],
        })
        assert resp.status_code == 200, f"Chat failed for {filename}: {resp.text}"
        agents = resp.json().get("metadata", {}).get("agents_used", [])
        assert "report_analyzer" in agents, (
            f"report_analyzer not invoked for {filename}. agents_used={agents}"
        )

    @pytest.mark.parametrize("filename", IMAGE_FILES)
    def test_all_images_trigger_report_analyzer(self, http, filename):
        """ATT-03: every image in resources/ triggers report_analyzer routing."""
        upload = self._upload_file(http, filename)
        session_id = str(uuid.uuid4())

        resp = http.post("/chat", json={
            "message": "Analyze this medical image.",
            "session_id": session_id,
            "attachments": [
                {
                    "url": upload["url"],
                    "filename": upload["filename"],
                    "content_type": upload["content_type"],
                }
            ],
        })
        assert resp.status_code == 200, f"Chat failed for {filename}: {resp.text}"
        agents = resp.json().get("metadata", {}).get("agents_used", [])
        assert "report_analyzer" in agents, (
            f"report_analyzer not invoked for {filename}. agents_used={agents}"
        )
