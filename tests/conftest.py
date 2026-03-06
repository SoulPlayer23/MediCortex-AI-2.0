"""
MediCortex Test Suite — Shared Fixtures

Provides mock LLM, mock MedGemma, mock httpx, sample envelopes, and
test data for all test categories.
"""

import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

# Ensure project root is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Mock MedGemma LLM ───────────────────────────────────────────────
class MockMedGemmaLLM:
    """Deterministic LLM replacement for testing."""

    def __init__(self, default_response: str = ""):
        self._default = default_response
        self._responses: list[str] = []

    def set_responses(self, responses: list[str]):
        """Queue deterministic responses for sequential calls."""
        self._responses = list(responses)

    def invoke(self, prompt: str, **kwargs) -> str:
        if self._responses:
            return self._responses.pop(0)
        return self._default

    def _call(self, prompt, **kwargs):
        return self.invoke(prompt)

    @property
    def _llm_type(self):
        return "mock_medgemma"


@pytest.fixture
def mock_medgemma():
    """Provide a MockMedGemmaLLM instance."""
    return MockMedGemmaLLM(default_response="Thought: Do I need to use a tool? No\nFinal Answer: Mock medical response.")


@pytest.fixture
def mock_medgemma_patch(mock_medgemma):
    """Patch MedGemmaLLM globally so agents use the mock."""
    with patch("specialized_agents.medgemma_llm.MedGemmaLLM", return_value=mock_medgemma):
        with patch("specialized_agents.base.MedGemmaLLM", return_value=mock_medgemma):
            yield mock_medgemma


# ── Sample A2A Protocol Objects ──────────────────────────────────────
@pytest.fixture
def sample_envelope():
    """Pre-built Envelope for agent tests."""
    from specialized_agents.protocols import Envelope
    return Envelope(
        trace_id="test-trace-001",
        sender_id="test-orchestrator",
        receiver_id="test-agent",
        payload={"input": "What are the symptoms of Type 2 Diabetes?"}
    )


@pytest.fixture
def sample_envelope_empty():
    """Envelope with missing input for error testing."""
    from specialized_agents.protocols import Envelope
    return Envelope(
        sender_id="test-orchestrator",
        receiver_id="test-agent",
        payload={}
    )


# ── Sample Patient Data ─────────────────────────────────────────────
@pytest.fixture
def sample_patient_record():
    """De-identified patient record for tool tests."""
    return {
        "patient_id": "PAT-001",
        "age": 45,
        "sex": "Male",
        "blood_type": "O+",
        "allergies": ["Penicillin"],
        "diagnoses": [
            {"condition": "Type 2 Diabetes", "status": "Active", "diagnosed": "2020-03-15"},
            {"condition": "Hypertension", "status": "Active", "diagnosed": "2019-07-22"}
        ],
        "medications": [
            {"name": "Metformin", "dosage": "500mg", "frequency": "Twice daily"},
            {"name": "Lisinopril", "dosage": "10mg", "frequency": "Once daily"}
        ],
        "vitals": {"bp": "130/85", "hr": 78, "temp": "98.6°F", "spo2": "97%"}
    }


@pytest.fixture
def sample_pii_mapping():
    """PII mapping for HIPAA tests."""
    return {
        "<PERSON_1>": "John Smith",
        "<PHONE_NUMBER_1>": "555-123-4567",
        "<EMAIL_ADDRESS_1>": "john.smith@email.com"
    }


# ── Mock Google Search HTML ──────────────────────────────────────────
@pytest.fixture
def mock_google_html():
    """Fake Google search results HTML for webcrawler tests."""
    return """
    <html><body>
    <div class="g">
        <a href="https://www.drugs.com/metformin.html"><h3>Metformin - Drugs.com</h3></a>
        <div class="VwiC3b">Metformin is used to treat type 2 diabetes.</div>
    </div>
    <div class="g">
        <a href="https://www.mayoclinic.org/drugs/metformin"><h3>Metformin - Mayo Clinic</h3></a>
        <div class="VwiC3b">Learn about Metformin dosage and side effects.</div>
    </div>
    </body></html>
    """


# ── Mock Medical Page HTML ───────────────────────────────────────────
@pytest.fixture
def mock_medical_page_html():
    """Fake medical article page for content extraction tests."""
    return """
    <html><body>
    <article>
        <h1>Metformin Drug Information</h1>
        <p>Metformin is a biguanide that decreases hepatic glucose production.</p>
        <div id="interactions">
            <h2>Drug Interactions</h2>
            <p>Metformin may interact with contrast dyes and alcohol.</p>
        </div>
    </article>
    </body></html>
    """


# ── Pytest Configuration ─────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch):
    """Set environment variables for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///test.db")
