"""
Unit tests for the PrivacyManager (PII redaction and restoration).
Tests the PrivacyManager class directly, independent of orchestrator initialization.
"""
import pytest
from unittest.mock import patch, MagicMock


class TestPrivacyManager:
    """Tests for PrivacyManager.redact_pii and restore_privacy."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Import PrivacyManager with mocked heavy dependencies."""
        # Mock the heavy orchestrator-level imports to avoid LLM/DB init
        with patch.dict("sys.modules", {
            "langchain_openai": MagicMock(),
        }):
            from orchestrator import PrivacyManager
            self.pm = PrivacyManager()

    def test_redact_name(self):
        text = "Patient John Smith has a headache."
        redacted, mapping = self.pm.redact_pii(text)
        assert "John Smith" not in redacted
        assert len(mapping) > 0

    def test_redact_phone_number(self):
        text = "Call me at 555-123-4567."
        redacted, mapping = self.pm.redact_pii(text)
        assert "555-123-4567" not in redacted

    def test_redact_email(self):
        text = "Email: patient@hospital.com"
        redacted, mapping = self.pm.redact_pii(text)
        assert "patient@hospital.com" not in redacted

    def test_no_pii(self):
        text = "What are the symptoms of diabetes?"
        redacted, mapping = self.pm.redact_pii(text)
        # With no PII, text should be unchanged
        assert "diabetes" in redacted

    def test_empty_input(self):
        redacted, mapping = self.pm.redact_pii("")
        assert redacted == ""
        assert mapping == {}

    def test_restore_privacy(self):
        mapping = {"<PERSON_1>": "Jane Doe"}
        text = "Patient <PERSON_1> diagnosed with flu."
        restored = self.pm.restore_privacy(text, mapping)
        assert "Jane Doe" in restored
        assert "<PERSON_1>" not in restored

    def test_restore_multiple_placeholders(self):
        mapping = {
            "<PERSON_1>": "Alice",
            "<PERSON_2>": "Bob",
        }
        text = "<PERSON_1> referred <PERSON_2> to specialist."
        restored = self.pm.restore_privacy(text, mapping)
        assert "Alice" in restored
        assert "Bob" in restored

    def test_restore_empty_mapping(self):
        text = "No placeholders here."
        restored = self.pm.restore_privacy(text, {})
        assert restored == text

    def test_roundtrip(self):
        """Redact then restore should approximately recover original."""
        original = "Patient John Smith has diabetes."
        redacted, mapping = self.pm.redact_pii(original)
        restored = self.pm.restore_privacy(redacted, mapping)
        assert "John Smith" in restored
