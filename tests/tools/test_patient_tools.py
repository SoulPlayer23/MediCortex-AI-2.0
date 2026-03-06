"""
Tests for patient retriever and history analyzer tools.
"""
import json
import pytest
from unittest.mock import patch, MagicMock


class TestRetrievePatientRecords:
    """Tests for retrieve_patient_records tool."""

    def test_retrieval_with_valid_pii_mapping(self):
        from tools.patient_retriever_tools import retrieve_patient_records
        mapping = json.dumps({"<PERSON_1>": "John Smith"})
        result = retrieve_patient_records.invoke({
            "redacted_identifier": "<PERSON_1>",
            "pii_mapping_json": mapping,
        })
        # Should return de-identified data without the real name
        assert "John Smith" not in result
        assert "<PERSON_1>" in result or "PAT-" in result

    def test_retrieval_with_direct_name(self):
        from tools.patient_retriever_tools import retrieve_patient_records
        result = retrieve_patient_records.invoke({
            "redacted_identifier": "Jane Doe",
            "pii_mapping_json": "{}",
        })
        # Should find or not find Jane Doe in simulated DB
        assert isinstance(result, str)
        assert len(result) > 0

    def test_patient_not_found(self):
        from tools.patient_retriever_tools import retrieve_patient_records
        result = retrieve_patient_records.invoke({
            "redacted_identifier": "Nonexistent Person",
            "pii_mapping_json": "{}",
        })
        assert "not found" in result.lower() or "no patient" in result.lower()

    def test_empty_identifier(self):
        from tools.patient_retriever_tools import retrieve_patient_records
        result = retrieve_patient_records.invoke({
            "redacted_identifier": "",
            "pii_mapping_json": "{}",
        })
        assert isinstance(result, str)

    def test_invalid_pii_json(self):
        from tools.patient_retriever_tools import retrieve_patient_records
        result = retrieve_patient_records.invoke({
            "redacted_identifier": "<PERSON_1>",
            "pii_mapping_json": "not valid json{{{",
        })
        # Should handle gracefully, not crash
        assert isinstance(result, str)

    def test_re_redaction(self):
        """Output should contain placeholders, not real names."""
        from tools.patient_retriever_tools import retrieve_patient_records
        mapping = json.dumps({"<PERSON_1>": "John Smith"})
        result = retrieve_patient_records.invoke({
            "redacted_identifier": "<PERSON_1>",
            "pii_mapping_json": mapping,
        })
        # Real name should NOT appear in output
        assert "John Smith" not in result


class TestAnalyzePatientHistory:
    """Tests for analyze_patient_history tool."""

    def test_analysis_with_record(self):
        from tools.patient_history_analyzer_tools import analyze_patient_history
        record = """
        Patient: <PERSON_1>, Age: 45, Sex: Male
        Diagnoses: Type 2 Diabetes (Active), Hypertension (Active)
        Medications: Metformin 500mg, Lisinopril 10mg
        """
        with patch("tools.patient_history_analyzer_tools.llm") as mock_llm:
            mock_llm.invoke.return_value = "Clinical analysis: Patient has well-controlled diabetes."
            result = analyze_patient_history.invoke({"patient_record": record})
        assert isinstance(result, str)

    def test_analysis_empty_record(self):
        from tools.patient_history_analyzer_tools import analyze_patient_history
        result = analyze_patient_history.invoke({"patient_record": ""})
        assert "Error" in result or "No" in result
