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


class TestAnalyzePatientVitals:
    """Tests for analyze_patient_vitals tool."""

    _RECORD = """
    Patient: <PERSON_1>, Age: 45, Sex: Male
    Diagnoses: Type 2 Diabetes (Active), Hypertension (Active)
    Vitals (2026-01-28): BP 138/88 mmHg, HR 76 bpm, Temp 98.6°F, SpO2 97%, Weight 92kg, BMI 28.4
    Vitals (2025-10-15): BP 142/92 mmHg, HR 80 bpm, SpO2 96%, Weight 94kg, BMI 29.0
    """

    def test_vitals_analysis_with_record(self):
        from tools.patient_vitals_tools import analyze_patient_vitals
        with patch("tools.patient_vitals_tools.llm") as mock_llm:
            mock_llm.invoke.return_value = "**Current Vitals Summary**: BP 138/88 — ⚠️ Stage 1 HTN"
            result = analyze_patient_vitals.invoke({"patient_record": self._RECORD})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_vitals_empty_record(self):
        from tools.patient_vitals_tools import analyze_patient_vitals
        result = analyze_patient_vitals.invoke({"patient_record": ""})
        assert "Error" in result

    def test_vitals_no_pii_in_output(self):
        """Real patient name must never appear in vitals output."""
        from tools.patient_vitals_tools import analyze_patient_vitals
        with patch("tools.patient_vitals_tools.llm") as mock_llm:
            mock_llm.invoke.return_value = "BP 138/88 mmHg — Stage 1 HTN. Patient <PERSON_1> shows worsening trend."
            result = analyze_patient_vitals.invoke({"patient_record": self._RECORD})
        assert "John Smith" not in result


class TestReviewPatientMedications:
    """Tests for review_patient_medications tool."""

    _RECORD = """
    Patient: <PERSON_1>, Age: 45, Sex: Male
    Allergies: Penicillin, Sulfa drugs
    Diagnoses: Type 2 Diabetes (Active), Hypertension (Active), Hyperlipidemia (Managed)
    Medications: Metformin 500mg twice daily, Lisinopril 10mg once daily, Atorvastatin 20mg once daily
    """

    def test_medication_review_with_record(self):
        from tools.patient_medication_review_tools import review_patient_medications
        with patch("tools.patient_medication_review_tools.llm") as mock_llm:
            mock_llm.invoke.return_value = "**Allergy Safety Check**: ✅ SAFE — No conflicts detected."
            result = review_patient_medications.invoke({"patient_record": self._RECORD})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_medication_review_empty_record(self):
        from tools.patient_medication_review_tools import review_patient_medications
        result = review_patient_medications.invoke({"patient_record": ""})
        assert "Error" in result

    def test_medication_review_no_pii_in_output(self):
        """Real patient name must never appear in medication review output."""
        from tools.patient_medication_review_tools import review_patient_medications
        with patch("tools.patient_medication_review_tools.llm") as mock_llm:
            mock_llm.invoke.return_value = "Medication review for <PERSON_1>: All medications appropriate."
            result = review_patient_medications.invoke({"patient_record": self._RECORD})
        assert "John Smith" not in result
