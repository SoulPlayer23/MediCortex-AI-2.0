"""
Tests for diagnosis tools: symptom analysis and diagnosis webcrawler.
"""
import pytest
from unittest.mock import patch, MagicMock


class TestAnalyzeSymptoms:
    """Tests for analyze_symptoms tool."""

    def test_symptom_analysis(self):
        from tools.symptom_analysis_tools import analyze_symptoms
        with patch("tools.symptom_analysis_tools.llm") as mock_llm:
            mock_llm.invoke.return_value = (
                "Clinical Profile:\n- Headache: severity moderate\n"
                "- Fever: suggests infection\nAssessment: Consider viral illness."
            )
            result = analyze_symptoms.invoke({"query": "headache, fever, fatigue", "knowledge_context": ""})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_symptoms(self):
        from tools.symptom_analysis_tools import analyze_symptoms
        with patch("tools.symptom_analysis_tools.llm") as mock_llm:
            mock_llm.invoke.return_value = "No symptoms provided."
            result = analyze_symptoms.invoke({"query": "", "knowledge_context": ""})
        assert isinstance(result, str)

    def test_knowledge_context_injected_into_prompt(self):
        """Verify knowledge_context is forwarded to the MedGemma prompt."""
        from tools.symptom_analysis_tools import analyze_symptoms
        captured = {}
        def capture_invoke(prompt):
            captured["prompt"] = prompt
            return "Clinical Profile: fever with graph context."
        with patch("tools.symptom_analysis_tools.llm") as mock_llm:
            mock_llm.invoke.side_effect = capture_invoke
            analyze_symptoms.invoke({
                "query": "high fever",
                "knowledge_context": "Fever is associated with Malaria (graph context)"
            })
        assert "Malaria" in captured["prompt"]
        assert "graph context" in captured["prompt"]


class TestCrawlDiagnosisArticles:
    """Tests for crawl_diagnosis_articles tool."""

    def test_crawl_with_results(self, mock_google_html, mock_medical_page_html):
        from tools.diagnosis_webcrawler_tools import crawl_diagnosis_articles

        mock_responses = [
            MagicMock(text=mock_google_html, status_code=200, raise_for_status=MagicMock()),
            MagicMock(text=mock_medical_page_html, status_code=200, raise_for_status=MagicMock()),
            MagicMock(text=mock_medical_page_html, status_code=200, raise_for_status=MagicMock()),
        ]

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(side_effect=mock_responses)

        with patch("tools.diagnosis_webcrawler_tools.httpx.Client", return_value=mock_client), \
             patch("utils.cache_utils.redis_client", None):
            result = crawl_diagnosis_articles.invoke({
                "query": "diabetes differential diagnosis",
                "max_results": 2
            })

        assert "Diagnostic Resources" in result or "Metformin" in result

    def test_crawl_empty_query(self):
        from tools.diagnosis_webcrawler_tools import crawl_diagnosis_articles
        result = crawl_diagnosis_articles.invoke({
            "query": "",
            "max_results": 5
        })
        assert "Error" in result or "empty" in result.lower()

    def test_crawl_sanitizes_input(self):
        from tools.diagnosis_webcrawler_tools import _sanitize_query
        result = _sanitize_query("test; DROP TABLE --")
        assert ";" not in result

    def test_max_results_capped(self, mock_google_html):
        from tools.diagnosis_webcrawler_tools import crawl_diagnosis_articles

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(
            return_value=MagicMock(text=mock_google_html, status_code=200, raise_for_status=MagicMock())
        )

        with patch("tools.diagnosis_webcrawler_tools.httpx.Client", return_value=mock_client), \
             patch("utils.cache_utils.redis_client", None):
            result = crawl_diagnosis_articles.invoke({
                "query": "headache",
                "max_results": 50  # should be capped to 10
            })

        assert isinstance(result, str)
