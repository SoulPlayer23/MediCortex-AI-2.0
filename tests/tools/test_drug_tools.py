"""
Tests for drug interaction and recommendation tools with mocked HTTP calls.
"""
import pytest
from unittest.mock import patch, MagicMock
import httpx


class TestCheckDrugInteractions:
    """Tests for check_drug_interactions tool."""

    @pytest.fixture
    def mock_http_responses(self, mock_google_html, mock_medical_page_html):
        """Mock httpx.Client to return fake Google results and page content."""
        mock_responses = [
            # First call: Google search
            MagicMock(text=mock_google_html, status_code=200, raise_for_status=MagicMock()),
            # Second call: Page content fetch
            MagicMock(text=mock_medical_page_html, status_code=200, raise_for_status=MagicMock()),
            # Third call: Page content fetch
            MagicMock(text=mock_medical_page_html, status_code=200, raise_for_status=MagicMock()),
        ]
        return mock_responses

    def test_interaction_check_basic(self, mock_http_responses):
        from tools.drug_interaction_tools import check_drug_interactions

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(side_effect=mock_http_responses)

        with patch("tools.drug_interaction_tools.httpx.Client", return_value=mock_client):
            result = check_drug_interactions.invoke({
                "medications": "Metformin, Lisinopril",
                "patient_conditions": ""
            })

        assert "Metformin" in result
        assert "drugs.com" in result.lower() or "Drug Interaction Check" in result

    def test_empty_medications(self):
        from tools.drug_interaction_tools import check_drug_interactions
        result = check_drug_interactions.invoke({
            "medications": "",
            "patient_conditions": ""
        })
        assert "Error" in result or "No medications" in result

    def test_with_conditions(self, mock_http_responses):
        from tools.drug_interaction_tools import check_drug_interactions

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(side_effect=mock_http_responses)

        with patch("tools.drug_interaction_tools.httpx.Client", return_value=mock_client):
            result = check_drug_interactions.invoke({
                "medications": "Ibuprofen",
                "patient_conditions": "kidney disease"
            })

        assert isinstance(result, str)

    def test_timeout_handling(self):
        from tools.drug_interaction_tools import check_drug_interactions

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(side_effect=Exception("Timeout"))

        with patch("tools.drug_interaction_tools.httpx.Client", return_value=mock_client):
            result = check_drug_interactions.invoke({
                "medications": "Aspirin",
                "patient_conditions": ""
            })

        assert "Error" in result


class TestRecommendDrugs:
    """Tests for recommend_drugs tool."""

    @pytest.fixture
    def mock_http(self, mock_google_html, mock_medical_page_html):
        mock_responses = [
            MagicMock(text=mock_google_html, status_code=200, raise_for_status=MagicMock()),
            MagicMock(text=mock_medical_page_html, status_code=200, raise_for_status=MagicMock()),
        ]
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(side_effect=mock_responses)
        return mock_client

    def test_recommendation_query(self, mock_http):
        from tools.drug_recommendation_tools import recommend_drugs
        with patch("tools.drug_recommendation_tools.httpx.Client", return_value=mock_http):
            result = recommend_drugs.invoke({
                "condition": "Type 2 Diabetes",
                "query_type": "recommendation",
                "patient_info": ""
            })
        assert isinstance(result, str)
        assert "Drug Query" in result or "Metformin" in result

    def test_dosage_query(self, mock_http):
        from tools.drug_recommendation_tools import recommend_drugs
        with patch("tools.drug_recommendation_tools.httpx.Client", return_value=mock_http):
            result = recommend_drugs.invoke({
                "condition": "Amoxicillin",
                "query_type": "dosage",
                "patient_info": ""
            })
        assert isinstance(result, str)

    def test_alternatives_query(self, mock_http):
        from tools.drug_recommendation_tools import recommend_drugs
        with patch("tools.drug_recommendation_tools.httpx.Client", return_value=mock_http):
            result = recommend_drugs.invoke({
                "condition": "Atorvastatin",
                "query_type": "alternatives",
                "patient_info": ""
            })
        assert isinstance(result, str)

    def test_empty_condition(self):
        from tools.drug_recommendation_tools import recommend_drugs
        result = recommend_drugs.invoke({
            "condition": "",
            "query_type": "recommendation",
            "patient_info": ""
        })
        assert "Error" in result
