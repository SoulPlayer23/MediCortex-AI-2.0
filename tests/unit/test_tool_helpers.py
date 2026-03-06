"""
Unit tests for helper functions in tool modules
(sanitizers, parsers, formatters — pure logic, no I/O).
"""
import pytest


class TestSanitizeQuery:
    """Tests for _sanitize_query from webcrawler tools."""

    def _sanitize(self, query: str) -> str:
        from tools.drug_interaction_tools import _sanitize_query
        return _sanitize_query(query)

    def test_normal_query(self):
        assert self._sanitize("Metformin interactions") == "Metformin interactions"

    def test_strips_injection_chars(self):
        result = self._sanitize("test; DROP TABLE; --")
        assert ";" not in result
        assert "DROP TABLE" in result  # text is kept, only special chars removed

    def test_strips_shell_chars(self):
        result = self._sanitize("test | cat /etc/passwd & rm -rf")
        assert "|" not in result
        assert "&" not in result

    def test_length_limit(self):
        long_query = "a" * 1000
        result = self._sanitize(long_query)
        assert len(result) <= 500

    def test_empty_query(self):
        assert self._sanitize("") == ""
        assert self._sanitize("   ") == ""


class TestBuildSiteFilter:
    """Tests for _build_site_filter from webcrawler tools."""

    def test_drug_site_filter(self):
        from tools.drug_interaction_tools import _build_site_filter
        result = _build_site_filter()
        assert "site:drugs.com" in result
        assert " OR " in result

    def test_diagnosis_site_filter(self):
        from tools.diagnosis_webcrawler_tools import _build_site_filter
        result = _build_site_filter()
        assert "site:mayoclinic.org" in result


class TestParseGoogleResults:
    """Tests for _parse_google_results from webcrawler tools."""

    def test_parses_results(self, mock_google_html):
        from tools.drug_interaction_tools import _parse_google_results
        results = _parse_google_results(mock_google_html, max_results=5)
        assert len(results) == 2
        assert results[0]["title"] == "Metformin - Drugs.com"
        assert "drugs.com" in results[0]["domain"]

    def test_respects_max_results(self, mock_google_html):
        from tools.drug_interaction_tools import _parse_google_results
        results = _parse_google_results(mock_google_html, max_results=1)
        assert len(results) == 1

    def test_empty_html(self):
        from tools.drug_interaction_tools import _parse_google_results
        results = _parse_google_results("<html><body></body></html>", max_results=5)
        assert results == []


class TestExtractContent:
    """Tests for content extraction from medical pages."""

    def test_extracts_article_content(self, mock_medical_page_html):
        from tools.drug_interaction_tools import _extract_interaction_content
        content = _extract_interaction_content(mock_medical_page_html)
        assert "Metformin" in content
        assert len(content) > 0

    def test_respects_max_chars(self, mock_medical_page_html):
        from tools.drug_interaction_tools import _extract_interaction_content
        content = _extract_interaction_content(mock_medical_page_html, max_chars=50)
        assert len(content) <= 55  # +5 for "..." suffix

    def test_empty_html(self):
        from tools.drug_interaction_tools import _extract_interaction_content
        content = _extract_interaction_content("<html></html>")
        assert content == ""


class TestFormatResults:
    """Tests for _format_results."""

    def test_formats_results(self):
        from tools.drug_interaction_tools import _format_results
        results = [
            {"title": "Drug A Info", "url": "https://drugs.com/a", "domain": "drugs.com", "snippet": "Info about Drug A"},
        ]
        output = _format_results(results, "Drug A interactions")
        assert "Drug A Info" in output
        assert "drugs.com" in output

    def test_empty_results(self):
        from tools.drug_interaction_tools import _format_results
        output = _format_results([], "missing drug")
        assert "No interaction information found" in output
