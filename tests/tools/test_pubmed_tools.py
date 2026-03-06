"""
Tests for PubMed search and medical webcrawler tools.
"""
import pytest
from unittest.mock import patch, MagicMock


# Sample PubMed XML response
MOCK_PUBMED_SEARCH_XML = """<?xml version="1.0"?>
<eSearchResult>
    <Count>2</Count>
    <IdList>
        <Id>12345678</Id>
        <Id>87654321</Id>
    </IdList>
</eSearchResult>"""

MOCK_PUBMED_FETCH_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
    <PubmedArticle>
        <MedlineCitation>
            <PMID>12345678</PMID>
            <Article>
                <ArticleTitle>Metformin and Type 2 Diabetes: A Review</ArticleTitle>
                <Abstract><AbstractText>Metformin is first-line therapy for T2DM.</AbstractText></Abstract>
                <Journal><Title>Medical Journal</Title></Journal>
                <AuthorList>
                    <Author><LastName>Smith</LastName><Initials>J</Initials></Author>
                </AuthorList>
                <ArticleDate><Year>2023</Year></ArticleDate>
            </Article>
        </MedlineCitation>
        <PubmedData>
            <ArticleIdList>
                <ArticleId IdType="doi">10.1234/test</ArticleId>
            </ArticleIdList>
        </PubmedData>
    </PubmedArticle>
</PubmedArticleSet>"""


class TestSearchPubmed:
    """Tests for search_pubmed tool."""

    def test_search_returns_results(self):
        from tools.pubmed_search_tools import search_pubmed

        mock_responses = [
            MagicMock(text=MOCK_PUBMED_SEARCH_XML, status_code=200, raise_for_status=MagicMock()),
            MagicMock(text=MOCK_PUBMED_FETCH_XML, status_code=200, raise_for_status=MagicMock()),
        ]

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(side_effect=mock_responses)

        with patch("tools.pubmed_search_tools.httpx.Client", return_value=mock_client):
            result = search_pubmed.invoke({
                "query": "metformin diabetes",
                "max_results": 2
            })

        assert "Metformin" in result
        assert "Smith" in result or "Medical Journal" in result

    def test_search_empty_query(self):
        from tools.pubmed_search_tools import search_pubmed
        result = search_pubmed.invoke({"query": "", "max_results": 5})
        assert isinstance(result, str)
        # Tool returns results or error; either is valid for empty query


class TestCrawlMedicalArticles:
    """Tests for crawl_medical_articles tool."""

    def test_crawl_returns_results(self, mock_google_html, mock_medical_page_html):
        from tools.medical_webcrawler_tools import crawl_medical_articles

        mock_responses = [
            MagicMock(text=mock_google_html, status_code=200, raise_for_status=MagicMock()),
            MagicMock(text=mock_medical_page_html, status_code=200, raise_for_status=MagicMock()),
            MagicMock(text=mock_medical_page_html, status_code=200, raise_for_status=MagicMock()),
        ]

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(side_effect=mock_responses)

        with patch("tools.medical_webcrawler_tools.httpx.Client", return_value=mock_client):
            result = crawl_medical_articles.invoke({
                "query": "diabetes treatment",
                "max_results": 2
            })

        assert isinstance(result, str)
        assert len(result) > 0
