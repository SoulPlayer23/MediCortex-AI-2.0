"""
Tests for orchestrator routing logic.
Uses direct function mocking to avoid heavy module initialization.
"""
import pytest
import json
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage


class TestRouteDecision:
    """Tests for route_decision function — lightweight, no LLM needed."""

    def test_valid_routes(self):
        from orchestrator import route_decision
        state = {"messages": [AIMessage(content="['pubmed', 'diagnosis']")]}
        routes = route_decision(state)
        assert "pubmed" in routes
        assert "diagnosis" in routes

    def test_caps_agents(self):
        from orchestrator import route_decision, MAX_CONCURRENT_AGENTS
        state = {
            "messages": [AIMessage(content="['pubmed', 'diagnosis', 'drug', 'patient', 'report']")]
        }
        routes = route_decision(state)
        assert len(routes) <= MAX_CONCURRENT_AGENTS

    def test_unknown_agents_filtered(self):
        from orchestrator import route_decision
        state = {"messages": [AIMessage(content="['fake_agent', 'pubmed']")]}
        routes = route_decision(state)
        assert "fake_agent" not in routes
        assert "pubmed" in routes

    def test_malformed_input_defaults(self):
        from orchestrator import route_decision
        state = {"messages": [AIMessage(content="not valid json")]}
        routes = route_decision(state)
        # Should default to diagnosis
        assert routes == ["diagnosis"]

    def test_single_agent(self):
        from orchestrator import route_decision
        state = {"messages": [AIMessage(content="['drug']")]}
        routes = route_decision(state)
        assert routes == ["drug"]


class TestNodeRouter:
    """Tests for node_router — requires mocking the LLM."""

    def test_drug_query_routes_to_drug(self):
        from orchestrator import node_router
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content='["drug"]')

        with patch("orchestrator.llm", mock_llm):
            state = {
                "redacted_input": "Check interactions between Metformin and Lisinopril",
                "messages": [],
            }
            result = node_router(state)
        assert "drug" in result["messages"][0].content

    def test_fallback_on_error(self):
        from orchestrator import node_router
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM failed")

        with patch("orchestrator.llm", mock_llm):
            state = {"redacted_input": "anything", "messages": []}
            result = node_router(state)
        # Should fallback to diagnosis
        assert "diagnosis" in result["messages"][0].content
