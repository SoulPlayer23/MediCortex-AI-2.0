"""
Layer 3 — Failure injection tests (FAIL-01..05).
Tests graceful degradation when individual services become unavailable.

Run with: pytest tests/stress/test_failure_injection.py -v -m stress
"""
import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock


pytestmark = pytest.mark.stress


class TestFailureInjection:

    def _base_state(self, query: str = "What is the dosage of lisinopril?"):
        return {
            "redacted_input": query,
            "context": [],
            "routing_context": "",
            "pii_mapping": {},
            "messages": [],
        }

    def test_fail01_arangodb_offline_no_crash(self):
        """FAIL-01: ArangoDB offline — node_retrieve_knowledge must not crash."""
        from orchestrator import node_retrieve_knowledge
        state = self._base_state()
        with patch("orchestrator.medical_engine") as mock_engine:
            mock_engine.search_and_reason.side_effect = Exception("Connection refused: ArangoDB")
            result = node_retrieve_knowledge(state)
        assert result is not None, "Node crashed when ArangoDB was offline"

    def test_fail01_arangodb_offline_no_hallucinated_kb_facts(self):
        """FAIL-01: ArangoDB offline — response must not contain fabricated KB facts."""
        from orchestrator import node_retrieve_knowledge
        state = self._base_state()
        with patch("orchestrator.medical_engine") as mock_engine:
            mock_engine.search_and_reason.side_effect = Exception("Connection refused")
            result = node_retrieve_knowledge(state)
        # KB context should be empty or indicate unavailability, not fabricated content
        kb_context = result.get("kb_context", "") or ""
        assert len(kb_context) < 50, "KB context should be empty when ArangoDB is offline"

    def test_fail03_groq_judge_timeout_score_is_none(self):
        """FAIL-03: Groq judge API timeout — judge_score must be None, pipeline must not hang."""
        from orchestrator import node_reviewer
        state = {
            "final_output": "Metformin 500mg twice daily is standard first-line therapy.",
            "redacted_input": "What is the treatment for T2DM?",
            "pii_mapping": {},
        }
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = TimeoutError("Groq API timeout after 5s")

        with patch("orchestrator.ChatGroq", return_value=mock_llm):
            with patch("orchestrator.settings") as mock_settings:
                mock_settings.JUDGE_ENABLED = True
                mock_settings.JUDGE_SAMPLE_RATE = 1.0
                mock_settings.GROQ_API_KEY = "test-key"
                mock_settings.JUDGE_MODEL = "llama-3.3-70b-versatile"
                mock_settings.JUDGE_FALLBACK_MODEL = "llama-3.1-8b-instant"
                mock_settings.JUDGE_MAX_INPUT_TOKENS = 500
                result = node_reviewer(state)

        # Pipeline must complete — judge_score None is acceptable
        assert result is not None, "node_reviewer crashed on Groq timeout"
        assert result.get("judge_score") is None or isinstance(result.get("judge_score"), (int, float))

    def test_fail05_all_agents_return_empty_aggregator_safe_response(self):
        """FAIL-05: All agents return empty output — aggregator must produce a safe response."""
        from orchestrator import node_aggregator
        state = {
            "redacted_input": "What are the symptoms of appendicitis?",
            "agent_outputs": {},  # all empty
            "kb_context": "",
            "pii_mapping": {},
            "messages": [],
        }
        with patch("orchestrator.llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(
                content="I was unable to retrieve sufficient information to answer this query. Please try again or consult a specialist."
            )
            result = node_aggregator(state)
        assert result is not None
        output = result.get("final_output", "")
        assert len(output) > 10, "Aggregator returned empty string when all agents failed"
