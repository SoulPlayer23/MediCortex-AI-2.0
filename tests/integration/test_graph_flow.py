"""
Tests for LangGraph node functions (privacy, aggregator, agent node factory).
"""
import pytest
from unittest.mock import patch, MagicMock


class TestNodeAnalyzePrivacy:
    """Tests for the privacy analysis node."""

    def test_redacts_pii(self):
        from orchestrator import node_analyze_privacy

        state = {
            "input": "Patient John Smith has diabetes.",
            "redacted_input": "",
            "pii_mapping": {},
        }

        with patch("orchestrator.privacy_manager") as mock_pm:
            mock_pm.redact_pii.return_value = (
                "Patient <PERSON_1> has diabetes.",
                {"<PERSON_1>": "John Smith"}
            )
            result = node_analyze_privacy(state)

        assert "<PERSON_1>" in result["redacted_input"]
        assert result["pii_mapping"]["<PERSON_1>"] == "John Smith"


class TestNodeRestorePrivacy:
    """Tests for the privacy restoration node."""

    def test_restores_names(self):
        from orchestrator import node_restore_privacy

        state = {
            "final_output": "Patient <PERSON_1> should take Metformin.",
            "pii_mapping": {"<PERSON_1>": "Alice"},
        }

        with patch("orchestrator.privacy_manager") as mock_pm:
            mock_pm.restore_privacy.return_value = "Patient Alice should take Metformin."
            result = node_restore_privacy(state)

        assert "Alice" in result["final_output"]


class TestNodeAggregator:
    """Tests for the aggregator node."""

    def test_combines_outputs(self):
        from orchestrator import node_aggregator

        state = {
            "agent_outputs": [
                "## Diagnosis Agent Response\nProbable flu.",
                "## Drug Agent Response\nNo interactions found."
            ]
        }

        with patch("orchestrator.llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(
                content="# Combined Report\n- Diagnosis: Flu\n- Drugs: Safe"
            )
            result = node_aggregator(state)

        assert "final_output" in result
        assert len(result["final_output"]) > 0

    def test_aggregator_fallback(self):
        """If formatting LLM fails, raw outputs are used."""
        from orchestrator import node_aggregator

        state = {
            "agent_outputs": ["Raw output 1", "Raw output 2"]
        }

        with patch("orchestrator.llm") as mock_llm:
            mock_llm.invoke.side_effect = Exception("LLM failed")
            result = node_aggregator(state)

        assert "Raw output" in result["final_output"]


class TestMakeAgentNode:
    """Tests for make_agent_node factory."""

    def test_unknown_agent_returns_error(self):
        from orchestrator import make_agent_node
        node = make_agent_node("nonexistent_agent")
        result = node({
            "redacted_input": "test",
            "context": [],
            "history": [],
            "trace_id": "t-001",
            "pii_mapping": {},
        })
        assert "Error" in result["agent_outputs"][0]

    def test_patient_node_injects_pii(self):
        """Patient agent node should inject pii_mapping into envelope."""
        from orchestrator import make_agent_node

        mock_agent = MagicMock()
        mock_agent.process.return_value = MagicMock(
            output="Patient data retrieved",
            thinking=["step 1"],
            error=None,
        )

        with patch("orchestrator.AGENT_REGISTRY", {"patient": mock_agent}):
            node = make_agent_node("patient")
            result = node({
                "redacted_input": "Get records for <PERSON_1>",
                "context": [],
                "history": [],
                "trace_id": "t-002",
                "pii_mapping": {"<PERSON_1>": "Jane Doe"},
            })

        mock_agent.process.assert_called_once()
        call_args = mock_agent.process.call_args
        envelope = call_args[0][0]
        assert "pii_mapping_json" in envelope.payload
