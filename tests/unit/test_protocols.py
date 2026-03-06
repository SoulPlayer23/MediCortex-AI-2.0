"""
Unit tests for A2A Protocol models (AgentCard, Envelope, AgentResponse).
"""
import pytest
from datetime import datetime
from specialized_agents.protocols import AgentCard, Envelope, AgentResponse


class TestAgentCard:
    """Tests for AgentCard model validation."""

    def test_card_creation_minimal(self):
        card = AgentCard(
            name="test-agent",
            description="A test agent",
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object", "properties": {}},
        )
        assert card.name == "test-agent"
        assert card.version == "1.0.0"  # default
        assert card.capabilities == []

    def test_card_with_capabilities(self):
        card = AgentCard(
            name="drug",
            description="Drug agent",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            version="2.0.0",
            capabilities=["drug-safety", "interaction-check"],
        )
        assert len(card.capabilities) == 2
        assert "drug-safety" in card.capabilities

    def test_card_requires_name(self):
        with pytest.raises(Exception):
            AgentCard(
                description="Missing name",
                input_schema={},
                output_schema={},
            )

    def test_card_serialization(self):
        card = AgentCard(
            name="test",
            description="desc",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )
        data = card.model_dump()
        assert "name" in data
        assert "version" in data
        assert data["name"] == "test"


class TestEnvelope:
    """Tests for A2A Envelope model."""

    def test_envelope_auto_generates_ids(self):
        env = Envelope(receiver_id="agent-1", payload={"input": "hello"})
        assert env.trace_id  # non-empty UUID
        assert env.idempotency_key
        assert env.trace_id != env.idempotency_key

    def test_envelope_preserves_trace_id(self):
        env = Envelope(
            trace_id="custom-trace",
            receiver_id="agent-1",
            payload={"input": "test"},
        )
        assert env.trace_id == "custom-trace"

    def test_envelope_default_sender(self):
        env = Envelope(receiver_id="agent-1", payload={})
        assert env.sender_id == "orchestrator"

    def test_envelope_timestamp(self):
        env = Envelope(receiver_id="agent-1", payload={})
        assert isinstance(env.timestamp, datetime)

    def test_envelope_payload_access(self):
        env = Envelope(
            receiver_id="test",
            payload={"input": "symptoms: headache, fever"},
        )
        assert env.payload["input"] == "symptoms: headache, fever"


class TestAgentResponse:
    """Tests for AgentResponse model."""

    def test_response_success(self):
        resp = AgentResponse(
            envelope_id="env-001",
            output="Diagnosis complete.",
            thinking=["Step 1: analyzed symptoms"],
        )
        assert resp.output == "Diagnosis complete."
        assert resp.error is None
        assert len(resp.thinking) == 1

    def test_response_with_error(self):
        resp = AgentResponse(
            envelope_id="env-002",
            output=None,
            error="Validation Error: input missing",
        )
        assert resp.output is None
        assert "Validation Error" in resp.error

    def test_response_empty_thinking(self):
        resp = AgentResponse(envelope_id="env-003", output="result")
        assert resp.thinking == []
        assert resp.usage == {}

    def test_response_timestamp(self):
        resp = AgentResponse(envelope_id="env-004", output="ok")
        assert isinstance(resp.timestamp, datetime)
