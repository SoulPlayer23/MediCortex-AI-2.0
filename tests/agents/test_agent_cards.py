"""
Tests that all 5 specialized agent cards are valid and well-formed.
"""
import pytest


class TestAgentCards:
    """Validate all agent cards have required fields and proper versions."""

    @pytest.fixture
    def all_agents(self):
        """Import all agents from the registry."""
        from specialized_agents.agents import AGENT_REGISTRY
        return AGENT_REGISTRY

    def test_all_agents_registered(self, all_agents):
        """All 5 expected agents are in the registry."""
        expected = {"pubmed", "diagnosis", "report_analyzer", "patient", "pharmacology"}
        assert expected == set(all_agents.keys())

    def test_cards_have_name(self, all_agents):
        for key, agent in all_agents.items():
            card = agent.get_card()
            assert card.name, f"Agent '{key}' card has no name"

    def test_cards_have_description(self, all_agents):
        for key, agent in all_agents.items():
            card = agent.get_card()
            assert len(card.description) > 10, f"Agent '{key}' card description is too short"

    def test_cards_have_schemas(self, all_agents):
        for key, agent in all_agents.items():
            card = agent.get_card()
            assert "type" in card.input_schema, f"Agent '{key}' missing input_schema type"
            assert "type" in card.output_schema, f"Agent '{key}' missing output_schema type"

    def test_cards_have_capabilities(self, all_agents):
        for key, agent in all_agents.items():
            card = agent.get_card()
            assert len(card.capabilities) > 0, f"Agent '{key}' has no capabilities"

    def test_cards_version_format(self, all_agents):
        """All cards should have semver-like version strings."""
        for key, agent in all_agents.items():
            card = agent.get_card()
            parts = card.version.split(".")
            assert len(parts) == 3, f"Agent '{key}' version '{card.version}' is not semver"

    def test_agents_have_tools(self, all_agents):
        """Every agent should have at least one tool."""
        for key, agent in all_agents.items():
            assert len(agent.tools) > 0, f"Agent '{key}' has no tools"

    def test_agents_have_system_prompt(self, all_agents):
        """Every agent should have a non-trivial system prompt."""
        for key, agent in all_agents.items():
            assert len(agent.system_prompt) > 50, f"Agent '{key}' system prompt is too short"
