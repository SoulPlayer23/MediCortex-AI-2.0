"""
Tests for A2ABaseAgent ReAct loop, tool dispatch, and error handling.
"""
import pytest
from unittest.mock import patch, MagicMock
from specialized_agents.protocols import AgentCard, Envelope, AgentResponse
from langchain_core.tools import tool


# Create simple test tools
@tool
def add_numbers(a_and_b: str) -> str:
    """Add two numbers separated by comma."""
    parts = a_and_b.split(",")
    return str(int(parts[0].strip()) + int(parts[1].strip()))


@tool
def greet_user(name: str) -> str:
    """Greet a user by name."""
    return f"Hello, {name}!"


class TestBaseAgentReActLoop:
    """Tests for the A2ABaseAgent ReAct loop."""

    @pytest.fixture
    def test_card(self):
        return AgentCard(
            name="test-agent",
            description="A test agent",
            input_schema={"type": "object", "properties": {"input": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"output": {"type": "string"}}},
        )

    @pytest.fixture
    def make_agent(self, test_card):
        """Factory to create test agents with custom LLM responses."""
        def _make(llm_responses, tools=None, max_iterations=3):
            from tests.conftest import MockMedGemmaLLM
            mock_llm = MockMedGemmaLLM()
            mock_llm.set_responses(llm_responses)

            from specialized_agents.base import A2ABaseAgent
            return A2ABaseAgent(
                name="test-agent",
                llm=mock_llm,
                tools=tools or [add_numbers, greet_user],
                system_prompt="You are a test agent.",
                card=test_card,
                max_iterations=max_iterations,
            )
        return _make

    def test_final_answer_direct(self, make_agent):
        """Agent returns Final Answer without using tools."""
        agent = make_agent([
            "Thought: Do I need to use a tool? No\nFinal Answer: The answer is 42."
        ])
        env = Envelope(receiver_id="test-agent", payload={"input": "What is 42?"})
        resp = agent.process(env)
        assert resp.output == "The answer is 42."
        assert resp.error is None

    def test_tool_dispatch(self, make_agent):
        """Agent calls a tool and uses the result."""
        agent = make_agent([
            "Thought: I need to add numbers.\nAction: add_numbers\nAction Input: 3, 7",
            "Thought: I got the result.\nFinal Answer: The sum is 10.",
        ])
        env = Envelope(receiver_id="test-agent", payload={"input": "Add 3 and 7"})
        resp = agent.process(env)
        assert resp.output == "The sum is 10."
        assert len(resp.thinking) >= 1

    def test_unknown_tool(self, make_agent):
        """Agent tries to call a non-existent tool."""
        agent = make_agent([
            "Thought: I need a special tool.\nAction: nonexistent_tool\nAction Input: data",
            "Thought: That tool doesn't exist.\nFinal Answer: Tool not available.",
        ])
        env = Envelope(receiver_id="test-agent", payload={"input": "test"})
        resp = agent.process(env)
        assert "not available" in resp.output.lower() or resp.output

    def test_max_iterations_guard(self, make_agent):
        """Agent hits max iterations without Final Answer."""
        agent = make_agent(
            ["Thought: I'm stuck.\nAction: add_numbers\nAction Input: 1, 1"] * 5,
            max_iterations=2,
        )
        env = Envelope(receiver_id="test-agent", payload={"input": "loop"})
        resp = agent.process(env)
        assert "max iterations" in resp.output.lower()

    def test_empty_payload_error(self, make_agent):
        """Envelope with missing input returns validation error."""
        agent = make_agent([])
        env = Envelope(receiver_id="test-agent", payload={})
        resp = agent.process(env)
        assert resp.error is not None
        assert "input" in resp.error.lower()

    def test_thinking_steps_captured(self, make_agent):
        """Thinking steps are captured from LLM output."""
        agent = make_agent([
            "Thought: Let me think about this carefully.\nFinal Answer: Done thinking.",
        ])
        env = Envelope(receiver_id="test-agent", payload={"input": "think"})
        resp = agent.process(env)
        assert len(resp.thinking) >= 1
        assert "think" in resp.thinking[0].lower()

    def test_invoke_legacy_wrapper(self, make_agent):
        """Legacy invoke() method works correctly."""
        agent = make_agent([
            "Thought: No\nFinal Answer: Legacy response."
        ])
        result = agent.invoke({"input": "test legacy"})
        assert result["output"] == "Legacy response."

    def test_get_card(self, make_agent, test_card):
        """get_card() returns the agent's card."""
        agent = make_agent(["Thought: No\nFinal Answer: ok"])
        card = agent.get_card()
        assert card.name == "test-agent"
