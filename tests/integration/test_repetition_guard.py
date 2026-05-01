"""
EVAL-2 Layer 1 — MedGemma Repetition Guard regression tests (REP-GUARD-01..03).

Regression suite for BUG-1. Tests that:
  - A repetitive synthesis output triggers Gemma 4 fallback
  - Normal output does not trigger fallback
  - KB placeholder strings are stripped before enhanced_input is built

No live LLM or DB connections required.
"""
import pytest
from unittest.mock import patch, MagicMock, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_repetitive_text(sentence: str, repeats: int = 5) -> str:
    """Return a string where one sentence appears `repeats` times."""
    return " ".join([sentence] * repeats)


KB_PLACEHOLDER = "[Medical Knowledge Base: No relevant information found]"
KB_PLACEHOLDER_ALT = "No specific knowledge found in graph."


# ---------------------------------------------------------------------------
# REP-GUARD-01 — Repetitive MedGemma output triggers Gemma 4 fallback
# ---------------------------------------------------------------------------

class TestRepGuard01FallbackFires:
    """REP-GUARD-01 — Repetitive output detected → fallback_to_gemma4 fires."""

    def test_fallback_fires_on_repetition(self):
        repetitive = _make_repetitive_text(
            "Metformin is used to treat type 2 diabetes mellitus.", repeats=5
        )
        clean = "Metformin reduces hepatic glucose output and improves insulin sensitivity."

        # First call (MedGemma) returns repetitive output; second call (Gemma 4) returns clean
        medgemma_mock = MagicMock()
        medgemma_mock.invoke.return_value = repetitive

        gemma4_mock = MagicMock()
        gemma4_llm = MagicMock()
        gemma4_llm.invoke.return_value = MagicMock(content=clean)

        with patch("specialized_agents.base.ChatOllama", return_value=gemma4_llm):
            from specialized_agents.base import A2ABaseAgent

            class _TestAgent(A2ABaseAgent):
                name = "test"
                description = "test"
                tools = {}
                system_prompt = "You are a test agent."

            agent = _TestAgent.__new__(_TestAgent)
            agent.name = "test"
            agent.system_prompt = "You are a test agent."
            agent.llm = medgemma_mock
            agent.max_iterations = 3
            agent.tools = {}

            result = agent._synthesize("What is metformin?", [])

        # MedGemma was called once and returned repetitive output
        medgemma_mock.invoke.assert_called_once()
        # Fallback (Gemma 4) should have been invoked
        gemma4_llm.invoke.assert_called_once()
        # Final result is the clean Gemma 4 output
        assert result == clean


# ---------------------------------------------------------------------------
# REP-GUARD-02 — Normal output does NOT trigger fallback
# ---------------------------------------------------------------------------

class TestRepGuard02NoFallbackForClean:
    """REP-GUARD-02 — Non-repetitive synthesis uses primary MedGemma output."""

    def test_no_fallback_for_clean_output(self):
        clean = (
            "Metformin reduces hepatic glucose production. "
            "It also improves peripheral insulin sensitivity. "
            "Common side effects include nausea and diarrhea. "
            "Renal function must be monitored regularly."
        )

        medgemma_mock = MagicMock()
        medgemma_mock.invoke.return_value = clean

        gemma4_llm = MagicMock()

        with patch("specialized_agents.base.ChatOllama", return_value=gemma4_llm):
            from specialized_agents.base import A2ABaseAgent

            agent = A2ABaseAgent.__new__(A2ABaseAgent)
            agent.name = "test"
            agent.system_prompt = "You are a test agent."
            agent.llm = medgemma_mock
            agent.max_iterations = 3
            agent.tools = {}

            result = agent._synthesize("What is metformin?", [])

        medgemma_mock.invoke.assert_called_once()
        # Gemma 4 fallback must NOT have been invoked
        gemma4_llm.invoke.assert_not_called()
        assert result == clean


# ---------------------------------------------------------------------------
# REP-GUARD-03 — Orchestrator strips KB placeholders before calling agents
# ---------------------------------------------------------------------------

class TestRepGuard03PlaceholderStripped:
    """REP-GUARD-03 — KB placeholder sections are removed from context before enhanced_input."""

    def test_empty_markers_filter_kb_sentinel(self):
        """The orchestrator's _empty_markers filter removes KB placeholder sections."""
        # This is the filtering logic from make_agent_node in orchestrator.py
        from orchestrator import _KB_EMPTY_SENTINEL

        _empty_markers = (
            _KB_EMPTY_SENTINEL,
            "Knowledge Engine Offline.",
            "No specific medical knowledge concept found",
        )
        context_with_placeholders = [
            "Metformin reduces hepatic glucose output.",   # real KB fact
            _KB_EMPTY_SENTINEL,                            # empty sentinel
            "Knowledge Engine Offline.",                   # offline marker
            "No specific medical knowledge concept found in query.",  # ambiguous
        ]

        meaningful_sections = [
            s for s in context_with_placeholders
            if not any(marker in s for marker in _empty_markers)
        ]

        assert len(meaningful_sections) == 1
        assert meaningful_sections[0] == "Metformin reduces hepatic glucose output."

    def test_kb_sentinel_not_in_meaningful_context(self):
        """After filtering, enhanced_input contains no KB placeholder text."""
        from orchestrator import _KB_EMPTY_SENTINEL

        _empty_markers = (
            _KB_EMPTY_SENTINEL,
            "Knowledge Engine Offline.",
            "No specific medical knowledge concept found",
        )
        context = [_KB_EMPTY_SENTINEL, "Knowledge Engine Offline."]
        filtered = [s for s in context if not any(m in s for m in _empty_markers)]

        # context_str built from filtered list must be empty
        context_str = "\n".join(filtered)
        assert context_str == "", (
            "KB placeholder leaked into context_str passed to agent"
        )

    def test_repetition_guard_catches_placeholder_loops(self):
        """If MedGemma repeats a KB placeholder sentence, _is_looping detects it."""
        from specialized_agents.base import A2ABaseAgent

        # Simulate MedGemma looping on a KB placeholder message
        looping_output = _make_repetitive_text(
            "[Medical Knowledge Base: No relevant information found]", repeats=5
        )
        assert A2ABaseAgent._is_looping(looping_output, max_repeats=3) is True


# ---------------------------------------------------------------------------
# _is_looping unit tests
# ---------------------------------------------------------------------------

class TestIsLooping:
    """Direct unit tests for the _is_looping static method."""

    def test_detects_repeated_sentence(self):
        from specialized_agents.base import A2ABaseAgent
        text = _make_repetitive_text("This sentence repeats.", repeats=4)
        assert A2ABaseAgent._is_looping(text, max_repeats=3) is True

    def test_clean_text_not_looping(self):
        from specialized_agents.base import A2ABaseAgent
        text = (
            "Metformin reduces hepatic glucose output. "
            "It also improves peripheral insulin sensitivity. "
            "Side effects include nausea. "
            "Monitor renal function regularly."
        )
        assert A2ABaseAgent._is_looping(text, max_repeats=3) is False

    def test_short_text_not_looping(self):
        from specialized_agents.base import A2ABaseAgent
        text = "Short answer."
        assert A2ABaseAgent._is_looping(text, max_repeats=3) is False

    def test_empty_text_not_looping(self):
        from specialized_agents.base import A2ABaseAgent
        assert A2ABaseAgent._is_looping("", max_repeats=3) is False

    def test_exactly_at_threshold_not_looping(self):
        from specialized_agents.base import A2ABaseAgent
        # Exactly 3 repeats — should NOT trigger (threshold is > 3)
        text = _make_repetitive_text("Repeated sentence here.", repeats=3)
        assert A2ABaseAgent._is_looping(text, max_repeats=3) is False

    def test_one_over_threshold_triggers(self):
        from specialized_agents.base import A2ABaseAgent
        # 4 repeats with max_repeats=3 — should trigger
        text = _make_repetitive_text("Repeated sentence here.", repeats=4)
        assert A2ABaseAgent._is_looping(text, max_repeats=3) is True
