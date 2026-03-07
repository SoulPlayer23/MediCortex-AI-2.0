"""
Tests for the Model-as-Judge node_reviewer (A2A §5.2).
"""
import json
import pytest
from unittest.mock import patch, MagicMock


class TestNodeReviewer:
    """Tests for the node_reviewer function."""

    @pytest.fixture
    def good_state(self):
        return {
            "final_output": "# Diabetes Management\nMetformin 500mg twice daily is recommended for Type 2 Diabetes.",
            "redacted_input": "What is the treatment for Type 2 Diabetes?",
            "pii_mapping": {},
        }

    @pytest.fixture
    def poor_state(self):
        return {
            "final_output": "I'm not sure about this medication.",
            "redacted_input": "What drug treats hypertension?",
            "pii_mapping": {},
        }

    @pytest.fixture
    def pii_leaked_state(self):
        return {
            "final_output": "Patient <PERSON_1> should take aspirin daily.",
            "redacted_input": "What should my patient take?",
            "pii_mapping": {"<PERSON_1>": "John Smith"},
        }

    def _mock_groq(self, score: int, reason: str = "Test reason"):
        """Helper to mock ChatGroq returning a given score."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content=json.dumps({"score": score, "reason": reason})
        )
        return mock_llm

    def test_high_score_no_disclaimer(self, good_state):
        """Score >= 3 — no disclaimer appended, judge_score set."""
        from orchestrator import node_reviewer

        with patch("orchestrator.ChatGroq", return_value=self._mock_groq(5)):
            with patch("orchestrator.settings") as mock_settings:
                mock_settings.JUDGE_ENABLED = True
                mock_settings.JUDGE_SAMPLE_RATE = 1.0
                mock_settings.GROQ_API_KEY = "test-key"
                mock_settings.JUDGE_MODEL = "llama-3.3-70b-versatile"
                mock_settings.JUDGE_FALLBACK_MODEL = "llama-3.1-8b-instant"
                mock_settings.JUDGE_MAX_INPUT_TOKENS = 500
                result = node_reviewer(good_state)

        assert result["judge_score"] == 5
        assert "Clinical Disclaimer" not in good_state["final_output"]

    def test_low_score_appends_disclaimer(self, poor_state):
        """Score < 3 — disclaimer appended, judge_score set."""
        from orchestrator import node_reviewer

        with patch("orchestrator.ChatGroq", return_value=self._mock_groq(2, "Response lacks evidence")):
            with patch("orchestrator.settings") as mock_settings:
                mock_settings.JUDGE_ENABLED = True
                mock_settings.JUDGE_SAMPLE_RATE = 1.0
                mock_settings.GROQ_API_KEY = "test-key"
                mock_settings.JUDGE_MODEL = "llama-3.3-70b-versatile"
                mock_settings.JUDGE_FALLBACK_MODEL = "llama-3.1-8b-instant"
                mock_settings.JUDGE_MAX_INPUT_TOKENS = 500
                result = node_reviewer(poor_state)

        assert result["judge_score"] == 2
        assert "Clinical Disclaimer" in result["final_output"]
        assert "2/5" in result["final_output"]
        assert "Response lacks evidence" in result["final_output"]

    def test_skipped_when_disabled(self, good_state):
        """JUDGE_ENABLED=False — node skips immediately."""
        from orchestrator import node_reviewer
        with patch("orchestrator.settings") as mock_settings:
            mock_settings.JUDGE_ENABLED = False
            mock_settings.JUDGE_SAMPLE_RATE = 1.0
            mock_settings.GROQ_API_KEY = "test-key"
            result = node_reviewer(good_state)
        assert result["judge_score"] is None

    def test_skipped_no_api_key(self, good_state):
        """Empty GROQ_API_KEY — node skips."""
        from orchestrator import node_reviewer
        with patch("orchestrator.settings") as mock_settings:
            mock_settings.JUDGE_ENABLED = True
            mock_settings.JUDGE_SAMPLE_RATE = 1.0
            mock_settings.GROQ_API_KEY = ""
            result = node_reviewer(good_state)
        assert result["judge_score"] is None

    def test_fallback_model_used_on_primary_failure(self, good_state):
        """Primary model fails → fallback model succeeds."""
        from orchestrator import node_reviewer

        call_count = {"n": 0}
        def groq_factory(**kwargs):
            mock = MagicMock()
            call_count["n"] += 1
            if call_count["n"] == 1:
                mock.invoke.side_effect = Exception("Rate limit hit")
            else:
                mock.invoke.return_value = MagicMock(
                    content=json.dumps({"score": 4, "reason": "Fallback worked"})
                )
            return mock

        with patch("orchestrator.ChatGroq", side_effect=groq_factory):
            with patch("orchestrator.settings") as mock_settings:
                mock_settings.JUDGE_ENABLED = True
                mock_settings.JUDGE_SAMPLE_RATE = 1.0
                mock_settings.GROQ_API_KEY = "test-key"
                mock_settings.JUDGE_MODEL = "llama-3.3-70b-versatile"
                mock_settings.JUDGE_FALLBACK_MODEL = "llama-3.1-8b-instant"
                mock_settings.JUDGE_MAX_INPUT_TOKENS = 500
                result = node_reviewer(good_state)

        assert result["judge_score"] == 4

    def test_all_models_fail_returns_none(self, good_state):
        """Both models fail → judge_score is None (fail open)."""
        from orchestrator import node_reviewer

        mock = MagicMock()
        mock.invoke.side_effect = Exception("All failed")

        with patch("orchestrator.ChatGroq", return_value=mock):
            with patch("orchestrator.settings") as mock_settings:
                mock_settings.JUDGE_ENABLED = True
                mock_settings.JUDGE_SAMPLE_RATE = 1.0
                mock_settings.GROQ_API_KEY = "test-key"
                mock_settings.JUDGE_MODEL = "llama-3.3-70b-versatile"
                mock_settings.JUDGE_FALLBACK_MODEL = "llama-3.1-8b-instant"
                mock_settings.JUDGE_MAX_INPUT_TOKENS = 500
                result = node_reviewer(good_state)

        assert result["judge_score"] is None

    def test_truncation_applied_to_long_response(self, good_state):
        """Very long final_output is truncated before sending to Groq."""
        from orchestrator import node_reviewer

        long_state = {
            "final_output": "A" * 10000,  # ~2500 tokens
            "redacted_input": "test query",
            "pii_mapping": {},
        }

        captured_prompt = {}
        def capture_invoke(messages):
            captured_prompt["content"] = messages[0].content
            return MagicMock(content=json.dumps({"score": 4, "reason": "ok"}))

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = capture_invoke

        with patch("orchestrator.ChatGroq", return_value=mock_llm):
            with patch("orchestrator.settings") as mock_settings:
                mock_settings.JUDGE_ENABLED = True
                mock_settings.JUDGE_SAMPLE_RATE = 1.0
                mock_settings.GROQ_API_KEY = "test-key"
                mock_settings.JUDGE_MODEL = "llama-3.3-70b-versatile"
                mock_settings.JUDGE_FALLBACK_MODEL = "llama-3.1-8b-instant"
                mock_settings.JUDGE_MAX_INPUT_TOKENS = 500
                node_reviewer(long_state)

        # Prompt should contain truncation marker
        assert "[truncated for evaluation]" in captured_prompt["content"]
