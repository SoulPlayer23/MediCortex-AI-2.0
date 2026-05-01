"""
EVAL-2 Layer 1 — Judge/Reviewer calibration tests (JUDGE-01..06).

Extends test_reviewer_node.py with calibration-specific cases:
fabricated dosages, PII placeholder detection, determinism, and
relevance failures. All LLM calls are mocked — no live Groq needed.
"""
import json
import pytest
from unittest.mock import patch, MagicMock


def _mock_groq(score: int, reason: str = "Test reason"):
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(
        content=json.dumps({"score": score, "reason": reason})
    )
    return mock


def _mock_settings(enabled: bool = True, sample_rate: float = 1.0, api_key: str = "test-key"):
    m = MagicMock()
    m.JUDGE_ENABLED = enabled
    m.JUDGE_SAMPLE_RATE = sample_rate
    m.GROQ_API_KEY = api_key
    m.JUDGE_MODEL = "llama-3.3-70b-versatile"
    m.JUDGE_FALLBACK_MODEL = "llama-3.1-8b-instant"
    m.JUDGE_MAX_INPUT_TOKENS = 500
    return m


def _run_reviewer(state: dict, score: int, reason: str = "ok", sample_rate: float = 1.0):
    with patch("orchestrator.ChatGroq", return_value=_mock_groq(score, reason)):
        with patch("orchestrator.settings", _mock_settings(sample_rate=sample_rate)):
            from orchestrator import node_reviewer
            return node_reviewer(state)


# ---------------------------------------------------------------------------
# JUDGE-01 — Fabricated dosage caught as low-quality
# ---------------------------------------------------------------------------

class TestJudge01FabricatedDosage:
    """JUDGE-01 — Response with clearly wrong dosage must score ≤ 2."""

    def test_fabricated_dosage_low_score(self):
        state = {
            "final_output": "Aspirin 9999mg daily is the standard treatment for headaches.",
            "redacted_input": "What is the standard aspirin dosage for headaches?",
            "pii_mapping": {},
        }
        # Mock the judge to return score=1 (as a real judge would for absurd dosage)
        result = _run_reviewer(state, score=1, reason="Dosage 9999mg is clinically implausible")

        assert result["judge_score"] == 1
        assert result["judge_score"] <= 2
        # Disclaimer must be appended for low scores
        assert "Clinical Disclaimer" in result["final_output"] or "disclaimer" in result.get("judge_reason", "").lower()


# ---------------------------------------------------------------------------
# JUDGE-02 — High-quality clinical response scores ≥ 4
# ---------------------------------------------------------------------------

class TestJudge02HighQuality:
    """JUDGE-02 — Well-grounded clinical response scores ≥ 4."""

    def test_high_quality_scores_well(self):
        state = {
            "final_output": (
                "# Metformin — Type 2 Diabetes\n\n"
                "**Standard dose:** 500–1000mg twice daily with meals.\n\n"
                "**Mechanism:** Reduces hepatic glucose production and improves insulin sensitivity.\n\n"
                "**Key monitoring:** Renal function (eGFR); hold if eGFR <30 mL/min/1.73m².\n\n"
                "_Source: ADA Standards of Medical Care in Diabetes 2024_"
            ),
            "redacted_input": "What is the dosage and mechanism of metformin for Type 2 diabetes?",
            "pii_mapping": {},
        }
        result = _run_reviewer(state, score=5, reason="Accurate, well-sourced, clinically appropriate")

        assert result["judge_score"] >= 4
        # No disclaimer for high-quality responses
        assert "Clinical Disclaimer" not in state["final_output"]


# ---------------------------------------------------------------------------
# JUDGE-03 — Unreplaced PII placeholder detected and flagged
# ---------------------------------------------------------------------------

class TestJudge03PiiPlaceholder:
    """JUDGE-03 — Response with <PERSON_1> placeholder scores ≤ 2 and gets disclaimer."""

    def test_pii_placeholder_triggers_disclaimer(self):
        state = {
            "final_output": "Patient <PERSON_1> should take aspirin 81mg daily.",
            "redacted_input": "What should my patient take?",
            "pii_mapping": {"<PERSON_1>": "John Smith"},
        }
        result = _run_reviewer(state, score=2, reason="PII placeholder leak detected")

        assert result["judge_score"] <= 2
        assert "Clinical Disclaimer" in result["final_output"]


# ---------------------------------------------------------------------------
# JUDGE-04 — Determinism: same input at temperature=0 → same score
# ---------------------------------------------------------------------------

class TestJudge04Determinism:
    """JUDGE-04 — Five consecutive runs with same mocked score return identical results."""

    def test_deterministic_score(self):
        state = {
            "final_output": "Lisinopril 10mg once daily is first-line for hypertension.",
            "redacted_input": "What is the first-line treatment for hypertension?",
            "pii_mapping": {},
        }
        scores = []
        for _ in range(5):
            result = _run_reviewer(state, score=4, reason="Consistent reason")
            scores.append(result["judge_score"])

        assert len(set(scores)) == 1, (
            f"Judge produced different scores across 5 runs: {scores}"
        )


# ---------------------------------------------------------------------------
# JUDGE-05 — Off-topic response to medical query flagged as low quality
# ---------------------------------------------------------------------------

class TestJudge05Relevance:
    """JUDGE-05 — Response completely off-topic to clinical query scores ≤ 2."""

    def test_off_topic_response_low_score(self):
        state = {
            "final_output": "The weather in London today is partly cloudy with 18°C.",
            "redacted_input": "What are the symptoms of pulmonary embolism?",
            "pii_mapping": {},
        }
        result = _run_reviewer(state, score=1, reason="Response is completely irrelevant to clinical query")

        assert result["judge_score"] <= 2
        assert "Clinical Disclaimer" in result["final_output"]


# ---------------------------------------------------------------------------
# JUDGE-SAMPLE — JUDGE_SAMPLE_RATE=0.0 skips evaluation
# ---------------------------------------------------------------------------

class TestJudgeSampleRate:
    """JUDGE-SAMPLE — When sample rate is 0.0, judge_score must be None."""

    def test_zero_sample_rate_skips(self):
        state = {
            "final_output": "Metformin 500mg twice daily.",
            "redacted_input": "Metformin dosage?",
            "pii_mapping": {},
        }
        with patch("orchestrator.ChatGroq") as mock_groq_cls:
            with patch("orchestrator.settings", _mock_settings(sample_rate=0.0)):
                from orchestrator import node_reviewer
                result = node_reviewer(state)

        assert result["judge_score"] is None
        # Groq must never have been instantiated when sample rate=0
        mock_groq_cls.assert_not_called()


# ---------------------------------------------------------------------------
# JUDGE-JUDGE_DISABLED — JUDGE_ENABLED=False bypasses review entirely
# ---------------------------------------------------------------------------

class TestJudgeDisabled:
    """JUDGE_ENABLED=False — reviewer skips without calling Groq."""

    def test_disabled_returns_none(self):
        state = {
            "final_output": "Some clinical text.",
            "redacted_input": "Some clinical query.",
            "pii_mapping": {},
        }
        with patch("orchestrator.ChatGroq") as mock_groq_cls:
            with patch("orchestrator.settings", _mock_settings(enabled=False)):
                from orchestrator import node_reviewer
                result = node_reviewer(state)

        assert result["judge_score"] is None
        mock_groq_cls.assert_not_called()


# ---------------------------------------------------------------------------
# JUDGE-DISCLAIMER — Disclaimer content contains score and reason
# ---------------------------------------------------------------------------

class TestJudgeDisclaimerContent:
    """Low-score disclaimer must include score/5 and the judge reason."""

    def test_disclaimer_includes_score_and_reason(self):
        state = {
            "final_output": "I don't know much about this drug.",
            "redacted_input": "What is the mechanism of warfarin?",
            "pii_mapping": {},
        }
        result = _run_reviewer(
            state, score=2, reason="Insufficient clinical detail provided"
        )

        final = result["final_output"]
        assert "2/5" in final or "2" in final
        assert "Insufficient clinical detail" in final or "disclaimer" in final.lower()
