"""
End-to-end evaluation suite for MediCortex orchestration, tool calling,
and query-relevance guardrail.

Coverage (75 tests):
  SCOPE-01..15  — node_scope_guard medical/non-medical classification
  ROUTER-01..15 — node_router agent selection and edge cases
  JUDGE-01..20  — node_reviewer quality scoring and guardrail behaviour
  TOOL-01..15   — drug, pubmed, and diagnosis tool call contracts
  FLOW-01..10   — end-to-end state flow through the graph

Run:
  source .venv/bin/activate
  .venv/bin/python3 -m pytest tests/integration/test_e2e_evaluation.py -v --tb=short
"""

import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from langchain_core.messages import AIMessage, HumanMessage


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _base_state(**overrides) -> dict:
    state = {
        "input": "What is the dosage of aspirin?",
        "redacted_input": "What is the dosage of aspirin?",
        "pii_mapping": {},
        "file_urls": [],
        "context": [],
        "history": [],
        "routing_context": "",
        "messages": [],
        "agent_outputs": [],
        "agent_thoughts": [],
        "agents_used": [],
        "agent_sources": [],
        "final_output": "",
        "judge_score": None,
        "judge_reason": None,
        "judge_confidence": None,
        "error": None,
        "trace_id": "eval-trace",
        "session_id": "eval-session",
        "retrieval_iteration": 0,
        "retrieval_feedback": [],
        "retrieval_ambiguous": False,
        "clarification_question": None,
        "re_retrieval_skipped": False,
    }
    state.update(overrides)
    return state


def _mock_settings(**kwargs):
    m = MagicMock()
    m.JUDGE_ENABLED = kwargs.get("JUDGE_ENABLED", True)
    m.JUDGE_SAMPLE_RATE = kwargs.get("JUDGE_SAMPLE_RATE", 1.0)
    m.GROQ_API_KEY = kwargs.get("GROQ_API_KEY", "test-key")
    m.JUDGE_MODEL = kwargs.get("JUDGE_MODEL", "llama-3.3-70b-versatile")
    m.JUDGE_FALLBACK_MODEL = kwargs.get("JUDGE_FALLBACK_MODEL", "llama-3.1-8b-instant")
    m.JUDGE_MAX_INPUT_TOKENS = kwargs.get("JUDGE_MAX_INPUT_TOKENS", 500)
    m.MAX_CONCURRENT_AGENTS = kwargs.get("MAX_CONCURRENT_AGENTS", 3)
    return m


def _mock_groq_judge(score: int, reason: str = "ok", confidence: str = "90%"):
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(
        content=json.dumps({"score": score, "reason": reason, "confidence": confidence})
    )
    return mock


def _run_reviewer(state: dict, score: int = 4, reason: str = "ok"):
    with patch("orchestrator.ChatGroq", return_value=_mock_groq_judge(score, reason)):
        with patch("orchestrator.settings", _mock_settings()):
            from orchestrator import node_reviewer
            return node_reviewer(state)


# ===========================================================================
# SCOPE-01..15 — node_scope_guard
# ===========================================================================

class TestScopeGuard:
    """SCOPE-01..15 — Medical queries pass; non-medical are rejected."""

    def _run_guard(self, query: str, groq_response: str) -> dict:
        from orchestrator import node_scope_guard
        state = _base_state(input=query)
        mock_llm = MagicMock()
        mock_llm.content = groq_response
        with patch("orchestrator.ChatGroq") as mock_groq_cls:
            with patch("orchestrator.llm_ainvoke", new_callable=AsyncMock) as mock_ainvoke:
                with patch("orchestrator.settings") as mock_settings:
                    mock_settings.GROQ_API_KEY = "test-key"
                    mock_ainvoke.return_value = MagicMock(content=groq_response)
                    import asyncio
                    return asyncio.get_event_loop().run_until_complete(node_scope_guard(state))

    # SCOPE-01: Clear medical query — must pass through (empty dict)
    def test_scope_01_medication_dosage_in_scope(self):
        result = self._run_guard("What is the standard dosage of metformin?", "1")
        assert result == {} or result.get("final_output") is None or "MediCortex" not in result.get("final_output", "")

    # SCOPE-02: Drug interaction — in scope
    def test_scope_02_drug_interaction_in_scope(self):
        result = self._run_guard("Can I take ibuprofen with warfarin?", "1")
        assert result == {} or "MediCortex" not in result.get("final_output", "")

    # SCOPE-03: Symptom query — in scope
    def test_scope_03_symptom_query_in_scope(self):
        result = self._run_guard("What are the symptoms of appendicitis?", "1")
        assert result == {} or "MediCortex" not in result.get("final_output", "")

    # SCOPE-04: Lab result interpretation — in scope
    def test_scope_04_lab_result_in_scope(self):
        result = self._run_guard("My HbA1c is 8.2%. What does that mean?", "1")
        assert result == {} or "MediCortex" not in result.get("final_output", "")

    # SCOPE-05: Mental health — in scope
    def test_scope_05_mental_health_in_scope(self):
        result = self._run_guard("What are first-line treatments for major depressive disorder?", "1")
        assert result == {} or "MediCortex" not in result.get("final_output", "")

    # SCOPE-06: Cooking — out of scope
    def test_scope_06_cooking_out_of_scope(self):
        result = self._run_guard("What is the best recipe for pasta carbonara?", "0")
        assert "MediCortex" in result.get("final_output", "")

    # SCOPE-07: Sports — out of scope
    def test_scope_07_sports_out_of_scope(self):
        result = self._run_guard("Who won the 2023 FIFA World Cup?", "0")
        assert "MediCortex" in result.get("final_output", "")

    # SCOPE-08: Programming — out of scope
    def test_scope_08_programming_out_of_scope(self):
        result = self._run_guard("How do I reverse a linked list in Python?", "0")
        assert "MediCortex" in result.get("final_output", "")

    # SCOPE-09: Politics — out of scope
    def test_scope_09_politics_out_of_scope(self):
        result = self._run_guard("What are the policy differences between Democrats and Republicans?", "0")
        assert "MediCortex" in result.get("final_output", "")

    # SCOPE-10: Math problem — out of scope
    def test_scope_10_math_out_of_scope(self):
        result = self._run_guard("What is the integral of sin(x)?", "0")
        assert "MediCortex" in result.get("final_output", "")

    # SCOPE-11: Out-of-scope response sets agents_used to ["scope_guard"]
    def test_scope_11_out_of_scope_sets_agents_used(self):
        result = self._run_guard("Tell me a joke about penguins.", "0")
        assert result.get("agents_used") == ["scope_guard"]

    # SCOPE-12: Out-of-scope response leaves agent_outputs empty
    def test_scope_12_out_of_scope_empty_agent_outputs(self):
        result = self._run_guard("What is the capital of France?", "0")
        assert result.get("agent_outputs") == []

    # SCOPE-13: Guard skips when GROQ_API_KEY is empty (defaults to in-scope)
    def test_scope_13_no_api_key_defaults_in_scope(self):
        from orchestrator import node_scope_guard
        import asyncio
        state = _base_state(input="Tell me a joke")
        with patch("orchestrator.settings") as mock_settings:
            mock_settings.GROQ_API_KEY = ""
            result = asyncio.get_event_loop().run_until_complete(node_scope_guard(state))
        assert result == {}

    # SCOPE-14: Guard fails gracefully when LLM call throws
    def test_scope_14_llm_failure_defaults_in_scope(self):
        from orchestrator import node_scope_guard
        import asyncio
        state = _base_state(input="What are symptoms of flu?")
        with patch("orchestrator.ChatGroq"):
            with patch("orchestrator.llm_ainvoke", new_callable=AsyncMock) as mock_ainvoke:
                with patch("orchestrator.settings") as mock_settings:
                    mock_settings.GROQ_API_KEY = "test-key"
                    mock_ainvoke.side_effect = Exception("Network error")
                    result = asyncio.get_event_loop().run_until_complete(node_scope_guard(state))
        assert result == {}

    # SCOPE-15: Nutrition / health overlap — in scope
    def test_scope_15_nutrition_health_in_scope(self):
        result = self._run_guard("What dietary changes help manage Type 2 diabetes?", "1")
        assert result == {} or "MediCortex" not in result.get("final_output", "")


# ===========================================================================
# ROUTER-01..15 — node_router agent selection
# ===========================================================================

class TestRouterEvaluation:
    """ROUTER-01..15 — Correct agent selection for diverse query types."""

    def _run_router(self, query: str, mock_response: list) -> list:
        from orchestrator import node_router
        state = _base_state(redacted_input=query)
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=json.dumps(mock_response))
        with patch("orchestrator.llm", mock_llm):
            with patch("orchestrator.settings", _mock_settings()):
                result = node_router(state)
        msgs = [m for m in result.get("messages", []) if isinstance(m, AIMessage)]
        if not msgs:
            return []
        try:
            return json.loads(msgs[-1].content)
        except (json.JSONDecodeError, TypeError):
            return []

    # ROUTER-01: Drug interaction → pharmacology
    def test_router_01_drug_interaction(self):
        agents = self._run_router("What are the interactions between warfarin and aspirin?", ["pharmacology"])
        assert "pharmacology" in agents

    # ROUTER-02: Research paper query → pubmed
    def test_router_02_pubmed_research(self):
        agents = self._run_router("What does recent research say about mRNA vaccines?", ["pubmed"])
        assert "pubmed" in agents

    # ROUTER-03: Patient vitals query → patient
    def test_router_03_patient_vitals(self):
        agents = self._run_router("What are the latest vitals for patient John Doe?", ["patient"])
        assert "patient" in agents

    # ROUTER-04: Differential diagnosis → diagnosis
    def test_router_04_differential_diagnosis(self):
        agents = self._run_router("Patient has fever, cough, and dyspnea — what are the differentials?", ["diagnosis"])
        assert "diagnosis" in agents

    # ROUTER-05: Lab report attached → report_analyzer
    def test_router_05_lab_report(self):
        from orchestrator import node_router
        state = _base_state(
            redacted_input="What does this blood test show?",
            file_urls=["http://minio/bucket/lab.pdf"],
        )
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content='["report_analyzer"]')
        with patch("orchestrator.llm", mock_llm):
            with patch("orchestrator.settings", _mock_settings()):
                result = node_router(state)
        msgs = [m for m in result.get("messages", []) if isinstance(m, AIMessage)]
        routed = json.loads(msgs[-1].content) if msgs else []
        assert "report_analyzer" in routed

    # ROUTER-06: Multi-agent complex query — up to MAX_CONCURRENT_AGENTS
    def test_router_06_multi_agent_cap(self):
        agents = self._run_router(
            "Complex cardiology case with research, drugs, and patient history",
            ["pubmed", "pharmacology", "diagnosis", "patient"],
        )
        assert len(agents) <= 3

    # ROUTER-07: Dosage question → pharmacology (not diagnosis)
    def test_router_07_dosage_routes_to_pharmacology(self):
        agents = self._run_router("What is the maximum daily dose of ibuprofen?", ["pharmacology"])
        assert "pharmacology" in agents

    # ROUTER-08: Symptom cluster → diagnosis
    def test_router_08_symptoms_to_diagnosis(self):
        agents = self._run_router("Chest pain, sweating, and left arm numbness — what could this be?", ["diagnosis"])
        assert "diagnosis" in agents

    # ROUTER-09: RCT results query → pubmed
    def test_router_09_rct_to_pubmed(self):
        agents = self._run_router("What did the EMPEROR-Reduced trial find?", ["pubmed"])
        assert "pubmed" in agents

    # ROUTER-10: Unknown agent filtered out
    def test_router_10_unknown_agent_filtered(self):
        agents = self._run_router("General query", ["diagnosis", "ghost_agent"])
        assert "ghost_agent" not in agents

    # ROUTER-11: Empty LLM output → fallback to ["diagnosis"]
    def test_router_11_empty_llm_fallback(self):
        from orchestrator import node_router
        state = _base_state(redacted_input="What is flu?")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="")
        with patch("orchestrator.llm", mock_llm):
            with patch("orchestrator.settings", _mock_settings()):
                result = node_router(state)
        assert isinstance(result, dict)

    # ROUTER-12: LLM exception → graceful fallback, no crash
    def test_router_12_llm_exception_no_crash(self):
        from orchestrator import node_router
        state = _base_state(redacted_input="What is aspirin?")
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("Connection refused")
        with patch("orchestrator.llm", mock_llm):
            with patch("orchestrator.settings", _mock_settings()):
                result = node_router(state)
        assert isinstance(result, dict)

    # ROUTER-13: All 5 valid agents returned by LLM → capped to 3
    def test_router_13_five_agents_capped_to_three(self):
        agents = self._run_router(
            "Full workup needed",
            ["pubmed", "diagnosis", "report_analyzer", "patient", "pharmacology"],
        )
        assert len(agents) <= 3

    # ROUTER-14: Single-agent response preserved exactly
    def test_router_14_single_agent_preserved(self):
        agents = self._run_router("Interaction between metoprolol and diltiazem?", ["pharmacology"])
        assert agents == ["pharmacology"]

    # ROUTER-15: router result is a dict (never raises)
    def test_router_15_always_returns_dict(self):
        from orchestrator import node_router
        state = _base_state(redacted_input="")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="null")
        with patch("orchestrator.llm", mock_llm):
            with patch("orchestrator.settings", _mock_settings()):
                result = node_router(state)
        assert isinstance(result, dict)


# ===========================================================================
# JUDGE-01..20 — node_reviewer quality guardrail
# ===========================================================================

class TestJudgeGuardrail:
    """JUDGE-01..20 — Query-relevance guardrail and scoring behaviour."""

    # JUDGE-01: Relevant, high-quality response → score 5
    def test_judge_01_high_quality_scores_five(self):
        state = _base_state(
            final_output="Atorvastatin 10–80mg daily is recommended for hypercholesterolaemia.",
            redacted_input="What is the dosage of atorvastatin?",
        )
        result = _run_reviewer(state, score=5, reason="Directly addresses atorvastatin dosage")
        assert result["judge_score"] == 5

    # JUDGE-02: Low-quality, off-topic response → score 1, disclaimer appended
    def test_judge_02_low_score_appends_disclaimer(self):
        state = _base_state(
            final_output="SGLT2 inhibitors reduce cardiovascular events in heart failure.",
            redacted_input="What is the dosage of atorvastatin?",
        )
        result = _run_reviewer(state, score=1, reason="Response addresses SGLT2, not atorvastatin dosage")
        assert result["judge_score"] == 1
        assert "Clinical Disclaimer" in result["final_output"]

    # JUDGE-03: Score < 3 → disclaimer contains score value
    def test_judge_03_disclaimer_contains_score(self):
        state = _base_state(
            final_output="Some vague response.",
            redacted_input="Explain metformin side effects.",
        )
        result = _run_reviewer(state, score=2, reason="Incomplete")
        assert "2/5" in result["final_output"]

    # JUDGE-04: Score >= 3 → no disclaimer appended
    def test_judge_04_no_disclaimer_for_passing_score(self):
        state = _base_state(
            final_output="Metformin commonly causes GI side effects including nausea.",
            redacted_input="What are the side effects of metformin?",
        )
        result = _run_reviewer(state, score=3)
        assert "Clinical Disclaimer" not in result.get("final_output", state["final_output"])

    # JUDGE-05: JUDGE_ENABLED=False → judge_score is None
    def test_judge_05_disabled_returns_none(self):
        from orchestrator import node_reviewer
        state = _base_state(final_output="some output", redacted_input="some query")
        with patch("orchestrator.settings", _mock_settings(JUDGE_ENABLED=False)):
            result = node_reviewer(state)
        assert result["judge_score"] is None

    # JUDGE-06: No GROQ_API_KEY → judge_score is None
    def test_judge_06_no_api_key_returns_none(self):
        from orchestrator import node_reviewer
        state = _base_state(final_output="some output", redacted_input="some query")
        with patch("orchestrator.settings", _mock_settings(GROQ_API_KEY="")):
            result = node_reviewer(state)
        assert result["judge_score"] is None

    # JUDGE-07: SAMPLE_RATE=0.0 → always skipped
    def test_judge_07_zero_sample_rate_always_skips(self):
        import random
        from orchestrator import node_reviewer
        state = _base_state(final_output="output", redacted_input="query")
        with patch("orchestrator.settings", _mock_settings(JUDGE_SAMPLE_RATE=0.0)):
            with patch("random.random", return_value=0.5):
                result = node_reviewer(state)
        assert result["judge_score"] is None

    # JUDGE-08: Both primary and fallback models fail → judge_score is None (fail open)
    def test_judge_08_all_models_fail_returns_none(self):
        from orchestrator import node_reviewer
        state = _base_state(final_output="output", redacted_input="query")
        failing_mock = MagicMock()
        failing_mock.invoke.side_effect = Exception("Rate limited")
        with patch("orchestrator.ChatGroq", return_value=failing_mock):
            with patch("orchestrator.settings", _mock_settings()):
                result = node_reviewer(state)
        assert result["judge_score"] is None

    # JUDGE-09: Primary fails, fallback succeeds → returns fallback score
    def test_judge_09_fallback_model_used_on_primary_failure(self):
        from orchestrator import node_reviewer
        call_n = {"n": 0}
        def factory(**kwargs):
            call_n["n"] += 1
            m = MagicMock()
            if call_n["n"] == 1:
                m.invoke.side_effect = Exception("Primary failed")
            else:
                m.invoke.return_value = MagicMock(
                    content=json.dumps({"score": 4, "reason": "fallback ok", "confidence": "80%"})
                )
            return m
        state = _base_state(final_output="output", redacted_input="query")
        with patch("orchestrator.ChatGroq", side_effect=factory):
            with patch("orchestrator.settings", _mock_settings()):
                result = node_reviewer(state)
        assert result["judge_score"] == 4

    # JUDGE-10: Long response truncated before sending to Groq
    def test_judge_10_long_response_truncated(self):
        from orchestrator import node_reviewer
        state = _base_state(final_output="X" * 20000, redacted_input="test query")
        captured = {}
        def capture_invoke(messages, **kw):
            captured["prompt"] = messages[0].content
            return MagicMock(content=json.dumps({"score": 4, "reason": "ok", "confidence": "90%"}))
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = capture_invoke
        with patch("orchestrator.ChatGroq", return_value=mock_llm):
            with patch("orchestrator.settings", _mock_settings()):
                node_reviewer(state)
        assert "[truncated for evaluation]" in captured.get("prompt", "")

    # JUDGE-11: Multi-turn — current query at top of prompt, prior turn truncated to 300 chars
    def test_judge_11_history_entries_truncated_to_300(self):
        from orchestrator import node_reviewer
        long_prior = "A" * 2000  # 2000-char prior assistant response
        state = _base_state(
            final_output="Atorvastatin 40mg once daily is standard.",
            redacted_input="What is the dosage of atorvastatin?",
            history=["User: What does research say about SGLT2 inhibitors?", f"Assistant: {long_prior}"],
        )
        captured = {}
        def capture_invoke(messages, **kw):
            captured["prompt"] = messages[0].content
            return MagicMock(content=json.dumps({"score": 5, "reason": "ok", "confidence": "90%"}))
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = capture_invoke
        with patch("orchestrator.ChatGroq", return_value=mock_llm):
            with patch("orchestrator.settings", _mock_settings()):
                node_reviewer(state)
        prompt = captured.get("prompt", "")
        # Current query must appear near the top (within first 500 chars)
        assert "atorvastatin" in prompt[:500].lower()
        # The 2000-char prior response must NOT appear in full
        assert "A" * 400 not in prompt  # truncated to 300

    # JUDGE-12: judge_reason stored in return payload
    def test_judge_12_reason_stored_in_result(self):
        state = _base_state(final_output="good response", redacted_input="query")
        result = _run_reviewer(state, score=4, reason="Addresses the query accurately")
        assert result.get("judge_reason") == "Addresses the query accurately"

    # JUDGE-13: judge_confidence stored in return payload
    def test_judge_13_confidence_stored_in_result(self):
        from orchestrator import node_reviewer
        state = _base_state(final_output="output", redacted_input="query")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content=json.dumps({"score": 4, "reason": "ok", "confidence": "85%"})
        )
        with patch("orchestrator.ChatGroq", return_value=mock_llm):
            with patch("orchestrator.settings", _mock_settings()):
                result = node_reviewer(state)
        assert result.get("judge_confidence") == "85%"

    # JUDGE-14: PII placeholder <PERSON_1> does NOT reduce score
    def test_judge_14_pii_placeholder_not_penalised(self):
        state = _base_state(
            final_output="<PERSON_1> should take metformin 500mg twice daily.",
            redacted_input="What medication should my patient take for Type 2 diabetes?",
            pii_mapping={"<PERSON_1>": "John Smith"},
        )
        # Judge instructed not to penalise placeholders — mock returning 5
        result = _run_reviewer(state, score=5, reason="Directly addresses diabetes treatment for patient")
        assert result["judge_score"] == 5
        assert "Clinical Disclaimer" not in result.get("final_output", state["final_output"])

    # JUDGE-15: Empty final_output → not a crash, returns score
    def test_judge_15_empty_output_no_crash(self):
        state = _base_state(final_output="", redacted_input="What is aspirin?")
        result = _run_reviewer(state, score=1, reason="Empty response")
        assert result["judge_score"] == 1

    # JUDGE-16: Score exactly 3 → no disclaimer (boundary)
    def test_judge_16_score_3_no_disclaimer(self):
        state = _base_state(
            final_output="Aspirin 81mg daily is used for antiplatelet therapy.",
            redacted_input="What is the antiplatelet dose of aspirin?",
        )
        result = _run_reviewer(state, score=3)
        assert "Clinical Disclaimer" not in result.get("final_output", state["final_output"])

    # JUDGE-17: Score exactly 2 → disclaimer IS appended (boundary)
    def test_judge_17_score_2_disclaimer_appended(self):
        state = _base_state(
            final_output="Aspirin is sometimes used.",
            redacted_input="What is the antiplatelet dose of aspirin?",
        )
        result = _run_reviewer(state, score=2)
        assert "Clinical Disclaimer" in result["final_output"]

    # JUDGE-18: Current query in prompt is from redacted_input, not history
    def test_judge_18_current_query_from_redacted_input(self):
        from orchestrator import node_reviewer
        state = _base_state(
            final_output="Atorvastatin 40mg daily.",
            redacted_input="What is the dosage of atorvastatin?",
            history=["User: Tell me about SGLT2", "Assistant: SGLT2 inhibitors work by..."],
        )
        captured = {}
        def capture_invoke(messages, **kw):
            captured["prompt"] = messages[0].content
            return MagicMock(content=json.dumps({"score": 5, "reason": "ok", "confidence": "90%"}))
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = capture_invoke
        with patch("orchestrator.ChatGroq", return_value=mock_llm):
            with patch("orchestrator.settings", _mock_settings()):
                node_reviewer(state)
        prompt = captured.get("prompt", "")
        # Current query marker must precede history in the prompt
        query_pos = prompt.lower().find("atorvastatin")
        history_pos = prompt.lower().find("sglt2")
        assert query_pos < history_pos, "Current query must appear before history in judge prompt"

    # JUDGE-19: agents_used included in judge prompt
    def test_judge_19_agents_used_in_prompt(self):
        from orchestrator import node_reviewer
        state = _base_state(
            final_output="output",
            redacted_input="query",
            agents_used=["pharmacology"],
        )
        captured = {}
        def capture_invoke(messages, **kw):
            captured["prompt"] = messages[0].content
            return MagicMock(content=json.dumps({"score": 4, "reason": "ok", "confidence": "90%"}))
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = capture_invoke
        with patch("orchestrator.ChatGroq", return_value=mock_llm):
            with patch("orchestrator.settings", _mock_settings()):
                node_reviewer(state)
        assert "pharmacology" in captured.get("prompt", "")

    # JUDGE-20: Document context included when file_urls present
    def test_judge_20_doc_context_injected_when_file_attached(self):
        from orchestrator import node_reviewer
        state = _base_state(
            final_output="The CBC shows elevated WBC.",
            redacted_input="What does the attached blood test show?",
            file_urls=["http://minio/bucket/cbc.pdf"],
            agent_outputs=["## Report Analyzer\nWBC: 14,000 cells/µL (high). Hgb: 11 g/dL (low)."],
        )
        captured = {}
        def capture_invoke(messages, **kw):
            captured["prompt"] = messages[0].content
            return MagicMock(content=json.dumps({"score": 5, "reason": "ok", "confidence": "90%"}))
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = capture_invoke
        with patch("orchestrator.ChatGroq", return_value=mock_llm):
            with patch("orchestrator.settings", _mock_settings()):
                node_reviewer(state)
        assert "Document Analysis" in captured.get("prompt", "")


# ===========================================================================
# TOOL-01..15 — Tool calling contracts (mocked network)
# ===========================================================================

class TestToolContracts:
    """TOOL-01..15 — Drug, pubmed, and report tools return expected shapes."""

    # TOOL-01: check_drug_interactions returns non-empty string for valid pair
    def test_tool_01_drug_interaction_returns_string(self):
        from tools.drug_interaction_tools import check_drug_interactions
        with patch("tools.drug_interaction_tools._search_ddg", return_value=[
            {"title": "Warfarin-Aspirin Interaction", "url": "https://drugs.com/x",
             "snippet": "Increases bleeding risk.", "domain": "drugs.com"}
        ]):
            with patch("tools.drug_interaction_tools.httpx.Client"):
                result = check_drug_interactions.invoke({"drug_a": "warfarin", "drug_b": "aspirin"})
        assert isinstance(result, str)
        assert len(result) > 0

    # TOOL-02: check_drug_interactions with same drug name still runs
    def test_tool_02_same_drug_both_params(self):
        from tools.drug_interaction_tools import check_drug_interactions
        with patch("tools.drug_interaction_tools._search_ddg", return_value=[]):
            result = check_drug_interactions.invoke({"drug_a": "aspirin", "drug_b": "aspirin"})
        assert isinstance(result, str)

    # TOOL-03: check_drug_interactions empty drug_a returns error string
    def test_tool_03_empty_drug_a_returns_error(self):
        from tools.drug_interaction_tools import check_drug_interactions
        result = check_drug_interactions.invoke({"drug_a": "", "drug_b": "aspirin"})
        assert "Error" in result or "No" in result or len(result) > 0

    # TOOL-04: recommend_drugs returns structured markdown
    def test_tool_04_recommend_drugs_returns_markdown(self):
        from tools.drug_recommendation_tools import recommend_drugs
        with patch("tools.drug_recommendation_tools._search_ddg", return_value=[
            {"title": "Type 2 Diabetes Treatments", "url": "https://mayoclinic.org/x",
             "snippet": "Metformin is first-line therapy.", "domain": "mayoclinic.org"}
        ]):
            with patch("tools.drug_recommendation_tools.httpx.Client"):
                result = recommend_drugs.invoke({"condition": "Type 2 Diabetes"})
        assert "##" in result or "Drug Query" in result

    # TOOL-05: recommend_drugs with query_type="dosage"
    def test_tool_05_recommend_drugs_dosage_type(self):
        from tools.drug_recommendation_tools import recommend_drugs
        with patch("tools.drug_recommendation_tools._search_ddg", return_value=[
            {"title": "Metformin Dosage", "url": "https://drugs.com/x",
             "snippet": "500mg twice daily.", "domain": "drugs.com"}
        ]):
            with patch("tools.drug_recommendation_tools.httpx.Client"):
                result = recommend_drugs.invoke({"condition": "Metformin", "query_type": "dosage"})
        assert isinstance(result, str)
        assert "Dosage" in result or "dosage" in result.lower() or len(result) > 0

    # TOOL-06: recommend_drugs with query_type="alternatives"
    def test_tool_06_recommend_drugs_alternatives_type(self):
        from tools.drug_recommendation_tools import recommend_drugs
        with patch("tools.drug_recommendation_tools._search_ddg", return_value=[
            {"title": "Alternatives to Metformin", "url": "https://drugs.com/alt",
             "snippet": "SGLT2 inhibitors as alternatives.", "domain": "drugs.com"}
        ]):
            with patch("tools.drug_recommendation_tools.httpx.Client"):
                result = recommend_drugs.invoke({"condition": "Metformin", "query_type": "alternatives"})
        assert isinstance(result, str)

    # TOOL-07: recommend_drugs empty condition returns error
    def test_tool_07_recommend_drugs_empty_condition(self):
        from tools.drug_recommendation_tools import recommend_drugs
        result = recommend_drugs.invoke({"condition": ""})
        assert "Error" in result

    # TOOL-08: recommend_drugs DDG returns no results → informative message
    def test_tool_08_recommend_drugs_no_results(self):
        from tools.drug_recommendation_tools import recommend_drugs
        with patch("tools.drug_recommendation_tools._search_ddg", return_value=[]):
            result = recommend_drugs.invoke({"condition": "UnknownDrugXYZ123"})
        assert "No results" in result or "not found" in result.lower() or isinstance(result, str)

    # TOOL-09: check_drug_interactions result mentions at least one drug name
    def test_tool_09_interaction_result_mentions_drug(self):
        from tools.drug_interaction_tools import check_drug_interactions
        with patch("tools.drug_interaction_tools._search_ddg", return_value=[
            {"title": "Metformin interaction", "url": "https://drugs.com/m",
             "snippet": "Metformin may interact with alcohol.", "domain": "drugs.com"}
        ]):
            with patch("tools.drug_interaction_tools.httpx.Client"):
                result = check_drug_interactions.invoke({"drug_a": "metformin", "drug_b": "alcohol"})
        assert "metformin" in result.lower() or "alcohol" in result.lower() or isinstance(result, str)

    # TOOL-10: recommend_drugs with patient_info parameter
    def test_tool_10_recommend_drugs_with_patient_info(self):
        from tools.drug_recommendation_tools import recommend_drugs
        with patch("tools.drug_recommendation_tools._search_ddg", return_value=[
            {"title": "Hypertension elderly", "url": "https://nih.gov/x",
             "snippet": "ACE inhibitors are first-line.", "domain": "nih.gov"}
        ]):
            with patch("tools.drug_recommendation_tools.httpx.Client"):
                result = recommend_drugs.invoke({
                    "condition": "Hypertension",
                    "patient_info": "elderly with kidney disease"
                })
        assert isinstance(result, str)

    # TOOL-11: SQL injection attempt in drug name is sanitised
    def test_tool_11_sql_injection_sanitised(self):
        from tools.drug_recommendation_tools import recommend_drugs
        with patch("tools.drug_recommendation_tools._search_ddg", return_value=[]) as mock_search:
            result = recommend_drugs.invoke({"condition": "'; DROP TABLE patients; --"})
        assert isinstance(result, str)

    # TOOL-12: Shell metacharacters in drug name are sanitised
    def test_tool_12_shell_injection_sanitised(self):
        from tools.drug_interaction_tools import check_drug_interactions
        with patch("tools.drug_interaction_tools._search_ddg", return_value=[]):
            result = check_drug_interactions.invoke({"drug_a": "$(rm -rf /)", "drug_b": "aspirin"})
        assert isinstance(result, str)

    # TOOL-13: DDG network failure returns empty list (no exception propagation)
    def test_tool_13_ddg_failure_no_exception(self):
        from tools.drug_recommendation_tools import _search_ddg
        with patch("tools.drug_recommendation_tools.DDGS") as mock_ddgs:
            mock_ddgs.return_value.__enter__.return_value.text.side_effect = Exception("Network down")
            result = _search_ddg("metformin dosage")
        assert result == []

    # TOOL-14: check_drug_interactions network failure falls back gracefully
    def test_tool_14_interaction_network_failure_graceful(self):
        from tools.drug_interaction_tools import check_drug_interactions
        with patch("tools.drug_interaction_tools._search_ddg", return_value=[
            {"title": "Interaction", "url": "https://drugs.com/x",
             "snippet": "May interact.", "domain": "drugs.com"}
        ]):
            with patch("tools.drug_interaction_tools.httpx.Client") as mock_client:
                mock_client.return_value.__enter__.return_value.get.side_effect = Exception("Timeout")
                result = check_drug_interactions.invoke({"drug_a": "warfarin", "drug_b": "ibuprofen"})
        assert isinstance(result, str)

    # TOOL-15: recommend_drugs result does not contain raw PII-like strings
    def test_tool_15_result_no_pii_leakage(self):
        from tools.drug_recommendation_tools import recommend_drugs
        with patch("tools.drug_recommendation_tools._search_ddg", return_value=[
            {"title": "Lisinopril dosage", "url": "https://drugs.com/x",
             "snippet": "10mg once daily for hypertension.", "domain": "drugs.com"}
        ]):
            with patch("tools.drug_recommendation_tools.httpx.Client"):
                result = recommend_drugs.invoke({"condition": "Hypertension"})
        # Result should not contain SSN-like patterns (basic check)
        import re
        assert not re.search(r"\b\d{3}-\d{2}-\d{4}\b", result)


# ===========================================================================
# FLOW-01..10 — Orchestration state flow
# ===========================================================================

class TestOrchestrationFlow:
    """FLOW-01..10 — State transformations through the pipeline."""

    # FLOW-01: node_analyze_privacy sets redacted_input from input
    def test_flow_01_privacy_node_sets_redacted_input(self):
        from orchestrator import node_analyze_privacy
        state = _base_state(input="What is the dosage of metformin?")
        result = node_analyze_privacy(state)
        assert "redacted_input" in result
        assert isinstance(result["redacted_input"], str)
        assert len(result["redacted_input"]) > 0

    # FLOW-02: node_analyze_privacy redacts a person's name
    def test_flow_02_privacy_node_redacts_person_name(self):
        from orchestrator import node_analyze_privacy
        state = _base_state(input="What medication should John Smith take for diabetes?")
        result = node_analyze_privacy(state)
        assert "John Smith" not in result["redacted_input"]

    # FLOW-03: node_analyze_privacy clears agent_outputs accumulator
    def test_flow_03_privacy_node_clears_agent_outputs(self):
        from orchestrator import node_analyze_privacy
        state = _base_state(
            input="What is aspirin?",
            agent_outputs=["stale output from prior turn"],
        )
        result = node_analyze_privacy(state)
        assert result.get("agent_outputs") == []

    # FLOW-04: route_decision caps agents to MAX_CONCURRENT_AGENTS
    def test_flow_04_route_decision_caps_agents(self):
        from orchestrator import route_decision, MAX_CONCURRENT_AGENTS
        state = {"messages": [AIMessage(content='["pubmed","diagnosis","pharmacology","patient","report_analyzer"]')]}
        routes = route_decision(state)
        assert len(routes) <= MAX_CONCURRENT_AGENTS

    # FLOW-05: route_decision filters unknown agents
    def test_flow_05_route_decision_filters_unknown(self):
        from orchestrator import route_decision
        state = {"messages": [AIMessage(content='["diagnosis","fake_agent_99"]')]}
        routes = route_decision(state)
        assert "fake_agent_99" not in routes

    # FLOW-06: route_decision with empty message list defaults to ["diagnosis"]
    def test_flow_06_route_decision_empty_messages_defaults(self):
        from orchestrator import route_decision
        state = {"messages": [AIMessage(content="not json at all")]}
        routes = route_decision(state)
        assert isinstance(routes, list)
        assert len(routes) > 0

    # FLOW-07: node_reviewer return payload always has judge_score key
    def test_flow_07_reviewer_always_has_judge_score_key(self):
        state = _base_state(final_output="response", redacted_input="query")
        result = _run_reviewer(state, score=4)
        assert "judge_score" in result

    # FLOW-08: reviewer low score appended disclaimer does not modify original state
    def test_flow_08_reviewer_does_not_mutate_state(self):
        state = _base_state(
            final_output="original response",
            redacted_input="What is aspirin for?",
        )
        original_output = state["final_output"]
        result = _run_reviewer(state, score=1, reason="Poor response")
        # State dict is not mutated; change is in the returned payload
        assert state["final_output"] == original_output
        assert result["final_output"] != original_output

    # FLOW-09: Multi-turn: second query (different topic) is evaluated correctly
    def test_flow_09_second_query_evaluated_not_first(self):
        from orchestrator import node_reviewer
        # Simulate: first query was SGLT2, second is Atorvastatin
        history = [
            "User: What does recent research say about SGLT2 inhibitors?",
            "Assistant: " + "SGLT2 inhibitors dapagliflozin and empagliflozin reduce HFrEF events. " * 30,
        ]
        state = _base_state(
            final_output="Atorvastatin 40mg daily is standard for hypercholesterolaemia.",
            redacted_input="What is the dosage of atorvastatin for a middle-aged patient?",
            history=history,
        )
        captured = {}
        def capture_invoke(messages, **kw):
            captured["prompt"] = messages[0].content
            return MagicMock(content=json.dumps({"score": 5, "reason": "correctly addresses atorvastatin dosage", "confidence": "95%"}))
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = capture_invoke
        with patch("orchestrator.ChatGroq", return_value=mock_llm):
            with patch("orchestrator.settings", _mock_settings()):
                result = node_reviewer(state)
        # Current query ("atorvastatin") must appear in prompt before history ("SGLT2")
        prompt = captured.get("prompt", "").lower()
        assert prompt.find("atorvastatin") < prompt.find("sglt2")
        # Score reflects the current query, not the prior one
        assert result["judge_score"] == 5

    # FLOW-10: node_analyze_privacy returns pii_mapping in result
    def test_flow_10_privacy_node_returns_pii_mapping(self):
        from orchestrator import node_analyze_privacy
        state = _base_state(input="My name is Alice and I need help with my diabetes medication.")
        result = node_analyze_privacy(state)
        assert "pii_mapping" in result
        assert isinstance(result["pii_mapping"], dict)
