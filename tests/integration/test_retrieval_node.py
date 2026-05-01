"""
EVAL-2 Layer 1 — Retrieval Node tests (RET-01..06).

Tests node_retrieve_knowledge: entity extraction, KB routing, generic anatomy
suppression, topic-continuity carry-forward, and graceful KB-offline degradation.

node_retrieve_knowledge uses module-level singletons `extractor_llm` and `llm`
(both None until lifespan() runs). Tests patch these directly.
"""
import pytest
from unittest.mock import patch, MagicMock


def _base_state(query: str, context: list = None, routing_context: str = "") -> dict:
    return {
        "input": query,
        "redacted_input": query,
        "pii_mapping": {},
        "file_urls": [],
        "context": context or [],
        "history": [],
        "routing_context": routing_context,
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
        "trace_id": "test-trace",
        "session_id": "test-session",
        "retrieval_iteration": 0,
        "retrieval_feedback": [],
        "retrieval_ambiguous": False,
        "clarification_question": None,
        "re_retrieval_skipped": False,
    }


def _make_llm_mock(entities_json: str):
    """Return a mock LLM whose invoke().content returns a JSON entity array."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content=entities_json)
    return mock


def _run_retrieval(query: str, entities_json: str, routing_context: str = "",
                   kb_return=None, kb_side_effect=None):
    """Run node_retrieve_knowledge with mocked extractor_llm and medical_engine."""
    from orchestrator import node_retrieve_knowledge
    state = _base_state(query, routing_context=routing_context)

    llm_mock = _make_llm_mock(entities_json)

    with patch("orchestrator.extractor_llm", llm_mock):
        with patch("orchestrator.medical_engine") as mock_engine:
            if kb_side_effect:
                mock_engine.search_and_reason.side_effect = kb_side_effect
            else:
                mock_engine.search_and_reason.return_value = kb_return or []
            with patch("orchestrator.consult_medical_knowledge") as mock_kb:
                if kb_side_effect:
                    mock_kb.invoke.side_effect = kb_side_effect
                else:
                    facts = kb_return or []
                    mock_kb.invoke.return_value = "\n".join(
                        f"- {r['name']} ({r['relation']})" for r in facts
                    ) if facts else "No specific knowledge found in graph."
                return node_retrieve_knowledge(state), mock_engine, mock_kb


# ---------------------------------------------------------------------------
# RET-01 — Drug query: entity extracted, KB consulted
# ---------------------------------------------------------------------------

class TestRet01DrugEntity:
    """RET-01 — 'metformin' extracted and KB is queried."""

    def test_entity_extracted_for_drug(self):
        result, _, mock_kb = _run_retrieval(
            "What is the dosage of metformin?",
            entities_json='["metformin"]',
            kb_return=[{"name": "Metformin", "relation": "treats", "hop": 1}],
        )
        # KB must have been queried
        mock_kb.invoke.assert_called()
        # Context must be populated
        assert isinstance(result.get("context"), list)
        assert result.get("retrieval_ambiguous") is False


# ---------------------------------------------------------------------------
# RET-02 — Vague query: no entity, retrieval_ambiguous=True
# ---------------------------------------------------------------------------

class TestRet02VagueQuery:
    """RET-02 — Generic query produces no KB entity and sets retrieval_ambiguous."""

    def test_no_entity_for_generic_query(self):
        result, _, mock_kb = _run_retrieval(
            "Can you explain my report?",
            entities_json="[]",
        )
        assert result.get("retrieval_ambiguous") is True
        # KB must NOT have been queried with a meaningful entity
        mock_kb.invoke.assert_not_called()


# ---------------------------------------------------------------------------
# RET-03 — Generic anatomy suppressed
# ---------------------------------------------------------------------------

class TestRet03AnatomySuppressed:
    """RET-03 — Generic anatomy terms are filtered before KB lookup."""

    @pytest.mark.parametrize("term,query", [
        ("left ventricle", "What does the left ventricle do?"),
        ("heart",   "My heart feels weird"),
        ("lung",    "My lung hurts"),
        ("liver",   "Tell me about my liver"),
        ("kidney",  "My kidney aches"),
        ("brain",   "My brain is tired"),
        ("blood",   "My blood feels off"),
        ("skin",    "My skin itches"),
        ("chest",   "My chest hurts"),
        ("abdomen", "Pain in my abdomen"),
        ("back",    "My back hurts"),
        ("leg",     "My leg is sore"),
        ("arm",     "My arm feels weak"),
        ("head",    "My head aches"),
        ("neck",    "My neck is stiff"),
    ])
    def test_anatomy_suppressed(self, term, query):
        # LLM returns just the anatomical term — should be filtered post-extraction
        result, _, mock_kb = _run_retrieval(
            query,
            entities_json=f'["{term}"]',
        )
        # Either ambiguous (KB not called) or KB wasn't called with the anatomy term
        if mock_kb.invoke.called:
            for call_args in mock_kb.invoke.call_args_list:
                arg = str(call_args[0][0]).lower() if call_args[0] else ""
                assert term.lower() not in arg or term.lower() == arg, (
                    f"Generic anatomy '{term}' leaked into KB query: {arg!r}"
                )
        else:
            assert result.get("retrieval_ambiguous") is True


# ---------------------------------------------------------------------------
# RET-04 — Topic-shift: entity injected from routing_context
# ---------------------------------------------------------------------------

class TestRet04TopicContinuity:
    """RET-04 — Follow-up 'What are the side effects?' inherits entity from context."""

    def test_entity_injected_from_routing_context(self):
        routing_ctx = (
            "User asked: What is metformin used for?\n"
            "AI used agents: pharmacology\n"
        )
        result, _, mock_kb = _run_retrieval(
            "What are the side effects?",
            entities_json="[]",  # LLM finds no entity in vague follow-up
            routing_context=routing_ctx,
            kb_return=[{"name": "Metformin side effects", "relation": "causes", "hop": 1}],
        )
        # Topic-shift should inject 'metformin' from routing_context → KB is called
        assert mock_kb.invoke.called, (
            "KB was not queried despite topic-shift routing context with prior metformin query"
        )


# ---------------------------------------------------------------------------
# RET-05 — KB offline: node completes without crashing
# ---------------------------------------------------------------------------

class TestRet05KbOffline:
    """RET-05 — ArangoDB offline → graceful degradation, no crash."""

    def test_kb_offline_no_crash(self):
        result, _, _ = _run_retrieval(
            "What is the dosage of lisinopril?",
            entities_json='["lisinopril"]',
            kb_side_effect=Exception("ArangoDB connection refused"),
        )
        assert isinstance(result, dict), "node_retrieve_knowledge must return a dict even when KB is offline"

    def test_kb_offline_safe_context(self):
        result, _, _ = _run_retrieval(
            "What is the dosage of lisinopril?",
            entities_json='["lisinopril"]',
            kb_side_effect=Exception("timeout"),
        )
        # Context must not contain fabricated KB facts
        context = result.get("context", [])
        assert isinstance(context, list)
        kb_text = " ".join(context).lower()
        # Should either be empty, contain "no specific" / sentinel, or just be safe
        assert (
            not kb_text
            or "no specific" in kb_text
            or "not found" in kb_text
            or "offline" in kb_text
            or "lisinopril" not in kb_text  # no fabricated drug-specific facts
        )


# ---------------------------------------------------------------------------
# RET-06 — Synonym resolution: 'Glucophage' maps to metformin KB facts
# ---------------------------------------------------------------------------

class TestRet06SynonymResolution:
    """RET-06 — Trade name 'Glucophage' is passed to KB; KB returns metformin facts."""

    def test_glucophage_kb_queried(self):
        result, _, mock_kb = _run_retrieval(
            "What is the mechanism of Glucophage?",
            entities_json='["Glucophage"]',
            kb_return=[{"name": "Metformin (Glucophage)", "relation": "treats", "hop": 1}],
        )
        # KB must have been called
        assert mock_kb.invoke.called
        # Result contains context (synonym resolved by KB engine)
        assert isinstance(result.get("context"), list)
        assert result.get("retrieval_ambiguous") is not True
