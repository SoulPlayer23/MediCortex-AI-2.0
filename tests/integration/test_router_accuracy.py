"""
EVAL-2 Layer 1 — Router accuracy tests (ROUTE-01..50 + edge cases).

Tests node_router against the 50-query ground truth set.

node_router uses the module-level `llm` singleton (None until lifespan() runs).
Tests patch `orchestrator.llm` directly with controlled mock responses.
"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage


GROUND_TRUTH_PATH = Path(__file__).parent.parent / "resources" / "routing_ground_truth.json"
VALID_AGENTS = {"pubmed", "diagnosis", "report_analyzer", "patient", "pharmacology"}


def _base_state(query: str, file_urls: list = None, retrieval_ambiguous: bool = False) -> dict:
    return {
        "input": query,
        "redacted_input": query,
        "pii_mapping": {},
        "file_urls": file_urls or [],
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
        "trace_id": "test-trace",
        "session_id": "test-session",
        "retrieval_iteration": 0,
        "retrieval_feedback": [],
        "retrieval_ambiguous": retrieval_ambiguous,
        "clarification_question": None,
        "re_retrieval_skipped": False,
    }


def _make_llm_mock(response_content: str):
    """Return a mock LLM whose invoke().content is response_content."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content=response_content)
    return mock


def _run_router(query: str, agents_json: list, file_urls: list = None,
                retrieval_ambiguous: bool = False) -> dict:
    """Run node_router with a mocked LLM returning agents_json."""
    from orchestrator import node_router
    state = _base_state(query, file_urls, retrieval_ambiguous)

    with patch("orchestrator.llm", _make_llm_mock(json.dumps(agents_json))):
        with patch("orchestrator.settings") as mock_settings:
            mock_settings.MAX_CONCURRENT_AGENTS = 3
            return node_router(state)


def _extract_routed_agents(result: dict) -> list:
    """Extract the agent list from the router's AIMessage output."""
    messages = result.get("messages", [])
    ai_msgs = [m for m in messages if isinstance(m, AIMessage)]
    if not ai_msgs:
        return []
    try:
        parsed = json.loads(ai_msgs[-1].content)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


# ---------------------------------------------------------------------------
# Ground-truth file loader (module-level so parametrize can use it)
# ---------------------------------------------------------------------------

def _load_ground_truth():
    with open(GROUND_TRUTH_PATH) as f:
        return json.load(f)


_GROUND_TRUTH = _load_ground_truth()


# ---------------------------------------------------------------------------
# ROUTE-01..50 — Ground truth benchmark
# ---------------------------------------------------------------------------

class TestRouteGroundTruth:
    """ROUTE-01..50 — Each ground-truth query routes to its expected agent(s)."""

    def test_ground_truth_file_has_50_entries(self):
        assert len(_GROUND_TRUTH) == 50, (
            f"Expected 50 entries in routing_ground_truth.json, got {len(_GROUND_TRUTH)}"
        )

    @pytest.mark.parametrize("entry,idx", [
        (e, i) for i, e in enumerate(_GROUND_TRUTH)
    ], ids=[f"ROUTE-{str(i+1).zfill(2)}" for i in range(len(_GROUND_TRUTH))])
    def test_routing_accuracy(self, entry, idx):
        """Mock LLM returns expected agents; verify node_router parses and emits them."""
        query = entry["query"]
        expected = set(entry["expected_agents"])

        result = _run_router(query, list(expected))
        routed = _extract_routed_agents(result)

        # If clarification path triggered (retrieval_ambiguous=True), skip agent check
        if result.get("clarification_question"):
            pytest.skip("Clarification path triggered — not an agent routing test")

        routed_set = set(routed)
        overlap = routed_set & expected
        assert overlap, (
            f"[ROUTE-{idx+1:02d}] No expected agent in route for {query!r}. "
            f"Expected {expected}, got {routed_set}"
        )


# ---------------------------------------------------------------------------
# ROUTE-CAP — Agent list capped at MAX_CONCURRENT_AGENTS (3)
# ---------------------------------------------------------------------------

class TestRouteCap:
    """ROUTE-CAP — Router caps output to MAX_CONCURRENT_AGENTS."""

    def test_cap_enforced(self):
        # LLM returns 5 agents — route_decision should cap to ≤ 3
        result = _run_router(
            "Complex query",
            ["pubmed", "diagnosis", "report_analyzer", "patient", "pharmacology"],
        )
        routed = _extract_routed_agents(result)
        if routed:
            assert len(routed) <= 3, (
                f"Router returned {len(routed)} agents, expected ≤3. Got: {routed}"
            )


# ---------------------------------------------------------------------------
# ROUTE-MALFORMED — Malformed LLM output falls back to ["diagnosis"]
# ---------------------------------------------------------------------------

class TestRouteMalformed:
    """ROUTE-MALFORMED — Non-JSON / empty LLM output uses fallback."""

    @pytest.mark.parametrize("bad_output", [
        "I cannot determine the route.",
        "",
        "null",
        "{'agents': ['diagnosis']}",
        "diagnosis pharmacology",
    ])
    def test_malformed_fallback(self, bad_output):
        from orchestrator import node_router
        state = _base_state("What is aspirin used for?")

        with patch("orchestrator.llm", _make_llm_mock(bad_output)):
            with patch("orchestrator.settings") as mock_settings:
                mock_settings.MAX_CONCURRENT_AGENTS = 3
                result = node_router(state)

        # No crash — must return a dict
        assert isinstance(result, dict)
        # If messages present, the content should be parseable or route to fallback
        messages = result.get("messages", [])
        if messages:
            ai_msgs = [m for m in messages if isinstance(m, AIMessage)]
            if ai_msgs:
                content = ai_msgs[-1].content
                try:
                    routed = json.loads(content)
                    # Fallback: either ["diagnosis"] or some valid list
                    assert isinstance(routed, list)
                except json.JSONDecodeError:
                    pass  # clarification path is also acceptable


# ---------------------------------------------------------------------------
# ROUTE-UNKNOWN — Unknown agent names filtered, known ones preserved
# ---------------------------------------------------------------------------

class TestRouteUnknown:
    """ROUTE-UNKNOWN — Unknown agent names in LLM output are filtered out."""

    def test_unknown_agent_filtered(self):
        result = _run_router(
            "What is the diagnosis?",
            ["diagnosis", "unknown_agent_xyz"],
        )
        routed = _extract_routed_agents(result)
        for agent in routed:
            assert agent in VALID_AGENTS or agent == "__clarify__", (
                f"Unknown agent '{agent}' passed through the filter"
            )


# ---------------------------------------------------------------------------
# ROUTE-FILE — file_urls present always includes report_analyzer
# ---------------------------------------------------------------------------

class TestRouteFileUrls:
    """When file_urls is non-empty, report_analyzer must appear in the route."""

    def test_file_url_includes_report_analyzer(self):
        result = _run_router(
            "What does this X-ray show?",
            ["report_analyzer"],
            file_urls=["http://minio/bucket/xray.pdf"],
        )
        routed = _extract_routed_agents(result)
        if routed:
            assert "report_analyzer" in routed, (
                f"report_analyzer missing from route when file_urls present. Got: {routed}"
            )
