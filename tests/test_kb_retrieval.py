"""
BUG-3 — ArangoDB KB Retrieval Verification Test

Run this standalone to confirm ArangoDB is populated and the knowledge
core returns real data for known medical entities.

Usage:
    python tests/test_kb_retrieval.py

Pre-requisites:
    - ArangoDB running and accessible (ARANGODB_HOST in .env)
    - knowledge_core assets built: python3 -m knowledge_core.build_fast_assets
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from knowledge_core.medical_engine import MedicalReasoningEngine
from config import settings

_EMPTY_SENTINEL = "No specific knowledge found in graph."
_MIN_CONTENT_CHARS = 200

TEST_TERMS = [
    "metformin",
    "hypertension",
    "Type 2 Diabetes",
]


def run():
    print(f"\nArangoDB host : {settings.ARANGODB_HOST}")
    print(f"Database      : {settings.ARANGODB_DB_NAME}\n")

    engine = MedicalReasoningEngine()

    passed = 0
    failed = 0

    for term in TEST_TERMS:
        results = engine.search_and_reason(term)
        formatted = [
            f"- {r['name']} ({r['relation']}, Hop: {r.get('hop', '?')})"
            for r in results
        ]
        content = "\n".join(formatted) if formatted else _EMPTY_SENTINEL

        ok = content != _EMPTY_SENTINEL and len(content) >= _MIN_CONTENT_CHARS
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        print(f"[{status}] '{term}'")
        if ok:
            lines = content.splitlines()
            preview = "\n".join(lines[:5])
            suffix = f"\n       ... ({len(lines)} facts total)" if len(lines) > 5 else ""
            print(f"       {preview}{suffix}")
        else:
            print(f"       Result: {content[:120]}")
        print()

    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        print(
            "\nArangoDB appears empty or unreachable. Run:\n"
            "  python3 -m knowledge_core.build_fast_assets\n"
            "and verify the host/credentials in .env before testing RAG-1."
        )
        sys.exit(1)
    else:
        print("KB is populated — safe to run RAG-1 tests.")


if __name__ == "__main__":
    run()
