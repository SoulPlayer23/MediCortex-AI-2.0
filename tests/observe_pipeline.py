"""
MediCortex Pipeline Observer
=============================
Fires a series of chat requests against the live /chat/stream endpoint and
prints every SSE event in real-time with timing and colour so you can watch
the routing, tool calls, and HIPAA redaction in action.

Usage:
    python tests/observe_pipeline.py

Requires the orchestrator to be running:
    python orchestrator.py
"""

import asyncio
import json
import os
import sys
import time
from typing import Optional

import httpx

# ── Colour helpers ────────────────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RED     = "\033[91m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
BLUE    = "\033[94m"
MAGENTA = "\033[95m"
CYAN    = "\033[96m"
WHITE   = "\033[97m"
GREY    = "\033[90m"

def _c(colour: str, text: str) -> str:
    return f"{colour}{text}{RESET}"

def _header(title: str, colour: str = CYAN) -> None:
    bar = "─" * 70
    print(f"\n{colour}{BOLD}{bar}{RESET}")
    print(f"{colour}{BOLD}  {title}{RESET}")
    print(f"{colour}{BOLD}{bar}{RESET}")

def _divider() -> None:
    print(f"{GREY}{'·' * 70}{RESET}")

# ── Event printer ─────────────────────────────────────────────────────────────

def print_event(event_type: str, content: str, elapsed: float) -> None:
    ts = f"{_c(GREY, f'[+{elapsed:5.1f}s]')}"

    if event_type == "session_id":
        print(f"{ts} {_c(BLUE,   '  SESSION')}  {_c(WHITE, content)}")

    elif event_type == "thought":
        # Distinguish routing thoughts vs tool call thoughts vs observations
        if content.startswith("Querying"):
            print(f"{ts} {_c(MAGENTA, '  BOOT   ')}  {content}")
        elif "Calling `" in content:
            # Tool call — highlight tool name
            print(f"{ts} {_c(YELLOW,  '  TOOL   ')}  {content}")
        elif content.startswith("**Observation"):
            snippet = content[:160] + ("…" if len(content) > 160 else "")
            print(f"{ts} {_c(CYAN,    '  OBS    ')}  {snippet}")
        elif "Synthesizing" in content:
            print(f"{ts} {_c(GREEN,   '  SYNTH  ')}  {content}")
        else:
            print(f"{ts} {_c(WHITE,   '  THINK  ')}  {content}")

    elif event_type == "metadata":
        try:
            meta = json.loads(content) if isinstance(content, str) else content
            score = meta.get("judge_score")
            reason = meta.get("judge_reason", "")
            conf   = meta.get("judge_confidence", "")
            colour = GREEN if score and score >= 4 else (YELLOW if score and score >= 3 else RED)
            print(f"{ts} {_c(colour,  '  JUDGE  ')}  Score {score}/5 | {reason} | {conf}")
        except Exception:
            print(f"{ts} {_c(GREY,    '  META   ')}  {content}")

    elif event_type == "response":
        print(f"\n{ts} {_c(GREEN, BOLD + '  FINAL  ' + RESET)}")
        # Print response with indentation, capped at 60 lines for readability
        lines = content.split("\n")
        cap = 60
        for line in lines[:cap]:
            print(f"          {_c(WHITE, line)}")
        if len(lines) > cap:
            print(f"          {_c(GREY, f'  … ({len(lines) - cap} more lines)')}")

    elif event_type == "error":
        print(f"{ts} {_c(RED,    '  ERROR  ')}  {content}")

    else:
        print(f"{ts} {_c(GREY,   f'  {event_type.upper():<7}')}  {content}")


# ── SSE streaming consumer ────────────────────────────────────────────────────

async def stream_query(
    client: httpx.AsyncClient,
    message: str,
    session_id: Optional[str] = None,
    base_url: str = "http://localhost:8001",
) -> Optional[str]:
    """
    Send a chat message to /chat/stream and print every SSE event as it arrives.
    Returns the session_id so it can be reused for multi-turn demos.
    """
    payload: dict = {"message": message}
    if session_id:
        payload["session_id"] = session_id

    print(f"\n{_c(BOLD, '  USER:')} {_c(YELLOW, message)}")
    _divider()

    returned_session_id = session_id
    start = time.monotonic()

    try:
        async with client.stream(
            "POST",
            f"{base_url}/chat/stream",
            json=payload,
            timeout=120.0,
        ) as response:
            if response.status_code != 200:
                print(_c(RED, f"  HTTP {response.status_code} — is the orchestrator running?"))
                return None

            async for raw_line in response.aiter_lines():
                if not raw_line.startswith("data: "):
                    continue
                data = raw_line[len("data: "):]
                if data == "[DONE]":
                    elapsed = time.monotonic() - start
                    print(f"\n{_c(GREY, f'  ✓ Done in {elapsed:.1f}s')}")
                    break

                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type", "unknown")
                content    = event.get("content", "")
                elapsed    = time.monotonic() - start

                # Track session_id for multi-turn scenarios
                if event_type == "session_id":
                    returned_session_id = content

                # Metadata content may be a dict — serialise for the printer
                if event_type == "metadata" and isinstance(content, dict):
                    content = json.dumps(content)

                print_event(event_type, content, elapsed)

    except httpx.ConnectError:
        print(_c(RED, "\n  Connection refused — start the orchestrator first:"))
        print(_c(GREY, "    python orchestrator.py"))
        sys.exit(1)
    except Exception as exc:
        print(_c(RED, f"\n  Stream error: {exc}"))

    return returned_session_id


# ── Scenarios ─────────────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "title": "SCENARIO 1 — Diagnosis routing (symptoms query)",
        "description": (
            "Expects router → ['diagnosis']\n"
            "  Phase 1: GPT-4o-mini calls symptom_analysis + webcrawler tools\n"
            "  Phase 2: MedGemma synthesizes clinical response"
        ),
        "message": "I have persistent chest pain, shortness of breath, and sweating. What could this be?",
    },
    {
        "title": "SCENARIO 2 — Pharmacology routing (drug query)",
        "description": (
            "Expects router → ['pharmacology']\n"
            "  Phase 1: GPT-4o-mini calls drug_interaction_tools\n"
            "  Phase 2: MedGemma synthesizes safety summary"
        ),
        "message": "What are the interactions between Metformin and Lisinopril?",
    },
    {
        "title": "SCENARIO 3 — PubMed routing (research query)",
        "description": (
            "Expects router → ['pubmed']\n"
            "  Phase 1: GPT-4o-mini searches PubMed and crawls abstracts\n"
            "  Phase 2: MedGemma summarises the evidence"
        ),
        "message": "What does recent research say about GLP-1 agonists for Type 2 Diabetes?",
    },
    {
        "title": "SCENARIO 4 — Patient routing + HIPAA redaction",
        "description": (
            "Expects router → ['patient']\n"
            "  Presidio redacts 'John Smith' → <PERSON_1> BEFORE any LLM sees the query\n"
            "  Phase 1: GPT-4o-mini calls retrieve_patient_records('<PERSON_1>')\n"
            "    Tool internally resolves placeholder → real name → DB lookup → re-redacts output\n"
            "  Phase 2: MedGemma synthesizes using only <PERSON_1> placeholder\n"
            "  node_restore_privacy replaces <PERSON_1> → 'John Smith' in final output only"
        ),
        "message": "Show me the full medical summary for patient John Smith.",
    },
    {
        "title": "SCENARIO 5 — Multi-agent routing (combined query)",
        "description": (
            "Expects router → ['diagnosis', 'pharmacology'] (two agents in parallel)\n"
            "  Both agents run concurrently up to MAX_CONCURRENT_AGENTS=3\n"
            "  Aggregator merges both outputs into one Markdown response"
        ),
        "message": "What are the symptoms of hypertension AND what are the common drugs used to treat it?",
    },
    {
        "title": "SCENARIO 6 — Multi-turn conversation (HIPAA history fix)",
        "description": (
            "Turn 1: patient query — 'John Smith' stored in DB after restore_privacy\n"
            "Turn 2: follow-up — history re-fetched from DB (contains real name)\n"
            "  privacy_manager.redact_pii() re-runs on history_str before enhanced_input\n"
            "  so GPT-4o-mini planner never sees 'John Smith' in the history context"
        ),
        "turns": [
            "What medications is John Smith currently taking?",
            "Are there any dangerous interactions between his current medications?",
        ],
    },
]


# ── Main runner ───────────────────────────────────────────────────────────────

async def main(base_url: str = "http://localhost:8001") -> None:
    print(_c(BOLD + CYAN, "\n╔══════════════════════════════════════════════════════════════════════╗"))
    print(_c(BOLD + CYAN,   "║           MediCortex Pipeline Observer                               ║"))
    print(_c(BOLD + CYAN,   "║  Watching: routing · tool calls · HIPAA redaction · judge scoring    ║"))
    print(_c(BOLD + CYAN,   "╚══════════════════════════════════════════════════════════════════════╝"))
    print(f"\n  Target: {_c(WHITE, base_url)}")
    print(f"  Tip: run with {_c(YELLOW, 'python tests/observe_pipeline.py')} while orchestrator is running\n")

    # Verify the server is reachable before starting
    try:
        async with httpx.AsyncClient(timeout=5.0) as probe:
            r = await probe.get(f"{base_url}/health")
            agents = r.json().get("agents", [])
            print(f"  {_c(GREEN, '✓')} Orchestrator healthy | agents: {_c(WHITE, ', '.join(agents))}")
    except Exception:
        print(_c(RED, f"  ✗ Cannot reach {base_url} — start with: python orchestrator.py"))
        sys.exit(1)

    async with httpx.AsyncClient() as client:
        for scenario in SCENARIOS:
            _header(scenario["title"])
            print(_c(DIM, f"\n  {scenario['description']}\n"))

            if "turns" in scenario:
                # Multi-turn: reuse session across turns
                session_id = None
                for i, turn_message in enumerate(scenario["turns"], 1):
                    print(f"\n{_c(BOLD, f'  ── Turn {i} ──')}")
                    session_id = await stream_query(
                        client, turn_message, session_id=session_id, base_url=base_url
                    )
            else:
                await stream_query(client, scenario["message"], base_url=base_url)

    _header("Observer complete", colour=GREEN)
    print(_c(GREEN, "  All scenarios finished.\n"))


if __name__ == "__main__":
    # Allow overriding the base URL via env var for CI / remote servers
    url = os.environ.get("MEDICORTEX_URL", "http://localhost:8001")
    asyncio.run(main(base_url=url))
