"""
MediCortex Per-Node Latency Profiler — Table 6.8

Measures wall-clock latency (mean ± SD, milliseconds) for each node in the
LangGraph pipeline by sending N queries to the running orchestrator and parsing
structured node_elapsed_ms log events captured from stdout.

The orchestrator must be running with JSON log output (DEBUG=false):
  python orchestrator.py 2>&1 | tee /tmp/medicortex_timing.log

Usage (requires running orchestrator):
  source .venv/bin/activate
  python3 tests/evaluation/run_latency_profiler.py

OR in standalone mode (no running server, direct component timing):
  python3 tests/evaluation/run_latency_profiler.py --standalone

Requirements (running mode):
  - Orchestrator running at port 8001 (python orchestrator.py)
  - Structlog JSON output: set DEBUG=false in .env

Requirements (standalone mode):
  - GROQ_API_KEY in .env (scope guard + judge timing)
  - homeserver Ollama reachable (router timing)
  - Presidio installed (PHI redaction timing)

Runtime: ~10-20 min for N=20 queries
"""

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ORCHESTRATOR_URL  = os.getenv("MEDICORTEX_URL", "http://homeserver:8000")
N_QUERIES         = 20
TIMING_LOG_FILE   = Path("/tmp/medicortex_timing.log")

# Clinical queries for profiling (single-agent, clear intent to get predictable routing)
PROFILE_QUERIES = [
    "What is the mechanism of action of metformin?",
    "What are the symptoms of hypothyroidism?",
    "Interpret: WBC 12.5k, Hgb 9.1, Plt 450k",
    "Is atorvastatin safe with warfarin?",
    "What is the first-line treatment for hypertension?",
    "Signs of pulmonary embolism",
    "Explain the CHADS2-VASc score",
    "What does an elevated troponin indicate?",
    "Drug interactions between SSRIs and MAOIs",
    "Management of type 2 diabetes in a patient with CKD stage 3",
    "What is the APACHE II score used for?",
    "Symptoms of congestive heart failure",
    "How does aspirin work as an antiplatelet?",
    "What are the contraindications of beta blockers?",
    "Interpret this ABG: pH 7.28, PaCO2 55, HCO3 24",
    "What is the Wells score for DVT?",
    "Management of acute exacerbation of COPD",
    "Amoxicillin dose for a 70kg adult with community-acquired pneumonia",
    "What does a positive D-dimer indicate?",
    "Signs of diabetic ketoacidosis",
]

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

R = "\033[91m"; G = "\033[92m"; Y = "\033[93m"; W = "\033[97m"
BOLD = "\033[1m"; DIM = "\033[2m"; RST = "\033[0m"
def col(c, s): return f"{c}{s}{RST}"

# ---------------------------------------------------------------------------
# Running-server mode: call /chat/stream, parse timing log
# ---------------------------------------------------------------------------

def _send_query(session_id: str, query: str) -> tuple[float, bool]:
    """Send one query to /chat/stream. Returns (wall_ms, success)."""
    t0 = time.perf_counter()
    try:
        with requests.post(
            f"{ORCHESTRATOR_URL}/chat/stream",
            json={"session_id": session_id, "message": query},
            stream=True,
            timeout=180,
        ) as resp:
            if resp.status_code != 200:
                return 0.0, False
            for _ in resp.iter_lines():
                pass  # drain stream
        wall_ms = (time.perf_counter() - t0) * 1000
        return wall_ms, True
    except Exception as e:
        print(col(R, f"    request failed: {e}"))
        return 0.0, False


def _parse_timing_log(log_path: Path) -> dict[str, list[int]]:
    """Parse structlog JSON lines for node_elapsed_ms events."""
    timings: dict[str, list[int]] = defaultdict(list)
    if not log_path.exists():
        return timings
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
            if event.get("event") == "node_elapsed_ms":
                node = event.get("node")
                ms   = event.get("elapsed_ms")
                if node and ms is not None:
                    timings[node].append(int(ms))
        except (json.JSONDecodeError, ValueError):
            # Console-rendered lines — not parseable in DEBUG mode
            if '"node_elapsed_ms"' in line or "node_elapsed_ms" in line:
                pass
    return timings


def run_server_mode():
    """Hit the live API N times, then parse the timing log."""
    import uuid
    print(f"\n{col(BOLD, 'Per-Node Latency Profiler — Server Mode')}")
    print(f"{col(DIM, f'Endpoint: {ORCHESTRATOR_URL}/chat/stream  |  N={N_QUERIES} queries')}")
    print(f"{col(DIM, f'Parsing structlog JSON from: {TIMING_LOG_FILE}')}\n")

    # Verify connectivity
    try:
        requests.get(f"{ORCHESTRATOR_URL}/health", timeout=5)
        print(col(G, f"  ✓ Orchestrator reachable at {ORCHESTRATOR_URL}\n"))
    except Exception as e:
        print(col(R, f"  ✗ Cannot reach orchestrator: {e}"))
        print(col(Y, "  Start the server: python orchestrator.py 2>&1 | tee /tmp/medicortex_timing.log"))
        print(col(Y, "  Ensure DEBUG=false in .env for JSON structlog output"))
        sys.exit(1)

    # Warn if log file missing
    if not TIMING_LOG_FILE.exists():
        print(col(Y, f"  ⚠ Timing log not found at {TIMING_LOG_FILE}"))
        print(col(Y, "  Re-run server with: python orchestrator.py 2>&1 | tee /tmp/medicortex_timing.log"))
        print(col(Y, "  Falling back to wall-clock timing only\n"))

    session_id = f"profiler-{uuid.uuid4().hex[:8]}"
    wall_times: list[float] = []

    for i, query in enumerate(PROFILE_QUERIES[:N_QUERIES], 1):
        print(f"  [{i:2d}/{N_QUERIES}] {query[:65]}", end="", flush=True)
        wall_ms, ok = _send_query(session_id, query)
        status = col(G, " ✓") if ok else col(R, " ✗")
        print(f"{status} {wall_ms:.0f}ms")
        if ok:
            wall_times.append(wall_ms)

    # Parse per-node timings from structlog
    timings = _parse_timing_log(TIMING_LOG_FILE)

    _print_results_table(timings, wall_times)


# ---------------------------------------------------------------------------
# Standalone mode: time each component independently
# ---------------------------------------------------------------------------

def run_standalone_mode():
    """Time each pipeline component directly without HTTP overhead."""
    print(f"\n{col(BOLD, 'Per-Node Latency Profiler — Standalone Mode')}")
    print(f"{col(DIM, f'Directly timing each component  |  N={N_QUERIES} queries per component')}\n")

    timings: dict[str, list[int]] = defaultdict(list)
    wall_times: list[float] = []

    # ── 1. PHI Redaction (Presidio) ──────────────────────────────────────
    print(col(BOLD, "  Timing: PHI Redaction (Presidio)..."))
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        analyzer  = AnalyzerEngine()
        anonymizer = AnonymizerEngine()
        sample_texts = [
            f"Patient John Smith, DOB 01/01/1975, asks: {q}" for q in PROFILE_QUERIES[:N_QUERIES]
        ]
        for text in sample_texts:
            t0 = time.perf_counter()
            results = analyzer.analyze(text=text, language="en")
            anonymizer.anonymize(text=text, analyzer_results=results)
            timings["analyze_privacy"].append(round((time.perf_counter() - t0) * 1000))
        print(col(G, f"    ✓ {len(timings['analyze_privacy'])} samples"))
    except Exception as e:
        print(col(Y, f"    ⚠ Presidio unavailable: {e}"))

    # ── 2. Scope Guard (Groq) ─────────────────────────────────────────────
    print(col(BOLD, "  Timing: Scope Guard (Groq llama-3.3-70b)..."))
    if not settings.GROQ_API_KEY:
        print(col(Y, "    ⚠ GROQ_API_KEY not set — skipping"))
    else:
        try:
            from groq import Groq
            client = Groq(api_key=settings.GROQ_API_KEY)
            scope_prompt = (
                "You are a medical AI scope filter. Reply ONLY '1' (in scope) or '0' (out of scope). "
                "IN SCOPE: medicine, diseases, symptoms, drugs, biology.\n"
                "OUT OF SCOPE: cooking, travel, programming, sports. Reply: 1 or 0"
            )
            for q in PROFILE_QUERIES[:N_QUERIES]:
                t0 = time.perf_counter()
                client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "system", "content": scope_prompt},
                              {"role": "user", "content": q}],
                    temperature=0, max_tokens=4,
                )
                timings["scope_guard"].append(round((time.perf_counter() - t0) * 1000))
                if len(timings["scope_guard"]) % 5 == 0:
                    time.sleep(1)  # rate-limit courtesy
            print(col(G, f"    ✓ {len(timings['scope_guard'])} samples"))
        except Exception as e:
            print(col(Y, f"    ⚠ Groq unavailable: {e}"))

    # ── 3. Router LLM (gemma4:31b-cloud) ─────────────────────────────────
    print(col(BOLD, "  Timing: Router (gemma4:31b-cloud via Ollama)..."))
    ollama_base = settings.OLLAMA_CLOUD_URL.removesuffix("/v1")
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_ollama import ChatOllama
        router_llm = ChatOllama(
            model="gemma4:31b-cloud",
            temperature=1.0, top_p=0.95, top_k=64, num_predict=64,
            base_url=ollama_base, timeout=90,
        )
        router_prompt = (
            "Select specialist agents. Return ONLY a JSON array from: "
            "[\"pubmed\", \"diagnosis\", \"report_analyzer\", \"patient\", \"pharmacology\"]. "
            "1-3 agents max."
        )
        for q in PROFILE_QUERIES[:N_QUERIES]:
            t0 = time.perf_counter()
            router_llm.invoke([
                SystemMessage(content=router_prompt),
                HumanMessage(content=f"Query: {q}"),
            ])
            timings["router"].append(round((time.perf_counter() - t0) * 1000))
        print(col(G, f"    ✓ {len(timings['router'])} samples"))
    except Exception as e:
        print(col(Y, f"    ⚠ Ollama/Router unavailable: {e}"))

    # ── 4. Agent tool planning / Phase 1 (gemma4:31b-cloud bind_tools) ───
    print(col(BOLD, "  Timing: Agent Phase 1 — Tool Planning (gemma4:31b-cloud)..."))
    try:
        from langchain_core.tools import tool as lc_tool
        from langchain_ollama import ChatOllama

        @lc_tool
        def search_pubmed(query: str) -> str:
            """Search PubMed for clinical evidence."""
            return "dummy"

        @lc_tool
        def get_drug_info(drug_name: str) -> str:
            """Get drug dosing, interactions and safety information."""
            return "dummy"

        planner_llm = ChatOllama(
            model="gemma4:31b-cloud",
            temperature=1.0, top_p=0.95, top_k=64, num_predict=256,
            base_url=ollama_base, timeout=90,
        ).bind_tools([search_pubmed, get_drug_info])

        for q in PROFILE_QUERIES[:N_QUERIES]:
            t0 = time.perf_counter()
            planner_llm.invoke([HumanMessage(content=q)])
            timings["agent_phase1"].append(round((time.perf_counter() - t0) * 1000))
        print(col(G, f"    ✓ {len(timings['agent_phase1'])} samples"))
    except Exception as e:
        print(col(Y, f"    ⚠ Agent Phase 1 timing unavailable: {e}"))

    # ── 5. Agent synthesis / Phase 2 (MedGemma — RunPod /runsync) ───────────
    print(col(BOLD, "  Timing: Agent Phase 2 — Synthesis (MedGemma @ RunPod)..."))
    medgemma_url = settings.MEDGEMMA_API_URL
    runpod_key   = getattr(settings, "RUNPOD_API_KEY", None)
    if not medgemma_url:
        print(col(Y, "    ⚠ MEDGEMMA_API_URL not set — skipping"))
    else:
        try:
            headers = {"Authorization": f"Bearer {runpod_key}"} if runpod_key else {}
            synthesis_prompt = (
                "You are a specialist medical agent. "
                "Provide a brief clinical answer (2-3 sentences). "
                "Query: {query}\n\nResponse:"
            )
            is_runpod = "runpod.ai" in medgemma_url or "/runsync" in medgemma_url
            for q in PROFILE_QUERIES[:N_QUERIES]:
                body = {"prompt": synthesis_prompt.format(query=q), "max_tokens": 256,
                        "temperature": 0.7, "top_k": 40, "top_p": 0.9, "min_p": 0.0}
                payload = {"input": body} if is_runpod else body
                t0 = time.perf_counter()
                requests.post(medgemma_url, json=payload, headers=headers, timeout=120)
                timings["agent_phase2"].append(round((time.perf_counter() - t0) * 1000))
            print(col(G, f"    ✓ {len(timings['agent_phase2'])} samples  [{medgemma_url[:50]}]"))
        except Exception as e:
            print(col(Y, f"    ⚠ MedGemma unavailable: {e}"))

    # ── 6. LLM-as-Judge (Groq) ────────────────────────────────────────────
    print(col(BOLD, "  Timing: LLM-as-Judge (Groq llama-3.3-70b)..."))
    if not settings.GROQ_API_KEY:
        print(col(Y, "    ⚠ GROQ_API_KEY not set — skipping"))
    else:
        try:
            from groq import Groq
            client_judge = Groq(api_key=settings.GROQ_API_KEY)
            judge_system = (
                "Score this medical response 1-5 on clinical accuracy. "
                "Reply ONLY with JSON: {\"score\": <1-5>, \"reason\": \"...\"}."
            )
            judge_responses = [
                "Metformin is a biguanide that reduces hepatic glucose production. Standard dose is 500-2000mg/day.",
                "Hypothyroidism presents with fatigue, weight gain, cold intolerance, and bradycardia.",
                "Elevated troponin indicates myocardial injury, most commonly acute MI.",
                "Beta blockers are contraindicated in cardiogenic shock, severe bradycardia, and high-degree AV block.",
                "ACE inhibitors reduce angiotensin II production, lowering afterload and blood pressure.",
            ] * 4  # repeat to get N_QUERIES samples

            for i, (q, r) in enumerate(zip(PROFILE_QUERIES[:N_QUERIES], judge_responses[:N_QUERIES])):
                t0 = time.perf_counter()
                client_judge.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": judge_system},
                        {"role": "user", "content": f"Query: {q}\n\nResponse: {r}"},
                    ],
                    temperature=0, max_tokens=128,
                )
                timings["reviewer"].append(round((time.perf_counter() - t0) * 1000))
                if (i + 1) % 5 == 0:
                    time.sleep(1)
            print(col(G, f"    ✓ {len(timings['reviewer'])} samples"))
        except Exception as e:
            print(col(Y, f"    ⚠ Groq judge unavailable: {e}"))

    # ── 7. Aggregation (gemma4:31b-cloud, longer output) ─────────────────────
    print(col(BOLD, "  Timing: Aggregation (gemma4:31b-cloud via Ollama)..."))
    try:
        from langchain_core.messages import HumanMessage as _HM
        from langchain_ollama import ChatOllama as _CO
        agg_llm = _CO(
            model="gemma4:31b-cloud",
            temperature=1.0, top_p=0.95, top_k=64, num_predict=512,
            base_url=ollama_base, timeout=90,
        )
        agg_prompt_tmpl = (
            "You are the MediCortex Interface. Format the following medical agent report "
            "into a human-readable Markdown response with ## headers and bullet points.\n\n"
            "Query: {query}\n\nAgent Report: The patient's condition relates to {query}. "
            "Standard clinical management applies.\n\nFormatted Response:"
        )
        for q in PROFILE_QUERIES[:N_QUERIES]:
            t0 = time.perf_counter()
            agg_llm.invoke([_HM(content=agg_prompt_tmpl.format(query=q))])
            timings["aggregator"].append(round((time.perf_counter() - t0) * 1000))
        print(col(G, f"    ✓ {len(timings['aggregator'])} samples"))
    except Exception as e:
        print(col(Y, f"    ⚠ Aggregator timing unavailable: {e}"))

    # ── 8. KB Retrieval (ArangoDB + vector index on homeserver) ──────────────
    print(col(BOLD, "  Timing: Entity Extraction + KB Retrieval (RAG-1A)..."))
    try:
        from knowledge_core.medical_engine import MedicalReasoningEngine
        engine = MedicalReasoningEngine()
        kb_queries = ["metformin", "hypertension", "troponin", "aspirin", "COPD",
                      "warfarin", "atorvastatin", "sepsis", "hypothyroidism", "DVT",
                      "lisinopril", "amoxicillin", "beta blocker", "ACE inhibitor", "diabetes",
                      "atrial fibrillation", "pulmonary embolism", "HbA1c", "creatinine", "INR"]
        for term in kb_queries[:N_QUERIES]:
            t0 = time.perf_counter()
            engine.query(term)
            timings["retrieve_knowledge"].append(round((time.perf_counter() - t0) * 1000))
        print(col(G, f"    ✓ {len(timings['retrieve_knowledge'])} samples"))
    except Exception as e:
        print(col(Y, f"    ⚠ KB Retrieval unavailable: {e}"))

    # ── 9. PHI Restoration (simple string replace, same as orchestrator) ─────
    print(col(BOLD, "  Timing: PHI Restoration..."))
    sample_output = "The patient <PERSON_1> was prescribed metformin 500mg twice daily."
    mapping = {"<PERSON_1>": "John Smith"}
    for _ in PROFILE_QUERIES[:N_QUERIES]:
        t0 = time.perf_counter()
        result = sample_output
        for placeholder, original in mapping.items():
            result = result.replace(placeholder, original)
        timings["restore_privacy"].append(round((time.perf_counter() - t0) * 1000))
    print(col(G, f"    ✓ {len(timings['restore_privacy'])} samples"))

    _print_results_table(timings, wall_times)


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

NODE_ORDER = [
    ("scope_guard",       "Scope Guard (Groq Llama-3.3-70B)"),
    ("analyze_privacy",   "PHI Redaction (Presidio)"),
    ("retrieve_knowledge","Entity Extraction + KB Retrieval (RAG-1A)"),
    ("router",            "Router (gemma4:31b-cloud)"),
    ("agent_phase1",      "Agent Execution — Phase 1 (gemma4:31b-cloud)"),
    ("agent_phase2",      "Agent Execution — Phase 2 (MedGemma)"),
    ("aggregator",        "Aggregation (gemma4:31b-cloud)"),
    ("reviewer",          "LLM-as-Judge (Groq Llama-3.3-70B)"),
    ("restore_privacy",   "PHI Restoration"),
]


def _stats(values: list[int]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = statistics.mean(values)
    sd   = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, sd


def _print_results_table(timings: dict[str, list[int]], wall_times: list[float]):
    print(f"\n{col(BOLD, '═' * 72)}")
    print(f"{col(BOLD, '  PER-NODE LATENCY BREAKDOWN — Table 6.8')}")
    print(f"  {col(DIM, 'Copy these values into main.tex tab:latency')}")
    print(f"{col(BOLD, '═' * 72)}\n")

    print(f"  {'Pipeline Node':<48} {'Mean (ms)':>10} {'SD (ms)':>10} {'N':>5}")
    print(f"  {'─'*48} {'─'*10} {'─'*10} {'─'*5}")

    for node_key, node_label in NODE_ORDER:
        vals = timings.get(node_key, [])
        mean, sd = _stats(vals)
        if math.isnan(mean):
            mean_s, sd_s = col(DIM, "N/A"), col(DIM, "N/A")
            n_s = col(DIM, "—")
        else:
            mean_s = f"{mean:,.0f}"
            sd_s   = f"{sd:,.0f}"
            n_s    = str(len(vals))
        print(f"  {node_label:<48} {mean_s:>10} {sd_s:>10} {n_s:>5}")

    if wall_times:
        wm, ws = _stats([round(w) for w in wall_times])
        print(f"  {'─'*48} {'─'*10} {'─'*10} {'─'*5}")
        print(f"  {col(BOLD,'End-to-End (full pipeline)'):<58} "
              f"{col(BOLD,f'{wm:,.0f}'):>10} {col(BOLD,f'{ws:,.0f}'):>10} "
              f"{col(BOLD,str(len(wall_times))):>5}")

    print(f"\n  {col(DIM, 'LaTeX snippet for main.tex (tab:latency):')}")
    for node_key, node_label in NODE_ORDER:
        vals = timings.get(node_key, [])
        mean, sd = _stats(vals)
        if math.isnan(mean):
            print(f"  {node_label} & \\textit{{[Pending]}} & \\textit{{[Pending]}} \\\\")
        else:
            print(f"  {node_label} & {mean:.0f} & {sd:.0f} \\\\")
    if wall_times:
        wm, ws = _stats([round(w) for w in wall_times])
        print(f"  \\textbf{{End-to-End (median query)}} & \\textbf{{{wm:.0f}}} & \\textbf{{{ws:.0f}}} \\\\")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediCortex Per-Node Latency Profiler")
    parser.add_argument(
        "--standalone", action="store_true",
        help="Time each component directly without a running server",
    )
    parser.add_argument(
        "--queries", type=int, default=N_QUERIES,
        help=f"Number of queries to send (default: {N_QUERIES})",
    )
    args = parser.parse_args()
    N_QUERIES = args.queries

    if args.standalone:
        run_standalone_mode()
    else:
        run_server_mode()
