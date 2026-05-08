"""
MediCortex Live Evaluation — Agentic RAG Metrics Report

Tests the actual gemma4:31b-cloud model with the exact production prompts to measure:
  1. Router Accuracy        — correct agent selection rate
  2. Router Reliability     — JSON parse rate + agent cap compliance
  3. Multi-turn Resolution  — follow-up pronoun/reference routing
  4. Aggregator Formatting  — valid Markdown output rate
  5. Aggregator Relevance   — response addresses the queried topic

Prints a metrics table + per-query breakdown at the end.

Usage:
  source .venv/bin/activate
  python3 tests/evaluation/run_live_metrics.py

Optional env override:
  MEDICORTEX_EVAL_QUERIES=20   # number of routing queries to sample (default 25)

Runtime: ~3-5 min (serial, 31B model @ homeserver)
"""

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL  = settings.OLLAMA_CLOUD_URL.removesuffix("/v1")
OLLAMA_MODEL     = "gemma4:31b-cloud"
INFERENCE_TIMEOUT = 90
N_ROUTING_QUERIES = int(os.getenv("MEDICORTEX_EVAL_QUERIES", "25"))
GROUND_TRUTH_PATH = Path(__file__).parent.parent / "resources" / "routing_ground_truth.json"
VALID_AGENTS      = {"pubmed", "diagnosis", "report_analyzer", "patient", "pharmacology"}

# Colour codes
R = "\033[91m"; G = "\033[92m"; Y = "\033[93m"; C = "\033[96m"; W = "\033[97m"
DIM = "\033[2m"; BOLD = "\033[1m"; RST = "\033[0m"

def col(c, s): return f"{c}{s}{RST}"

# ---------------------------------------------------------------------------
# Production prompts (exact copies from orchestrator.py and drug_agent.py)
# ---------------------------------------------------------------------------

ROUTER_SYSTEM_PROMPT = (
    "You are the MediCortex Orchestrator. Your ONLY job is to select which specialist agents to call.\n\n"
    "VALID KEYS — use ONLY these, never invent new ones:\n"
    "- \"pubmed\"          → latest research, studies, evidence, guidelines\n"
    "- \"diagnosis\"       → symptoms, differential diagnosis, clinical assessment\n"
    "- \"report_analyzer\" → lab results, imaging reports, ECG, pathology, uploaded files\n"
    "- \"patient\"         → specific named/identified patient history, records, vitals\n"
    "- \"pharmacology\"    → drugs, medications, dosing, interactions, side effects, contraindications\n\n"
    "RULES:\n"
    "1. Return ONLY a valid JSON array containing keys from the list above — never invent new keys\n"
    "2. Select 1-3 agents maximum\n"
    "3. NEVER route to 'pubmed' unless research papers or evidence are explicitly requested\n"
    "4. Symptoms/diagnosis only → [\"diagnosis\"]\n"
    "5. Named drug question → [\"pharmacology\"]\n"
    "6. Symptoms + treatment → [\"diagnosis\", \"pharmacology\"]\n"
    "7. Uploaded file/report/image → always include \"report_analyzer\"\n\n"
    "EXAMPLES:\n"
    "\"What is the dose of amoxicillin for a child?\" → [\"pharmacology\"]\n"
    "\"Patient has fever, cough and low SpO2\" → [\"diagnosis\"]\n"
    "\"Interpret this CBC: WBC 14k, Hgb 8.2\" → [\"report_analyzer\"]\n"
    "\"Latest trials on checkpoint inhibitors\" → [\"pubmed\"]\n"
    "\"John Smith's last visit and his beta blocker dose\" → [\"patient\", \"pharmacology\"]\n"
    "\"Is metformin safe in CKD stage 4?\" → [\"pharmacology\", \"pubmed\"]\n"
    "\"45yo with chest pain and diaphoresis — diagnosis and treatment?\" → [\"diagnosis\", \"pharmacology\"]\n\n"
    "FOLLOW-UP RESOLUTION — if the query uses pronouns (his/her/their/it/the patient/the drug), "
    "resolve using the Recent Session Context below:\n"
    "- Prior [patient] + asks about drugs → [\"pharmacology\"]\n"
    "- Prior [diagnosis] + asks about treatment → [\"pharmacology\"]\n\n"
    "Return ONLY the JSON array, no explanation, no prose."
)

AGGREGATOR_FORMAT_PROMPT = (
    "You are the MediCortex Interface. Format the following medical agent reports into "
    "a beautiful, human-readable Markdown response.\n"
    "Rules:\n"
    "- Use ## headers for each topic\n"
    "- Use bullet points for lists\n"
    "- Bold key terms\n"
    "- Keep it medically accurate and avoid patient-specific advice\n"
    "- End with a 'Sources' section if source URLs are mentioned\n\n"
)

# ---------------------------------------------------------------------------
# Multi-turn routing scenarios
# ---------------------------------------------------------------------------

MULTITURN_SCENARIOS = [
    {
        "id": "MT-01",
        "description": "Prior pharmacology + pronoun 'it' → pharmacology",
        "context": "Prior session: [pharmacology] — user asked about metformin for diabetes.",
        "query": "Is it safe during pregnancy?",
        "expected": ["pharmacology"],
    },
    {
        "id": "MT-02",
        "description": "Prior diagnosis + asks about treatment → pharmacology",
        "context": "Prior session: [diagnosis] — user asked about hypertension symptoms.",
        "query": "What drugs are used to treat it?",
        "expected": ["pharmacology"],
    },
    {
        "id": "MT-03",
        "description": "Prior patient + asks about drug dose → pharmacology",
        "context": "Prior session: [patient] — retrieved records for patient John Smith.",
        "query": "What is his current beta blocker dose?",
        "expected": ["pharmacology", "patient"],
    },
    {
        "id": "MT-04",
        "description": "Prior pubmed + asks about same drug study → pubmed",
        "context": "Prior session: [pubmed] — user asked about SGLT2 inhibitor trials.",
        "query": "What were the primary endpoints in that trial?",
        "expected": ["pubmed"],
    },
    {
        "id": "MT-05",
        "description": "Prior diagnosis + follow-up symptom → diagnosis",
        "context": "Prior session: [diagnosis] — user asked about chest pain differentials.",
        "query": "Could it also explain the shortness of breath?",
        "expected": ["diagnosis"],
    },
]

# ---------------------------------------------------------------------------
# Aggregator test scenarios
# ---------------------------------------------------------------------------

AGGREGATOR_SCENARIOS = [
    {
        "id": "AGG-01",
        "query": "What is atorvastatin used for?",
        "agent_output": (
            "## Pharmacology Agent Response\n"
            "Atorvastatin is a statin that lowers LDL cholesterol. "
            "Standard dose: 10–80mg once daily. "
            "Side effects: myalgia, elevated liver enzymes. "
            "Source: drugs.com"
        ),
        "expected_keywords": ["atorvastatin", "statin", "cholesterol", "dose"],
    },
    {
        "id": "AGG-02",
        "query": "Patient has chest pain and sweating — diagnosis and treatment?",
        "agent_output": (
            "## Diagnosis Agent Response\n"
            "Chest pain with diaphoresis is classic for acute coronary syndrome (ACS).\n\n"
            "## Pharmacology Agent Response\n"
            "Aspirin 300mg loading dose + nitroglycerin sublingual for ACS. "
            "Beta blockers and anticoagulation are standard of care."
        ),
        "expected_keywords": ["chest", "aspirin", "acs"],
    },
    {
        "id": "AGG-03",
        "query": "What does recent research say about GLP-1 agonists?",
        "agent_output": (
            "## PubMed Agent Response\n"
            "SUSTAIN-6 trial: semaglutide reduced MACE by 26% in T2D patients with high CV risk. "
            "LEADER trial: liraglutide reduced CV death by 22%. "
            "Source: NEJM 2016, NEJM 2019."
        ),
        "expected_keywords": ["glp-1", "semaglutide", "trial"],
    },
    {
        "id": "AGG-04",
        "query": "CBC shows WBC 14k, Hgb 8.2 — interpret this.",
        "agent_output": (
            "## Report Analyzer Response\n"
            "WBC 14,000 cells/µL — elevated (normal 4,500–11,000). Suggests infection or inflammation.\n"
            "Hgb 8.2 g/dL — below normal (normal 13.5–17.5 male / 12.0–15.5 female). Indicates anaemia."
        ),
        "expected_keywords": ["wbc", "hgb", "elevated", "anaemia"],
    },
    {
        "id": "AGG-05",
        "query": "What are the interactions between warfarin and aspirin?",
        "agent_output": (
            "## Pharmacology Agent Response\n"
            "**Major interaction** — concurrent warfarin + aspirin significantly increases bleeding risk. "
            "Aspirin inhibits platelet aggregation; warfarin inhibits clotting factors. "
            "Combined use increases risk of GI or intracranial haemorrhage. "
            "Source: FDA Drug Safety Communication, Drugs.com"
        ),
        "expected_keywords": ["warfarin", "aspirin", "bleeding", "interaction"],
    },
]

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

@dataclass
class RouteResult:
    id: str
    query: str
    expected: list[str]
    actual: list[str]
    raw_output: str
    parse_ok: bool
    latency_s: float
    correct: bool = field(init=False)
    cap_ok: bool = field(init=False)
    unknown_agents: list[str] = field(init=False)

    def __post_init__(self):
        self.correct = bool(set(self.actual) & set(self.expected))
        self.cap_ok = len(self.actual) <= 3
        self.unknown_agents = [a for a in self.actual if a not in VALID_AGENTS]


@dataclass
class AggResult:
    id: str
    query: str
    output: str
    latency_s: float
    has_headers: bool = field(init=False)
    non_empty: bool = field(init=False)
    keyword_hits: list[str] = field(default_factory=list)
    keywords_expected: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.has_headers = bool(re.search(r'^#{1,3} ', self.output, re.MULTILINE))
        self.non_empty = len(self.output.strip()) > 50

    @property
    def keyword_rate(self) -> float:
        if not self.keywords_expected:
            return 1.0
        return len(self.keyword_hits) / len(self.keywords_expected)


def _call_router(llm: ChatOllama, query: str, context: str = "") -> tuple[list[str], str, bool]:
    """Returns (parsed_agents, raw_output, parse_ok)."""
    user_msg = f"User Query: {query}"
    if context:
        user_msg += f"\n\nRecent Session Context (use this to resolve follow-up references):\n{context}"
    user_msg += "\n\nKnowledge Core Context (for your awareness): "

    messages = [SystemMessage(content=ROUTER_SYSTEM_PROMPT), HumanMessage(content=user_msg)]
    raw = llm.invoke(messages).content
    clean = raw.replace("```json", "").replace("```", "").strip().replace("'", '"')

    try:
        parsed = json.loads(clean)
        if isinstance(parsed, list):
            return parsed, raw, True
        return ["diagnosis"], raw, False
    except json.JSONDecodeError:
        m = re.search(r'\[.*?\]', clean, re.DOTALL)
        if m:
            try:
                return json.loads(m.group()), raw, True
            except Exception:
                pass
        return ["diagnosis"], raw, False


def _call_aggregator(llm: ChatOllama, query: str, agent_output: str) -> str:
    prompt = (
        AGGREGATOR_FORMAT_PROMPT
        + f"Current User Query: {query}\n\n"
        + f"Agent Reports:\n{agent_output}\n\nFormatted Response:"
    )
    return llm.invoke([HumanMessage(content=prompt)]).content


# ---------------------------------------------------------------------------
# Section runners
# ---------------------------------------------------------------------------

def run_routing_accuracy(llm: ChatOllama) -> list[RouteResult]:
    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text())

    # Sample evenly across agent types to get N_ROUTING_QUERIES entries
    # Sort by agent so we get even distribution, then pick every N//50-th
    step = max(1, len(ground_truth) // N_ROUTING_QUERIES)
    sample = ground_truth[::step][:N_ROUTING_QUERIES]

    results = []
    for i, entry in enumerate(sample, 1):
        qid = entry["id"]
        query = entry["query"]
        expected = entry.get("expected", entry.get("expected_agents", []))

        print(f"  [{i:2d}/{len(sample)}] {qid}: {query[:65]}...", end="", flush=True)
        t0 = time.perf_counter()
        agents, raw, parse_ok = _call_router(llm, query)
        latency = time.perf_counter() - t0

        r = RouteResult(
            id=qid, query=query, expected=expected,
            actual=agents, raw_output=raw,
            parse_ok=parse_ok, latency_s=round(latency, 1),
        )
        status = col(G, "✓") if r.correct else col(R, "✗")
        print(f" {status} {agents} ({latency:.1f}s)")
        results.append(r)

    return results


def run_multiturn(llm: ChatOllama) -> list[RouteResult]:
    results = []
    for i, sc in enumerate(MULTITURN_SCENARIOS, 1):
        print(f"  [{i}/{len(MULTITURN_SCENARIOS)}] {sc['id']}: {sc['description']}", end="", flush=True)
        t0 = time.perf_counter()
        agents, raw, parse_ok = _call_router(llm, sc["query"], context=sc["context"])
        latency = time.perf_counter() - t0

        r = RouteResult(
            id=sc["id"], query=sc["query"], expected=sc["expected"],
            actual=agents, raw_output=raw,
            parse_ok=parse_ok, latency_s=round(latency, 1),
        )
        status = col(G, "✓") if r.correct else col(R, "✗")
        print(f" {status} got {agents} (expected {sc['expected']}) ({latency:.1f}s)")
        results.append(r)
    return results


def run_aggregator(llm: ChatOllama) -> list[AggResult]:
    # Use a longer num_predict for aggregator (needs to produce a full response)
    agg_llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        num_predict=512,
        base_url=OLLAMA_BASE_URL,
        timeout=INFERENCE_TIMEOUT,
    )
    results = []
    for i, sc in enumerate(AGGREGATOR_SCENARIOS, 1):
        print(f"  [{i}/{len(AGGREGATOR_SCENARIOS)}] {sc['id']}: {sc['query'][:65]}", end="", flush=True)
        t0 = time.perf_counter()
        output = _call_aggregator(agg_llm, sc["query"], sc["agent_output"])
        latency = time.perf_counter() - t0

        lower = output.lower()
        hits = [kw for kw in sc["expected_keywords"] if kw in lower]

        r = AggResult(
            id=sc["id"], query=sc["query"], output=output,
            latency_s=round(latency, 1),
            keyword_hits=hits,
            keywords_expected=sc["expected_keywords"],
        )
        status = col(G, "✓") if r.has_headers and r.non_empty and r.keyword_rate >= 0.5 else col(R, "✗")
        print(f" {status} headers={r.has_headers} keywords={len(hits)}/{len(sc['expected_keywords'])} ({latency:.1f}s)")
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Metrics table printer
# ---------------------------------------------------------------------------

def _pct(n, d): return f"{100*n/d:.1f}%" if d else "N/A"
def _bar(n, d, width=20):
    filled = int(width * n / d) if d else 0
    colour = G if n/d >= 0.8 else (Y if n/d >= 0.6 else R)
    return col(colour, "█" * filled) + col(DIM, "░" * (width - filled))


def print_report(
    routing: list[RouteResult],
    multiturn: list[RouteResult],
    aggregator: list[AggResult],
    total_elapsed: float,
):
    print(f"\n{col(BOLD+C, '╔' + '═'*62 + '╗')}")
    print(f"{col(BOLD+C, '║')}  {col(BOLD+W, 'MediCortex Agentic RAG — Live Evaluation Report')}         {col(BOLD+C, '║')}")
    print(f"{col(BOLD+C, '║')}  {col(DIM, f'Model: {OLLAMA_MODEL} @ {OLLAMA_BASE_URL}'):<63}{col(BOLD+C, '║')}")
    print(f"{col(BOLD+C, '╚' + '═'*62 + '╝')}\n")

    # ── 1. Routing accuracy ──────────────────────────────────────────
    n = len(routing)
    correct   = sum(1 for r in routing if r.correct)
    parse_ok  = sum(1 for r in routing if r.parse_ok)
    cap_ok    = sum(1 for r in routing if r.cap_ok)
    no_unk    = sum(1 for r in routing if not r.unknown_agents)
    avg_lat   = sum(r.latency_s for r in routing) / n if n else 0

    # Per-agent accuracy
    agent_stats: dict[str, dict] = {}
    for r in routing:
        for exp_agent in r.expected:
            if exp_agent not in agent_stats:
                agent_stats[exp_agent] = {"total": 0, "correct": 0}
            agent_stats[exp_agent]["total"] += 1
            if exp_agent in r.actual:
                agent_stats[exp_agent]["correct"] += 1

    print(col(BOLD, "  ① ROUTING ACCURACY") + col(DIM, f"  ({n} queries from ground truth)"))
    print(f"  {'Metric':<38} {'Score':<10} {'Bar'}")
    print(f"  {'─'*38} {'─'*10} {'─'*22}")

    rows = [
        ("Correct agent selected",     correct,  n),
        ("Valid JSON output",           parse_ok, n),
        ("Agent cap compliance (≤3)",   cap_ok,   n),
        ("No unknown agent keys",       no_unk,   n),
    ]
    for label, num, den in rows:
        pct = _pct(num, den)
        bar = _bar(num, den)
        score_col = G if num/den >= 0.8 else (Y if num/den >= 0.6 else R)
        print(f"  {label:<38} {col(score_col, f'{num}/{den} ({pct})'):<22} {bar}")

    print(f"\n  {'Per-agent accuracy:':<38}")
    for agent, s in sorted(agent_stats.items()):
        pct = _pct(s['correct'], s['total'])
        bar = _bar(s['correct'], s['total'], width=12)
        score_col = G if s['correct']/s['total'] >= 0.8 else (Y if s['correct']/s['total'] >= 0.6 else R)
        score_str = f"{s['correct']}/{s['total']} ({pct})"
        print(f"    {agent:<34} {col(score_col, score_str):<22} {bar}")

    print(f"\n  Avg latency per call: {col(W, f'{avg_lat:.1f}s')}\n")

    # ── 2. Multi-turn resolution ─────────────────────────────────────
    mt_n = len(multiturn)
    mt_correct = sum(1 for r in multiturn if r.correct)
    print(col(BOLD, "  ② MULTI-TURN FOLLOW-UP RESOLUTION") + col(DIM, f"  ({mt_n} scenarios)"))
    print(f"  {'Metric':<38} {'Score':<10} {'Bar'}")
    print(f"  {'─'*38} {'─'*10} {'─'*22}")
    pct = _pct(mt_correct, mt_n)
    bar = _bar(mt_correct, mt_n)
    score_col = G if mt_correct/mt_n >= 0.8 else (Y if mt_correct/mt_n >= 0.6 else R)
    print(f"  {'Pronoun/reference correctly resolved':<38} {col(score_col, f'{mt_correct}/{mt_n} ({pct})'):<22} {bar}")

    failures = [r for r in multiturn if not r.correct]
    if failures:
        print(f"\n  {col(Y, 'Failed scenarios:')}")
        for r in failures:
            print(f"    {r.id}: got {r.actual}, expected {r.expected}")
    print()

    # ── 3. Aggregator quality ────────────────────────────────────────
    agg_n = len(aggregator)
    agg_headers  = sum(1 for r in aggregator if r.has_headers)
    agg_nonempty = sum(1 for r in aggregator if r.non_empty)
    agg_kw       = sum(1 for r in aggregator if r.keyword_rate >= 0.5)
    agg_full_kw  = sum(r.keyword_rate for r in aggregator) / agg_n if agg_n else 0
    agg_avg_lat  = sum(r.latency_s for r in aggregator) / agg_n if agg_n else 0

    print(col(BOLD, "  ③ AGGREGATOR FORMATTING QUALITY") + col(DIM, f"  ({agg_n} scenarios)"))
    print(f"  {'Metric':<38} {'Score':<10} {'Bar'}")
    print(f"  {'─'*38} {'─'*10} {'─'*22}")

    agg_rows = [
        ("Markdown headers present",       agg_headers,  agg_n),
        ("Non-empty response (>50 chars)",  agg_nonempty, agg_n),
        ("Query topic coverage (≥50% kw)", agg_kw,       agg_n),
    ]
    for label, num, den in agg_rows:
        pct = _pct(num, den)
        bar = _bar(num, den)
        score_col = G if num/den >= 0.8 else (Y if num/den >= 0.6 else R)
        print(f"  {label:<38} {col(score_col, f'{num}/{den} ({pct})'):<22} {bar}")

    print(f"  {'Avg keyword coverage':<38} {col(W, f'{agg_full_kw*100:.1f}%')}")
    print(f"  Avg latency per call: {col(W, f'{agg_avg_lat:.1f}s')}\n")

    # ── 4. Overall summary ───────────────────────────────────────────
    total_checks = n + mt_n + agg_n
    total_pass   = correct + mt_correct + agg_kw

    overall_pct  = 100 * total_pass / total_checks if total_checks else 0
    overall_col  = G if overall_pct >= 80 else (Y if overall_pct >= 60 else R)

    print(col(BOLD, "  ④ OVERALL SUMMARY"))
    print(f"  {'─'*62}")
    print(f"  Total checks:   {total_checks}")
    print(f"  Passed:         {col(overall_col, str(total_pass))}")
    print(f"  Overall score:  {col(BOLD + overall_col, f'{overall_pct:.1f}%')}")
    print(f"  Total runtime:  {col(W, f'{total_elapsed:.0f}s')}")
    print(f"  {'─'*62}\n")

    # Verdict
    if overall_pct >= 85:
        verdict = col(BOLD + G, "  ✓ PASS — System is operating effectively")
    elif overall_pct >= 65:
        verdict = col(BOLD + Y, "  ⚠ PARTIAL — Some routing or formatting gaps detected")
    else:
        verdict = col(BOLD + R, "  ✗ FAIL — Significant routing or quality issues found")
    print(verdict + "\n")

    # Missed routing breakdown
    missed = [r for r in routing if not r.correct]
    if missed:
        print(col(Y, f"  Routing misses ({len(missed)}):"))
        for r in missed:
            print(f"    {r.id}: {col(DIM, r.query[:60])}")
            print(f"          expected={r.expected}  got={r.actual}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{col(BOLD, '  MediCortex Live Evaluation')}")
    print(f"  {col(DIM, f'Model: {OLLAMA_MODEL} @ {OLLAMA_BASE_URL}')}\n")

    # Connectivity check
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if OLLAMA_MODEL not in models:
            print(col(R, f"  ✗ {OLLAMA_MODEL} not found on homeserver. Available: {models}"))
            sys.exit(1)
        print(col(G, f"  ✓ Homeserver reachable — {OLLAMA_MODEL} available\n"))
    except Exception as e:
        print(col(R, f"  ✗ Cannot reach homeserver: {e}"))
        sys.exit(1)

    # Router LLM — short num_predict (only need a JSON array)
    router_llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        num_predict=64,
        base_url=OLLAMA_BASE_URL,
        timeout=INFERENCE_TIMEOUT,
    )

    t_start = time.perf_counter()

    # ── Section 1: Routing accuracy ──────────────────────────────────
    print(col(BOLD, "  Running ① Routing Accuracy..."))
    routing_results = run_routing_accuracy(router_llm)

    # ── Section 2: Multi-turn resolution ─────────────────────────────
    print(f"\n{col(BOLD, '  Running ② Multi-turn Resolution...')}")
    multiturn_results = run_multiturn(router_llm)

    # ── Section 3: Aggregator quality ────────────────────────────────
    print(f"\n{col(BOLD, '  Running ③ Aggregator Formatting Quality...')}")
    aggregator_results = run_aggregator(router_llm)

    total_elapsed = time.perf_counter() - t_start

    # ── Print report ──────────────────────────────────────────────────
    print_report(routing_results, multiturn_results, aggregator_results, total_elapsed)


if __name__ == "__main__":
    main()
