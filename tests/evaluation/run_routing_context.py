"""
MediCortex Routing Context Evaluator — Table 6.7

Measures routing accuracy on multi-turn follow-up queries:
  - WITHOUT routing context (blind follow-up resolution)
  - WITH routing context (session-aware follow-up resolution)

Produces the two-row comparison for Table 6.7 in the dissertation.

Usage:
  source .venv/bin/activate
  python3 tests/evaluation/run_routing_context.py

Requirements:
  - homeserver Ollama reachable (gemma4:31b-cloud)

Runtime: ~4-6 min (40 router calls at ~3s each)
"""

import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL   = settings.OLLAMA_CLOUD_URL.removesuffix("/v1")
OLLAMA_MODEL      = "gemma4:31b-cloud"
INFERENCE_TIMEOUT = 90
VALID_AGENTS      = {"pubmed", "diagnosis", "report_analyzer", "patient", "pharmacology"}

# ---------------------------------------------------------------------------
# Router system prompt (exact copy from orchestrator.py)
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
    "FOLLOW-UP RESOLUTION — if the query uses pronouns (his/her/their/it/the patient/the drug), "
    "resolve using the Recent Session Context below:\n"
    "- Prior [patient] + asks about drugs → [\"pharmacology\"]\n"
    "- Prior [diagnosis] + asks about treatment → [\"pharmacology\"]\n\n"
    "Return ONLY the JSON array, no explanation, no prose."
)

# ---------------------------------------------------------------------------
# Multi-turn test set (20 follow-up scenarios)
# Each has: follow_up query, prior_context (what was discussed), expected_agents
# ---------------------------------------------------------------------------

SCENARIOS = [
    # Drug follow-ups (context: a drug was previously discussed)
    {
        "id": "MT-01",
        "description": "metformin → pregnancy safety",
        "context": "Prior session: [pharmacology] — user asked about metformin for type 2 diabetes.",
        "query": "Is it safe during pregnancy?",
        "expected": ["pharmacology"],
    },
    {
        "id": "MT-02",
        "description": "atorvastatin → side effects",
        "context": "Prior session: [pharmacology] — user asked about atorvastatin for high cholesterol.",
        "query": "What are the common side effects?",
        "expected": ["pharmacology"],
    },
    {
        "id": "MT-03",
        "description": "amoxicillin → dose in renal failure",
        "context": "Prior session: [pharmacology] — user asked about amoxicillin for strep throat.",
        "query": "Does the dose need adjustment in renal failure?",
        "expected": ["pharmacology"],
    },
    {
        "id": "MT-04",
        "description": "lisinopril → drug interactions",
        "context": "Prior session: [pharmacology] — user asked about lisinopril for hypertension.",
        "query": "What drugs does it interact with?",
        "expected": ["pharmacology"],
    },
    {
        "id": "MT-05",
        "description": "warfarin → INR monitoring",
        "context": "Prior session: [pharmacology] — user asked about warfarin for atrial fibrillation.",
        "query": "How often should INR be monitored?",
        "expected": ["pharmacology"],
    },

    # Diagnosis follow-ups (context: a condition/symptoms was discussed)
    {
        "id": "MT-06",
        "description": "hypertension → treatment options",
        "context": "Prior session: [diagnosis] — user asked about hypertension symptoms and staging.",
        "query": "What drugs are used to treat it?",
        "expected": ["pharmacology"],
    },
    {
        "id": "MT-07",
        "description": "chest pain differentials → follow-up SOB",
        "context": "Prior session: [diagnosis] — user asked about chest pain differentials.",
        "query": "Could it also explain the shortness of breath?",
        "expected": ["diagnosis"],
    },
    {
        "id": "MT-08",
        "description": "PE diagnosis → anticoagulation",
        "context": "Prior session: [diagnosis] — user asked about pulmonary embolism diagnosis.",
        "query": "What anticoagulant should be started first?",
        "expected": ["pharmacology"],
    },
    {
        "id": "MT-09",
        "description": "diabetes → long-term complications",
        "context": "Prior session: [diagnosis] — user asked about type 2 diabetes diagnosis criteria.",
        "query": "What are the long-term complications?",
        "expected": ["diagnosis"],
    },
    {
        "id": "MT-10",
        "description": "hypothyroidism → TSH monitoring",
        "context": "Prior session: [diagnosis] — user asked about hypothyroidism symptoms.",
        "query": "How often should TSH be checked after starting treatment?",
        "expected": ["pharmacology"],
    },

    # Patient record follow-ups
    {
        "id": "MT-11",
        "description": "patient records → drug dose query",
        "context": "Prior session: [patient] — retrieved records for patient John Smith with hypertension.",
        "query": "What is his current beta blocker dose?",
        "expected": ["pharmacology", "patient"],
    },
    {
        "id": "MT-12",
        "description": "patient records → allergy check",
        "context": "Prior session: [patient] — user queried patient Emily Davis.",
        "query": "Does she have any documented penicillin allergy?",
        "expected": ["patient"],
    },
    {
        "id": "MT-13",
        "description": "patient CBC → follow-up on abnormal Hgb",
        "context": "Prior session: [report_analyzer] — interpreted CBC showing Hgb 7.8 g/dL.",
        "query": "What could cause this level of anaemia?",
        "expected": ["diagnosis"],
    },
    {
        "id": "MT-14",
        "description": "report_analyzer → follow-up treatment",
        "context": "Prior session: [report_analyzer] — reviewed chest X-ray showing bilateral infiltrates.",
        "query": "What antibiotic regimen would be appropriate?",
        "expected": ["pharmacology"],
    },
    {
        "id": "MT-15",
        "description": "patient records → recent medication changes",
        "context": "Prior session: [patient] — retrieved patient records for Michael Brown, T2DM.",
        "query": "Was his insulin dose changed recently?",
        "expected": ["patient"],
    },

    # PubMed follow-ups
    {
        "id": "MT-16",
        "description": "SGLT2 trial → primary endpoints",
        "context": "Prior session: [pubmed] — user asked about SGLT2 inhibitor cardiovascular trials.",
        "query": "What were the primary endpoints in that trial?",
        "expected": ["pubmed"],
    },
    {
        "id": "MT-17",
        "description": "GLP-1 studies → adverse events",
        "context": "Prior session: [pubmed] — user asked about GLP-1 agonist weight loss studies.",
        "query": "What adverse events were reported?",
        "expected": ["pubmed"],
    },

    # Ambiguous pronoun scenarios
    {
        "id": "MT-18",
        "description": "statin → hepatotoxicity risk",
        "context": "Prior session: [pharmacology] — user asked about rosuvastatin dosing.",
        "query": "Is there a risk of hepatotoxicity with this?",
        "expected": ["pharmacology"],
    },
    {
        "id": "MT-19",
        "description": "prior diagnosis → drug selection",
        "context": "Prior session: [diagnosis] — user asked about heart failure with reduced EF.",
        "query": "Which agents have mortality benefit in this condition?",
        "expected": ["pharmacology"],
    },
    {
        "id": "MT-20",
        "description": "prior report → clinical significance",
        "context": "Prior session: [report_analyzer] — interpreted ECG showing new LBBB.",
        "query": "What is the clinical significance of this finding?",
        "expected": ["diagnosis"],
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

R = "\033[91m"; G = "\033[92m"; Y = "\033[93m"; W = "\033[97m"
BOLD = "\033[1m"; DIM = "\033[2m"; RST = "\033[0m"
def col(c, s): return f"{c}{s}{RST}"


@dataclass
class RouteResult:
    id: str
    query: str
    expected: list
    actual: list
    latency_s: float
    with_context: bool
    correct: bool = field(init=False)

    def __post_init__(self):
        self.correct = bool(set(self.actual) & set(self.expected))


def _call_router(llm: ChatOllama, query: str, context: str = "") -> tuple[list[str], float]:
    user_msg = f"User Query: {query}"
    if context:
        user_msg += f"\n\nRecent Session Context (use this to resolve follow-up references):\n{context}"
    user_msg += "\n\nKnowledge Core Context (for your awareness): "

    messages = [SystemMessage(content=ROUTER_SYSTEM_PROMPT), HumanMessage(content=user_msg)]
    t0 = time.perf_counter()
    raw = llm.invoke(messages).content
    latency = time.perf_counter() - t0

    clean = raw.replace("```json", "").replace("```", "").strip().replace("'", '"')
    try:
        parsed = json.loads(clean)
        if isinstance(parsed, list):
            return parsed, latency
    except json.JSONDecodeError:
        m = re.search(r'\[.*?\]', clean, re.DOTALL)
        if m:
            try:
                return json.loads(m.group()), latency
            except Exception:
                pass
    return ["diagnosis"], latency


def _pct(n, d): return f"{100*n/d:.1f}%" if d else "N/A"
def _bar(n, d, width=16):
    filled = int(width * n / d) if d else 0
    c = G if n/d >= 0.8 else (Y if n/d >= 0.6 else R)
    return col(c, "█" * filled) + col(DIM, "░" * (width - filled))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{col(BOLD, 'MediCortex Routing Context Evaluation — Table 6.7')}")
    print(f"{col(DIM, f'Model: {OLLAMA_MODEL} @ {OLLAMA_BASE_URL}  |  {len(SCENARIOS)} scenarios × 2 conditions')}\n")

    import requests as _req
    try:
        r = _req.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if OLLAMA_MODEL not in models:
            print(col(R, f"ERROR: {OLLAMA_MODEL} not found. Available: {models}"))
            sys.exit(1)
        print(col(G, f"  ✓ Homeserver reachable — {OLLAMA_MODEL} loaded\n"))
    except Exception as e:
        print(col(R, f"ERROR: Cannot reach homeserver: {e}"))
        sys.exit(1)

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        num_predict=64,
        base_url=OLLAMA_BASE_URL,
        timeout=INFERENCE_TIMEOUT,
    )

    # ── WITHOUT context ──────────────────────────────────────────────
    print(col(BOLD, "  ① Without Routing Context (baseline)"))
    without_results: list[RouteResult] = []
    for i, sc in enumerate(SCENARIOS, 1):
        print(f"  [{i:2d}/{len(SCENARIOS)}] {sc['id']}: {sc['description']}", end="", flush=True)
        agents, lat = _call_router(llm, sc["query"], context="")
        r = RouteResult(id=sc["id"], query=sc["query"], expected=sc["expected"],
                        actual=agents, latency_s=lat, with_context=False)
        status = col(G, " ✓") if r.correct else col(R, " ✗")
        print(f"{status} got {agents} (expected {sc['expected']}) ({lat:.1f}s)")
        without_results.append(r)

    print()

    # ── WITH context ─────────────────────────────────────────────────
    print(col(BOLD, "  ② With Routing Context (ours)"))
    with_results: list[RouteResult] = []
    for i, sc in enumerate(SCENARIOS, 1):
        print(f"  [{i:2d}/{len(SCENARIOS)}] {sc['id']}: {sc['description']}", end="", flush=True)
        agents, lat = _call_router(llm, sc["query"], context=sc["context"])
        r = RouteResult(id=sc["id"], query=sc["query"], expected=sc["expected"],
                        actual=agents, latency_s=lat, with_context=True)
        status = col(G, " ✓") if r.correct else col(R, " ✗")
        print(f"{status} got {agents} (expected {sc['expected']}) ({lat:.1f}s)")
        with_results.append(r)

    # ── Compute metrics ──────────────────────────────────────────────
    n = len(SCENARIOS)

    def _agent_acc(results):
        return sum(1 for r in results if r.correct)

    # "Follow-up resolved" = pronoun/reference queries where correct agent selected
    # All scenarios in this set are follow-up queries (they all use pronouns/implicit refs)
    def _followup_resolved(results):
        return sum(1 for r in results if r.correct)

    wo_correct = _agent_acc(without_results)
    wi_correct = _agent_acc(with_results)

    # ── Print results table ──────────────────────────────────────────
    print(f"\n{col(BOLD, '═' * 70)}")
    print(f"{col(BOLD, '  ROUTING CONTEXT EVALUATION RESULTS — Table 6.7')}")
    print(f"  {col(DIM, 'Copy these values into main.tex tab:routingctx')}")
    print(f"{col(BOLD, '═' * 70)}\n")

    print(f"  {'Condition':<30} {'Correct Agent (%)':>20} {'Follow-Up Resolved (%)':>22}")
    print(f"  {'─'*30} {'─'*20} {'─'*22}")

    wo_pct  = 100 * wo_correct / n
    wi_pct  = 100 * wi_correct / n
    wo_col  = G if wo_pct >= 80 else (Y if wo_pct >= 60 else R)
    wi_col  = G if wi_pct >= 80 else (Y if wi_pct >= 60 else R)

    print(f"  {'Without routing context':<30} {col(wo_col, f'{wo_correct}/{n} ({wo_pct:.1f}%)'):>30} "
          f"{col(wo_col, f'{wo_correct}/{n} ({wo_pct:.1f}%)'):>32}")
    print(f"  {'With routing context (ours)':<30} {col(wi_col, f'{wi_correct}/{n} ({wi_pct:.1f}%)'):>30} "
          f"{col(wi_col, f'{wi_correct}/{n} ({wi_pct:.1f}%)'):>32}")

    delta = wi_pct - wo_pct
    delta_col = G if delta > 0 else R
    print(f"\n  Δ (with − without context): {col(delta_col, f'{delta:+.1f} pp')}")

    print(f"\n  {col(DIM, 'LaTeX snippet for main.tex:')}")
    print(f"  Without routing context & {wo_pct:.1f}\\% & {wo_pct:.1f}\\% \\\\")
    print(f"  With routing context (ours) & {wi_pct:.1f}\\% & {wi_pct:.1f}\\% \\\\")
    print()

    # ── Failure breakdown for WITH context ──────────────────────────
    failures = [r for r in with_results if not r.correct]
    if failures:
        print(col(Y, f"  WITH context failures ({len(failures)}):"))
        for r in failures:
            print(f"    {r.id}: got {r.actual}, expected {r.expected}")
            print(f"          query: {r.query}")
        print()


if __name__ == "__main__":
    main()
