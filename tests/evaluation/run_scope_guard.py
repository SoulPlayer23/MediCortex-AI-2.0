"""
MediCortex Scope Guard Evaluator — Table 6.6

Measures Precision / Recall / F1 for the medical scope guard (Groq llama-3.3-70b)
on a balanced 60-query test set (30 in-scope + 30 out-of-scope).

Usage:
  source .venv/bin/activate
  python3 tests/evaluation/run_scope_guard.py

Requirements:
  - GROQ_API_KEY in .env
  - groq Python SDK installed

Runtime: ~2-3 min (60 Groq API calls, rate-limited)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings

try:
    from groq import Groq
except ImportError:
    print("ERROR: groq SDK not installed. Run: pip install groq")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Labelled test set  (label=1 → in-scope / clinical, label=0 → out-of-scope)
# ---------------------------------------------------------------------------

TEST_SET = [
    # ── IN-SCOPE (30) ────────────────────────────────────────────────────────
    {"query": "What are the symptoms of type 2 diabetes?",                       "label": 1},
    {"query": "What is the maximum daily dose of metformin?",                    "label": 1},
    {"query": "My patient has chest pain and diaphoresis — differential?",       "label": 1},
    {"query": "Interpret this CBC: WBC 14k, Hgb 8.2 g/dL",                     "label": 1},
    {"query": "Is atorvastatin safe in patients with hepatic impairment?",       "label": 1},
    {"query": "What are the first-line antibiotics for community-acquired pneumonia?", "label": 1},
    {"query": "Patient has SOB, bilateral crackles, and elevated BNP",          "label": 1},
    {"query": "Explain the mechanism of action of ACE inhibitors",               "label": 1},
    {"query": "What is the recommended HbA1c target for diabetic patients?",     "label": 1},
    {"query": "Signs of acute kidney injury in a post-surgical patient",         "label": 1},
    {"query": "Can warfarin be used alongside aspirin for atrial fibrillation?", "label": 1},
    {"query": "What does an elevated troponin indicate?",                        "label": 1},
    {"query": "MRI shows T2 hyperintensity in right basal ganglia — significance?", "label": 1},
    {"query": "Management of anaphylaxis in an adult patient",                   "label": 1},
    {"query": "What is COPD and how is it staged?",                              "label": 1},
    {"query": "Interpret this spirometry: FEV1/FVC 0.62, FEV1 68% predicted",   "label": 1},
    {"query": "Drug interactions between sertraline and tramadol",               "label": 1},
    {"query": "What vaccinations are recommended for immunocompromised patients?","label": 1},
    {"query": "Explain the CHADS2-VASc scoring system",                          "label": 1},
    {"query": "Post-op day 3: patient has fever, tachycardia, and wound redness","label": 1},
    {"query": "What is the Glasgow Coma Scale?",                                 "label": 1},
    {"query": "Serotonin syndrome — symptoms and treatment",                     "label": 1},
    {"query": "Recommended calcium and vitamin D supplementation in osteoporosis","label": 1},
    {"query": "What does the APGAR score assess in newborns?",                   "label": 1},
    {"query": "Is lisinopril contraindicated in pregnancy?",                     "label": 1},
    {"query": "Pathophysiology of heart failure with reduced ejection fraction", "label": 1},
    {"query": "What is the normal eGFR range and what does a low value indicate?","label": 1},
    {"query": "Thyroid-stimulating hormone (TSH) reference range",               "label": 1},
    {"query": "Treatment protocol for community-acquired sepsis in adults",      "label": 1},
    {"query": "What are the side effects of long-term corticosteroid use?",      "label": 1},

    # ── OUT-OF-SCOPE (30) ────────────────────────────────────────────────────
    {"query": "How do I make pasta carbonara?",                                  "label": 0},
    {"query": "What are the best places to visit in Rome?",                      "label": 0},
    {"query": "Write me a Python function to sort a list",                       "label": 0},
    {"query": "Who won the 2022 FIFA World Cup?",                                "label": 0},
    {"query": "Recommend a good sci-fi novel",                                   "label": 0},
    {"query": "What is the capital of Australia?",                               "label": 0},
    {"query": "How do I invest in index funds?",                                 "label": 0},
    {"query": "Summarise the plot of Hamlet",                                    "label": 0},
    {"query": "What is the speed of light?",                                     "label": 0},
    {"query": "Tell me a joke",                                                  "label": 0},
    {"query": "How do I fix a JavaScript async/await error?",                    "label": 0},
    {"query": "Best workout routines for weight loss",                           "label": 0},
    {"query": "How does machine learning work?",                                 "label": 0},
    {"query": "What are the rules of chess?",                                    "label": 0},
    {"query": "Translate 'good morning' into Spanish",                          "label": 0},
    {"query": "What is the latest iPhone model?",                                "label": 0},
    {"query": "Give me a recipe for banana bread",                               "label": 0},
    {"query": "How do I get a mortgage?",                                        "label": 0},
    {"query": "Who is the CEO of Tesla?",                                        "label": 0},
    {"query": "Write a cover letter for a software engineer role",               "label": 0},
    {"query": "How do electric cars work?",                                      "label": 0},
    {"query": "What are the best programming languages to learn in 2025?",       "label": 0},
    {"query": "How do I improve my chess rating?",                               "label": 0},
    {"query": "What happened in World War 2?",                                   "label": 0},
    {"query": "How do I start a YouTube channel?",                               "label": 0},
    {"query": "What is the GDP of Germany?",                                     "label": 0},
    {"query": "How do I grow tomatoes at home?",                                 "label": 0},
    {"query": "Best podcasts about history",                                     "label": 0},
    {"query": "How do I learn to play guitar?",                                  "label": 0},
    {"query": "What are the most popular social media platforms?",               "label": 0},
]

# ---------------------------------------------------------------------------
# Scope guard prompt (exact copy from orchestrator.py)
# ---------------------------------------------------------------------------

SCOPE_GUARD_PROMPT = (
    "You are a medical AI scope filter. Reply with ONLY '1' (in scope) or '0' (out of scope). "
    "No explanation, no punctuation — a single digit.\n\n"
    "IN SCOPE: medicine, diseases, symptoms, drugs, pharmacology, anatomy, physiology, biology, "
    "lab results, medical procedures, patient care, mental health, genetics, nutrition related "
    "to health, public health, veterinary medicine.\n\n"
    "OUT OF SCOPE: cooking, travel, sports, politics, programming, history, entertainment, "
    "celebrity news, general trivia, math problems, creative writing, anything unrelated to "
    "health or biology.\n\n"
    "Reply: 1 or 0"
)

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

R = "\033[91m"; G = "\033[92m"; Y = "\033[93m"; W = "\033[97m"
BOLD = "\033[1m"; DIM = "\033[2m"; RST = "\033[0m"
def col(c, s): return f"{c}{s}{RST}"

# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def run_scope_guard_eval():
    if not settings.GROQ_API_KEY:
        print(col(R, "ERROR: GROQ_API_KEY not set in .env"))
        sys.exit(1)

    client = Groq(api_key=settings.GROQ_API_KEY)

    print(f"\n{col(BOLD, 'MediCortex Scope Guard Evaluation — Table 6.6')}")
    print(f"{col(DIM, f'Model: llama-3.3-70b-versatile @ Groq  |  {len(TEST_SET)} queries')}\n")

    results = []
    for i, item in enumerate(TEST_SET, 1):
        query, label = item["query"], item["label"]
        print(f"  [{i:2d}/{len(TEST_SET)}] {'IN ' if label else 'OUT'} | {query[:65]}", end="", flush=True)

        try:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SCOPE_GUARD_PROMPT},
                    {"role": "user",   "content": query},
                ],
                temperature=0,
                max_tokens=4,
            )
            latency_ms = round((time.perf_counter() - t0) * 1000)
            raw = resp.choices[0].message.content.strip()
            predicted = 0 if raw.startswith("0") else 1
            correct = predicted == label
            status = col(G, " ✓") if correct else col(R, " ✗")
            print(f"{status} → {raw} ({latency_ms}ms)")
        except Exception as e:
            print(col(R, f" ERROR: {e}"))
            predicted = 1  # default to in-scope on failure
            correct = predicted == label
            latency_ms = 0

        results.append({"label": label, "predicted": predicted, "correct": correct})

        # Rate-limit courtesy pause (Groq free tier: ~30 req/min)
        if i % 10 == 0 and i < len(TEST_SET):
            time.sleep(2)

    # ── Compute metrics ──────────────────────────────────────────────
    in_scope  = [r for r in results if r["label"] == 1]
    out_scope = [r for r in results if r["label"] == 0]

    def metrics(subset, positive_class):
        tp = sum(1 for r in subset if r["predicted"] == positive_class)
        fn = sum(1 for r in subset if r["predicted"] != positive_class)
        # False positives: items from the OTHER class predicted as positive_class
        other = [r for r in results if r["label"] != positive_class]
        fp = sum(1 for r in other if r["predicted"] == positive_class)
        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
        return prec, recall, f1

    in_p,  in_r,  in_f1  = metrics(in_scope,  1)
    out_p, out_r, out_f1 = metrics(out_scope, 0)

    total_correct = sum(1 for r in results if r["correct"])
    overall_acc   = total_correct / len(results)
    overall_p     = (in_p + out_p) / 2
    overall_r     = (in_r + out_r) / 2
    overall_f1    = (in_f1 + out_f1) / 2

    # ── Print table ──────────────────────────────────────────────────
    print(f"\n{col(BOLD, '═' * 70)}")
    print(f"{col(BOLD, '  SCOPE GUARD EVALUATION RESULTS — Table 6.6')}")
    print(f"  {col(DIM, 'Copy these values into main.tex tab:scopeguard')}")
    print(f"{col(BOLD, '═' * 70)}")
    print(f"\n  {'Category':<28} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'─'*28} {'─'*10} {'─'*10} {'─'*10}")
    print(f"  {'In-scope (clinical)':<28} {in_p:>10.3f} {in_r:>10.3f} {in_f1:>10.3f}")
    print(f"  {'Out-of-scope (general)':<28} {out_p:>10.3f} {out_r:>10.3f} {out_f1:>10.3f}")
    print(f"  {'─'*28} {'─'*10} {'─'*10} {'─'*10}")
    print(f"  {col(BOLD,'Overall (macro avg)'):<28} {col(BOLD,f'{overall_p:>10.3f}')} "
          f"{col(BOLD,f'{overall_r:>10.3f}')} {col(BOLD,f'{overall_f1:>10.3f}')}")
    print(f"\n  Overall accuracy: {total_correct}/{len(results)} ({overall_acc*100:.1f}%)")

    print(f"\n  {col(DIM, 'LaTeX snippet for main.tex:')}")
    print(f"  In-scope   & {in_p:.3f} & {in_r:.3f} & {in_f1:.3f} \\\\")
    print(f"  Out-of-scope & {out_p:.3f} & {out_r:.3f} & {out_f1:.3f} \\\\")
    print(f"  \\textbf{{Overall}} & \\textbf{{{overall_p:.3f}}} & \\textbf{{{overall_r:.3f}}} & \\textbf{{{overall_f1:.3f}}} \\\\")
    print()


if __name__ == "__main__":
    run_scope_guard_eval()
