"""
MediCortex — Fetch Judge Scores for Table 6.4 (LLM-as-Judge Calibration)

Runs all 30 queries from human_ratings_template.csv against the live orchestrator,
captures judge_score + judge_reason + response text, and writes:

    tests/resources/human_ratings.csv

The output CSV has all auto-fillable columns populated. The 8 rater columns
(rater_a_* / rater_b_*) are left blank for manual entry.

Usage:
  source .venv/bin/activate
  python3 tests/evaluation/fetch_judge_scores.py

Options:
  --resume          Skip rows already present in human_ratings.csv (safe to re-run)
  --url URL         Orchestrator base URL (default: http://homeserver:8000)
  --delay SECONDS   Sleep between queries in seconds (default: 3)

Requirements:
  - Full orchestrator running: python orchestrator.py
  - JUDGE_ENABLED=True and GROQ_API_KEY set in .env
  - JUDGE_SAMPLE_RATE=1.0 recommended (ensures every query is judged)

Runtime: ~20-40 min for 30 queries (depends on MedGemma + Groq latency)
"""

import argparse
import csv
import json
import os
import sys
import time
import uuid
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT              = Path(__file__).parent.parent.parent
TEMPLATE_PATH     = ROOT / "tests" / "resources" / "human_ratings_template.csv"
EVAL_SET_PATH     = ROOT / "tests" / "resources" / "eval_test_set.json"
OUTPUT_PATH       = ROOT / "tests" / "resources" / "human_ratings.csv"

# ---------------------------------------------------------------------------
# CSV columns (output)
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "item_id", "domain", "query_preview", "response_preview",
    "rater_a_accuracy", "rater_a_completeness", "rater_a_safety", "rater_a_clarity",
    "rater_b_accuracy", "rater_b_completeness", "rater_b_safety", "rater_b_clarity",
    "judge_score", "judge_reason",
]

# Colour helpers
R = "\033[91m"; G = "\033[92m"; Y = "\033[93m"; W = "\033[97m"
BOLD = "\033[1m"; DIM = "\033[2m"; RST = "\033[0m"
def col(c, s): return f"{c}{s}{RST}"

# ---------------------------------------------------------------------------
# SSE streaming call
# ---------------------------------------------------------------------------

def query_orchestrator(base_url: str, query: str, session_id: str, timeout: int = 240
                       ) -> dict:
    """
    POST to /chat/stream, drain SSE events.
    Returns dict with keys: response_text, judge_score, judge_reason, agents_used, error.
    """
    result = {
        "response_text": "",
        "judge_score": None,
        "judge_reason": "",
        "agents_used": [],
        "error": None,
    }

    try:
        with requests.post(
            f"{base_url}/chat/stream",
            json={"session_id": session_id, "message": query},
            stream=True,
            timeout=timeout,
        ) as resp:
            if resp.status_code != 200:
                result["error"] = f"HTTP {resp.status_code}"
                return result

            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data: "):
                    continue
                payload = raw_line[6:]
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                etype = event.get("type", "")
                content = event.get("content", "")

                if etype == "response":
                    result["response_text"] = content
                elif etype == "metadata":
                    result["judge_score"]   = content.get("judge_score")
                    result["judge_reason"]  = content.get("judge_reason") or ""
                    result["agents_used"]   = content.get("agents_used", [])

    except requests.exceptions.Timeout:
        result["error"] = "timeout"
    except Exception as e:
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, n: int = 220) -> str:
    text = text.replace("\n", " ").strip()
    return text[:n] + "…" if len(text) > n else text


def _load_existing(path: Path) -> set[str]:
    """Return set of item_ids already written to the output CSV."""
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["item_id"] for row in reader if row.get("response_preview")}


def _load_full_queries(eval_set_path: Path) -> dict[str, str]:
    """item_id → full query text from eval_test_set.json."""
    with open(eval_set_path, encoding="utf-8") as f:
        items = json.load(f)
    return {item["id"]: item["query"] for item in items}


def _load_template(template_path: Path) -> list[dict]:
    with open(template_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch judge scores for calibration CSV")
    parser.add_argument("--url",    default=os.getenv("MEDICORTEX_URL", "http://homeserver:8001"),
                        help="Orchestrator base URL")
    parser.add_argument("--delay",  type=float, default=3.0,        help="Seconds between queries")
    parser.add_argument("--resume", action="store_true",            help="Skip already-fetched rows")
    parser.add_argument("--timeout", type=int, default=240,         help="Per-query timeout seconds")
    args = parser.parse_args()

    # ── Preflight checks ───────────────────────────────────────────────────
    print(f"\n{col(BOLD, 'MediCortex — Fetch Judge Scores')}")
    print(f"{col(DIM, f'Orchestrator: {args.url}  |  delay: {args.delay}s  |  resume: {args.resume}')}\n")

    try:
        requests.get(f"{args.url}/health", timeout=5)
        print(col(G, "  ✓ Orchestrator reachable\n"))
    except Exception as e:
        print(col(R, f"  ✗ Cannot reach orchestrator at {args.url}: {e}"))
        print(col(Y, "  Start it first: source .venv/bin/activate && python orchestrator.py"))
        sys.exit(1)

    if not TEMPLATE_PATH.exists():
        print(col(R, f"  ✗ Template not found: {TEMPLATE_PATH}"))
        sys.exit(1)

    if not EVAL_SET_PATH.exists():
        print(col(R, f"  ✗ eval_test_set.json not found: {EVAL_SET_PATH}"))
        sys.exit(1)

    # ── Load data ──────────────────────────────────────────────────────────
    template_rows  = _load_template(TEMPLATE_PATH)
    full_queries   = _load_full_queries(EVAL_SET_PATH)
    already_done   = _load_existing(OUTPUT_PATH) if args.resume else set()

    if args.resume and already_done:
        print(col(DIM, f"  Resume mode: {len(already_done)} rows already fetched — skipping\n"))

    # ── Load existing output rows (for resume) ─────────────────────────────
    existing_rows: dict[str, dict] = {}
    if args.resume and OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing_rows[row["item_id"]] = row

    # ── Run queries ────────────────────────────────────────────────────────
    results: list[dict] = []
    total = len(template_rows)

    for i, tmpl in enumerate(template_rows, 1):
        item_id = tmpl["item_id"]
        domain  = tmpl["domain"]

        # Resume: re-use cached row
        if item_id in already_done:
            print(f"  [{i:2d}/{total}] {col(DIM, item_id)} — {col(DIM, 'skipped (already fetched)')}")
            results.append(existing_rows[item_id])
            continue

        # Get full query (fall back to template preview if missing from eval set)
        full_query = full_queries.get(item_id, tmpl.get("query_preview", ""))
        if not full_query:
            print(col(Y, f"  [{i:2d}/{total}] {item_id} — no query found, skipping"))
            continue

        print(f"  [{i:2d}/{total}] {col(BOLD, item_id)} ({domain}): {full_query[:65]}", end="", flush=True)

        session_id = str(uuid.uuid4())
        t0 = time.perf_counter()
        result = query_orchestrator(args.url, full_query, session_id, timeout=args.timeout)
        elapsed = time.perf_counter() - t0

        if result["error"]:
            print(col(R, f" ✗ ERROR: {result['error']} ({elapsed:.1f}s)"))
            # Write an error row so resume can detect it
            out_row = {
                "item_id":           item_id,
                "domain":            domain,
                "query_preview":     _truncate(full_query, 120),
                "response_preview":  f"[ERROR: {result['error']}]",
                "rater_a_accuracy": "", "rater_a_completeness": "",
                "rater_a_safety":   "", "rater_a_clarity":      "",
                "rater_b_accuracy": "", "rater_b_completeness": "",
                "rater_b_safety":   "", "rater_b_clarity":      "",
                "judge_score":       "",
                "judge_reason":      f"[ERROR: {result['error']}]",
            }
        else:
            score_str = str(result["judge_score"]) if result["judge_score"] is not None else ""
            score_col = G if result["judge_score"] and result["judge_score"] >= 4 else (
                        Y if result["judge_score"] and result["judge_score"] >= 3 else R)
            agents_str = ", ".join(result["agents_used"]) if result["agents_used"] else "—"
            print(col(score_col, f" ✓ judge={score_str}/5") +
                  col(DIM, f" agents=[{agents_str}] ({elapsed:.1f}s)"))

            out_row = {
                "item_id":           item_id,
                "domain":            domain,
                "query_preview":     _truncate(full_query, 120),
                "response_preview":  _truncate(result["response_text"], 220),
                "rater_a_accuracy": "", "rater_a_completeness": "",
                "rater_a_safety":   "", "rater_a_clarity":      "",
                "rater_b_accuracy": "", "rater_b_completeness": "",
                "rater_b_safety":   "", "rater_b_clarity":      "",
                "judge_score":       score_str,
                "judge_reason":      _truncate(result["judge_reason"], 200),
            }

        results.append(out_row)

        # Write incrementally after every row (safe to interrupt + resume)
        _write_csv(OUTPUT_PATH, results)

        if i < total:
            time.sleep(args.delay)

    # ── Final write + summary ──────────────────────────────────────────────
    _write_csv(OUTPUT_PATH, results)

    fetched  = sum(1 for r in results if r["judge_score"] and not r["judge_score"].startswith("["))
    no_score = sum(1 for r in results if r["judge_score"] == "")
    errors   = sum(1 for r in results if str(r.get("response_preview", "")).startswith("[ERROR"))

    print(f"\n{col(BOLD, '─' * 60)}")
    print(f"  {col(BOLD, 'Done')}  →  {OUTPUT_PATH.relative_to(ROOT)}")
    print(f"  Rows written : {len(results)}")
    print(f"  Judge scored : {col(G, str(fetched))}")
    if no_score:
        print(f"  No judge score (judge disabled/sampled out): {col(Y, str(no_score))}")
    if errors:
        print(f"  Errors        : {col(R, str(errors))}")
    print(f"{col(BOLD, '─' * 60)}")
    print(f"\n  {col(BOLD, 'Next step:')} open {OUTPUT_PATH.name} and fill in the 8 rater columns,")
    print(f"  then run:  python3 tests/evaluation/run_judge_calibration.py\n")


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
