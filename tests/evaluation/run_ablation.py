"""
EVAL-1: Ablation Study — Table VI (and Table V full-system latency).

Runs the 50-query test set across 4 ablation configurations:
  1. full       — normal system (baseline, also used for Table V latency)
  2. no_kg      — ArangoDB disabled (ARANGODB_HOST="")
  3. no_judge   — LLM-as-judge bypassed (JUDGE_ENABLED=False)
  4. sequential — single concurrent agent (MAX_CONCURRENT_AGENTS=1)
  5. no_adapt   — base Gemma 3 4B instead of MedGemma (MEDGEMMA_MODEL=gemma3:4b)

Each config writes results/ablation_{config}.json.
After all runs, prints Table VI comparison.

Usage:
    python tests/evaluation/run_ablation.py [--configs full,no_kg,no_judge,sequential,no_adapt]

Prereqs:
    Orchestrator must be restarted between configs (env vars change server behaviour).
    This script prompts you to restart the server before each config run.
"""

import argparse
import asyncio
import json
import os
import time
import uuid
import httpx
from pathlib import Path

TEST_SET_PATH = Path(__file__).parent.parent / "resources" / "eval_test_set.json"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
BASE_URL = os.getenv("MEDICORTEX_URL", "http://localhost:8001")

ABLATION_CONFIGS = {
    "full": {
        "description": "Full system — all components enabled",
        "env_instructions": "Standard .env — no changes needed.",
        "table_label": "MediCortex AI 2.0 (Full)",
    },
    "no_kg": {
        "description": "No KG traversal — ArangoDB disabled",
        "env_instructions": "Set ARANGODB_HOST='' in .env, restart orchestrator.",
        "table_label": "No KG Traversal",
    },
    "no_judge": {
        "description": "No LLM-as-Judge gate",
        "env_instructions": "Set JUDGE_ENABLED=False in .env, restart orchestrator.",
        "table_label": "No Judge Gate",
    },
    "sequential": {
        "description": "Sequential agent execution (MAX_CONCURRENT_AGENTS=1)",
        "env_instructions": "Set MAX_CONCURRENT_AGENTS=1 in orchestrator.py constant, restart.",
        "table_label": "Sequential Execution",
    },
    "no_adapt": {
        "description": "No domain adaptation (gemma3:4b instead of MedGemma)",
        "env_instructions": "Pull gemma3:4b via Ollama; set MEDGEMMA_MODEL=gemma3:4b in .env, restart.",
        "table_label": "No Domain Adaptation",
    },
}


async def query_with_timing(query: str, session_id: str) -> tuple[str, dict, float]:
    """Returns (response_text, metadata, wall_clock_seconds)."""
    response_text = ""
    metadata = {}
    t0 = time.perf_counter()

    async with httpx.AsyncClient(timeout=180.0) as client:
        async with client.stream(
            "POST",
            f"{BASE_URL}/chat/stream",
            json={"message": query, "session_id": session_id},
        ) as resp:
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                    if event.get("type") == "response":
                        response_text += event.get("content", "")
                    elif event.get("type") == "metadata":
                        metadata = event.get("content", {})
                except json.JSONDecodeError:
                    continue

    elapsed = time.perf_counter() - t0
    return response_text.strip(), metadata, elapsed


async def run_config(config_name: str, test_set: list) -> list:
    rows = []
    print(f"\n  Running {len(test_set)} queries for config '{config_name}'...")
    for i, item in enumerate(test_set):
        print(f"    [{i+1}/{len(test_set)}] {item['id']}")
        session_id = str(uuid.uuid4())
        try:
            response, metadata, elapsed = await query_with_timing(item["query"], session_id)
        except Exception as e:
            print(f"      ERROR: {e}")
            response, metadata, elapsed = "", {}, 0.0

        node_timings = metadata.get("node_timings", {})
        total_node_time = sum(node_timings.values()) if node_timings else None

        rows.append({
            "item_id": item["id"],
            "domain": item["domain"],
            "query": item["query"],
            "response": response,
            "judge_score": metadata.get("judge_score"),
            "agents_used": metadata.get("agents_used", []),
            "node_timings": node_timings,
            "total_node_latency_s": total_node_time,
            "wall_clock_latency_s": round(elapsed, 2),
            "kb_available": None,
        })
        await asyncio.sleep(1)
    return rows


def print_table_vi(all_results: dict):
    import statistics

    print("\n" + "=" * 70)
    print("TABLE VI — ABLATION STUDY RESULTS")
    print("=" * 70)
    header = f"{'Config':<30} {'Accuracy':>10} {'Mean Latency':>14} {'N Queries':>10}"
    print(header)
    print("-" * 70)

    for config_name, rows in all_results.items():
        label = ABLATION_CONFIGS[config_name]["table_label"]
        scores = [r["judge_score"] for r in rows if r.get("judge_score") is not None]
        accuracy = sum(1 for s in scores if s >= 4) / len(scores) if scores else 0
        latencies = [r["wall_clock_latency_s"] for r in rows if r.get("wall_clock_latency_s")]
        mean_lat = statistics.mean(latencies) if latencies else 0
        print(f"{label:<30} {accuracy:>9.1%} {mean_lat:>13.1f}s {len(rows):>10}")

    print("=" * 70)
    print("Accuracy = fraction of responses with judge_score >= 4")
    print("Latency  = mean wall-clock end-to-end time per query")


async def main(configs_to_run: list):
    with open(TEST_SET_PATH) as f:
        test_set = json.load(f)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for config_name in configs_to_run:
        cfg = ABLATION_CONFIGS[config_name]
        print(f"\n{'='*60}")
        print(f"CONFIG: {config_name.upper()} — {cfg['description']}")
        print(f"{'='*60}")
        print(f"Setup required: {cfg['env_instructions']}")
        input("Press ENTER when the orchestrator is ready for this config...")

        rows = await run_config(config_name, test_set)
        all_results[config_name] = rows

        output_path = RESULTS_DIR / f"ablation_{config_name}.json"
        with open(output_path, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"  Saved → {output_path}")

    print_table_vi(all_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        default="full,no_kg,no_judge,sequential,no_adapt",
        help="Comma-separated list of configs to run",
    )
    args = parser.parse_args()
    configs = [c.strip() for c in args.configs.split(",")]
    asyncio.run(main(configs))
