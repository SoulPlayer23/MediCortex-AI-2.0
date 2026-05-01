"""
EVAL-1: RAGAS Evaluation — Table III (and non-agentic baseline for Table V).

Usage:
    # Full system run (Table III):
    python tests/evaluation/run_ragas.py

    # Non-agentic baseline (Table V):
    EVAL_FORCE_NOAGENT=1 python tests/evaluation/run_ragas.py --output results/ragas_noagent.json

Prereqs:
    pip install ragas datasets httpx asyncio
    OBS-1 must be complete (node_timings + retrieval in message_metadata).
    Live orchestrator running on port 8001.
"""

import argparse
import asyncio
import json
import os
import time
import httpx
from pathlib import Path

# Install: pip install ragas datasets
try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from datasets import Dataset
except ImportError:
    raise SystemExit("Run: pip install ragas datasets")

BASE_URL = os.getenv("MEDICORTEX_URL", "http://localhost:8001")
TEST_SET_PATH = Path(__file__).parent.parent / "resources" / "eval_test_set.json"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"


async def query_medicortex(query: str, session_id: str) -> tuple[str, dict]:
    """Send a query to /chat/stream and collect the full response + final metadata."""
    response_text = ""
    metadata = {}

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Create or reuse session
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
                    if event.get("type") == "content":
                        response_text += event.get("content", "")
                    elif event.get("type") == "metadata":
                        metadata = event.get("data", {})
                except json.JSONDecodeError:
                    continue

    return response_text.strip(), metadata


async def run_evaluation(test_set_path: Path, output_path: Path):
    with open(test_set_path) as f:
        test_set = json.load(f)

    print(f"Running evaluation on {len(test_set)} queries...")
    rows = []

    for i, item in enumerate(test_set):
        print(f"  [{i+1}/{len(test_set)}] {item['id']}: {item['query'][:60]}...")
        session_id = f"eval-{item['id']}-{int(time.time())}"

        try:
            response, metadata = await query_medicortex(item["query"], session_id)
        except Exception as e:
            print(f"    ERROR: {e}")
            response = ""
            metadata = {}

        retrieval_meta = metadata.get("retrieval", {})
        context = retrieval_meta.get("refined_context", "")
        if not context:
            context = retrieval_meta.get("raw_facts", "")

        rows.append({
            "question": item["query"],
            "answer": response,
            "contexts": [context] if context else [""],
            "ground_truth": item.get("ground_truth_answer", ""),
            "domain": item["domain"],
            "item_id": item["id"],
            "judge_score": metadata.get("judge_score"),
            "agents_used": metadata.get("agents_used", []),
            "node_timings": metadata.get("node_timings", {}),
        })

        # Avoid hammering the endpoint
        await asyncio.sleep(2)

    # RAGAS evaluation
    print("\nRunning RAGAS scoring...")
    ragas_rows = [
        {
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["contexts"],
            "ground_truth": r["ground_truth"],
        }
        for r in rows
    ]
    dataset = Dataset.from_list(ragas_rows)
    scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
    scores_df = scores.to_pandas()

    # Merge back domain and metadata
    for i, row in enumerate(rows):
        rows[i]["faithfulness"] = float(scores_df.iloc[i].get("faithfulness", 0))
        rows[i]["answer_relevancy"] = float(scores_df.iloc[i].get("answer_relevancy", 0))
        rows[i]["context_precision"] = float(scores_df.iloc[i].get("context_precision", 0))

    # Save raw results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(rows, f, indent=2)

    # Print per-domain summary (Table III format)
    import pandas as pd
    df = pd.DataFrame(rows)
    print("\n=== TABLE III — RAGAS SCORES PER DOMAIN ===")
    summary = df.groupby("domain")[["faithfulness", "answer_relevancy", "context_precision"]].mean()
    print(summary.round(3).to_string())
    print(f"\nOverall — Faithfulness: {df['faithfulness'].mean():.3f}, "
          f"Answer Relevancy: {df['answer_relevancy'].mean():.3f}, "
          f"Context Precision: {df['context_precision'].mean():.3f}")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set", default=str(TEST_SET_PATH))
    parser.add_argument("--output", default=str(RESULTS_DIR / "ragas_scores.json"))
    args = parser.parse_args()

    asyncio.run(run_evaluation(Path(args.test_set), Path(args.output)))
