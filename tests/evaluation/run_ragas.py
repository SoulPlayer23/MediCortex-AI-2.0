"""
EVAL-1: RAGAS Evaluation — Table III (and non-agentic baseline for Table V).

Usage:
    # Full system run (Table III):
    python tests/evaluation/run_ragas.py

    # Non-agentic baseline (Table V):
    EVAL_FORCE_NOAGENT=1 python tests/evaluation/run_ragas.py --output results/ragas_noagent.json

Prereqs:
    pip install ragas datasets httpx
    Live orchestrator running on port 8001.
    GROQ_API_KEY set in .env (used for RAGAS scoring — no OpenAI key required).
"""

import argparse
import asyncio
import json
import os
import sys
import time
import httpx
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from ragas import evaluate
    from ragas.metrics import Faithfulness, ContextPrecision
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI
    from datasets import Dataset
    import pandas as pd
except ImportError:
    raise SystemExit("Run: pip install ragas datasets pandas langchain-openai")

from config import settings

BASE_URL = os.getenv("MEDICORTEX_URL", "http://localhost:8001")
TEST_SET_PATH = Path(__file__).parent.parent / "resources" / "eval_test_set.json"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"


def _build_ragas_metrics():
    """Build RAGAS metrics backed by Groq (OpenAI-compatible, no OpenAI key needed)."""
    llm = LangchainLLMWrapper(ChatOpenAI(
        model="llama-3.3-70b-versatile",
        api_key=settings.GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    ))
    return [
        Faithfulness(llm=llm),
        ContextPrecision(llm=llm),
    ]


async def query_medicortex(query: str, session_id: str) -> tuple[str, dict]:
    """Send a query to /chat/stream and collect the full response + final metadata."""
    response_text = ""
    metadata = {}

    async with httpx.AsyncClient(timeout=120.0) as client:
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

    # Save raw responses before RAGAS scoring (allows re-running scoring without re-querying)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_path.with_name(output_path.stem + "_raw.json")
    with open(raw_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nRaw responses saved to {raw_path}")

    # RAGAS evaluation (Groq-backed, no OpenAI key needed)
    print("Running RAGAS scoring via Groq llama-3.3-70b-versatile...")
    metrics = _build_ragas_metrics()
    ragas_rows = [
        {
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["contexts"],
            "ground_truth": r["ground_truth"],
        }
        for r in rows
        if r["answer"]  # skip empty responses (timeouts)
    ]
    valid_ids = [r["item_id"] for r in rows if r["answer"]]
    dataset = Dataset.from_list(ragas_rows)
    scores = evaluate(dataset, metrics=metrics)
    scores_df = scores.to_pandas()

    # Merge scores back by position (skipped rows get NaN)
    score_map = {vid: scores_df.iloc[i] for i, vid in enumerate(valid_ids)}
    for i, row in enumerate(rows):
        s = score_map.get(row["item_id"])
        rows[i]["faithfulness"] = float(s["faithfulness"]) if s is not None else None
        rows[i]["context_precision"] = float(s["context_precision"]) if s is not None else None

    # Save final results
    with open(output_path, "w") as f:
        json.dump(rows, f, indent=2)

    # Print per-domain summary (Table III format)
    df = pd.DataFrame(rows)
    df_valid = df.dropna(subset=["faithfulness"])
    print("\n=== TABLE III — RAGAS SCORES PER DOMAIN ===")
    summary = df_valid.groupby("domain")[["faithfulness", "context_precision"]].mean()
    print(summary.round(3).to_string())
    print(f"\nOverall — Faithfulness: {df_valid['faithfulness'].mean():.3f}, "
          f"Context Precision: {df_valid['context_precision'].mean():.3f}")
    avg_judge = df_valid["judge_score"].dropna().mean()
    mean_latency = df_valid["node_timings"].apply(
        lambda t: sum(t.values()) / 1000 if t else None
    ).dropna().mean()
    print(f"\nMean judge score: {avg_judge:.2f}/5")
    print(f"Mean total node latency: {mean_latency:.1f}s" if mean_latency else "")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set", default=str(TEST_SET_PATH))
    parser.add_argument("--output", default=str(RESULTS_DIR / "ragas_scores.json"))
    args = parser.parse_args()

    asyncio.run(run_evaluation(Path(args.test_set), Path(args.output)))
