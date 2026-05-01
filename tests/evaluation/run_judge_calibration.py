"""
EVAL-1: LLM-as-Judge Calibration — Table IV.

Reads human_ratings.csv (30 queries rated by two expert raters on 4 dimensions 1-5)
and computes:
  - ICC (Intraclass Correlation Coefficient) between judge and each rater
  - Weighted Cohen's kappa between judge and each rater, and between raters

Usage:
    python tests/evaluation/run_judge_calibration.py \
        --ratings tests/resources/human_ratings.csv \
        --output results/judge_calibration.json

    Fill in human_ratings.csv from human_ratings_template.csv first.
    judge_score column is read from the CSV (copy from message_metadata queries
    or run: python tests/evaluation/fetch_judge_scores.py to auto-populate).

Prereqs:
    pip install pingouin scikit-learn pandas
"""

import argparse
import json
from pathlib import Path

try:
    import pandas as pd
    import pingouin
    from sklearn.metrics import cohen_kappa_score
except ImportError:
    raise SystemExit("Run: pip install pingouin scikit-learn pandas")


def compute_calibration(ratings_path: Path, output_path: Path):
    df = pd.read_csv(ratings_path)

    # Validate required columns
    required = [
        "item_id", "judge_score",
        "rater_a_accuracy", "rater_a_completeness", "rater_a_safety", "rater_a_clarity",
        "rater_b_accuracy", "rater_b_completeness", "rater_b_safety", "rater_b_clarity",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in ratings CSV: {missing}")

    df = df.dropna(subset=required)
    n = len(df)
    if n < 10:
        raise ValueError(f"Only {n} complete rows found. Need at least 10 for meaningful ICC.")

    print(f"Computing calibration on {n} queries...")

    # Compute composite human scores (mean of 4 dimensions per rater)
    df["rater_a_composite"] = df[
        ["rater_a_accuracy", "rater_a_completeness", "rater_a_safety", "rater_a_clarity"]
    ].mean(axis=1).round().astype(int)

    df["rater_b_composite"] = df[
        ["rater_b_accuracy", "rater_b_completeness", "rater_b_safety", "rater_b_clarity"]
    ].mean(axis=1).round().astype(int)

    judge = df["judge_score"].round().astype(int)
    rater_a = df["rater_a_composite"]
    rater_b = df["rater_b_composite"]

    # ICC — Judge vs Rater A
    icc_data_a = pd.DataFrame({
        "item": list(df["item_id"]) * 2,
        "rater": ["judge"] * n + ["rater_a"] * n,
        "score": list(judge) + list(rater_a),
    })
    icc_a = pingouin.intraclass_corr(
        data=icc_data_a, targets="item", raters="rater", ratings="score"
    )
    icc_a_value = float(icc_a[icc_a["Type"] == "ICC2"]["ICC"].values[0])
    icc_a_ci = icc_a[icc_a["Type"] == "ICC2"][["CI95%"]].values[0][0]

    # ICC — Judge vs Rater B
    icc_data_b = pd.DataFrame({
        "item": list(df["item_id"]) * 2,
        "rater": ["judge"] * n + ["rater_b"] * n,
        "score": list(judge) + list(rater_b),
    })
    icc_b = pingouin.intraclass_corr(
        data=icc_data_b, targets="item", raters="rater", ratings="score"
    )
    icc_b_value = float(icc_b[icc_b["Type"] == "ICC2"]["ICC"].values[0])

    # ICC — Inter-rater (A vs B)
    icc_data_inter = pd.DataFrame({
        "item": list(df["item_id"]) * 2,
        "rater": ["rater_a"] * n + ["rater_b"] * n,
        "score": list(rater_a) + list(rater_b),
    })
    icc_inter = pingouin.intraclass_corr(
        data=icc_data_inter, targets="item", raters="rater", ratings="score"
    )
    icc_inter_value = float(icc_inter[icc_inter["Type"] == "ICC2"]["ICC"].values[0])

    # Weighted Cohen's kappa
    kappa_judge_a = cohen_kappa_score(judge, rater_a, weights="quadratic")
    kappa_judge_b = cohen_kappa_score(judge, rater_b, weights="quadratic")
    kappa_inter = cohen_kappa_score(rater_a, rater_b, weights="quadratic")

    results = {
        "n_queries": n,
        "icc_judge_vs_rater_a": round(icc_a_value, 3),
        "icc_judge_vs_rater_a_ci95": str(icc_a_ci),
        "icc_judge_vs_rater_b": round(icc_b_value, 3),
        "icc_inter_rater_a_vs_b": round(icc_inter_value, 3),
        "kappa_judge_vs_rater_a": round(kappa_judge_a, 3),
        "kappa_judge_vs_rater_b": round(kappa_judge_b, 3),
        "kappa_inter_rater_a_vs_b": round(kappa_inter, 3),
        "interpretation": {
            "icc_threshold_excellent": 0.75,
            "kappa_threshold_substantial": 0.60,
            "judge_icc_excellent": icc_a_value >= 0.75,
            "judge_kappa_substantial": kappa_judge_a >= 0.60,
        },
    }

    # Dimension-level breakdown
    dim_results = {}
    for dim in ["accuracy", "completeness", "safety", "clarity"]:
        a_col = f"rater_a_{dim}"
        b_col = f"rater_b_{dim}"
        if a_col in df.columns and b_col in df.columns:
            a_dim = df[a_col].dropna().round().astype(int)
            b_dim = df[b_col].dropna().round().astype(int)
            shared_idx = a_dim.index.intersection(b_dim.index)
            if len(shared_idx) >= 5:
                dim_results[dim] = {
                    "inter_rater_kappa": round(
                        cohen_kappa_score(a_dim[shared_idx], b_dim[shared_idx], weights="quadratic"), 3
                    ),
                    "rater_a_mean": round(float(a_dim.mean()), 2),
                    "rater_b_mean": round(float(b_dim.mean()), 2),
                }
    results["dimension_breakdown"] = dim_results

    # Print Table IV summary
    print("\n=== TABLE IV — LLM-AS-JUDGE CALIBRATION ===")
    print(f"  Queries evaluated: {n}")
    print(f"  ICC (Judge vs Rater A): {icc_a_value:.3f}  95% CI: {icc_a_ci}  {'✅ Excellent' if icc_a_value >= 0.75 else '⚠️  Below threshold'}")
    print(f"  ICC (Judge vs Rater B): {icc_b_value:.3f}")
    print(f"  ICC (Rater A vs B):     {icc_inter_value:.3f}")
    print(f"  Weighted kappa (Judge vs Rater A): {kappa_judge_a:.3f}  {'✅ Substantial' if kappa_judge_a >= 0.60 else '⚠️  Below threshold'}")
    print(f"  Weighted kappa (Judge vs Rater B): {kappa_judge_b:.3f}")
    print(f"  Weighted kappa (A vs B):           {kappa_inter:.3f}")
    if dim_results:
        print("\n  Dimension-level inter-rater kappa (A vs B):")
        for dim, v in dim_results.items():
            print(f"    {dim:15s}: κ={v['inter_rater_kappa']:.3f}  (mean A={v['rater_a_mean']}, B={v['rater_b_mean']})")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", default="tests/resources/human_ratings.csv")
    parser.add_argument("--output", default="results/judge_calibration.json")
    args = parser.parse_args()
    compute_calibration(Path(args.ratings), Path(args.output))
