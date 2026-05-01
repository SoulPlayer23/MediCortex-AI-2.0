"""
EVAL-1: Generate all 4 Section 6.2 performance plots.

Plots produced:
  1. fig_ragas_faithfulness.pdf   — RAGAS faithfulness bar chart per domain
  2. fig_latency_boxplot.pdf      — Latency distribution box plot (3 system configs)
  3. fig_judge_distribution.pdf   — Judge score histogram from test set
  4. fig_rgcn_roc.pdf             — R-GCN ROC curve (copy from notebook output)

Usage:
    python tests/evaluation/plots/generate_all.py \
        --ragas results/ragas_scores.json \
        --ablation-full results/ablation_full.json \
        --ablation-sequential results/ablation_sequential.json \
        --ablation-noagent results/ablation_noagent.json \
        --rgcn-roc path/to/existing_roc.pdf \
        --output results/

Prereqs:
    pip install matplotlib pandas
"""

import argparse
import json
import shutil
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import pandas as pd
except ImportError:
    raise SystemExit("Run: pip install matplotlib pandas")

DOMAIN_COLORS = {
    "pubmed": "#4472C4",
    "diagnosis": "#ED7D31",
    "pharmacology": "#A9D18E",
    "patient": "#FFC000",
    "report_analyzer": "#9E5DCF",
}

DOMAIN_LABELS = {
    "pubmed": "Literature\nRetrieval",
    "diagnosis": "Differential\nDiagnosis",
    "pharmacology": "Pharmacology",
    "patient": "Patient Data",
    "report_analyzer": "Radiology /\nReports",
}


def plot_ragas_faithfulness(ragas_path: Path, output_dir: Path):
    with open(ragas_path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    domain_means = df.groupby("domain")["faithfulness"].mean().sort_values(ascending=True)
    colors = [DOMAIN_COLORS.get(d, "#888888") for d in domain_means.index]
    labels = [DOMAIN_LABELS.get(d, d) for d in domain_means.index]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(labels, domain_means.values, color=colors, edgecolor="white", height=0.6)

    # Value annotations
    for bar, val in zip(bars, domain_means.values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10, fontweight="bold")

    overall = df["faithfulness"].mean()
    ax.axvline(overall, color="#C00000", linestyle="--", linewidth=1.5, label=f"Overall mean: {overall:.3f}")

    ax.set_xlabel("Faithfulness Score (0 – 1)", fontsize=12)
    ax.set_title("RAGAS Faithfulness by Agent Domain\nMediCortex AI 2.0", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1.12)
    ax.legend(fontsize=10, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    out = output_dir / "fig_ragas_faithfulness.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_latency_boxplot(full_path: Path, sequential_path: Path, noagent_path: Path, output_dir: Path):
    configs = {}
    for name, path in [("Full System", full_path), ("Sequential", sequential_path), ("Non-agentic", noagent_path)]:
        if path and path.exists():
            with open(path) as f:
                data = json.load(f)
            configs[name] = [r["wall_clock_latency_s"] for r in data if r.get("wall_clock_latency_s")]

    if not configs:
        print("  [SKIP] No ablation result files found for latency box plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(configs.keys())
    values = list(configs.values())
    colors = ["#4472C4", "#ED7D31", "#A9D18E"]

    bp = ax.boxplot(values, labels=labels, patch_artist=True, notch=False,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_ylabel("End-to-End Latency (seconds)", fontsize=12)
    ax.set_title("Latency Distribution — System Comparison\nMediCortex AI 2.0", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    out = output_dir / "fig_latency_boxplot.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_judge_distribution(ragas_path: Path, output_dir: Path):
    with open(ragas_path) as f:
        data = json.load(f)
    scores = [r["judge_score"] for r in data if r.get("judge_score") is not None]

    if not scores:
        print("  [SKIP] No judge scores found in RAGAS results.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    counts = {s: scores.count(s) for s in range(1, 6)}
    bar_colors = ["#C00000", "#FF4444", "#FFC000", "#92D050", "#00B050"]
    score_labels = ["1 — Poor", "2 — Below avg", "3 — Average", "4 — Good", "5 — Excellent"]

    ax.bar(list(counts.keys()), list(counts.values()),
           color=bar_colors, edgecolor="white", width=0.6)

    for x, y in counts.items():
        ax.text(x, y + 0.3, str(y), ha="center", fontsize=11, fontweight="bold")

    legend_patches = [mpatches.Patch(color=bar_colors[i], label=score_labels[i]) for i in range(5)]
    ax.legend(handles=legend_patches, fontsize=9, loc="upper left")

    mean_score = sum(scores) / len(scores)
    ax.axvline(mean_score, color="navy", linestyle="--", linewidth=1.5, label=f"Mean: {mean_score:.2f}")

    ax.set_xlabel("LLM Judge Score (1 – 5)", fontsize=12)
    ax.set_ylabel("Number of Responses", fontsize=12)
    ax.set_title(f"LLM-as-Judge Score Distribution (n={len(scores)})\nMediCortex AI 2.0 Test Set",
                 fontsize=13, fontweight="bold")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    out = output_dir / "fig_judge_distribution.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def copy_rgcn_roc(roc_source: Path, output_dir: Path):
    if roc_source and roc_source.exists():
        out = output_dir / "fig_rgcn_roc.pdf"
        shutil.copy(roc_source, out)
        print(f"  Saved: {out}")
    else:
        print("  [SKIP] R-GCN ROC source file not found. Copy manually from notebook output.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ragas", default="results/ragas_scores.json")
    parser.add_argument("--ablation-full", default="results/ablation_full.json")
    parser.add_argument("--ablation-sequential", default="results/ablation_sequential.json")
    parser.add_argument("--ablation-noagent", default="results/ablation_noagent.json")
    parser.add_argument("--rgcn-roc", default=None, help="Path to existing R-GCN ROC PDF from notebook")
    parser.add_argument("--output", default="results/")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    ragas_path = Path(args.ragas)
    if not ragas_path.exists():
        raise SystemExit(f"RAGAS results not found at {ragas_path}. Run run_ragas.py first.")

    print("Generating Section 6.2 plots...")
    plot_ragas_faithfulness(ragas_path, output_dir)
    plot_latency_boxplot(
        Path(args.ablation_full),
        Path(args.ablation_sequential),
        Path(args.ablation_noagent),
        output_dir,
    )
    plot_judge_distribution(ragas_path, output_dir)
    copy_rgcn_roc(Path(args.rgcn_roc) if args.rgcn_roc else None, output_dir)
    print(f"\nAll plots saved to {output_dir}")
