"""
Aggregates judge_score from ragas_scores.json and prints Table III.
Run: python results/summarize_results.py
"""
import json
import statistics
from collections import defaultdict
from pathlib import Path

data = json.loads(Path(__file__).parent.joinpath("ragas_scores.json").read_text())

domain_scores = defaultdict(list)
for item in data:
    score = item.get("judge_score")
    domain = item.get("domain", "unknown")
    if score is not None:
        domain_scores[domain].append(score)

all_scores = [s for scores in domain_scores.values() for s in scores]

print("\n=== TABLE III — LLM-AS-JUDGE SCORES PER DOMAIN ===\n")
print(f"{'Domain':<20} {'N':>4} {'Mean':>6} {'Median':>8} {'Std':>6} {'Min':>5} {'Max':>5}")
print("-" * 55)

for domain in sorted(domain_scores):
    scores = domain_scores[domain]
    print(
        f"{domain:<20} {len(scores):>4} {statistics.mean(scores):>6.2f} "
        f"{statistics.median(scores):>8.2f} {statistics.stdev(scores) if len(scores) > 1 else 0.0:>6.2f} "
        f"{min(scores):>5} {max(scores):>5}"
    )

print("-" * 55)
print(
    f"{'OVERALL':<20} {len(all_scores):>4} {statistics.mean(all_scores):>6.2f} "
    f"{statistics.median(all_scores):>8.2f} {statistics.stdev(all_scores):>6.2f} "
    f"{min(all_scores):>5} {max(all_scores):>5}"
)

print("\n=== AGENT ROUTING COVERAGE ===\n")
from collections import Counter
agent_combos = Counter()
for item in data:
    agents = tuple(sorted(item.get("agents_used", [])))
    agent_combos[agents] += 1

for combo, count in agent_combos.most_common():
    print(f"  {' + '.join(combo) if combo else '(none)':<40} {count:>4} questions")

print(f"\n  Total questions evaluated: {len(data)}")
