#!/usr/bin/env python3
"""Summarise the decomposition across several seeds.

A single run of an LLM simulation is an anecdote. Running the same three
scenarios under several seeds and reporting the spread is what turns it into a
result worth reading.

    python scripts/aggregate_trials.py 42 43 44
"""
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUCKETS = ["new", "stolen", "ghost", "lost"]


def load(seed: int):
    path = os.path.join(REPO, "demo", f"ghost_summary_s{seed}.json")
    if not os.path.exists(path):
        print(f"  (no summary for seed {seed}: {os.path.basename(path)})")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def spread(values):
    """mean and range, written the way a one-line table cell wants it."""
    if not values:
        return "-"
    mean = statistics.mean(values)
    if len(values) == 1:
        return f"{mean:.0f}"
    return f"{mean:.1f} ({min(values)}-{max(values)})"


def main() -> None:
    seeds = [int(s) for s in sys.argv[1:]]
    if not seeds:
        sys.exit("usage: aggregate_trials.py <seed> [<seed> ...]")

    trials = [t for t in (load(s) for s in seeds) if t]
    if not trials:
        sys.exit("no trial summaries found")

    # Every trial holds one summary per treated scenario, in the same order
    by_campaign = {}
    for trial in trials:
        for summary in trial:
            name = summary["treated_dir"].replace("output_", "").split("_s")[0]
            by_campaign.setdefault(name, []).append(summary)

    print()
    print(f"{len(trials)} trials, seeds {seeds}")
    print("=" * 78)
    header = f"{'campaign':<12}" + "".join(f"{b.upper():>14}" for b in BUCKETS)
    print(header + f"{'reported':>12}{'real':>10}{'ratio':>10}")
    print("-" * 78)

    rows = {}
    for name, summaries in by_campaign.items():
        counts = {b: [len(s["buckets"][b]) for s in summaries] for b in BUCKETS}
        reported = [s["attributed_cv"] for s in summaries]
        real = [s["incremental_brand"] for s in summaries]
        ratios = [s["ghost_multiple"] for s in summaries if s["ghost_multiple"] is not None]

        line = f"{name:<12}" + "".join(f"{spread(counts[b]):>14}" for b in BUCKETS)
        line += f"{spread(reported):>12}{spread(real):>10}"
        line += f"{(f'{statistics.mean(ratios):.2f}x' if ratios else 'n/a'):>10}"
        print(line)
        rows[name] = {
            "trials": len(summaries),
            "buckets": counts,
            "attributed_cv": reported,
            "incremental_brand": real,
            "ghost_multiple": ratios,
        }
    print("=" * 78)
    print("ratio = conversions the platform reports / conversions that were actually gained")
    print()

    out = os.path.join(REPO, "demo", "trials_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"seeds": seeds, "campaigns": rows}, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
