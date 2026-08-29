#!/usr/bin/env python3
"""List the runs present in the repository so the dashboard can offer them.

Run directories are named output_<scenario>[_s<seed>]. Which ones exist depends
on what was run - one trial, several seeds, or a subset - so the dashboard reads
this manifest instead of hard-coding names.
"""
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATTERN = re.compile(r"^output_(?P<scenario>[a-z]+)(?:_s(?P<seed>\d+))?$")


def steps_in(path: str) -> int:
    metrics = os.path.join(path, "metrics.jsonl")
    if not os.path.exists(metrics):
        return 0
    with open(metrics, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main() -> None:
    runs = []
    for entry in sorted(os.listdir(REPO)):
        match = PATTERN.match(entry)
        if not match or not os.path.isdir(os.path.join(REPO, entry)):
            continue
        steps = steps_in(os.path.join(REPO, entry))
        if steps == 0:
            continue
        runs.append({
            "dir": entry,
            "scenario": match.group("scenario"),
            "seed": int(match.group("seed")) if match.group("seed") else None,
            "steps": steps,
        })

    # Every treated run is paired with the no-ad run of the same seed; without
    # that partner there is nothing to compare it against
    controls = {r["seed"]: r["dir"] for r in runs if r["scenario"] == "noad"}
    pairs = [
        {"control": controls[r["seed"]], "treated": r["dir"],
         "scenario": r["scenario"], "seed": r["seed"],
         "steps": min(r["steps"], steps_in(os.path.join(REPO, controls[r["seed"]])))}
        for r in runs
        if r["scenario"] != "noad" and r["seed"] in controls
    ]

    out = os.path.join(REPO, "demo", "runs.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"runs": runs, "pairs": pairs}, f, ensure_ascii=False, indent=2)

    print(f"Wrote {out}: {len(runs)} run(s), {len(pairs)} comparable pair(s)")
    for p in pairs:
        seed = f"seed {p['seed']}" if p["seed"] is not None else "default seed"
        print(f"  {p['control']:<24} vs {p['treated']:<24} ({seed}, {p['steps']} steps)")


if __name__ == "__main__":
    main()
