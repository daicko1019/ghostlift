#!/usr/bin/env python3
"""Check that two runs really are the same world until the ad lands.

The whole argument of this project rests on one claim: before the campaign
starts, the advertised world and the counterfactual world are not merely
similar, they are identical - same people, same positions, same words, same
reasoning. Every difference after that step is therefore caused by the ad and
nothing else.

That is a checkable claim, so this checks it rather than asserting it.

    python scripts/verify_twins.py output_noad output_broadcast --ad-step 4

Exits non-zero if the runs diverge before the campaign, which is the only
outcome that would invalidate the decomposition.
"""
import argparse
import json
import os
import sys
from typing import Dict, List

# Files whose contents must match step for step before the campaign begins
TRACKED = ["metrics.jsonl", "messages.jsonl", "memory_reasoning.jsonl"]


def load(output_dir: str, name: str) -> List[Dict]:
    path = os.path.join(output_dir, name)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def before(records: List[Dict], step: int) -> List[Dict]:
    return [r for r in records if r.get("step", 0) < step]


def first_difference(a: List[Dict], b: List[Dict]) -> str:
    """Describe the first place the two lists stop agreeing."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            keys = sorted(set(x) | set(y))
            for k in keys:
                if x.get(k) != y.get(k):
                    return (f"record {i} (step {x.get('step')}), field '{k}':\n"
                            f"      A: {json.dumps(x.get(k), ensure_ascii=False)[:220]}\n"
                            f"      B: {json.dumps(y.get(k), ensure_ascii=False)[:220]}")
    if len(a) != len(b):
        return f"different number of records before the campaign: {len(a)} vs {len(b)}"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("control", help="output dir of the no-ad run")
    parser.add_argument("treated", help="output dir of the advertised run")
    parser.add_argument("--ad-step", type=int, required=True,
                        help="the campaign's start_step; steps before this must match")
    args = parser.parse_args()

    print(f"Comparing {args.control} and {args.treated}, steps 1..{args.ad_step - 1}")
    print()

    ok = True
    for name in TRACKED:
        a = before(load(args.control, name), args.ad_step)
        b = before(load(args.treated, name), args.ad_step)

        if not a and not b:
            print(f"  {name:<24} (no records before the campaign)")
            continue

        # metrics.jsonl records the ad delivery itself, which is legitimately
        # empty in the control run; comparing it as-is would flag a difference
        # that is not a divergence in behaviour
        if name == "metrics.jsonl":
            a = [{k: v for k, v in r.items() if k not in ("ads_delivered", "impressions_cum")} for r in a]
            b = [{k: v for k, v in r.items() if k not in ("ads_delivered", "impressions_cum")} for r in b]
            b = [{**r, "agents": [{**ag, "ads_seen": 0} for ag in r["agents"]]} for r in b]
            a = [{**r, "agents": [{**ag, "ads_seen": 0} for ag in r["agents"]]} for r in a]

        if a == b:
            print(f"  {name:<24} identical  ({len(a)} records)")
        else:
            ok = False
            print(f"  {name:<24} DIVERGED")
            print(f"      {first_difference(a, b)}")

    print()
    if ok:
        print("The two worlds are identical up to the campaign. Every later "
              "difference is attributable to the ad.")
        sys.exit(0)

    print("The two worlds diverged before the campaign, so the decomposition "
          "cannot attribute later differences to the ad alone.")
    print("Usual causes: the seed is not set in one of the configs, the two "
          "runs used different models, or the Ollama server ignored the seed.")
    sys.exit(1)


if __name__ == "__main__":
    main()
