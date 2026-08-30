#!/usr/bin/env python3
"""Decompose the conversions an ad platform would claim, into what the ad
actually did.

Two runs of the same world are compared agent by agent: the counterfactual run
(no advertising) and a treated run (some campaign). Because both runs share a
seed, agent 3 in one run is the same person as agent 3 in the other, which is
the comparison no real advertiser can ever make.

Every conversion the platform would attribute to the campaign falls into
exactly one of three buckets:

  NEW     bought nothing in the counterfactual, bought the brand here
          -> demand the ad genuinely created
  STOLEN  bought the rival in the counterfactual, bought the brand here
          -> demand moved from a competitor, real for the brand, zero for the market
  GHOST   bought the brand in BOTH runs
          -> the ad was billed for a sale that was going to happen anyway

And one bucket the platform never reports:

  LOST    bought the brand in the counterfactual, did NOT buy it here
          -> the campaign destroyed a sale it would otherwise have had

Usage:
    python scripts/ghost_analysis.py output_noad output_broadcast [--brand brandco]
    python scripts/ghost_analysis.py output_noad output_broadcast output_retarget
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional


def load_final_state(output_dir: str) -> Dict:
    """Read metrics.jsonl and return the last step's record."""
    path = os.path.join(output_dir, "metrics.jsonl")
    if not os.path.exists(path):
        sys.exit(f"No metrics.jsonl in {output_dir}. Did the run finish?")

    last = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)
    if last is None:
        sys.exit(f"metrics.jsonl in {output_dir} is empty.")
    return last


def load_series(output_dir: str) -> List[Dict]:
    """Read every step record, in order."""
    path = os.path.join(output_dir, "metrics.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def purchases_by_agent(state: Dict) -> Dict[int, Optional[str]]:
    """agent_id -> place it bought at, or None if it never bought."""
    return {a["id"]: a.get("purchase_place") for a in state["agents"]}


def classify(control: Dict, treated: Dict, brand: str) -> Dict[str, List[int]]:
    """Sort every agent into one bucket by comparing its two outcomes."""
    ctrl = purchases_by_agent(control)
    trt = purchases_by_agent(treated)

    buckets: Dict[str, List[int]] = {
        "new": [], "stolen": [], "ghost": [], "lost": [], "untouched": [],
    }

    for agent_id in sorted(trt):
        before = ctrl.get(agent_id)
        after = trt.get(agent_id)

        bought_brand_before = before == brand
        bought_brand_after = after == brand

        if bought_brand_after and not bought_brand_before:
            buckets["stolen" if before is not None else "new"].append(agent_id)
        elif bought_brand_after and bought_brand_before:
            buckets["ghost"].append(agent_id)
        elif bought_brand_before and not bought_brand_after:
            buckets["lost"].append(agent_id)
        else:
            buckets["untouched"].append(agent_id)

    return buckets


def summarise(control_dir: str, treated_dir: str, brand: str, cpi: float) -> Dict:
    control = load_final_state(control_dir)
    treated = load_final_state(treated_dir)
    buckets = classify(control, treated, brand)

    # What the ad platform reports: everyone who saw an ad and then bought the
    # brand. It has no way to know which of them would have bought regardless.
    attributed = len(buckets["new"]) + len(buckets["stolen"]) + len(buckets["ghost"])

    # What the brand's P&L actually gained: sales it did not have before,
    # minus the ones the campaign cost it.
    incremental_brand = len(buckets["new"]) + len(buckets["stolen"]) - len(buckets["lost"])

    # What the market gained. This has to come from the actual totals, not from
    # the brand's buckets: a campaign can pull people towards a shelf that then
    # empties, and those people may buy nowhere at all. Counting only the
    # brand's own gains and losses misses them entirely, which is what an
    # earlier version of this script did.
    total_control = sum(1 for p in purchases_by_agent(control).values() if p)
    total_treated = sum(1 for p in purchases_by_agent(treated).values() if p)
    incremental_market = total_treated - total_control

    # Sales the other shops lost. Nobody reports these either, and for a
    # campaign that empties its own shelf they can dwarf the brand's gain.
    others_control = total_control - sum(
        1 for p in purchases_by_agent(control).values() if p == brand)
    others_treated = total_treated - sum(
        1 for p in purchases_by_agent(treated).values() if p == brand)

    spend = treated["impressions_cum"] * cpi

    return {
        "control_dir": control_dir,
        "treated_dir": treated_dir,
        "brand": brand,
        "buckets": buckets,
        "attributed_cv": attributed,
        "incremental_brand": incremental_brand,
        "incremental_market": incremental_market,
        "impressions": treated["impressions_cum"],
        "spend": spend,
        # How many times bigger the reported number is than the real one.
        # None when the real number is zero or negative: the ratio stops
        # meaning anything there, and that case deserves saying out loud.
        "ghost_multiple": (attributed / incremental_brand) if incremental_brand > 0 else None,
        "reported_cpa": (spend / attributed) if attributed else None,
        "true_cpa": (spend / incremental_brand) if incremental_brand > 0 else None,
        "other_sales_control": others_control,
        "other_sales_treated": others_treated,
        "brand_sales_control": sum(1 for p in purchases_by_agent(control).values() if p == brand),
        "brand_sales_treated": sum(1 for p in purchases_by_agent(treated).values() if p == brand),
        "total_sales_control": sum(1 for p in purchases_by_agent(control).values() if p),
        "total_sales_treated": sum(1 for p in purchases_by_agent(treated).values() if p),
    }


def print_report(s: Dict) -> None:
    b = s["buckets"]
    print()
    print("=" * 66)
    print(f"  {s['treated_dir']}  vs  {s['control_dir']}   (brand: {s['brand']})")
    print("=" * 66)
    print(f"  Impressions bought        : {s['impressions']}  (spend {s['spend']:.0f})")
    print()
    print(f"  Conversions the platform would report : {s['attributed_cv']}")
    print(f"    NEW    demand the ad created        : {len(b['new']):>2}  agents {b['new']}")
    print(f"    STOLEN taken from the rival         : {len(b['stolen']):>2}  agents {b['stolen']}")
    print(f"    GHOST  would have bought anyway     : {len(b['ghost']):>2}  agents {b['ghost']}")
    print()
    print(f"  Never reported by the platform:")
    print(f"    LOST   sales the campaign destroyed : {len(b['lost']):>2}  agents {b['lost']}")
    print()
    print(f"  True incremental sales for the brand  : {s['incremental_brand']}")
    print(f"  Sales at every other shop             : "
          f"{s['other_sales_control']} -> {s['other_sales_treated']} "
          f"({s['other_sales_treated'] - s['other_sales_control']:+d})")
    print(f"  Purchases in the whole market         : "
          f"{s['total_sales_control']} -> {s['total_sales_treated']} "
          f"({s['incremental_market']:+d})")
    if s["ghost_multiple"] is not None:
        print(f"  Reported / real                       : {s['ghost_multiple']:.2f}x")
    else:
        print(f"  Reported / real                       : undefined "
              f"(the campaign produced no net gain)")
    if s["reported_cpa"] is not None:
        true_cpa = f"{s['true_cpa']:.1f}" if s["true_cpa"] is not None else "infinite"
        print(f"  Cost per conversion  reported {s['reported_cpa']:.1f}  /  real {true_cpa}")
    print()


def plot(summaries: List[Dict], control_dir: str, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # Left: the decomposition, one stacked bar per campaign
    labels = [os.path.basename(s["treated_dir"].rstrip("/")).replace("output_", "")
              for s in summaries]
    new = [len(s["buckets"]["new"]) for s in summaries]
    stolen = [len(s["buckets"]["stolen"]) for s in summaries]
    ghost = [len(s["buckets"]["ghost"]) for s in summaries]
    lost = [-len(s["buckets"]["lost"]) for s in summaries]

    x = range(len(summaries))
    ax1.bar(x, new, label="NEW (ad created it)", color="#2a9d8f")
    ax1.bar(x, stolen, bottom=new, label="STOLEN (from rival)", color="#e9c46a")
    ax1.bar(x, ghost, bottom=[n + s for n, s in zip(new, stolen)],
            label="GHOST (would have bought anyway)", color="#bfbfbf")
    ax1.bar(x, lost, label="LOST (campaign destroyed it)", color="#e76f51")
    ax1.axhline(0, color="black", linewidth=0.8)

    for i, s in enumerate(summaries):
        ax1.text(i, s["attributed_cv"] + 0.14,
                 f"platform reports {s['attributed_cv']}\nreally gained {s['incremental_brand']}",
                 ha="center", va="bottom", fontsize=9)

    # Headroom for the annotations and the legend, which otherwise sit on the bars
    top = max([s["attributed_cv"] for s in summaries] + [1])
    bottom = min(lost + [0])
    ax1.set_ylim(bottom - 0.6, top + 1.9)

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("agents")
    ax1.set_title("What the reported conversions were actually made of")
    ax1.legend(fontsize=8, loc="lower left", framealpha=0.95)
    ax1.grid(axis="y", alpha=0.3)

    # Right: cumulative brand sales over time, treated runs against the control
    def brand_curve(output_dir: str, brand: str) -> List[int]:
        return [
            sum(1 for a in rec["agents"] if a.get("purchase_place") == brand)
            for rec in load_series(output_dir)
        ]

    brand = summaries[0]["brand"]
    ctrl_curve = brand_curve(control_dir, brand)
    ax2.plot(range(1, len(ctrl_curve) + 1), ctrl_curve,
             label="no ad (counterfactual)", color="black", linestyle="--", linewidth=2)
    for s in summaries:
        curve = brand_curve(s["treated_dir"], brand)
        ax2.plot(range(1, len(curve) + 1), curve,
                 label=os.path.basename(s["treated_dir"].rstrip("/")).replace("output_", ""),
                 linewidth=2)

    ax2.set_xlabel("step")
    ax2.set_ylabel(f"cumulative {brand} sales")
    n_agents = len(load_series(control_dir)[0]["agents"])
    ax2.set_title(f"{brand} sales: every run is the same {n_agents} people")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("control", help="output dir of the no-ad run")
    parser.add_argument("treated", nargs="+", help="output dir(s) of the advertised run(s)")
    parser.add_argument("--brand", default="brandco", help="place the advertiser owns")
    parser.add_argument("--cost-per-impression", type=float, default=1.0)
    parser.add_argument("-o", "--out", default="demo/ghost_decomposition.png")
    parser.add_argument("--json-out", default=None, help="also write the summary as JSON")
    args = parser.parse_args()

    summaries = [
        summarise(args.control, t, args.brand, args.cost_per_impression)
        for t in args.treated
    ]
    for s in summaries:
        print_report(s)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)
        print(f"Wrote {args.json_out}")

    plot(summaries, args.control, args.out)


if __name__ == "__main__":
    main()
