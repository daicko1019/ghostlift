#!/usr/bin/env python3
"""Run the scenarios, optionally across several seeds, and analyse the results.

Works the same on Windows, macOS and Linux, which the shell scripts next to it
do not. Two modes:

    # one trial with the seed written in the scenario files
    python scripts/run_all.py

    # several trials, so the decomposition can be reported as a distribution
    # rather than a single number
    python scripts/run_all.py --seeds 42 43 44

Runs can overlap with --parallel N. Ollama batches concurrent requests into one
GPU batch, so N worlds cost far less than N times one world - but the server has
to have been started with OLLAMA_NUM_PARALLEL>=N or the extra requests just
queue up.
"""
import argparse
import concurrent.futures
import os
import re
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENARIOS = ["noad", "broadcast", "retarget"]


def venv_python() -> str:
    """Prefer the project venv, so the script works when run by a bare python."""
    for rel in ("venv/bin/python", "venv/Scripts/python.exe"):
        candidate = os.path.join(REPO, rel)
        if os.path.exists(candidate):
            return candidate
    return sys.executable


def config_for(scenario: str, seed: int = None) -> tuple:
    """Return (config_path, output_dir).

    Without a seed the scenario file is used untouched. With one, a temporary
    copy is written with the seed and the output directory swapped, so trials
    never overwrite each other.
    """
    src = os.path.join(REPO, f"scenario_{scenario}.yaml")
    if seed is None:
        return src, os.path.join(REPO, f"output_{scenario}")

    with open(src, encoding="utf-8") as f:
        text = f.read()

    out_dir = f"output_{scenario}_s{seed}"
    text = re.sub(r"^seed:\s*\d+", f"seed: {seed}", text, count=1, flags=re.M)
    text = re.sub(r"^(\s*)seed:\s*\d+(\s*$)", rf"\g<1>seed: {seed}\g<2>", text, flags=re.M)
    text = text.replace(f'output_dir: "output_{scenario}"', f'output_dir: "{out_dir}"')
    text = text.replace(f'log_file: "simulation_{scenario}.log"',
                        f'log_file: "simulation_{scenario}_s{seed}.log"')

    fd, path = tempfile.mkstemp(prefix=f"{scenario}_s{seed}_", suffix=".yaml", dir=REPO)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    return path, os.path.join(REPO, out_dir)


def run_one(job: tuple) -> tuple:
    scenario, seed = job
    config, out_dir = config_for(scenario, seed)
    label = scenario if seed is None else f"{scenario}(seed {seed})"
    log_name = f"run_{scenario}.out" if seed is None else f"run_{scenario}_s{seed}.out"
    log_path = os.path.join(REPO, log_name)

    print(f"  start  {label}")
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(
            [venv_python(), os.path.join(REPO, "src", "main.py"), "--config", config],
            cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
        )

    # Only the generated copies are ours to delete
    if seed is not None and os.path.exists(config):
        os.remove(config)

    status = "done" if proc.returncode == 0 else f"FAILED (see {log_name})"
    print(f"  {status:<8} {label}")
    return scenario, seed, proc.returncode, out_dir


def analyse(seed, out_dirs: dict) -> None:
    suffix = "" if seed is None else f"_s{seed}"
    cmd = [
        venv_python(), os.path.join(REPO, "scripts", "ghost_analysis.py"),
        out_dirs["noad"], out_dirs["broadcast"], out_dirs["retarget"],
        "--json-out", os.path.join(REPO, "demo", f"ghost_summary{suffix}.json"),
        "-o", os.path.join(REPO, "demo", f"ghost_decomposition{suffix}.png"),
    ]
    subprocess.run(cmd, cwd=REPO, check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="run one trial per seed; omit to use the scenario files as they are")
    parser.add_argument("--parallel", type=int, default=1,
                        help="how many runs to overlap (match OLLAMA_NUM_PARALLEL)")
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else [None]
    jobs = [(s, seed) for seed in seeds for s in args.scenarios]

    print(f"{len(jobs)} run(s), {args.parallel} at a time")
    results = []
    if args.parallel > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
            results = list(pool.map(run_one, jobs))
    else:
        results = [run_one(job) for job in jobs]

    failed = [r for r in results if r[2] != 0]
    if failed:
        print(f"\n{len(failed)} run(s) failed; skipping analysis")
        sys.exit(1)

    os.makedirs(os.path.join(REPO, "demo"), exist_ok=True)
    for seed in seeds:
        out_dirs = {sc: od for sc, sd, _, od in results if sd == seed}
        if len(out_dirs) < 3:
            print(f"seed {seed}: need all three scenarios to decompose, skipping")
            continue
        analyse(seed, out_dirs)

    if len(seeds) > 1:
        print("\nAggregate across seeds:")
        subprocess.run(
            [venv_python(), os.path.join(REPO, "scripts", "aggregate_trials.py")]
            + [str(s) for s in seeds],
            cwd=REPO, check=False,
        )


if __name__ == "__main__":
    main()
