#!/usr/bin/env python3
"""Run the scenarios, optionally across several seeds, and analyse the results.

Works the same on Windows, macOS and Linux, which the shell scripts next to it
do not. Two modes:

    # one trial with the seed written in the scenario files
    python scripts/run_all.py

    # several trials, so the decomposition can be reported as a distribution
    # rather than a single number
    python scripts/run_all.py --seeds 42 43 44

Runs can overlap with --parallel N. Whatever N is, the model is warmed up
first: a run that triggers the model load generates different messages and
memories from one that finds it resident, and those feed back into the world,
so twin runs started cold drift apart before the ad ever lands.
"""
import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENARIOS = ["noad", "broadcast", "retarget"]


def warm_up(url: str = None) -> None:
    """Load the model before any run starts.

    A run that triggers the model load sees different generation from one that
    finds it already resident - not in the actions taken, but in the messages
    and memories the agents produce, which do feed back into the world. Twin
    runs must therefore all start warm, so one throwaway request is made first.
    """
    import urllib.request

    base = (url or "http://localhost:11434").rstrip("/")
    model = None
    for name in SCENARIOS:
        path = os.path.join(REPO, f"scenario_{name}.yaml")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("model:"):
                        model = line.split(":", 1)[1].strip().strip('"\'')
                        break
        if model:
            break
    if not model:
        return

    body = json.dumps({"model": model, "prompt": "warm up", "stream": False,
                       "options": {"num_predict": 1, "seed": 42}}).encode()
    req = urllib.request.Request(f"{base}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    print(f"  warming {model} on {base} ...", end="", flush=True)
    try:
        with urllib.request.urlopen(req, timeout=600):
            pass
        print(" ready")
    except Exception as e:
        print(f" failed ({e}); runs may not be comparable")


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
    scenario, seed, llm_url = job
    config, out_dir = config_for(scenario, seed)
    label = scenario if seed is None else f"{scenario}(seed {seed})"
    log_name = f"run_{scenario}.out" if seed is None else f"run_{scenario}_s{seed}.out"
    log_path = os.path.join(REPO, log_name)

    cmd = [venv_python(), os.path.join(REPO, "src", "main.py"), "--config", config]
    if llm_url:
        cmd += ["--llm-url", llm_url]

    print(f"  start  {label}" + (f"  -> {llm_url}" if llm_url else ""))
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)

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
    parser.add_argument("--llm-urls", nargs="+", default=None,
                        help="one or more Ollama endpoints; runs are dealt out "
                             "across them round-robin, so several GPUs can share "
                             "the work (e.g. http://gpu1:11434 http://gpu2:11434)")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else [None]

    # A seed's scenarios are twins: they must produce identical LLM output for
    # identical prompts, which is only safe on one and the same endpoint. Two
    # GPUs can differ in the last bits of a float and that is enough to break
    # the comparison. So endpoints are dealt out per seed, never per scenario.
    urls = args.llm_urls or [None]
    jobs = [
        (s, seed, urls[i % len(urls)])
        for i, seed in enumerate(seeds)
        for s in args.scenarios
    ]

    for url in urls:
        warm_up(url)

    print(f"{len(jobs)} run(s), {args.parallel} at a time"
          + (f", across {len(urls)} endpoint(s)" if args.llm_urls else ""))
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

    subprocess.run([venv_python(), os.path.join(REPO, "scripts", "make_manifest.py")],
                   cwd=REPO, check=False)

    if len(seeds) > 1:
        print("\nAggregate across seeds:")
        subprocess.run(
            [venv_python(), os.path.join(REPO, "scripts", "aggregate_trials.py")]
            + [str(s) for s in seeds],
            cwd=REPO, check=False,
        )


if __name__ == "__main__":
    main()
