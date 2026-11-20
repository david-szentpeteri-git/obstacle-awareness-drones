# -*- coding: utf-8 -*-
#NEWLY ADDED FOR CS5478 PROJECT
"""
Parallel evaluation launcher: every trained model (condition x seed) evaluated
in its own process on one scenario suite, with a live progress table, then one
merged and count-verified records.csv.

  wall-gap suite (default; evaluation/wall_gap.py):
    .\\venv\\Scripts\\python.exe -B -u run_wall_gap_eval.py --episodes 200 --results-dir results/compact_wall_gap
  unseen training-like worlds with an observation probe (evaluation/obs_variants.py):
    .\\venv\\Scripts\\python.exe -B -u run_wall_gap_eval.py --suite preset --probe v1 --conditions sensing_only,full --results-dir results/perception/p1_v1_preset
  sampled instead of deterministic actions:
    .\\venv\\Scripts\\python.exe -B -u run_wall_gap_eval.py --suite preset --sample-actions --results-dir results/perception/sampled_preset

Follow progress from another terminal with
  Get-Content .\\<results-dir>_launch.log -Wait -Tail 12
when the launcher's output is redirected to that log.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from config.experiment import CONDITIONS
from config.ppo_defaults import get_preset
from evaluation.obs_variants import PROBE_NAMES
from evaluation.scenarios import resolve_scenarios
from evaluation.wall_gap import wall_gap_scenarios

ROOT = Path(__file__).resolve().parent
LINE_RE = re.compile(r"^\s+\[(\S+) \| (\S+) s(\d+) \| (\d+)/(\d+)\] seed=\d+ reached=(\d) collided=(\d)")


#PARSE CLI ARGS
def parse_args():
    parser = argparse.ArgumentParser(description="Parallel evaluation of all trained models on one scenario suite")
    parser.add_argument("--preset", default="compact")
    parser.add_argument("--suite", default="wall_gap", choices=["wall_gap", "preset"])
    parser.add_argument("--scenarios", default="all", help="all, or comma-separated scenario names of the suite")
    parser.add_argument("--worlds-preset", default=None, help="take the scenario suite from this preset (worlds only; model checks use --preset)")
    parser.add_argument("--probe", default=None, choices=list(PROBE_NAMES), help="observation variant fed to the models (default: normal input)")
    parser.add_argument("--sample-actions", action="store_true", help="sample actions instead of the deterministic mean")
    parser.add_argument("--checkpoint", default=None, help="evaluate a saved checkpoint instead of final models: latest, or a name such as stage_2 (needed for stalled runs)")
    parser.add_argument("--conditions", default="all")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--episodes", type=int, default=200, help="worlds per scenario")
    parser.add_argument("--results-dir", default="results/compact_wall_gap")
    parser.add_argument("--progress-interval", type=int, default=60)

    return parser.parse_args()


#SCENARIO NAMES THE RUN COVERS
def scenario_names(args) -> list[str]:
    preset = get_preset(args.worlds_preset or args.preset)
    suite = wall_gap_scenarios(preset) if args.suite == "wall_gap" else resolve_scenarios(preset)
    names = list(suite) if args.scenarios == "all" else [s.strip() for s in args.scenarios.split(",") if s.strip()]
    unknown = [n for n in names if n not in suite]

    if unknown:
        raise SystemExit(f"[ABORT] unknown scenarios for suite {args.suite}: {unknown}")

    return names


#ONE EVALUATE COMMAND PER (CONDITION, SEED)
def build_commands(args, names, seeds, eval_dir: Path):
    cmds = {}
    extra = []

    if args.scenarios != "all":
        extra += ["--scenarios", args.scenarios]
    if args.worlds_preset:
        extra += ["--worlds-preset", args.worlds_preset]
    if args.probe:
        extra += ["--probe", args.probe]
    if args.sample_actions:
        extra += ["--sample-actions"]
    if args.checkpoint:
        extra += ["--checkpoint", args.checkpoint]

    for name in names:
        for seed in seeds:
            cmds[(name, seed)] = [
                sys.executable, "-B", "-u", "-m", "evaluation.evaluate",
                "--suite", args.suite, "--preset", args.preset,
                "--conditions", name, "--train-seeds", str(seed), "--seed", "0",
                "--episodes-random", str(args.episodes),
                "--out-dir", str(eval_dir / f"_part_{name}_s{seed}"),
            ] + extra

    return cmds


#COUNT FINISHED EPISODES & OUTCOMES IN ONE PART LOG
def scan_log(path: Path):
    done = reached = collided = 0
    current = "-"

    if not path.exists():
        return done, reached, collided, current

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            match = LINE_RE.match(line)

            if match:
                done += 1
                reached += int(match.group(6))
                collided += int(match.group(7))
                current = f"{match.group(1)} {match.group(4)}/{match.group(5)}"

    return done, reached, collided, current


#PRINT PER-CONDITION PROGRESS TABLE
def print_progress(names, seeds, log_dir: Path, per_run: int, start: float, procs, label: str) -> None:
    elapsed = max(time.time() - start, 1.0)
    worst_eta = 0.0
    lines = [f"---- {label} @ {elapsed / 60:.1f} min ({datetime.now():%H:%M}) ----",
             f"{'condition':<14}{'done':>14}{'pct':>7}{'reach':>8}{'crash':>8}{'t/o':>7}{'eps/min':>9}{'eta min':>9}  running"]

    for name in names:
        done = reached = collided = 0
        current = []

        for seed in seeds:
            d_run, r_run, c_run, cur = scan_log(log_dir / f"eval_{name}_s{seed}.log")
            done, reached, collided = done + d_run, reached + r_run, collided + c_run
            current.append(f"s{seed}:{cur.split(' ')[0] if procs[(name, seed)].poll() is None else 'done'}")

        total = per_run * len(seeds)
        rate = done / (elapsed / 60)
        eta = (total - done) / rate if rate > 0 else float("inf")
        worst_eta = max(worst_eta, eta)
        pct = lambda k: f"{100 * k / done:.1f}%" if done else "-"
        eta_txt = f"{eta:.0f}" if eta != float("inf") else "-"
        lines.append(f"{name:<14}{done:>7,}/{total:<6,}{100 * done / total:>6.1f}%{pct(reached):>8}{pct(collided):>8}"
                     f"{pct(done - reached - collided):>7}{rate:>9.1f}{eta_txt:>9}  {' '.join(current)}")

    finish = f"{datetime.fromtimestamp(time.time() + worst_eta * 60):%H:%M}" if worst_eta != float("inf") else "-"
    lines.append(f"estimated finish {finish} (rates pooled over scenarios so far)")
    print("\n".join(lines) + "\n", flush=True)


#MERGE PART RECORDS IN CANONICAL ORDER & VERIFY COMPLETENESS
def merge_records(names, seeds, eval_dir: Path, per_run: int) -> int:
    header = None
    rows = []

    for name in names:
        for seed in seeds:
            path = eval_dir / f"_part_{name}_s{seed}" / "records.csv"
            lines = path.read_text(encoding="utf-8").splitlines()

            if header is None:
                header = lines[0]
            elif lines[0] != header:
                raise SystemExit(f"[ABORT] record schema differs for {name} s{seed}")

            part = [line for line in lines[1:] if line.strip()]

            if len(part) != per_run:
                raise SystemExit(f"[ABORT] {name} s{seed}: {len(part)} records, expected {per_run}")

            rows.extend(part)

    (eval_dir / "records.csv").write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")

    return len(rows)


#LAUNCH ALL PARTS, TRACK PROGRESS, MERGE
def main() -> None:
    args = parse_args()
    names = list(CONDITIONS) if args.conditions == "all" else [n.strip() for n in args.conditions.split(",")]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    res_dir = ROOT / args.results_dir
    eval_dir = res_dir / "eval"
    log_dir = res_dir / "logs"
    scenarios = scenario_names(args)
    per_run = args.episodes * len(scenarios)
    label = f"{args.suite} eval" + (f" probe={args.probe}" if args.probe else "") + (" sampled" if args.sample_actions else "")

    if res_dir.exists() and any(res_dir.iterdir()):
        raise SystemExit(f"[ABORT] {res_dir} is not empty; choose a fresh --results-dir")

    log_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    cmds = build_commands(args, names, seeds, eval_dir)
    plan = {"started": datetime.now().isoformat(timespec="seconds"), "suite": args.suite, "scenarios": scenarios,
            "probe": args.probe, "sample_actions": args.sample_actions, "episodes_per_scenario": args.episodes,
            "runs": len(cmds), "episodes_per_run": per_run, "commands": {f"{n}_s{s}": c for (n, s), c in cmds.items()}}
    (res_dir / "run_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    print(f"{label}: {len(cmds)} models x {per_run:,} episodes ({len(scenarios)} scenarios x {args.episodes} worlds)", flush=True)
    print(f"results: {res_dir}\n", flush=True)

    start = time.time()
    procs = {}
    handles = []

    for (name, seed), cmd in cmds.items():
        handle = open(log_dir / f"eval_{name}_s{seed}.log", "w", encoding="utf-8")
        handles.append(handle)
        procs[(name, seed)] = subprocess.Popen(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        print(f"  [LAUNCH] {name} s{seed}", flush=True)

    print("", flush=True)
    last = 0.0

    try:
        while any(proc.poll() is None for proc in procs.values()):
            if time.time() - last >= args.progress_interval:
                print_progress(names, seeds, log_dir, per_run, start, procs, label)
                last = time.time()

            time.sleep(5)
    finally:
        for handle in handles:
            handle.close()

    print_progress(names, seeds, log_dir, per_run, start, procs, label)
    failed = [f"{n} s{s}" for (n, s), proc in procs.items() if proc.returncode != 0]

    if failed:
        raise SystemExit(f"[ABORT] evaluation failed for: {', '.join(failed)} (see {log_dir})")

    total = merge_records(names, seeds, eval_dir, per_run)
    print(f"[SAVED] merged {total:,} episode records -> {eval_dir / 'records.csv'}", flush=True)
    print(f"[DONE] {label} finished in {(time.time() - start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
