# -*- coding: utf-8 -*-
#NEWLY ADDED FOR CS5478 PROJECT
"""
One-shot experiment runner - the single command behind the report.

    python run_experiment.py            (train every run -> evaluate -> analyze)
    python run_experiment.py --smoke    (tiny end-to-end pipeline check, ~minutes,
                                         outputs isolated in models/smoke and results/smoke)
    python run_experiment.py --preset compact --train-only --parallel 12 --torch-threads 1
    python -B run_experiment.py --preset compact --train-only --dry-run

A "run" is one (condition, seed) pair: 4 conditions x 3 seeds = 12 runs by
default, so the report can quote a spread across seeds instead of a single
point estimate per condition.

Runs, in order, as subprocesses of the current interpreter:
  1. training.train_ppo   - PPO curriculum training per run (validates completed
                            artifacts before reuse; incomplete runs restart
                            from stage zero)
  2. evaluation.evaluate    - full scenario suite, every run, shared episode seeds
  3. analysis.analyze     - every figure + table + results/results.json

Training is CPU-bound in PyBullet, not in the policy: the gradient update is
about a tenth of each rollout's wall time and the net is a small MLP, so GPU
buys almost nothing here. Wall-clock parallelism comes from --parallel, which
runs several training processes side by side (SuperSuit's env copies step
sequentially inside one process, so --n-envs does not add any).

After one successful run, everything needed to write the report is on disk:
  models/<preset>/ppo_<condition>_s<seed>/  models, per-stage + periodic checkpoints,
                                            VecNormalize stats, run manifest (incl.
                                            every env seed), train summary
  results/<preset>/train_logs/              per-rollout learning-curve CSVs, per run
  results/<preset>/eval/                    per-episode records, summaries
  results/<preset>/figures|tables/          report-ready PNGs and CSV/Markdown tables
  results/<preset>/results.json             every aggregate number, the paired tests,
                                            the headline block and the run manifests
"""
from __future__ import annotations

import argparse
import json
import math
import csv
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from drone_env.curriculum import Curriculum
from config.blending import resolve_stage_ceiling
from config.experiment import apply_preset_to_condition, get_condition, resolve_condition_names, run_name
from config.ppo_defaults import TRAIN_SEEDS, get_preset

PROJECT_DIR = Path(__file__).resolve().parent


#RUN ONE PIPELINE PHASE AS A SUBPROCESS, ABORTING THE RUN ON FAILURE
def run_phase(label: str, cmd: list[str], env: dict) -> None:
    print(f"\n======== {label} ========")
    print("$", " ".join(cmd))

    log_dir = Path(env.get("CS5478_RESULTS_DIR", "results/default")) / "train_logs"

    if not log_dir.is_absolute():
        log_dir = PROJECT_DIR / log_dir

    log_dir.mkdir(parents=True, exist_ok=True)

    with (log_dir / f"phase_{label.lower()}.log").open("a", encoding="utf-8") as handle:
        handle.write("\n$ " + " ".join(cmd) + "\n")
        handle.flush()
        proc = subprocess.Popen(cmd, env=env, cwd=PROJECT_DIR, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")

        try:
            for line in proc.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
            code = proc.wait()
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
            proc.stdout.close()

    if code != 0:
        sys.exit(f"[ABORT] phase '{label}' failed with exit code {code}")


#RUN THE (CONDITION, SEED) RUNS THROUGH A FIXED-WIDTH PROCESS POOL
#READ ONE RUN'S LATEST ROLLOUT ROW FROM ITS LEARNING-CURVE CSV
def read_run_progress(results_dir: str, name: str, seed: int) -> dict | None:
    """
    The per-rollout CSV is the cheapest honest progress source: the training
    process already writes it, so reading it costs nothing and cannot disturb
    the run. Returns None until the first rollout lands.
    """
    path = os.path.join(results_dir, "train_logs", f"{name}_s{seed}.csv")

    if not os.path.isfile(path):
        return None

    try:
        with open(path, "r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return None

    if not rows:
        return None

    last = rows[-1]

    #LAST 50 ROLLOUTS = RECENT BEHAVIOUR, NOT THE WHOLE-RUN AVERAGE
    window = rows[-50:]
    episodes = sum(int(float(r["episodes"])) for r in window)
    reaches = sum(int(float(r["reaches"])) for r in window)

    return {
        "stage": int(float(last["stage"])),
        "decisions": int(float(last["decisions"])),
        "recent_reach": (reaches / episodes) if episodes else 0.0,
        "ep_rew_mean": float(last["ep_rew_mean"]),
        "rollouts": len(rows),
    }


#LATEST GATE VERDICT FOR A RUN, IF IT IS A SUCCESS-GATED RUN
def read_gate_progress(models_dir: str, name: str, seed: int) -> dict | None:
    """
    For a gated run the honest progress signal is the last VALIDATED success
    rate and the gate state, not a fraction of a budget the gate ignores.
    """
    path = os.path.join(models_dir, f"ppo_{name}_s{seed}", "stage_gate_history.json")

    if not os.path.isfile(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None

    log = data.get("validation_log") or []

    if not log:
        return None

    last = log[-1]

    return {
        "check": last.get("check"),
        "success": float(last.get("success_rate", 0.0)),
        "beta": float(last.get("beta_next_block", 0.0)),
        "state": str(last.get("state", "?")),
    }


#PRINT ONE CONSOLIDATED PROGRESS TABLE FOR EVERY LIVE/FINISHED RUN
def print_progress_table(results_dir: str, slots: list, total_decisions: int, started: float,
                         models_dir: str = "", gated: bool = False) -> None:
    elapsed = (time.time() - started) / 60.0
    print("")
    print(f"  ---- progress @ {elapsed:.1f} min ----", flush=True)

    if gated:
        print(f"  {'run':<20}{'stage':>6}{'decisions':>12}{'reach(50)':>11}"
              f"{'ep_rew':>10}{'chk':>5}{'val succ':>10}{'beta':>7}{'gate':>15}", flush=True)
    else:
        print(f"  {'run':<20}{'stage':>6}{'decisions':>12}{'of budget':>11}"
              f"{'reach(50)':>11}{'ep_rew':>10}{'state':>10}", flush=True)

    for name, seed, state in slots:
        info = read_run_progress(results_dir, name, seed)
        label = f"{name}_s{seed}"

        if info is None:
            print(f"  {label:<20}{'-':>6}{'-':>12}{'-':>11}{'-':>11}{'-':>10}{state:>10}", flush=True)

            continue

        if gated:
            gate = read_gate_progress(models_dir, name, seed)
            chk = str(gate["check"]) if gate else "-"
            succ = f"{100.0 * gate['success']:.1f}%" if gate else "-"
            beta = f"{gate['beta']:.3f}" if gate else "-"
            gstate = gate["state"] if gate else state
            print(
                f"  {label:<20}{info['stage']:>6}{info['decisions']:>12,}"
                f"{100.0 * info['recent_reach']:>10.2f}%{info['ep_rew_mean']:>10.1f}"
                f"{chk:>5}{succ:>10}{beta:>7}{gstate:>15}",
                flush=True,
            )

            continue

        pct = 100.0 * info["decisions"] / total_decisions if total_decisions else 0.0
        print(
            f"  {label:<20}{info['stage']:>6}{info['decisions']:>12,}{pct:>10.1f}%"
            f"{100.0 * info['recent_reach']:>10.2f}%{info['ep_rew_mean']:>10.1f}{state:>10}",
            flush=True,
        )


#LAUNCH ONE TRAINING SUBPROCESS PER (CONDITION, SEED) THROUGH A FIXED-WIDTH POOL
def run_training_parallel(
    names: list[str],
    seeds: list[int],
    base_cmd: list[str],
    env: dict,
    log_dir: str,
    max_parallel: int,
    progress_interval: int = 0,
    results_dir: str = "results",
    total_decisions: int = 0,
    models_dir: str = "",
    gated: bool = False,
) -> None:
    """
    One process per run, at most max_parallel alive at a time. Each process
    owns its own PyBullet instance and its own model folder, so there is no
    shared state to coordinate - the only thing to manage is not
    oversubscribing the CPU, which CS5478_TORCH_THREADS handles.

    A finished slot is refilled immediately rather than waiting for a whole
    wave, so a slow run never leaves cores idle.
    """
    os.makedirs(log_dir, exist_ok=True)

    #SEED-MAJOR ORDER, SO AN ABORTED RUN STILL LEAVES COMPLETE SEEDS BEHIND
    queue = [(name, seed) for seed in seeds for name in names]
    running: list[tuple[str, int, subprocess.Popen, object, float]] = []
    failures: list[str] = []
    done = 0
    total = len(queue)

    #LAUNCH ONE RUN & PRESERVE PREVIOUS ATTEMPT LOGS
    def launch(name: str, seed: int) -> None:
        cmd = base_cmd + ["--condition", name, "--seeds", str(seed)]
        log_path = os.path.join(log_dir, f"train_{name}_s{seed}.log")
        handle = open(log_path, "a", encoding="utf-8")
        handle.write("\n$ " + " ".join(cmd) + "\n")
        handle.flush()

        print(f"  [LAUNCH] {name} s{seed} -> {log_path}", flush=True)
        running.append(
            (
                name,
                seed,
                subprocess.Popen(cmd, env=env, cwd=PROJECT_DIR, stdout=handle, stderr=subprocess.STDOUT),
                handle,
                time.time(),
            )
        )

    run_started = time.time()
    last_report = run_started
    finished: list[tuple[str, int, str]] = []

    try:
        while queue or running:
            while queue and len(running) < max_parallel:
                launch(*queue.pop(0))

            time.sleep(5)

            #PERIODIC CONSOLIDATED TABLE: 12 INTERLEAVED STDOUT STREAMS WOULD BE
            #UNREADABLE, SO EACH RUN GETS ONE LINE READ FROM ITS OWN CSV
            if progress_interval > 0 and time.time() - last_report >= progress_interval:
                last_report = time.time()
                slots = [(n, s, "running") for n, s, _, _, _ in running] + finished
                print_progress_table(results_dir, slots, total_decisions, run_started,
                                     models_dir=models_dir, gated=gated)

            for entry in list(running):
                name, seed, proc, handle, started = entry
                code = proc.poll()

                if code is None:
                    continue

                running.remove(entry)
                handle.close()
                done += 1

                mins = (time.time() - started) / 60.0
                status = "OK" if code == 0 else f"FAILED (exit {code})"
                print(
                    f"  [DONE {done}/{total}] {name} s{seed}: {status} "
                    f"({mins:.1f} min)",
                    flush=True,
                )

                finished.append((name, seed, "OK" if code == 0 else "FAILED"))

                if code != 0:
                    failures.append(f"{name} s{seed}")

    finally:
        #A STOPPED LAUNCHER MUST NOT LEAVE WORKERS WRITING THE SLOTS BEING RESUMED
        for _, _, proc, handle, _ in running:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
            handle.close()

    if failures:
        sys.exit(
            f"[ABORT] training failed for: {', '.join(failures)} "
            f"- see the logs in {log_dir}"
        )


#EVALUATE ONE CONDITION PER PROCESS, THEN MERGE THE PER-EPISODE RECORDS
def run_eval_parallel(
    names: list[str],
    eval_cmd: list[str],
    env: dict,
    results_dir: str,
    max_parallel: int,
) -> None:
    """
    Safe to split by condition because the episode seeds are derived
    deterministically from the scenario name (CRC32, not Python's salted
    hash()), so every process faces exactly the same worlds and the records
    still pair episode-by-episode once merged.
    """
    eval_dir = os.path.join(results_dir, "eval")
    log_dir = os.path.join(results_dir, "train_logs")
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    parts: list[tuple[str, str]] = []
    running: list[tuple[str, subprocess.Popen, object, float]] = []
    failures: list[str] = []
    queue = list(names)

    #LAUNCH ONE CONDITION INTO ITS OWN PART FOLDER
    def launch(name: str) -> None:
        out_dir = os.path.join(eval_dir, f"_part_{name}")
        cmd = eval_cmd + ["--conditions", name, "--out-dir", out_dir]
        log_path = os.path.join(log_dir, f"eval_{name}.log")
        handle = open(log_path, "a", encoding="utf-8")
        handle.write("\n$ " + " ".join(cmd) + "\n")
        handle.flush()

        print(f"  [LAUNCH] eval {name} -> {log_path}", flush=True)
        parts.append((name, os.path.join(out_dir, "records.csv")))
        running.append(
            (name, subprocess.Popen(cmd, env=env, cwd=PROJECT_DIR, stdout=handle, stderr=subprocess.STDOUT), handle, time.time())
        )

    while queue or running:
        while queue and len(running) < max_parallel:
            launch(queue.pop(0))

        time.sleep(5)

        for entry in list(running):
            name, proc, handle, started = entry
            code = proc.poll()

            if code is None:
                continue

            running.remove(entry)
            handle.close()

            mins = (time.time() - started) / 60.0
            status = "OK" if code == 0 else f"FAILED (exit {code})"
            print(f"  [DONE] eval {name}: {status} ({mins:.1f} min)", flush=True)

            if code != 0:
                failures.append(name)

    if failures:
        sys.exit(f"[ABORT] evaluation failed for: {', '.join(failures)}")

    #MERGE: ONE HEADER, THEN EVERY PART'S ROWS, IN CANONICAL CONDITION ORDER
    merged = os.path.join(eval_dir, "records.csv")
    header = None
    rows: list[str] = []

    for name, path in parts:
        if not os.path.isfile(path):
            sys.exit(f"[ABORT] evaluation produced no records for {name} ({path})")

        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.read().splitlines()

        if not lines:
            sys.exit(f"[ABORT] empty records file for {name} ({path})")

        if header is None:
            header = lines[0]
        elif lines[0] != header:
            sys.exit(f"[ABORT] record schema differs for {name}: {lines[0]!r} != {header!r}")

        rows.extend(line for line in lines[1:] if line.strip())

    with open(merged, "w", encoding="utf-8", newline="") as fh:
        fh.write(header + "\n")
        fh.write("\n".join(rows) + "\n")

    print(f"[SAVED] merged {len(rows)} episode records -> {merged}")


#PARSE CLI ARGS
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="CS5478 experiment: train + evaluate + analyze")
    parser.add_argument("--conditions", default="all", help="all, both, or comma-separated names")
    parser.add_argument(
        "--seeds",
        default=",".join(str(s) for s in TRAIN_SEEDS),
        help="comma-separated run seeds (one independent training run each)",
    )
    parser.add_argument(
        "--preset",
        default="compact",
        help="named preset: compact (the reported study) or phase2 (wall training "
        "warm-started from the compact models). Passed to training and evaluation so "
        "control rates and reward profiles match.",
    )
    parser.add_argument("--scale", type=float, default=1.0, help="curriculum timestep multiplier")
    parser.add_argument("--n-envs", type=int, default=None, help="env copies per training process")
    parser.add_argument("--device", default="cpu", help="auto, cpu, or cuda (cpu is faster here)")
    parser.add_argument(
        "--parallel",
        type=int,
        default=0,
        help="train this many conditions at once (0 = sequential). Each extra "
        "process is a real wall-clock win, unlike --n-envs.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=2,
        help="torch intra-op threads per training process; keep "
        "parallel x threads under the core count",
    )
    parser.add_argument("--episodes-random", type=int, default=None, help="evaluation episodes per scenario")
    parser.add_argument("--seed", type=int, default=0, help="base seed for evaluation worlds")
    parser.add_argument(
        "--stage-ceiling", type=int, default=0,
        help="forwarded to training: max decisions a gated stage may spend before "
        "it is recorded STALLED (0 = unbounded, advance on validated success only)",
    )
    parser.add_argument(
        "--progress-interval", type=int, default=120,
        help="seconds between live progress tables during parallel training (0 disables)",
    )
    parser.add_argument("--retrain", action="store_true", help="retrain finished runs too")
    parser.add_argument("--resume", default=None, metavar="latest|CHECKPOINT",
                        help="continue stopped runs from committed checkpoints; latest also starts fresh missing slots")
    parser.add_argument("--resume-stage", type=int, default=None,
                        help="with --resume and one condition/seed, manually enter a later stage and record bypassed gates")
    parser.add_argument("--resume-reason", default=None,
                        help="report explanation for --resume-stage")
    parser.add_argument("--eval-checkpoint", default=None, metavar="latest|CHECKPOINT",
                        help="with --skip-train, evaluate committed checkpoints from unfinished runs")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-analyze", action="store_true")
    parser.add_argument("--train-only", action="store_true", help="train and save all training artifacts; postpone evaluation and analysis")
    parser.add_argument("--dry-run", action="store_true", help="print resolved configuration, budgets, output paths and commands; launch nothing and write no files")
    parser.add_argument("--models-dir", default=None, help="override preset model output directory")
    parser.add_argument("--results-dir", default=None, help="override preset results/log output directory")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="end-to-end check: 20,480 steps/run at current rollout size, few episodes, "
        "isolated output folders (models/smoke/, results/smoke/)",
    )

    return parser.parse_args(argv)


#RESOLVE CLI CONFIG, OUTPUTS & PHASE COMMANDS WITHOUT WRITING OR LAUNCHING
def build_plan(args, inherited_env=None) -> tuple[dict, dict]:
    env = dict(os.environ)
    if inherited_env is not None:
        env = dict(inherited_env)

    preset = get_preset(args.preset)
    models_dir = args.models_dir or preset.get("models_dir") or env.get("CS5478_MODELS_DIR", "models/default")
    results_dir = args.results_dir or preset.get("results_dir") or env.get("CS5478_RESULTS_DIR", "results/default")
    if args.smoke:
        models_dir = args.models_dir or "models/smoke"
        results_dir = args.results_dir or "results/smoke"

    env["CS5478_MODELS_DIR"] = str(models_dir)
    env["CS5478_RESULTS_DIR"] = str(results_dir)
    #-u KEEPS CHILD STDOUT UNBUFFERED. WITHOUT IT, A PHASE WRITING TO A LOG
    #FILE BLOCK-BUFFERS ITS PROGRESS LINES, SO A PERFECTLY HEALTHY RUN LOOKS
    #LIKE A HUNG ONE FOR MINUTES AT A TIME.
    py = [sys.executable, "-u", "-m"]

    if args.torch_threads > 0:
        env["CS5478_TORCH_THREADS"] = str(args.torch_threads)

    #RESOLVED EARLY SO A TYPO FAILS BEFORE HOURS OF TRAINING
    names = resolve_condition_names(args.conditions)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    if not names or not seeds or len(set(names)) != len(names) or len(set(seeds)) != len(seeds):
        raise ValueError("conditions and seeds must be nonempty and unique")
    if not math.isfinite(args.scale) or args.scale <= 0:
        raise ValueError("scale must be finite and positive")
    if args.n_envs not in (None, 1):
        raise ValueError("use --n-envs 1 and --parallel for independent training processes")
    if any(seed < 0 or seed > 9 for seed in seeds):
        raise ValueError("run seeds must be 0 through 9 to preserve the training seed band")
    if args.parallel < 0 or args.torch_threads < 0:
        raise ValueError("parallel and torch-threads must be nonnegative")
    if args.episodes_random is not None and args.episodes_random <= 0:
        raise ValueError("evaluation episode count must be positive")
    if args.train_only and args.skip_train:
        raise ValueError("--train-only cannot be combined with --skip-train")
    if args.resume and (args.retrain or args.smoke or args.skip_train):
        raise ValueError("--resume cannot be combined with --retrain, --smoke or --skip-train")
    if args.eval_checkpoint and (not args.skip_train or args.skip_eval):
        raise ValueError("--eval-checkpoint requires --skip-train and evaluation enabled")
    if any(selector and selector != "latest" for selector in (args.resume, args.eval_checkpoint)) and len(names) * len(seeds) != 1:
        raise ValueError("an explicit checkpoint requires one condition and seed")
    if args.resume and not preset.get("stage_gate"):
        raise ValueError("checkpoint continuation requires a success-gated preset")
    if args.resume_stage is not None and (not args.resume or len(names) * len(seeds) != 1):
        raise ValueError("--resume-stage requires --resume and exactly one condition/seed")
    if args.resume_reason and args.resume_stage is None:
        raise ValueError("--resume-reason requires --resume-stage")

    for name in names:
        get_condition(name)

    curriculum = Curriculum(variant=preset["curriculum"])

    if args.resume_stage is not None and not 0 < args.resume_stage < len(curriculum.stages):
        raise ValueError("--resume-stage must select a later stage within the curriculum")

    scale = 0.002 if args.smoke else args.scale
    rollout_steps = int(preset["n_steps"]) * (args.n_envs or 1)
    gated = bool(preset.get("stage_gate"))
    stage_ceiling = resolve_stage_ceiling(args.stage_ceiling, rollout_steps)

    if args.smoke and gated and stage_ceiling is None:
        raise ValueError("a success-gated smoke needs --stage-ceiling; --scale does not limit gated training")

    stages = []

    for idx, stage in enumerate(curriculum.stages):
        requested = max(1, int(stage[4] * scale))
        decisions = max(1, math.ceil(int(stage[4] * scale) / rollout_steps)) * rollout_steps
        extra = dict(preset.get("stage_settings", [{} for _ in curriculum.stages])[idx])
        stages.append({
            "stage": idx, "world_scale": float(stage[0]), "num_drones": int(stage[1]),
            "num_static_obstacles": int(stage[2]), "num_dynamic_obstacles": int(stage[3]),
            "requested_policy_decisions": None if gated else requested,
            "policy_decisions": None if gated else decisions,
            "reference_policy_decisions": decisions,
            "max_policy_decisions": stage_ceiling if gated else decisions,
            "max_physics_ticks": (
                stage_ceiling * preset["action_repeat"] if gated and stage_ceiling is not None
                else None if gated else decisions * preset["action_repeat"]
            ),
            "episode_horizon_physics_ticks": int(stage[5]), "lidar_enabled": bool(stage[6]),
            "blocking_obstacle_prob": extra.get("blocking_obstacle_prob", preset.get("blocking_obstacle_prob", 0.0)),
            **extra,
        })

    reference_total = sum(stage["reference_policy_decisions"] for stage in stages)
    total = None if gated else reference_total
    upper_bound = stage_ceiling * len(stages) if gated and stage_ceiling is not None else total
    paths = {
        "models_dir": str((PROJECT_DIR / models_dir).resolve()),
        "results_dir": str((PROJECT_DIR / results_dir).resolve()),
        "training_logs": str((PROJECT_DIR / results_dir / "train_logs").resolve()),
        "tensorboard_logs": str((PROJECT_DIR / models_dir / "tb_logs").resolve()),
        "runs": [str((PROJECT_DIR / models_dir / run_name(name, seed)).resolve()) for seed in seeds for name in names],
    }
    plan = {
        "preset": args.preset.strip().lower(), "preset_settings": preset,
        "train_only": bool(args.train_only), "conditions": names, "seeds": seeds,
        "resume_checkpoint": args.resume, "evaluation_checkpoint": args.eval_checkpoint,
        "manual_resume_stage": args.resume_stage, "manual_resume_reason": args.resume_reason,
        "run_count": len(names) * len(seeds), "scale": scale, "num_envs": args.n_envs or 1,
        "device": args.device, "parallel_processes": max(1, args.parallel),
        "torch_threads_per_process": args.torch_threads, "stages": stages,
        "decision_budget_mode": "success_gated" if gated else "fixed",
        "stage_ceiling": stage_ceiling,
        "requested_stage_ceiling": args.stage_ceiling or None,
        "reference_policy_decisions_per_run": reference_total,
        "policy_decisions_per_run": total,
        "policy_decisions_all_runs": None if gated else total * len(names) * len(seeds),
        "max_policy_decisions_per_run": upper_bound,
        "max_physics_ticks_per_run": None if upper_bound is None else upper_bound * preset["action_repeat"],
        "tick_count_note": "success gates determine actual cost; no upper bound without an explicit ceiling" if gated else "maximum estimate; terminal decisions can stop before action_repeat ticks",
        "effective_reward_profiles": {name: vars(apply_preset_to_condition(get_condition(name), preset).reward) for name in names},
        "outputs": paths, "phases": [],
    }

    if not args.skip_train:
        base = py + ["training.train_ppo", "--device", args.device, "--preset", args.preset]

        #FORWARD THE SAFETY VALVE; 0 MEANS UNBOUNDED SO IT IS LEFT OFF
        if stage_ceiling is not None:
            base += ["--stage-ceiling", str(stage_ceiling)]
        if args.resume:
            base += ["--resume", args.resume]
        if args.resume_stage is not None:
            base += ["--resume-stage", str(args.resume_stage)]
        if args.resume_reason:
            base += ["--resume-reason", args.resume_reason]

        if args.smoke:
            base += ["--scale", "0.002", "--n-envs", "1", "--retrain"]
        else:
            base += ["--scale", str(args.scale)]
            if args.n_envs is not None:
                base += ["--n-envs", str(args.n_envs)]
            if args.retrain:
                base += ["--retrain"]

        if args.parallel and args.parallel > 1:
            commands = [base + ["--condition", name, "--seeds", str(seed)] for seed in seeds for name in names]
            plan["phases"].append({"phase": "train", "parallel": True, "base_command": base, "commands": commands})
        else:
            cmd = base + [
                "--condition", ",".join(names),
                "--seeds", ",".join(str(s) for s in seeds),
            ]
            plan["phases"].append({"phase": "train", "parallel": False, "commands": [cmd]})

    if not args.skip_eval and not args.train_only:
        cmd = py + [
            "evaluation.evaluate",
            "--train-seeds", ",".join(str(s) for s in seeds),
            "--seed", str(args.seed),
            "--preset", args.preset,
        ]
        if args.eval_checkpoint:
            cmd += ["--checkpoint", args.eval_checkpoint]

        if args.smoke:
            cmd += ["--episodes-random", "3"]
        elif args.episodes_random is not None:
            cmd += ["--episodes-random", str(args.episodes_random)]

        if args.parallel and args.parallel > 1 and len(names) > 1:
            commands = [cmd + ["--conditions", name, "--out-dir", str(Path(results_dir) / "eval" / f"_part_{name}")] for name in names]
            plan["phases"].append({"phase": "evaluate", "parallel": True, "base_command": cmd, "commands": commands})
        else:
            plan["phases"].append({"phase": "evaluate", "parallel": False,
                                   "commands": [cmd + ["--conditions", ",".join(names), "--out-dir", str(Path(results_dir) / "eval")]]})

    if not args.skip_analyze and not args.train_only:
        plan["phases"].append({"phase": "analyze", "parallel": False, "commands": [py + ["analysis.analyze"]]})

    return plan, env


#ATOMICALLY PERSIST RESOLVED RUN PLAN & ORCHESTRATOR STATUS
def save_execution_plan(path: Path, plan: dict) -> None:
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    temp_path.replace(path)


#CHECK ALL REQUESTED TRAINING SLOTS BEFORE ANY CHILD OR OUTPUT WRITE
def preflight_training_slots(plan: dict, args) -> None:
    if args.skip_train or args.retrain or args.smoke:
        return

    models_root = Path(plan["outputs"]["models_dir"])
    log_root = Path(plan["outputs"]["training_logs"])
    complete = []
    resumable = []

    for seed in plan["seeds"]:
        for name in plan["conditions"]:
            out_dir = models_root / run_name(name, seed)
            log_path = log_root / f"{name}_s{seed}.csv"
            ep_path = log_path.with_name(log_path.stem + "_episodes.csv")
            exists = (out_dir.exists() and any(out_dir.iterdir())) or log_path.exists() or ep_path.exists()

            if not exists:
                if args.resume_stage is not None:
                    raise ValueError("--resume-stage requires an existing committed checkpoint")

                continue

            manifest_path = out_dir / "run_manifest.json"
            marked_complete = (json.loads(manifest_path.read_text(encoding="utf-8")).get("status") == "complete"
                               if manifest_path.is_file() else False)
            if args.resume and (args.resume_stage is not None or not (out_dir / "final.zip").is_file() or not marked_complete or args.resume != "latest"):
                resumable.append((name, seed, out_dir, log_path))
                continue

            if not (out_dir / "final.zip").is_file():
                raise ValueError(f"existing incomplete training attempt {out_dir}; use --resume latest to continue, --retrain to restart, or fresh output directories")

            complete.append((name, seed, out_dir, log_path))

    if not complete and not resumable:
        return

    from training.train_ppo import build_run_manifest, resolve_device, validate_complete_run
    from training.checkpoint_io import select_checkpoint, validate_resume
    stages = Curriculum(plan["preset_settings"]["curriculum"]).stages

    for name, seed, out_dir, log_path in complete:
        expected = build_run_manifest(get_condition(name), out_dir, stages, resolve_device(args.device), plan["scale"], seed, plan["num_envs"], plan["preset"], plan.get("stage_ceiling"))
        validate_complete_run(out_dir, expected, log_path, allow_bypassed=bool(args.resume))

    for name, seed, out_dir, log_path in resumable:
        expected = build_run_manifest(get_condition(name), out_dir, stages, resolve_device(args.device), plan["scale"], seed, plan["num_envs"], plan["preset"], plan.get("stage_ceiling"))
        _, state = select_checkpoint(out_dir, args.resume)
        validate_resume(state, expected, log_path, out_dir, args.resume_stage)


#DRIVE RESOLVED PHASES OR PRINT ZERO-WRITE PREVIEW
def main() -> None:
    args = parse_args()
    plan, env = build_plan(args)

    if args.dry_run:
        print(json.dumps(plan, indent=2))

        return

    preflight_training_slots(plan, args)
    start = time.time()
    os.chdir(PROJECT_DIR)
    print(f"conditions: {plan['conditions']}")
    print(f"run seeds: {plan['seeds']} ({plan['run_count']} runs total)")
    budget = "determined by success gates" if plan["policy_decisions_per_run"] is None else f"{plan['policy_decisions_per_run']:,}"
    print(f"preset: {plan['preset']}; policy decisions/run: {budget}")
    print(f"models: {plan['outputs']['models_dir']}\nresults: {plan['outputs']['results_dir']}")
    results_root = Path(plan["outputs"]["results_dir"])
    results_root.mkdir(parents=True, exist_ok=True)
    plan_path = results_root / "training_plan.json"

    if plan_path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        plan_path = results_root / f"training_plan_{stamp}.json"

    plan.update(status="running", started_utc=datetime.now(timezone.utc).isoformat(), runner_arguments=vars(args), runner_command=[sys.executable, *sys.argv])
    save_execution_plan(plan_path, plan)
    print(f"run plan: {plan_path}")

    try:
        for phase in plan["phases"]:
            phase["status"] = "running"
            phase["started_utc"] = datetime.now(timezone.utc).isoformat()
            save_execution_plan(plan_path, plan)

            if phase["phase"] == "train" and phase["parallel"]:
                run_training_parallel(
                    names=plan["conditions"], seeds=plan["seeds"], base_cmd=phase["base_command"],
                    env=env, log_dir=str(Path(env["CS5478_RESULTS_DIR"]) / "train_logs"), max_parallel=args.parallel,
                    progress_interval=args.progress_interval,
                    results_dir=env["CS5478_RESULTS_DIR"],
                    total_decisions=int(plan.get("policy_decisions_per_run") or 0),
                    models_dir=env["CS5478_MODELS_DIR"],
                    gated=bool(plan["preset_settings"].get("stage_gate")),
                )
            elif phase["phase"] == "evaluate" and phase["parallel"]:
                run_eval_parallel(
                    names=plan["conditions"], eval_cmd=phase["base_command"], env=env,
                    results_dir=env["CS5478_RESULTS_DIR"], max_parallel=args.parallel,
                )
            else:
                run_phase(phase["phase"].upper(), phase["commands"][0], env)

            phase["status"] = "complete"
            phase["completed_utc"] = datetime.now(timezone.utc).isoformat()
            save_execution_plan(plan_path, plan)
    except BaseException as error:
        plan.update(status="failed", error=f"{type(error).__name__}: {error}", completed_utc=datetime.now(timezone.utc).isoformat())

        for phase in plan["phases"]:
            if phase.get("status") == "running":
                phase["status"] = "failed"

        save_execution_plan(plan_path, plan)

        raise

    plan.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(), wall_time_s=time.time() - start)
    save_execution_plan(plan_path, plan)

    mins = (time.time() - start) / 60.0
    if args.train_only:
        print(f"\n[DONE] training finished in {mins:.1f} min - models: {env['CS5478_MODELS_DIR']}; logs: {env['CS5478_RESULTS_DIR']}/train_logs")
    else:
        out = env["CS5478_RESULTS_DIR"]
        print(f"\n[DONE] pipeline finished in {mins:.1f} min - see {out}/figures, {out}/tables, {out}/results.json")


if __name__ == "__main__":
    main()
