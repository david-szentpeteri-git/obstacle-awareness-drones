# -*- coding: utf-8 -*-
#NEWLY ADDED FOR CS5478 PROJECT
"""
Reference rows for the evaluation tables: two scripted controllers flown on
exactly the worlds the trained models faced (same scenario names, same episode
seeds, same environment builder, same episode runner).

  expert   - the shared full-information teacher used in training
             (training/teacher.py: ExpertPolicy with its original radii,
             unmasked observation incl. all six rays): an upper reference
  straight - the same expert with every avoidance term switched off (no
             obstacle steering, no ray bias, no wall push): flies straight at
             the goal with the expert's cruise speed and goal braking, the
             "no obstacle-awareness at all" baseline

  .\\venv\\Scripts\\python.exe -B -u -m evaluation.reference_eval --controller expert --suite preset --out-dir results/reference/preset_expert
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config.experiment import apply_preset_to_condition, get_condition
from config.ppo_defaults import get_preset
from training.env_factory import safe_close
from evaluation.evaluate import RECORD_FIELDS, make_scenario_env, run_episode, scenario_episode_seeds
from evaluation.scenarios import resolve_scenarios
from training.teacher import REFERENCE_RADII, make_teacher, teacher_settings
from evaluation.wall_gap import WALL_RECORD_FIELDS, wall_gap_scenarios

CONTROLLERS = ("expert", "straight", "gap_expert")


#SCRIPTED POLICY BEHIND THE MODEL.predict INTERFACE THE EPISODE RUNNER USES
class ScriptedPolicy:

    #WRAP POLICY WITH .action(obs)
    def __init__(self, policy):
        self.policy = policy

    #ONE ACTION PER RAW OBSERVATION ROW
    def predict(self, obs_batch, deterministic=True):
        return np.stack([self.policy.action(obs) for obs in obs_batch]), None


#BUILD CONTROLLER FOR A WORLD SCALE
def make_controller(name: str, world_scale: float) -> ScriptedPolicy:
    if name == "expert":
        return ScriptedPolicy(make_teacher(world_scale))
    if name == "gap_expert":
        #PHASE-2 TEACHER: KNOWS THE WALL LAYOUT (ATTACHED TO THE ENV BY THE CALLER)
        return ScriptedPolicy(make_teacher(world_scale, kind="gap_aware"))

    from drone_env.expert_policy import ExpertPolicy

    #EXPERT MINUS ALL AVOIDANCE: SAME CRUISE SPEED & GOAL BRAKING
    straight = ExpertPolicy(world_scale=float(world_scale), use_lidar=False, avoid_radius=0.0,
                            wall_margin=1e-6, teammate_avoid_radius=0.0,
                            target_brake_dist=REFERENCE_RADII["target_brake_dist"])

    return ScriptedPolicy(straight)


#PARSE CLI ARGS
def parse_args():
    parser = argparse.ArgumentParser(description="Scripted reference controllers on the evaluation worlds")
    parser.add_argument("--controller", required=True, choices=CONTROLLERS)
    parser.add_argument("--suite", default="preset", choices=["preset", "wall_gap"])
    parser.add_argument("--preset", default="compact")
    parser.add_argument("--episodes-random", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0, help="base seed for scenario generation (models used 0)")
    parser.add_argument("--out-dir", required=True)

    return parser.parse_args()


#RUN CONTROLLER ON EVERY SCENARIO & EPISODE SEED, WRITE records.csv + MANIFEST
def main() -> None:
    args = parse_args()
    preset = get_preset(args.preset)
    scenarios = wall_gap_scenarios(preset) if args.suite == "wall_gap" else resolve_scenarios(preset)
    fields = RECORD_FIELDS + (WALL_RECORD_FIELDS if args.suite == "wall_gap" else [])
    out_dir = Path(args.out_dir)

    if (out_dir / "records.csv").exists():
        raise FileExistsError(f"reference evidence already exists: {out_dir / 'records.csv'}")

    out_dir.mkdir(parents=True, exist_ok=True)
    #FULL CONDITION = UNMASKED OBS WITH RAYS (REWARD PROFILE IRRELEVANT HERE)
    env_cond = apply_preset_to_condition(get_condition("full"), preset)
    seeds = {name: scenario_episode_seeds(name, args.seed, args.episodes_random) for name in scenarios}
    expert_src = Path(__file__).resolve().parents[1] / "drone_env" / "expert_policy.py"
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args),
        "controller": args.controller,
        "teacher_settings": {f"{scenarios[n]['env_cfg']['world_scale']}": teacher_settings(scenarios[n]["env_cfg"]["world_scale"], kind="gap_aware" if args.controller == "gap_expert" else "expert") for n in scenarios},
        "expert_source_sha256": hashlib.sha256(expert_src.read_bytes()).hexdigest(),
        "episode_seeds": seeds,
    }

    with (out_dir / "records.csv").open("x", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()

        for name, scenario in scenarios.items():
            ctrl = make_controller(args.controller, scenario["env_cfg"]["world_scale"])
            env = make_scenario_env(env_cond, scenario, preset, preset["action_repeat"])

            if hasattr(ctrl.policy, "attach_env"):
                ctrl.policy.attach_env(env)

            try:
                for idx, seed in enumerate(seeds[name], start=1):
                    record, _traj = run_episode(env, ctrl, env_cond, None, scenario, seed)
                    record.update(scenario=name, condition=args.controller, train_seed=0)
                    writer.writerow(record)
                    fh.flush()
                    print(f"  [{name} | {args.controller} s0 | {idx}/{len(seeds[name])}] seed={seed} "
                          f"reached={record['reached']} collided={record['collided']} steps={record['steps']}", flush=True)
            finally:
                safe_close(env)

    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[SAVED] {out_dir / 'records.csv'}")


if __name__ == "__main__":
    main()
