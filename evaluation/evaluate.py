# -*- coding: utf-8 -*-
#NEWLY ADDED FOR CS5478 PROJECT
"""
Paired evaluation of every trained run (4 conditions x 3 seeds) on one
scenario suite: the held-out worlds of the training curriculum (one scenario
per stage) or the wall-gap worlds (evaluation/wall_gap.py).

Every run is replayed on the SAME seeded scenarios - the blind policies are
deliberately tested in worlds they were never given the information to see.
Episode seeds come from a band (>= EVAL_SEED_LO) that no training seed can
reach, so "held out" is a checkable property rather than an assumption, and
they are derived with a stable CRC32 rather than Python's per-process hash()
so a rerun reproduces the same worlds.

Per-episode records (incl. min obstacle clearance and time-to-goal) go to
<results_dir>/eval/records.csv, and aggregate tables are printed + saved as JSON.

    python -m evaluation.evaluate --preset compact --episodes-random 200
    python -m evaluation.evaluate --preset compact --conditions degenerate,full --scenarios stage6_6s3d
    python -m evaluation.evaluate --preset compact --suite wall_gap --episodes-random 200
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from tabulate import tabulate

from drone_env.defaults import POS_SLICE
from config.experiment import (
    apply_preset_to_condition,
    FULL_OBS_DIM,
    Condition,
    get_condition,
    resolve_condition_names,
    run_name,
)
from config.ppo_defaults import (
    training_semantics,
    EVAL_EPISODES_RANDOM,
    EVAL_RESULTS_DIR,
    EVAL_SEED_BASE,
    EVAL_SEED_HI,
    EVAL_SEED_LO,
    PPO_MODELS_DIR,
    TRAIN_SEEDS,
    get_preset,
)
from training.env_factory import make_raw_env, safe_close
from training.checkpoint_io import read_checkpoint, select_checkpoint
from evaluation.scenarios import reset_scenario_episode, resolve_scenarios
from evaluation.obs_variants import PROBE_NAMES, make_obs_transform
from evaluation.wall_gap import WALL_RECORD_FIELDS, wall_gap_scenarios
from training.wrappers.obs_mask import apply_obs_mask

RECORD_FIELDS = [
    "scenario", "condition", "train_seed", "seed", "reached", "collided", "wall_hit",
    "obstacle_hit", "drone_hit", "timeout", "steps", "physics_ticks", "time_s", "reach_step",
    "time_to_goal_s", "path_len", "straight_dist", "path_ratio", "min_clearance",
    "start_x", "start_y", "start_z", "goal_x", "goal_y", "goal_z",
]


#EPISODE SEEDS FOR ONE SCENARIO - STABLE ACROSS PROCESSES AND RERUNS
def scenario_episode_seeds(scenario_name: str, base_seed: int, n_eps: int) -> List[int]:
    """
    Python's hash() is salted per process (PYTHONHASHSEED), so deriving seeds
    from it made every rerun face different worlds. CRC32 is stable, and the
    draw band starts above EVAL_SEED_LO so no episode can land on a seed the
    training runs used.
    """
    name_offset = zlib.crc32(scenario_name.encode("utf-8")) % 1_000_003
    rng = np.random.default_rng(EVAL_SEED_BASE + int(base_seed) + name_offset)

    if n_eps <= 0:
        raise ValueError("evaluation episode count must be positive")

    return [int(s) + EVAL_SEED_LO for s in rng.choice(EVAL_SEED_HI - EVAL_SEED_LO, size=n_eps, replace=False)]


#LOAD SAVED OBS-NORMALIZATION STATISTICS (STANDALONE, NO VEC ENV NEEDED)
def load_vec_normalize(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"required VecNormalize statistics missing: {path}")

    with open(path, "rb") as fh:
        vec_norm = pickle.load(fh)

    vec_norm.training = False
    vec_norm.norm_reward = False

    return vec_norm


#LOCATE ONE RUN'S FINAL MODEL + NORMALIZATION STATS
def resolve_artifacts(condition: Condition, train_seed: int, models_dir: str, preset: Optional[dict] = None,
                      checkpoint: Optional[str] = None):
    run_dir = Path(models_dir) / run_name(condition.name, train_seed)
    model_path = run_dir / "final.zip"
    norm_path = run_dir / "final_vecnormalize.pkl"
    checkpoint_state = None
    if checkpoint:
        model_path, checkpoint_state = select_checkpoint(run_dir, checkpoint)
        norm_path = model_path.with_name(model_path.stem + "_vecnormalize.pkl")

    if not model_path.is_file():
        raise FileNotFoundError(
            f"no trained model for '{condition.name}' seed {train_seed} at "
            f"{model_path} - run: python -m training.train_ppo "
            f"--condition {condition.name} --seeds {train_seed}"
        )

    manifest_path = run_dir / "run_manifest.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(f"training provenance missing: {manifest_path}")

    manifest = (checkpoint_state["training_manifest"] if checkpoint_state else
                json.loads(manifest_path.read_text(encoding="utf-8")))

    if checkpoint_state is None and manifest.get("status") != "complete":
        raise ValueError(f"training run is not marked complete: {manifest_path}")

    expected = {
        "condition": condition.name,
        "run_seed": int(train_seed),
        "masked_slices": [[sl.start, sl.stop] for sl in condition.masked_slices],
        "reward_profile": vars(condition.reward),
        "use_lidar": condition.use_lidar,
        "reward_shaped": condition.reward_shaped,
        "obstacle_sensing": condition.obstacle_sensing,
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]

    if mismatches:
        raise ValueError(f"training manifest does not match requested condition/run ({mismatches}): {manifest_path}")

    if preset is not None:
        #COMPARE TRAINING SEMANTICS ONLY (CONTROL RATE, REWARD, CURRICULUM) -
        #NOT OUTPUT DIRS OR EVALUATION-SIDE SETTINGS
        saved_preset = manifest.get("preset_settings")
        saved_preset = None if saved_preset is None else training_semantics(saved_preset)
        expected_preset = json.loads(json.dumps(training_semantics(preset)))

        if saved_preset is None and preset.get("curriculum") == "default":
            saved_preset = expected_preset

        if saved_preset != expected_preset:
            raise ValueError(f"training preset/control rate does not match requested evaluation preset: {manifest_path}")

    for path in (model_path, norm_path):
        expected_hash = (checkpoint_state or manifest).get("artifact_sha256", {}).get(path.name)

        if not path.is_file() or expected_hash is None or file_sha256(path) != expected_hash:
            raise ValueError(f"missing, modified or mismatched evaluation artifact: {path}")

    return model_path, norm_path


#TRANSFORM RAW ENV OBS THE SAME WAY THE TRAINING WRAPPER STACK DID
def prepare_obs(obs_batch: np.ndarray, condition: Condition, vec_norm) -> np.ndarray:
    obs = apply_obs_mask(obs_batch, condition.masked_slices, FULL_OBS_DIM)

    if vec_norm is not None:
        obs = vec_norm.normalize_obs(obs)

    return np.asarray(obs, dtype=np.float32)


#MIN SURFACE DISTANCE FROM THE DRONE TO ANY OBSTACLE BODY (NAN IF NONE)
def min_obstacle_clearance(env) -> float:
    if not env._obstacles or env._p is None or not env._drone_ids:
        return float("nan")

    p = env._p
    did = env._drone_ids[0]
    best = float("inf")

    for obs in env._obstacles:
        body_id = getattr(obs, "body_id", None)

        if body_id is None:
            continue

        try:
            pts = env._call_physics(
                p.getClosestPoints, bodyA=did, bodyB=body_id, distance=100.0
            )
        except Exception:
            continue

        for pt in pts or []:
            best = min(best, float(pt[8]))

    return best if np.isfinite(best) else float("nan")


#RUN ONE SEEDED EPISODE, RETURN ITS RECORD + POSITION TRAJECTORY
def run_episode(
    env,
    model,
    condition: Condition,
    vec_norm,
    scenario: dict,
    seed: int,
    obs_transform=None,
    sample_actions: bool = False,
) -> tuple[Dict, np.ndarray]:
    obs_dict = reset_scenario_episode(env, scenario, seed)
    agent = env.possible_agents[0]

    #SAMPLED ACTIONS DRAW FROM TORCH'S RNG; SEED IT PER EPISODE SO RERUNS REPRODUCE
    if sample_actions:
        set_random_seed(int(seed) % (2**31 - 1))

    ctrl_freq = float(getattr(env._env, "CTRL_FREQ", 48.0))
    start_pos = np.array(obs_dict[agent][POS_SLICE], dtype=np.float32)
    goal_pos = np.array(env._targets[0].start_pos, dtype=np.float32)
    straight_dist = float(np.linalg.norm(goal_pos - start_pos))

    prev_pos = start_pos.copy()
    traj = [start_pos.copy()]
    path_len = 0.0
    min_clear = min_obstacle_clearance(env)
    reach_step = -1
    reach_tick = -1
    steps = 0
    #PHYSICS TICKS, WHICH IS WHAT env.max_steps AND ALL TIMING ARE IN. WITH
    #action_repeat > 1 ONE DECISION ADVANCES SEVERAL TICKS, SO COUNTING
    #DECISIONS HERE WOULD OVERRUN THE HORIZON AND NEVER FLAG A TIMEOUT.
    ticks = 0
    truncated = False

    while True:
        obs_batch = obs_dict[agent][None, :]

        #OPTIONAL PROBE: EDIT THE RAW OBSERVATION BEFORE THE CONDITION MASK
        if obs_transform is not None:
            obs_batch = obs_transform(env, obs_batch)

        obs_batch = prepare_obs(obs_batch, condition, vec_norm)
        act, _ = model.predict(obs_batch, deterministic=not sample_actions)

        obs_dict, _rewards, term_dict, trunc_dict, info_dict = env.step({agent: act[0]})
        steps += 1
        ticks += int(info_dict[agent].get("simulation_steps", 1))

        #TERMINAL POSITION/CLEARANCE CAPTURED BEFORE SIMULATOR DESPAWNS DRONE
        info = info_dict[agent]
        substeps = info.get("substep_positions")

        if substeps is None:
            substeps = [info.get("position", obs_dict[agent][POS_SLICE])]

        for position in substeps:
            cur_pos = np.asarray(position, dtype=np.float32)
            path_len += float(np.linalg.norm(cur_pos - prev_pos))
            prev_pos = cur_pos
            traj.append(cur_pos.copy())
        clear = info.get("min_obstacle_clearance", float("nan"))

        if np.isfinite(clear):
            min_clear = clear if not np.isfinite(min_clear) else min(min_clear, clear)

        if reach_step < 0 and info_dict[agent].get("reached", False):
            reach_step = steps
            reach_tick = ticks

        truncated = bool(trunc_dict[agent])

        if (term_dict[agent] or truncated) or ticks >= env.max_steps:
            break

    reached = bool(env._reached_once[0])
    wall_hit = bool(env._wall_collision_once[0])
    obs_hit = bool(env._obstacle_collision_once[0])
    drone_hit = bool(env._drone_collision_once[0])
    collided = wall_hit or obs_hit or drone_hit
    #TRUNCATION IS THE AUTHORITATIVE HORIZON SIGNAL; THE TICK GUARD IS A
    #BACKSTOP FOR THE CASE WHERE THE LOOP EXITS WITHOUT ONE
    timeout = bool((truncated or ticks >= env.max_steps) and not reached and not collided)

    record = {
        "seed": seed,
        "reached": int(reached),
        "collided": int(collided),
        "wall_hit": int(wall_hit),
        "obstacle_hit": int(obs_hit),
        "drone_hit": int(drone_hit),
        "timeout": int(timeout),
        "steps": int(steps),
        "physics_ticks": int(ticks),
        #TIMING FROM PHYSICS TICKS, SO SECONDS STAY COMPARABLE ACROSS
        #action_repeat SETTINGS
        "time_s": round(ticks / ctrl_freq, 2),
        "reach_step": int(reach_step),
        "time_to_goal_s": round(reach_tick / ctrl_freq, 2) if reach_tick > 0 else float("nan"),
        "path_len": round(path_len, 3),
        "straight_dist": round(straight_dist, 3),
        #PATH RATIO ONLY MEANINGFUL WHEN THE DRONE ACTUALLY ARRIVED
        "path_ratio": round(path_len / straight_dist, 3)
        if reached and straight_dist > 1e-6
        else float("nan"),
        "min_clearance": round(min_clear, 3) if np.isfinite(min_clear) else float("nan"),
        **{f"start_{axis}": float(value) for axis, value in zip("xyz", start_pos)},
        **{f"goal_{axis}": float(value) for axis, value in zip("xyz", goal_pos)},
        **(getattr(env, "eval_layout", None) or {}),
    }

    return record, np.array(traj, dtype=np.float32)


#BUILD RAW EVAL ENV FOR ONE SCENARIO (SHARED BY MODELS & REFERENCE CONTROLLERS)
def make_scenario_env(condition: Condition, scenario: dict, preset: Optional[dict], action_repeat: int,
                      render_mode: Optional[str] = None):
    cfg = scenario["env_cfg"]
    env_options = {key: cfg[key] for key in ("obstacle_layout", "blocking_obstacle_prob", "obstacle_size_range", "dynamic_speed_range", "dynamic_travel_range", "obstacle_dt", "wall_coverage") if key in cfg}

    if preset and "evaluation_obstacle_dt" in preset:
        env_options.setdefault("obstacle_dt", preset["evaluation_obstacle_dt"])

    env = make_raw_env(
        condition=condition,
        world_scale=cfg["world_scale"],
        num_drones=1,
        num_static_obs=cfg["num_static_obs"],
        num_dynamic_obs=cfg["num_dynamic_obs"],
        episode_horizon=cfg["episode_horizon"],
        stage_lidar=True,
        render_mode=render_mode,
        #MUST MATCH TRAINING: A POLICY TRAINED AT 24 DECISIONS/s EVALUATED AT
        #240 IS A DIFFERENT CONTROLLER ENTIRELY
        action_repeat=action_repeat,
        **env_options,
    )
    env.record_eval_metrics = True

    return env


#EVALUATE ONE RUN ON ONE SCENARIO ACROSS ALL EPISODE SEEDS
def evaluate_condition_on_scenario(
    condition: Condition,
    train_seed: int,
    scenario_name: str,
    scenario: dict,
    seeds: List[int],
    models_dir: str,
    render_mode: Optional[str] = None,
    record_writer=None,
    record_file=None,
    action_repeat: int = 1,
    preset: Optional[dict] = None,
    checkpoint: Optional[str] = None,
    obs_probe: Optional[str] = None,
    sample_actions: bool = False,
) -> List[Dict]:
    obs_transform = make_obs_transform(obs_probe)
    artifact_options = {"checkpoint": checkpoint} if checkpoint else {}
    model_path, norm_path = resolve_artifacts(condition, train_seed, models_dir, preset=preset, **artifact_options)

    model = PPO.load(str(model_path), device="cpu")
    vec_norm = load_vec_normalize(norm_path)
    env = make_scenario_env(condition, scenario, preset, action_repeat, render_mode)
    records = []

    try:
        for idx, seed in enumerate(seeds, start=1):
            record, _traj = run_episode(
                env, model, condition, vec_norm, scenario, seed,
                obs_transform=obs_transform, sample_actions=sample_actions,
            )
            record["scenario"] = scenario_name

            if obs_probe or sample_actions:
                record["obs_probe"] = obs_probe or "none"
                record["action_mode"] = "sampled" if sample_actions else "deterministic"

            record["condition"] = condition.name
            record["train_seed"] = int(train_seed)
            records.append(record)

            if record_writer is not None:
                record_writer.writerow(record)
                record_file.flush()

            print(
                f"  [{scenario_name} | {condition.name} s{train_seed} | "
                f"{idx}/{len(seeds)}] seed={record['seed']} "
                f"reached={record['reached']} collided={record['collided']} "
                f"steps={record['steps']}"
            )
    finally:
        safe_close(env)

    return records


#AGGREGATE PER-EPISODE RECORDS INTO THE HEADLINE COMPARISON METRICS
def summarize(records: List[Dict]) -> Dict:
    n = max(1, len(records))
    path_ratios = [r["path_ratio"] for r in records if np.isfinite(r["path_ratio"])]
    reach_times = [r["time_to_goal_s"] for r in records if np.isfinite(r["time_to_goal_s"])]
    clearances = [r["min_clearance"] for r in records if np.isfinite(r["min_clearance"])]

    return {
        "episodes": len(records),
        "success_rate": sum(r["reached"] for r in records) / n,
        "collision_free_rate": sum(1 - r["collided"] for r in records) / n,
        "wall_hit_rate": sum(r["wall_hit"] for r in records) / n,
        "obstacle_hit_rate": sum(r["obstacle_hit"] for r in records) / n,
        "timeout_rate": sum(r["timeout"] for r in records) / n,
        "mean_steps": float(np.mean([r["steps"] for r in records])),
        "mean_time_to_goal_s": float(np.mean(reach_times)) if reach_times else float("nan"),
        "path_ratio": float(np.mean(path_ratios)) if path_ratios else float("nan"),
        "mean_min_clearance": float(np.mean(clearances)) if clearances else float("nan"),
    }


#PRINT ONE SCENARIO'S SIDE-BY-SIDE COMPARISON TABLE
def print_comparison(scenario_name: str, summaries: Dict[str, Dict]) -> None:
    rows = [
        ("Episodes", "episodes", "{:.0f}"),
        ("Success rate", "success_rate", "{:.1%}"),
        ("Collision-free rate", "collision_free_rate", "{:.1%}"),
        ("Wall-hit rate", "wall_hit_rate", "{:.1%}"),
        ("Obstacle-hit rate", "obstacle_hit_rate", "{:.1%}"),
        ("Timeout rate", "timeout_rate", "{:.1%}"),
        ("Mean episode length (steps)", "mean_steps", "{:.0f}"),
        ("Mean time-to-goal (s)", "mean_time_to_goal_s", "{:.1f}"),
        ("Path ratio (1.0 = straight line)", "path_ratio", "{:.3f}"),
        ("Mean min obstacle clearance (m)", "mean_min_clearance", "{:.3f}"),
    ]

    names = list(summaries)
    table = []

    for label, key, fmt in rows:
        row = [label]

        for name in names:
            value = summaries[name][key]
            row.append("n/a" if isinstance(value, float) and np.isnan(value) else fmt.format(value))

        table.append(row)

    print(f"\n### {scenario_name}")
    print(tabulate(table, headers=["Metric", *names], tablefmt="github"))


#WRITE EVERY PER-EPISODE RECORD FOR LATER PLOTTING/STATS
def write_csv(records: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RECORD_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    print(f"\n[SAVED] per-episode records -> {out_path}")


#PARSE CLI ARGS
def parse_args():
    parser = argparse.ArgumentParser(description="Scenario-suite evaluation of trained conditions")
    parser.add_argument("--conditions", default="all", help="all, both, or comma-separated names")
    parser.add_argument(
        "--scenarios",
        default="all",
        help="all, or comma-separated scenario names of the chosen suite "
        "(stage0_0s0d ... stage6_6s3d for the preset suite; wall_s2_x10 ... wall_s3_x90 for wall_gap)",
    )
    parser.add_argument("--episodes-random", type=int, default=EVAL_EPISODES_RANDOM, help="episodes per scenario")
    parser.add_argument("--seed", type=int, default=0, help="base seed for scenario generation")
    parser.add_argument(
        "--train-seeds",
        default=",".join(str(s) for s in TRAIN_SEEDS),
        help="which trained run seeds to evaluate",
    )
    parser.add_argument(
        "--preset",
        default="compact",
        help="experiment preset the models were TRAINED with; its action_repeat "
        "must match or the policy is evaluated at the wrong control rate",
    )
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--checkpoint", default=None, metavar="latest|CHECKPOINT",
                        help="evaluate a committed checkpoint, including from a stopped/incomplete run; default requires completed final models")
    parser.add_argument("--suite", default="preset", choices=["preset", "wall_gap"],
                        help="preset = the preset's own scenario suite; wall_gap = one wall across the passage covering 10-90%% at scales 2 and 3 (evaluation/wall_gap.py)")
    parser.add_argument("--worlds-preset", default=None,
                        help="take the scenario suite from this preset instead of the models' preset (worlds only; checks use --preset)")
    parser.add_argument("--probe", default=None, choices=list(PROBE_NAMES),
                        help="observation variant fed to the model instead of its normal input (evaluation/obs_variants.py); default: normal input")
    parser.add_argument("--sample-actions", action="store_true",
                        help="sample actions from the policy (seeded per episode) instead of taking the deterministic mean")
    parser.add_argument("--render", action="store_true", help="show the PyBullet GUI")
    parser.add_argument("--out-dir", default=None)

    return parser.parse_args()


#HASH ARTIFACT CONTENT FOR EVALUATION PROVENANCE
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


#PERSIST MANIFEST WITHOUT LEAVING PARTIALLY WRITTEN JSON
def save_manifest(path: Path, manifest: dict) -> None:
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temp_path.replace(path)


#RUN EVERY (CONDITION, SEED) RUN ON EVERY SCENARIO WITH SHARED EPISODE SEEDS
def main() -> None:
    args = parse_args()
    condition_names = resolve_condition_names(args.conditions)
    train_seeds = [int(s) for s in args.train_seeds.split(",") if s.strip()]

    if not train_seeds or len(train_seeds) != len(set(train_seeds)):
        raise ValueError("--train-seeds must contain distinct run seeds")
    if args.episodes_random <= 0:
        raise ValueError("episode count must be positive")

    #THE PRESET THE MODELS WERE TRAINED WITH; ITS action_repeat DEFINES THE
    #CONTROL RATE THE POLICY EXPECTS
    #getattr KEEPS main() CALLABLE WITH A SYNTHETIC ARGS NAMESPACE, WHICH
    #THE REPORT REGRESSION SUITE DOES
    eval_preset = get_preset(getattr(args, "preset", "compact"))
    suite = getattr(args, "suite", "preset")
    #WORLDS MAY COME FROM ANOTHER PRESET (e.g. PHASE-2 MODELS ON THE compact TRAINING-LIKE WORLDS);
    #ARTIFACT AND SEMANTIC CHECKS ALWAYS USE THE MODELS' OWN PRESET
    worlds_preset = get_preset(args.worlds_preset) if getattr(args, "worlds_preset", None) else eval_preset
    scenarios = wall_gap_scenarios(worlds_preset) if suite == "wall_gap" else resolve_scenarios(worlds_preset)
    record_fields = RECORD_FIELDS + (WALL_RECORD_FIELDS if suite == "wall_gap" else [])
    obs_probe = getattr(args, "probe", None)
    sample_actions = bool(getattr(args, "sample_actions", False))

    if obs_probe or sample_actions:
        record_fields = record_fields + ["obs_probe", "action_mode"]

    #"all" RESOLVES AGAINST THE CHOSEN SUITE
    if args.scenarios.strip().lower() == "all":
        scenario_names = list(scenarios)
    else:
        scenario_names = [s.strip() for s in args.scenarios.split(",") if s.strip()]

    if not scenario_names or len(scenario_names) != len(set(scenario_names)):
        raise ValueError("--scenarios must contain distinct scenario names")
    if any(name not in scenarios for name in scenario_names):
        raise ValueError(f"unknown scenario; choose from {list(scenarios)}")

    args.models_dir = args.models_dir or eval_preset.get("models_dir", PPO_MODELS_DIR)
    args.out_dir = args.out_dir or str(Path(eval_preset.get("results_dir", Path(EVAL_RESULTS_DIR).parent)) / "eval")
    out_dir = Path(args.out_dir)

    for fname in ("records.csv", "summary.json", "eval_manifest.json", "source_snapshot.zip"):
        if (out_dir / fname).exists():
            raise FileExistsError(f"evaluation evidence already exists: {out_dir / fname}; choose a fresh --out-dir / CS5478_RESULTS_DIR")

    artifacts = {}
    checkpoint_selector = getattr(args, "checkpoint", None)

    if checkpoint_selector and checkpoint_selector != "latest" and len(condition_names) * len(train_seeds) != 1:
        raise ValueError("an explicit evaluation checkpoint requires one condition and seed")

    for name in condition_names:
        for train_seed in train_seeds:
            paths = resolve_artifacts(
                apply_preset_to_condition(get_condition(name), eval_preset),
                train_seed, args.models_dir, preset=eval_preset,
                **({"checkpoint": checkpoint_selector} if checkpoint_selector else {}),
            )
            load_vec_normalize(paths[1])
            artifacts[run_name(name, train_seed)] = {
                label: {"path": str(path.resolve()), "sha256": file_sha256(path)}
                for label, path in zip(("model", "normalization"), paths)
            }
            artifacts[run_name(name, train_seed)]["training_manifest"] = json.loads((paths[0].parent / "run_manifest.json").read_text(encoding="utf-8"))
            if checkpoint_selector:
                state, state_path = read_checkpoint(paths[0])
                artifacts[run_name(name, train_seed)]["checkpoint_state"] = state
                artifacts[run_name(name, train_seed)]["checkpoint_state_sha256"] = file_sha256(state_path)
                artifacts[run_name(name, train_seed)]["training_manifest"] = state["training_manifest"]

    episode_seeds = {name: scenario_episode_seeds(name, args.seed, args.episodes_random) for name in scenario_names}
    from training.train_ppo import source_files

    eval_sources = source_files()
    bypasses = {run: data["training_manifest"].get("bypassed_stages", []) for run, data in artifacts.items()}
    manifest = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args),
        "conditions": condition_names,
        "train_seeds": train_seeds,
        "episode_seeds": episode_seeds,
        "suite": suite,
        "worlds_preset": getattr(args, "worlds_preset", None) or getattr(args, "preset", "compact"),
        "obs_probe": obs_probe or "none",
        "action_mode": "sampled" if sample_actions else "deterministic",
        "preset_settings": eval_preset,
        "scenarios": {name: scenarios[name] for name in scenario_names},
        "artifacts": artifacts,
        "checkpoint_evaluation": bool(checkpoint_selector),
        "manual_curriculum_bypasses": bypasses,
        "manual_stage_advances": {run: data["training_manifest"].get("manual_stage_advances", []) for run, data in artifacts.items()},
        "completion_claim": ("checkpoint evaluation; curriculum completion is not implied" if checkpoint_selector else
                             "finished training with manual curriculum bypasses; all-stage mastery is not established" if any(bypasses.values()) else
                             "completed training models"),
        "source_sha256": {name: file_sha256(path) for name, path in eval_sources.items()},
        "source_snapshot": "source_snapshot.zip",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = out_dir / "source_snapshot.zip"

    with zipfile.ZipFile(snapshot_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in eval_sources.items():
            archive.write(path, arcname=name)

    manifest["source_snapshot_sha256"] = file_sha256(snapshot_path)
    manifest_path = out_dir / "eval_manifest.json"
    save_manifest(manifest_path, manifest)

    render_mode = "human" if args.render else None
    all_records: List[Dict] = []
    all_summaries: Dict[str, Dict[str, Dict]] = {}

    print(f"conditions: {condition_names} | train seeds: {train_seeds}\n")

    with (out_dir / "records.csv").open("x", newline="", encoding="utf-8") as record_file:
        record_writer = csv.DictWriter(record_file, fieldnames=record_fields)
        record_writer.writeheader()
        record_file.flush()

        for scenario_name in scenario_names:
            seeds = episode_seeds[scenario_name]
            summaries: Dict[str, Dict] = {}

            for name in condition_names:
                condition = apply_preset_to_condition(get_condition(name), eval_preset)
                per_condition: List[Dict] = []

                for train_seed in train_seeds:
                    records = evaluate_condition_on_scenario(
                        condition=condition,
                        train_seed=train_seed,
                        scenario_name=scenario_name,
                        scenario=scenarios[scenario_name],
                        seeds=seeds,
                        models_dir=args.models_dir,
                        render_mode=render_mode,
                        record_writer=record_writer,
                        record_file=record_file,
                        action_repeat=eval_preset["action_repeat"],
                        preset=eval_preset,
                        obs_probe=obs_probe,
                        sample_actions=sample_actions,
                        **({"checkpoint": artifacts[run_name(name, train_seed)]["model"]["path"]} if checkpoint_selector else {}),
                    )
                    all_records.extend(records)
                    per_condition.extend(records)
                    summaries[f"{condition.name}_s{train_seed}"] = summarize(records)

                summaries[condition.name] = summarize(per_condition)

            print_comparison(scenario_name, {name: summaries[name] for name in condition_names})
            all_summaries[scenario_name] = summaries

    summary_path = Path(args.out_dir) / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(all_summaries, fh, indent=2)

    manifest.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(), episodes=len(all_records))
    manifest["records_sha256"] = file_sha256(out_dir / "records.csv")
    save_manifest(manifest_path, manifest)
    print(f"[SAVED] aggregate summaries -> {summary_path}")


if __name__ == "__main__":
    main()
