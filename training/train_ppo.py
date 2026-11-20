# -*- coding: utf-8 -*-
#NEWLY ADDED FOR CS5478 PROJECT
"""
PPO curriculum training for the experiment conditions.

    python -m training.train_ppo                     (4 conditions x 3 seeds)
    python -m training.train_ppo --condition full --seeds 0
    python -m training.train_ppo --condition degenerate,reward_only
    python -m training.train_ppo --scale 0.002 --n-envs 1   (quick smoke run)

All conditions run identical hyperparameters, curriculum and policy shape -
config/experiment.py is the only thing that differs between them. A complete
run is reused only when its configuration, source and artifact hashes match.
An explicit --retrain archives the previous attempt and restarts from stage
zero. --resume restores a committed checkpoint and curriculum state, with new
simulator episodes; it is not an exact physics replay.

One run = one (condition, seed) pair, writing to models/<preset>/ppo_<cond>_s<seed>/.
Three seeds per condition gives the report a spread to quote rather than a
single point estimate. Seeding goes through env_factory.seed_vec_env, not
PPO(seed=...), which this wrapper stack cannot support - see that function.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import pickle
import shutil
import sys
import time
import warnings
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import gym_pybullet_drones
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from training.checkpoint_io import (
    BYPASSED, advance_checkpoint_stage, commit_state, log_prefixes, restore_rng,
    select_checkpoint, validate_resume, validate_stage_completion,
)

from config.blending import (
    BETA_START,
    CHECK_INTERVAL_DECISIONS,
    IMITATION_BATCH_SIZE,
    IMITATION_COEF,
    IMITATION_EPOCHS,
    blending_settings,
    resolve_stage_ceiling,
    validate_blending_settings,
)
from training.blend_scheduler import PASSED, STALLED, StageScheduler
from training.stage_gate import run_validation_batch
from training.imitation_ppo import ExpertActionRecorder, ImitationPPO
from training.teacher import make_teacher, teacher_settings
from training.warm_start import load_warm_start
from drone_env.curriculum import Curriculum
from config.experiment import (
    apply_preset_to_condition,
    Condition,
    get_condition,
    resolve_condition_names,
    run_name,
)
from config.ppo_defaults import (
    BATCH_SIZE,
    CHECKPOINT_EVERY_STEPS,
    CLIP_RANGE,
    ENTROPY_COEF,
    GAE_LAMBDA,
    GAMMA,
    LEARNING_RATE,
    MAX_GRAD_NORM,
    TARGET_KL,
    NET_ARCH,
    N_ENVS,
    N_EPOCHS,
    N_STEPS,
    PPO_MODELS_DIR,
    TB_LOG_DIR,
    TORCH_NUM_THREADS,
    TRAIN_LOG_DIR,
    TRAIN_SEEDS,
    VALUE_COEF,
    get_preset,
    train_env_seed,
)
from training.callbacks.stats_cb import RolloutStatsCallback
from training.wrappers.expert_blend import VecExpertBlend
from training.wrappers.reward_scale import ScaleReward
from training.env_factory import (
    copy_vec_normalize_stats,
    make_vec_env,
    safe_close,
    stage_to_kwargs,
    unwrap_vec_normalize,
)

warnings.filterwarnings("ignore", message=".*Box low's precision lowered.*")
warnings.filterwarnings("ignore", message=".*Box high's precision lowered.*")


#OUTPUT DIR FOR ONE RUN'S CHECKPOINTS & FINAL MODEL
def condition_dir(condition: Condition, seed: int) -> Path:
    path = Path(PPO_MODELS_DIR) / run_name(condition.name, seed)
    path.mkdir(parents=True, exist_ok=True)

    return path


#RESOLVE TRAINING DEVICE, PREFERRING GPU WHEN AVAILABLE
def resolve_device(device: str) -> str:
    if device != "auto":
        return device

    return "cuda" if torch.cuda.is_available() else "cpu"


#SAVE MODEL + ITS OBS-NORMALIZATION STATISTICS UNDER ONE NAME
def save_checkpoint(model, env, out_dir: Path, name: str) -> None:
    if not all(torch.isfinite(parameter).all().item() for parameter in model.policy.parameters()):
        raise RuntimeError(f"refusing to save non-finite policy parameters: {name}")

    vec_norm = unwrap_vec_normalize(env)
    if vec_norm is not None:
        obs_stats = vec_norm.obs_rms
        stats = list(obs_stats.values()) if isinstance(obs_stats, dict) else [obs_stats]
        stats.append(vec_norm.ret_rms)

        for item in stats:
            if not all(np.isfinite(value).all() for value in (item.mean, item.var, item.count)):
                raise RuntimeError(f"refusing to save non-finite normalization statistics: {name}")
            if np.any(item.var < 0) or item.count < 0:
                raise RuntimeError(f"refusing to save invalid normalization variance/count: {name}")

    model_path = out_dir / f"{name}.zip"
    pending_model = out_dir / f"{name}.pending.zip"
    model.save(str(pending_model))

    if vec_norm is not None:
        norm_path = out_dir / f"{name}_vecnormalize.pkl"
        pending_norm = out_dir / f"{name}_vecnormalize.pending.pkl"
        vec_norm.save(str(pending_norm))
        pending_norm.replace(norm_path)
        print(f"[SAVED VECNORMALIZE] {norm_path}")
    pending_model.replace(model_path)
    print(f"[SAVED MODEL] {model_path}")


#PER-RUN LEARNING-CURVE CSV PATH
def train_log_path(condition: Condition, seed: int) -> Path:
    return Path(TRAIN_LOG_DIR) / f"{run_name(condition.name, seed)[4:]}.csv"


#SEPARATE RAW PER-EPISODE TRAINING EVIDENCE FROM ROLLOUT AVERAGES
def episode_log_path(log_path: Path) -> Path:
    return log_path.with_name(log_path.stem + "_episodes.csv")


#RESOLVE EXACT PHYSICAL STAGE SETTINGS WITHOUT UNDECLARED PRESET OVERRIDES
def stage_env_settings(cfg_preset: dict, stage_idx: int, stages) -> dict:
    settings = {"blocking_obstacle_prob": cfg_preset.get("blocking_obstacle_prob", 0.0)}
    overrides = cfg_preset.get("stage_settings", [])

    if overrides:
        if len(overrides) != len(stages):
            raise ValueError("preset stage_settings must match the curriculum stage count")

        settings.update(overrides[stage_idx])

    allowed = {
        "blocking_obstacle_prob", "obstacle_layout", "obstacle_size_range",
        "dynamic_speed_range", "dynamic_travel_range", "obstacle_dt", "wall_coverage",
    }

    if set(settings) - allowed:
        raise ValueError(f"unknown stage environment settings: {sorted(set(settings) - allowed)}")

    return settings


#MOVE THE PREVIOUS ATTEMPT'S OUTPUTS ASIDE BEFORE A RESTART WRITES ITS SLOT
def archive_run_attempt(out_dir: Path, log_path: Path, reason: str) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    out_dir = out_dir.resolve()
    log_path = log_path.resolve()
    archive_dir = out_dir.with_name(out_dir.name + f"_attempt_{time.time_ns()}")

    for path in (out_dir, log_path, episode_log_path(log_path), archive_dir):
        if not path.is_relative_to(project_root) or path == project_root:
            raise ValueError(f"attempt archive path must stay inside the project: {path}")

    if archive_dir.exists():
        raise ValueError(f"attempt archive already exists: {archive_dir}")

    #WINDOWS SCANNERS MAY HOLD A FILE HANDLE BRIEFLY AFTER WRITERS CLOSE --> RETRY
    for attempt in range(8):
        try:
            out_dir.rename(archive_dir)
            break
        except PermissionError:
            if attempt == 7:
                raise

            time.sleep(0.25)
    saved_logs = {}

    for path in (log_path, episode_log_path(log_path)):
        if path.is_file():
            destination = archive_dir / ("archived_" + path.name)
            path.rename(destination)
            saved_logs[str(path)] = destination.name

    write_json(archive_dir / "attempt_archive.json", {
        "reason": reason, "original_run_dir": str(out_dir), "training_logs": saved_logs,
        "note": "Preserved previous attempt; restarting restores no environment or RNG state.",
    })
    print(f"[ARCHIVED PREVIOUS ATTEMPT] {archive_dir}")

    return archive_dir


#CALLBACK STACK FOR ONE CURRICULUM STAGE
def make_stage_callbacks(
    condition: Condition, seed: int, stage_idx: int, out_dir: Path, num_envs: int,
    run_start_time: float,
    physics_ticks_before: int = 0,
):
    ckpt_cb = CheckpointCallback(
        save_freq=max(1, CHECKPOINT_EVERY_STEPS // max(1, num_envs)),
        save_path=str(out_dir / "checkpoints"),
        name_prefix=f"{run_name(condition.name, seed)}_stage{stage_idx}",
        save_vecnormalize=True,
    )

    stats_cb = RolloutStatsCallback(
        stage_idx=stage_idx,
        condition_name=condition.name,
        seed=seed,
        csv_path=str(train_log_path(condition, seed)),
        verbose=1,
        run_start_time=run_start_time,
        physics_ticks_before=physics_ticks_before,
    )

    return CallbackList([stats_cb, ckpt_cb])


#ROUND REQUESTED STAGE BUDGETS TO COMPLETE PPO ROLLOUTS
def stage_budgets(stages, scale: float, num_envs: int, n_steps: int = N_STEPS) -> list[int]:
    rollout_steps = int(n_steps) * num_envs

    return [max(1, math.ceil(int(stage[4] * scale) / rollout_steps)) * rollout_steps for stage in stages]


#FINGERPRINT FILE CONTENTS FOR REPRODUCIBLE ARTIFACT PAIRING
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


#COLLECT PROJECT & EDITABLE SIMULATOR SOURCE WITHOUT GENERATED OUTPUTS
def source_files() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    files = {}

    for folder in ("config", "drone_env", "training", "evaluation", "analysis"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".json"} and "__pycache__" not in path.parts:
                files[path.relative_to(root).as_posix()] = path

    for name in ("run_experiment.py", "run_wall_gap_eval.py", "watch_model.py", "requirements.txt"):
        path = root / name

        if path.is_file():
            files[name] = path

    sim_root = Path(gym_pybullet_drones.__file__).resolve().parent

    for path in sorted(sim_root.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".urdf", ".dae", ".obj", ".mtl", ".png", ".json", ".txt", ".csv", ".bin", ".xml"}:
            files["external/gym_pybullet_drones/" + path.relative_to(sim_root).as_posix()] = path

    return files


#IDENTIFY SOURCES THAT CHANGE TRAINING SEMANTICS
def training_source_hashes(files: dict[str, Path]) -> dict[str, str]:
    prefixes = ("config/", "drone_env/", "training/", "external/")

    return {
        name: file_sha256(path) for name, path in files.items()
        if name.startswith(prefixes)
    }


#WRITE JSON ATOMICALLY SO INTERRUPTED WRITES CANNOT LOOK COMPLETE
def write_json(path: Path, data: dict) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")

    with temp_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

    temp_path.replace(path)


#BUILD EXACT CONDITION, SOURCE, DEPENDENCY & ROLLOUT METADATA
def build_run_manifest(
    condition: Condition,
    out_dir: Path,
    stages,
    device: str,
    scale: float,
    seed: int,
    num_envs: int,
    preset: str = "compact",
    stage_ceiling: int | None = None,
) -> dict:
    cfg_preset = get_preset(preset)
    condition = apply_preset_to_condition(condition, cfg_preset)
    budgets = stage_budgets(stages, scale, num_envs, cfg_preset["n_steps"])
    gated = bool(cfg_preset.get("stage_gate"))
    stage_ceiling = resolve_stage_ceiling(stage_ceiling, cfg_preset["n_steps"] * num_envs)
    boundaries = []
    total = 0

    for steps in budgets:
        total += steps
        boundaries.append(total)

    files = source_files()
    manifest = {
        "schema_version": 5,
        "status": "running",
        "condition": condition.name,
        "run_seed": int(seed),
        #EXACT PER-COPY SEEDS, SO ANY RUN CAN BE REPRODUCED OR AUDITED FOR
        #OVERLAP WITH THE EVALUATION SEED BAND
        "stage_env_seeds": [
            {
                "stage": i,
                "base": train_env_seed(seed, i),
                "copies": [train_env_seed(seed, i) + c for c in range(num_envs)],
            }
            for i in range(len(stages))
        ],
        "reward_shaped": condition.reward_shaped,
        "obstacle_sensing": condition.obstacle_sensing,
        "use_lidar": condition.use_lidar,
        "masked_slices": [[sl.start, sl.stop] for sl in condition.masked_slices],
        "reward_profile": vars(condition.reward),
        "preset": preset,
        "preset_settings": cfg_preset,
        "curriculum_variant": cfg_preset["curriculum"],
        "effective_reward_profile": vars(condition.reward),
        "resolved_stage_settings": [stage_env_settings(cfg_preset, i, stages) for i in range(len(stages))],
        "timestep_unit": "policy decisions",
        "episode_horizon_unit": "physics ticks",
        "physics_hz": 240,
        "decisions_per_second": 240.0 / cfg_preset["action_repeat"],
        "decision_budget_mode": "success_gated" if gated else "fixed",
        "physics_ticks_budget_upper_bound": (
            stage_ceiling * len(stages) * cfg_preset["action_repeat"]
            if gated and stage_ceiling is not None else
            None if gated else sum(budgets) * cfg_preset["action_repeat"]
        ),
        "physics_ticks_budget_estimate": None if gated else sum(budgets) * cfg_preset["action_repeat"],
        "reward_aggregation": "undiscounted sum of physics-tick rewards per decision",
        "gamma_unit": "policy decisions",
        "restart_policy": "--resume restores a committed PPO/normalizer/scheduler/RNG checkpoint in an archived branch; simulator episodes restart; --retrain starts from stage zero",
        "training_csv_units": {"timesteps": "policy decisions", "ep_len_mean": "policy decisions", "physics_ticks": "actual physics ticks"},
        "stage_step_boundaries": None if gated else boundaries,
        "requested_stage_steps": None if gated else [int(stage[4] * scale) for stage in stages],
        "actual_stage_steps": None if gated else budgets,
        "reference_requested_stage_steps": [int(stage[4] * scale) for stage in stages],
        "reference_stage_steps": budgets,
        "training_source_sha256": training_source_hashes(files),
        "source_sha256": {name: file_sha256(path) for name, path in files.items()},
        "source_snapshot": "source_snapshot.zip",
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch_threads": torch.get_num_threads(),
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device) if device.startswith("cuda") else platform.processor(),
        },
        "packages": dict(sorted((dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions() if dist.metadata["Name"])),
        "device": device,
        "timestep_scale": scale,
        "n_envs": num_envs,
        "hyperparameters": {
            "algorithm": "PPO+imitation" if cfg_preset.get("imitation") else "PPO",
            "learning_rate": cfg_preset.get("learning_rate", LEARNING_RATE),
            "n_steps": cfg_preset["n_steps"],
            "batch_size": BATCH_SIZE,
            "n_epochs": N_EPOCHS,
            "gamma": GAMMA,
            "gae_lambda": GAE_LAMBDA,
            "clip_range": CLIP_RANGE,
            "ent_coef": ENTROPY_COEF,
            "vf_coef": VALUE_COEF,
            "max_grad_norm": MAX_GRAD_NORM,
            "target_kl": TARGET_KL,
            "net_arch": {k: list(v) for k, v in NET_ARCH.items()},
        },
        "curriculum": [list(s) for s in stages],
        "stage_gate_enabled": bool(cfg_preset.get("stage_gate")),
        "stage_gate_settings": blending_settings() if gated else None,
        "stage_ceiling": stage_ceiling,
        "blending_enabled": bool(cfg_preset.get("blending")),
        "blending_settings": blending_settings() if cfg_preset.get("blending") else None,
        "teacher": teacher_settings(stages[0][0], kind=cfg_preset.get("teacher", "expert")) if cfg_preset.get("blending") else None,
        "imitation_enabled": bool(cfg_preset.get("imitation")),
        "manual_stage_advances": [],
        "bypassed_stages": [],
        "curriculum_mastery_complete": None,
    }

    return manifest


#CHECK COMPLETE RUN AGAINST CURRENT CONFIG, SOURCE & SAVED ARTIFACTS
def validate_complete_run(out_dir: Path, expected: dict, log_path: Path, allow_bypassed: bool = False) -> None:
    manifest_path = out_dir / "run_manifest.json"

    if not manifest_path.is_file():
        raise ValueError(f"missing manifest in {out_dir}; use --retrain or fresh output directories")

    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    bypassed_stages = set(saved.get("bypassed_stages", []))

    if bypassed_stages and not allow_bypassed:
        raise ValueError("run contains manual curriculum bypasses; explicit --resume latest is required for reuse")

    stage_names = saved.get("stage_checkpoint_names", {})
    match_keys = (
        "schema_version", "condition", "run_seed", "stage_env_seeds", "reward_profile",
        "masked_slices", "use_lidar", "timestep_scale", "n_envs", "hyperparameters",
        "curriculum", "actual_stage_steps", "training_source_sha256", "packages",
        "preset", "preset_settings", "curriculum_variant", "resolved_stage_settings",
        "reward_shaped", "obstacle_sensing", "requested_stage_steps", "stage_step_boundaries",
        "blending_enabled", "blending_settings", "stage_gate_enabled", "stage_gate_settings",
        "stage_ceiling", "decision_budget_mode", "reference_stage_steps", "reference_requested_stage_steps",
        "teacher", "imitation_enabled",
    )
    if expected.get("stage_gate_enabled"):
        match_keys = tuple(key for key in match_keys if key not in {"actual_stage_steps", "stage_step_boundaries"})
    mismatches = [key for key in match_keys if saved.get(key) != expected.get(key)]

    if saved.get("status") != "complete" or mismatches:
        raise ValueError(f"incomplete or incompatible run {out_dir} ({mismatches}); use --retrain or fresh output directories")

    required = {"final.zip", "final_vecnormalize.pkl", "train_summary.json", "source_snapshot.zip"}
    if expected.get("stage_gate_enabled"):
        required.add("stage_gate_history.json")
    if saved.get("schema_version", 0) >= 3:
        required.update({"initial.zip", "initial_vecnormalize.pkl"})

        for stage_idx in range(len(expected["curriculum"])):
            if stage_idx not in bypassed_stages:
                checkpoint_name = stage_names.get(str(stage_idx), f"stage_{stage_idx}")
                required.update({f"{checkpoint_name}.zip", f"{checkpoint_name}_vecnormalize.pkl"})
    if saved.get("schema_version", 0) >= 5 and expected.get("stage_gate_enabled"):
        for checkpoint_name in ("initial", "final", *(stage_names.get(str(i), f"stage_{i}")
                                for i in range(len(expected["curriculum"])) if i not in bypassed_stages)):
            required.update({f"{checkpoint_name}_training_state.json", f"{checkpoint_name}_rng.pkl"})
    artifacts = saved.get("artifact_sha256", {})

    if not required.issubset(artifacts):
        raise ValueError(f"missing completion artifact hashes in {out_dir}")

    for name, digest in artifacts.items():
        path = out_dir / name

        if not path.resolve().is_relative_to(out_dir.resolve()):
            raise ValueError(f"run artifact path escapes its directory: {name}")

        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"missing or modified run artifact: {path}")

    if not log_path.is_file() or file_sha256(log_path) != saved.get("train_log_sha256"):
        raise ValueError(f"missing or modified learning log: {log_path}")

    if saved.get("schema_version", 0) >= 3:
        episode_path = episode_log_path(log_path)

        if not episode_path.is_file() or file_sha256(episode_path) != saved.get("train_episode_log_sha256"):
            raise ValueError(f"missing or modified episode log: {episode_path}")
    if expected.get("stage_gate_enabled"):
        gates = json.loads((out_dir / "stage_gate_history.json").read_text(encoding="utf-8"))
        verdicts = gates.get("stage_verdicts", [])
        counts = saved.get("actual_stage_steps")
        actual_bypassed = validate_stage_completion(verdicts, gates.get("manual_stage_advances", []), len(expected["curriculum"]))

        if (len(verdicts) != len(expected["curriculum"])
                or set(actual_bypassed) != bypassed_stages
                or gates.get("manual_stage_advances", []) != saved.get("manual_stage_advances", [])
                or saved.get("curriculum_mastery_complete", not actual_bypassed) != (not actual_bypassed)
                or not isinstance(counts, list)
                or counts != [item.get("decisions") for item in verdicts]
                or sum(counts) != saved.get("actual_total_decisions")):
            raise ValueError(f"gated run did not pass every stage with consistent actual counters: {out_dir}")


#REJECT INVALID BUDGETS, UNSUPPORTED VECTORIZATION & OVERLAPPING SEED BANDS
def validate_training_args(seed: int, scale: float, num_envs: int, stages) -> None:
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    if num_envs != 1 or any(stage[1] != 1 for stage in stages):
        raise ValueError("verified training requires --n-envs 1 and one drone")
    if seed < 0 or train_env_seed(seed, len(stages) - 1) >= 100_000:
        raise ValueError("run seed must keep training environment seeds below 100000 (use 0 through 9)")


#TRAIN ONE (CONDITION, SEED) RUN THROUGH EVERY CURRICULUM STAGE
#SHARED FULL-INFORMATION TEACHER; EACH STUDENT RETAINS ITS CONDITION'S MASK
def make_condition_expert(condition: Condition, world_scale: float, kind: str = "expert"):
    """
    Every condition gets the SAME full-information expert (decided 2026-10-12;
    see training/teacher.py). The blend wrapper sits below the observation mask,
    so the teacher reads the raw observation even when its student is blind; the
    student's own observation is still masked as its condition requires.
    `condition` is accepted for call-site compatibility and deliberately unused.
    `kind` selects the plain expert or the gap-aware one (phase 2).
    """
    return make_teacher(world_scale, kind=kind)


#FIND THE BLEND WRAPPER INSIDE A BUILT VEC CHAIN
def find_blend_wrapper(env):
    node = env

    for _ in range(12):
        if node is None:
            return None
        if isinstance(node, VecExpertBlend):
            return node

        node = getattr(node, "venv", None)

    return None


#WRITE THE AUDITABLE GATE RECORD (SETTINGS, SCHEDULER, VERDICTS, VALIDATION LOG)
def save_gate_history(out_dir, scheduler, blending_on, stage_ceiling, stage_verdicts, validation_log, manual_advances=None):
    write_json(out_dir / "stage_gate_history.json", {
        "stage_gate_settings": blending_settings(),
        "blending_enabled": bool(blending_on),
        "stage_ceiling": stage_ceiling,
        "current_stage": scheduler.snapshot(),
        "stage_verdicts": stage_verdicts,
        "validation_log": validation_log,
        "manual_stage_advances": manual_advances or [],
    })


#ARCHIVE THE STOPPED ATTEMPT & BRANCH FROM THE SELECTED COMMITTED CHECKPOINT
def prepare_resume(out_dir: Path, log_path: Path, selector: str, expected: dict,
                   resume_stage: int | None = None, resume_reason: str | None = None) -> dict:
    """Preserve the stopped attempt; branch from the selected committed boundary."""
    model_path, state = select_checkpoint(out_dir, selector)
    validate_resume(state, expected, log_path, out_dir, resume_stage)
    archived = archive_run_attempt(out_dir, log_path, f"checkpoint continuation from {model_path.name}")
    out_dir.mkdir(parents=True)
    names = {"source_snapshot.zip"}
    checkpoints = {"initial", model_path.stem}
    verdict_by_stage = {item["stage"]: item for item in state["stage_verdicts"]}
    checkpoints.update(verdict_by_stage.get(item["stage"], {}).get("checkpoint") or f"stage_{item['stage']}"
                       for item in state["stage_times"]
                       if item["stage"] < state["stage_index"] and item.get("gate_state") != BYPASSED)
    checkpoints.update(item["checkpoint"] for item in state["stage_verdicts"] if item.get("checkpoint"))

    for checkpoint in checkpoints:
        names.update(f"{checkpoint}{suffix}" for suffix in (
            ".zip", "_vecnormalize.pkl", "_rng.pkl", "_training_state.json"))

    for name in names:
        source = archived / name
        if source.is_file():
            shutil.copy2(source, out_dir / name)

    for prefix in state["log_prefixes"]:
        source = archived / ("archived_" + prefix["name"])
        target = log_path.parent / prefix["name"]
        target.parent.mkdir(parents=True, exist_ok=True)
        remaining = prefix["bytes"]
        with source.open("rb") as reader, target.open("xb") as writer:
            while remaining:
                chunk = reader.read(min(1024 * 1024, remaining))

                if not chunk:
                    raise ValueError("archived log is shorter than committed checkpoint")

                writer.write(chunk)
                remaining -= len(chunk)
    ancestry_path = archived / "resume_ancestry.json"
    ancestors = json.loads(ancestry_path.read_text(encoding="utf-8")) if ancestry_path.exists() else []
    ancestors.append({"archived_run": str(archived), "checkpoint": model_path.name,
                      "total_decisions": state["total_decisions"],
                      "state_sha256": file_sha256(archived / (model_path.stem + "_training_state.json")),
                      "note": "All prior logs/checkpoints/diagnostics preserved here; active CSVs contain only the committed prefix."})
    write_json(out_dir / "resume_ancestry.json", ancestors)
    state["resume_ancestry"] = ancestors
    if resume_stage is not None:
        state = advance_checkpoint_stage(state, resume_stage, resume_reason)

    return state


#TRAIN ONE (CONDITION, SEED) RUN; ON ANY ABORT MARK THE MANIFEST stopped/failed & CLOSE RESOURCES
def train(condition: Condition, seed: int = 0, device: str = "auto", scale: float = 1.0,
          num_envs: int = N_ENVS, preset: str = "compact", retrain: bool = False,
          stage_ceiling: int | None = None, resume: str | None = None,
          resume_stage: int | None = None, resume_reason: str | None = None,
          init_from: str | None = None) -> None:
    resources = []
    try:
        _train_impl(condition, seed, device, scale, num_envs, preset, retrain,
                    stage_ceiling, resume, resources, resume_stage, resume_reason, init_from)
    except BaseException as error:
        out_dir = Path(PPO_MODELS_DIR) / run_name(condition.name, seed)
        path = out_dir / "run_manifest.json"
        if resources and path.is_file():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("status") == "running":
                saved.update(status="stopped" if isinstance(error, KeyboardInterrupt) else "failed",
                             stop_reason=f"{type(error).__name__}: {error}",
                             continuation_note="Resume the last committed checkpoint; unfinished work is preserved in the archived attempt.")
                write_json(path, saved)
        raise
    finally:
        for resource in reversed(resources):
            safe_close(resource)


#FULL TRAINING LOOP: ENV STACK, PPO (+IMITATION), STAGE GATES, CHECKPOINTS, MANIFEST
def _train_impl(
    condition: Condition,
    seed: int = 0,
    device: str = "auto",
    scale: float = 1.0,
    num_envs: int = N_ENVS,
    preset: str = "compact",
    retrain: bool = False,
    stage_ceiling: int | None = None,
    resume: str | None = None,
    resources: list | None = None,
    resume_stage: int | None = None,
    resume_reason: str | None = None,
    init_from: str | None = None,
) -> None:
    device_ = resolve_device(device)
    cfg_preset = get_preset(preset)
    #WARM START SOURCE: CLI VALUE, ELSE THE PRESET'S (PHASE 2); NEVER WHEN RESUMING
    init_from = init_from or cfg_preset.get("init_from")
    stages = Curriculum(cfg_preset["curriculum"]).stages
    validate_training_args(seed, scale, num_envs, stages)
    stage_ceiling = resolve_stage_ceiling(stage_ceiling, cfg_preset["n_steps"] * num_envs)
    out_dir = condition_dir(condition, seed)
    log_path = train_log_path(condition, seed)

    if resume and retrain:
        raise ValueError("--resume and --retrain are mutually exclusive")
    if resume_stage is not None and (not resume or not cfg_preset.get("stage_gate")):
        raise ValueError("--resume-stage requires --resume and a success-gated preset")
    if resume_reason and resume_stage is None:
        raise ValueError("--resume-reason requires --resume-stage")

    if TORCH_NUM_THREADS > 0:
        torch.set_num_threads(TORCH_NUM_THREADS)
    manifest = build_run_manifest(condition, out_dir, stages, device_, scale, seed, num_envs, preset, stage_ceiling)
    resume_state = None

    if any(out_dir.iterdir()) or log_path.exists() or episode_log_path(log_path).exists():
        if resume:
            resume_state = prepare_resume(out_dir, log_path, resume, manifest, resume_stage, resume_reason)
        else:
            if not retrain:
                raise ValueError(f"existing attempt {out_dir}; use --resume latest, --retrain, or fresh output directories")

            archive_run_attempt(out_dir, log_path, "explicit --retrain")
            out_dir.mkdir(parents=True)
    elif resume_stage is not None or (resume and resume != "latest"):
        raise ValueError("selected checkpoint run does not exist")
    budgets = stage_budgets(stages, scale, num_envs, cfg_preset["n_steps"])

    #SHARED HELPER, SO TRAINING AND EVALUATION CANNOT DRIFT APART
    base_condition = condition
    condition = apply_preset_to_condition(condition, cfg_preset)
    run_start_time = time.monotonic() - (resume_state["run_wall_time_s"] if resume_state else 0.0)

    #POLICY INIT, ACTION SAMPLING AND TORCH ALL PINNED TO THE RUN SEED. THIS
    #REPLACES PPO(seed=...), WHICH RAISES ON THIS WRAPPER STACK BECAUSE IT
    #CALLS VecEnv.seed() - SEE env_factory.seed_vec_env.
    set_random_seed(seed, using_cuda=device_.startswith("cuda"))

    if TORCH_NUM_THREADS > 0:
        torch.set_num_threads(TORCH_NUM_THREADS)

    print(
        f"\n===== CONDITION: {condition.name} | seed={seed} | "
        f"device={device_} | stages={len(stages)} ====="
    )
    print(f"reward profile: {condition.reward.name} | lidar: {condition.use_lidar}")

    if resume_state:
        manifest["resume_ancestry"] = resume_state["resume_ancestry"]
        manifest["resume_source_sha256"] = manifest["source_sha256"]
        manifest["source_sha256"] = resume_state["training_manifest"]["source_sha256"]
        manifest["manual_stage_advances"] = resume_state.get("manual_stage_advances", resume_state["training_manifest"].get("manual_stage_advances", []))
        manifest["bypassed_stages"] = sorted(item["stage"] for item in resume_state["stage_verdicts"] if item["state"] == BYPASSED)
    diag_dir = out_dir / "diagnostics" / str(time.time_ns())
    manifest["diagnostics_dir"] = diag_dir.relative_to(out_dir).as_posix()
    write_json(out_dir / "run_manifest.json", manifest)

    if not resume_state:
        with ZipFile(out_dir / "source_snapshot.zip", "w", ZIP_DEFLATED) as archive:
            for name, path in source_files().items():
                archive.write(path, name)

    #FRESH LEARNING-CURVE CSV FOR THIS RUN; PRIOR ATTEMPTS WERE ARCHIVED
    log_path.parent.mkdir(parents=True, exist_ok=True)

    stage_times = resume_state["stage_times"] if resume_state else []

    model = None
    train_env = None
    physics_ticks = resume_state["physics_ticks"] if resume_state else 0

    #ADAPTIVE ASSISTANCE IS OPT-IN PER PRESET; AN UNASSISTED RUN TAKES THE
    #IDENTICAL PATH IT TOOK BEFORE AND BUILDS NO EXPERT AT ALL
    #TWO INDEPENDENT SWITCHES. THE GATE DECIDES HOW A STAGE IS LEFT (VALIDATED
    #SUCCESS, NOT A FIXED BUDGET) AND APPLIES TO ASSISTED AND UNASSISTED RUNS
    #ALIKE, SO THE TWO ARE COMPARABLE. BLENDING ONLY DECIDES WHETHER A TEACHER
    #EXISTS AT ALL; AN UNASSISTED GATED RUN TRAINS AT beta 0 THROUGHOUT.
    gating_on = bool(cfg_preset.get("stage_gate"))
    blending_on = bool(cfg_preset.get("blending"))
    imitation_on = bool(cfg_preset.get("imitation")) and blending_on
    algo_cls = ImitationPPO if imitation_on else PPO
    algo_kwargs = (dict(imitation_epochs=IMITATION_EPOCHS, imitation_batch_size=IMITATION_BATCH_SIZE)
                   if imitation_on else {})
    stage_verdicts = resume_state["stage_verdicts"] if resume_state else []
    validation_log = resume_state["validation_log"] if resume_state else []
    stalled_stage = None

    if gating_on:
        validate_blending_settings(len(stages))

    start_stage = resume_stage if resume_stage is not None else (resume_state["stage_index"] if resume_state else 0)
    resume_finished = bool(resume_stage is None and resume_state and resume_state["scheduler"]["state"] == PASSED
                           and any(item["stage"] == start_stage for item in stage_times))
    if resume_finished:
        start_stage = min(start_stage + 1, len(stages) - 1)
        #LAST-STAGE BOUNDARY CAN FINALIZE WITHOUT ANOTHER TRAINING BLOCK
        resume_finished = start_stage != resume_state["stage_index"]
    if resume_state:
        stage_times = [item for item in stage_times if item["stage"] < start_stage]
        stage_verdicts = [item for item in stage_verdicts if item["stage"] < start_stage]

    #SAVE MODEL + NORMALIZER, THEN COMMIT THE RESUMABLE STATE BUNDLE WHEN GATING
    def checkpoint(name):
        save_checkpoint(model, train_env, out_dir, name)

        if not gating_on:
            return

        stats = callbacks.callbacks[0]
        state = {
            "training_manifest": manifest, "stage_index": stage_idx,
            "stage_decisions": stage_decisions, "total_decisions": int(model.num_timesteps),
            "physics_ticks": int(stats.physics_ticks), "stage_physics_ticks": int(stats.stage_physics_ticks),
            "stage_episodes": int(getattr(stats, "stage_episodes", 0)),
            "run_wall_time_s": time.monotonic() - run_start_time,
            "stage_wall_time_s": time.monotonic() - stage_start,
            "scheduler": scheduler.snapshot(), "stage_times": stage_times,
            "stage_verdicts": stage_verdicts, "validation_log": validation_log,
            "log_prefixes": log_prefixes([log_path, episode_log_path(log_path)]),
        }
        committed = commit_state(model, out_dir, name, state)
        manifest["latest_checkpoint"] = {"model": name + ".zip", "training_state": committed.name,
                                         "total_decisions": int(model.num_timesteps)}
        write_json(out_dir / "run_manifest.json", manifest)

    for stage_idx, stage in enumerate(stages):
        if stage_idx < start_stage:
            continue

        continuing_stage = bool(resume_state and stage_idx == resume_state["stage_index"] and not resume_finished)
        stage_start = time.monotonic() - (resume_state["stage_wall_time_s"] if continuing_stage else 0.0)
        scheduler = (StageScheduler.restore(resume_state["scheduler"], ceiling=stage_ceiling)
                     if continuing_stage else
                     StageScheduler(stage_idx, ceiling=stage_ceiling, assistance_enabled=blending_on)) if gating_on else None
        if resume_stage == stage_idx:
            scheduler._log("manual_entry", 0, None,
                           f"continued from stage {resume_state['stage_index']} checkpoint {resume_state['checkpoint']}; {manifest['manual_stage_advances'][-1]['reason']}")
        stage_decisions = int(resume_state["stage_decisions"]) if continuing_stage else 0
        cfg = stage_to_kwargs(stage)
        stage_steps = budgets[stage_idx]
        stage_settings_kwargs = stage_env_settings(cfg_preset, stage_idx, stages)
        stage_seed = train_env_seed(seed, stage_idx)

        print(
            f"\n--- Stage {stage_idx} | world={cfg['world_scale']} "
            f"drones={cfg['num_drones']} static={cfg['num_static_obs']} "
            f"dynamic={cfg['num_dynamic_obs']} steps={stage_steps} "
            f"env_seed={stage_seed} ---"
        )

        new_env = make_vec_env(
            condition=condition,
            world_scale=cfg["world_scale"],
            num_drones=cfg["num_drones"],
            num_static_obs=cfg["num_static_obs"],
            num_dynamic_obs=cfg["num_dynamic_obs"],
            episode_horizon=cfg["episode_horizon"],
            stage_lidar=cfg["stage_lidar"],
            num_envs=num_envs,
            seed=stage_seed,
            action_repeat=cfg_preset["action_repeat"],
            blend_expert=make_condition_expert(condition, cfg["world_scale"], kind=cfg_preset.get("teacher", "expert")) if blending_on else None,
            blend_beta=scheduler.beta if blending_on else 0.0,
            **stage_settings_kwargs,
        )

        #OPTIMIZER-SIDE REWARD SCALING ONLY. OUTSIDE VecMonitor, SO THE
        #LEARNING CSVS KEEP RECORDING UNSCALED EPISODE RETURNS.
        if cfg_preset["reward_scale"] != 1.0:
            new_env = ScaleReward(new_env, cfg_preset["reward_scale"])
        if resources is not None:
            resources.append(new_env)

        if model is None and resume_state:
            with (out_dir / (resume_state["checkpoint"] + "_vecnormalize.pkl")).open("rb") as stream:
                saved_norm = pickle.load(stream)
            current_norm = unwrap_vec_normalize(new_env)
            current_norm.obs_rms = saved_norm.obs_rms
            current_norm.ret_rms = saved_norm.ret_rms
            model = algo_cls.load(str(out_dir / (resume_state["checkpoint"] + ".zip")),
                                  env=new_env, device=device_, force_reset=True)

            if int(model.num_timesteps) != resume_state["total_decisions"]:
                raise ValueError("loaded PPO counter does not match committed checkpoint")

            if not continuing_stage:
                #PPO SAVES EPISODE-AVERAGE BUFFERS TOO; A NEW STAGE NEEDS ITS OWN STATISTICS
                if model.ep_info_buffer is not None:
                    model.ep_info_buffer.clear()
                if model.ep_success_buffer is not None:
                    model.ep_success_buffer.clear()
            restore_rng(out_dir, resume_state["checkpoint"], model)
            model.set_logger(configure(str(diag_dir), ["csv", "tensorboard"]))
            if resources is not None:
                resources.append(model.logger)
            print(f"[RESUMED] stage={stage_idx} decisions={model.num_timesteps} beta={scheduler.beta:.3f}; simulator episode reset")
        elif model is None:
            model = algo_cls(
                "MlpPolicy",
                new_env,
                learning_rate=cfg_preset.get("learning_rate", LEARNING_RATE),
                n_steps=cfg_preset["n_steps"],
                batch_size=BATCH_SIZE,
                n_epochs=N_EPOCHS,
                gamma=GAMMA,
                gae_lambda=GAE_LAMBDA,
                clip_range=CLIP_RANGE,
                ent_coef=ENTROPY_COEF,
                vf_coef=VALUE_COEF,
                max_grad_norm=MAX_GRAD_NORM,
                target_kl=TARGET_KL,
                policy_kwargs=dict(net_arch=NET_ARCH),
                tensorboard_log=f"{TB_LOG_DIR}/{run_name(condition.name, seed)}",
                verbose=0,
                device=device_,
                **algo_kwargs,
            )
            #ACTION-SPACE SAMPLING IS THE ONE PIECE set_random_seed MISSES
            model.action_space.seed(seed)
            model.set_logger(configure(str(diag_dir), ["csv", "tensorboard"]))
            if resources is not None:
                resources.append(model.logger)
            #WARM START BEFORE THE INITIAL CHECKPOINT, SO initial.zip HOLDS THE LOADED WEIGHTS
            if init_from:
                manifest["warm_start"] = load_warm_start(model, new_env, init_from, condition.name, seed)
                print(f"[WARM START] policy weights and normalisation loaded from {manifest['warm_start']['source_run']}", flush=True)
        else:
            #CARRY OBS STATISTICS FORWARD SO THE POLICY SEES A STABLE INPUT SCALE
            copy_vec_normalize_stats(train_env, new_env)

            if model.observation_space != new_env.observation_space:
                safe_close(new_env)
                raise ValueError(
                    f"observation space changed across stages: "
                    f"{model.observation_space} -> {new_env.observation_space}"
                )

            model.set_env(new_env)
            safe_close(train_env)

            #STAGE-SPECIFIC REWARD MEANS MUST EXCLUDE PREVIOUS STAGE EPISODES
            if model.ep_info_buffer is not None:
                model.ep_info_buffer.clear()
            if model.ep_success_buffer is not None:
                model.ep_success_buffer.clear()

        train_env = new_env

        callbacks = make_stage_callbacks(condition, seed, stage_idx, out_dir, num_envs, run_start_time, physics_ticks)
        stats_cb = callbacks.callbacks[0]

        #IMITATION TARGETS: COPY THE BLEND WRAPPER'S EXPERT ACTIONS INTO THE MODEL
        if imitation_on:
            callbacks.callbacks.append(ExpertActionRecorder())
        if continuing_stage:
            stats_cb.stage_physics_ticks = resume_state["stage_physics_ticks"]
            stats_cb.stage_episodes = resume_state["stage_episodes"]
            stats_cb._start_time = stage_start
        decisions_before_stage = int(model.num_timesteps) - stage_decisions
        if not resume_state and stage_idx == 0:
            checkpoint("initial")
        if resume_stage == stage_idx:
            checkpoint(f"stage_{stage_idx}_manual_entry")
            save_gate_history(out_dir, scheduler, blending_on, stage_ceiling, stage_verdicts,
                              validation_log, manifest["manual_stage_advances"])

        if not gating_on:
            model.learn(
                total_timesteps=stage_steps,
                reset_num_timesteps=False,
                callback=callbacks,
                tb_log_name="run",
            )
        else:
            blend_env = find_blend_wrapper(train_env)

            if blending_on and blend_env is None:
                raise RuntimeError("blending preset built an env without the blend wrapper")

            if blend_env is not None:
                blend_env.set_beta(scheduler.beta)

            #IMITATION FADES WITH ASSISTANCE: WEIGHT = COEF * CURRENT BETA
            if imitation_on:
                model.imitation_weight = IMITATION_COEF * scheduler.beta

            #TRAIN IN FIXED BLOCKS, THEN VALIDATE. beta ONLY EVER CHANGES
            #BETWEEN BLOCKS, SO IT IS CONSTANT WITHIN EVERY PPO ROLLOUT.
            while not scheduler.finished:
                rollout_steps = int(cfg_preset["n_steps"]) * num_envs
                block = math.ceil(CHECK_INTERVAL_DECISIONS / rollout_steps) * rollout_steps

                if stage_ceiling is not None:
                    block = int(min(block, stage_ceiling - stage_decisions))

                if blend_env is not None:
                    blend_env.reset_diagnostics()
                decisions_before_block = int(model.num_timesteps)
                model.learn(
                    total_timesteps=block,
                    reset_num_timesteps=False,
                    callback=callbacks,
                    tb_log_name="run",
                )
                actual_block = int(model.num_timesteps) - decisions_before_block

                if actual_block != block:
                    raise RuntimeError(f"PPO block counter mismatch: requested {block}, executed {actual_block}")

                stage_decisions += actual_block
                action_diag = blend_env.diagnostics() if blend_env is not None else {}
                #VALIDATE WITH THE CURRENT BLEND (AT BETA 0 THAT IS THE STUDENT ALONE)
                check = run_validation_batch(
                    model, train_env, condition, cfg, cfg_preset,
                    stage_idx, seed, stage_settings_kwargs,
                    beta=scheduler.beta if blending_on else 0.0,
                )
                beta_before = scheduler.beta
                beta = scheduler.observe(check["success_rate"], stage_decisions)

                if blend_env is not None:
                    blend_env.set_beta(beta)
                if imitation_on:
                    model.imitation_weight = IMITATION_COEF * beta
                validation_log.append({
                    "stage": stage_idx,
                    "check": scheduler.checks,
                    "decisions": stage_decisions,
                    "beta_during_block": beta_before,
                    "beta_next_block": beta,
                    "state": scheduler.state,
                    "assisted": beta_before > 0.0,
                    **check,
                    **action_diag,
                })
                #AN UNBOUNDED STAGE CAN RUN A LONG TIME, SO EVERY CHECK LEAVES A
                #RECOVERABLE MODEL/NORMALIZER PAIR, NOT ONLY STAGE ENDS
                model.logger.record("time/total_timesteps", model.num_timesteps)
                model.logger.dump(step=model.num_timesteps)
                checkpoint(f"stage_{stage_idx}_check_{scheduler.checks:03d}")
                save_gate_history(out_dir, scheduler, blending_on, stage_ceiling, stage_verdicts, validation_log, manifest["manual_stage_advances"])
                print(
                    f"    [gate] stage {stage_idx} check {scheduler.checks} "
                    f"decisions={stage_decisions} success={check['success_rate']:.3f} "
                    f"beta {beta_before:.3f}->{beta:.3f} state={scheduler.state}",
                    flush=True,
                )

            #THE LOOP'S BUDGET CHECK AND THE SCHEDULER'S LIVE IN DIFFERENT
            #MODULES. IF THE LOOP STOPS FIRST THE STAGE IS STILL A STALL, AND
            #THE RECORDED VERDICT MUST SAY SO RATHER THAN READ "ASSISTED".
            if not scheduler.finished:
                scheduler.state = STALLED
                scheduler._log("stalled", stage_decisions, None,
                               f"training loop exhausted its {stage_decisions}-decision stage budget "
                               f"without passing the unassisted gate")

            stage_verdicts.append({
                **scheduler.snapshot(),
                "stage": stage_idx,
                "state": scheduler.state,
                "checks": scheduler.checks,
                "decisions": stage_decisions,
                "final_beta": scheduler.beta,
                "history": scheduler.history,
            })
            save_gate_history(out_dir, scheduler, blending_on, stage_ceiling, stage_verdicts, validation_log, manifest["manual_stage_advances"])
            if scheduler.state == STALLED:
                stalled_stage = stage_idx

        #SB3 LOGS BEFORE OPTIMIZATION; FLUSH FINAL UPDATE BEFORE NEXT LEARN CALL
        model.logger.record("time/total_timesteps", model.num_timesteps)
        model.logger.record("rollout_stats/stage", stage_idx)
        model.logger.dump(step=model.num_timesteps)
        stats_cb = callbacks.callbacks[0]
        physics_ticks = stats_cb.physics_ticks

        stage_times.append(
            {
                "stage": stage_idx,
                "planned_steps": None if gating_on else stage_steps,
                "reference_steps": stage_steps,
                "stage_decisions": int(model.num_timesteps) - decisions_before_stage,
                "gate_state": scheduler.state if gating_on else None,
                "timesteps_after": int(model.num_timesteps),
                "decisions_after": int(model.num_timesteps),
                "physics_ticks_after": int(physics_ticks),
                "stage_physics_ticks": int(stats_cb.stage_physics_ticks),
                "env_seed": stage_seed,
                "wall_time_s": round(time.monotonic() - stage_start, 1),
            }
        )
        checkpoint(f"stage_{stage_idx}")

        write_json(out_dir / "train_summary.json", {
            "condition": condition.name,
            "run_seed": int(seed),
            "status": "running",
            "total_timesteps": int(model.num_timesteps),
            "total_decisions": int(model.num_timesteps),
            "total_physics_ticks": int(physics_ticks),
            "total_wall_time_s": round(time.monotonic() - run_start_time, 1),
            "stages": stage_times,
        })

        if stalled_stage is not None:
            break

    if not gating_on and int(model.num_timesteps) != sum(budgets):
        raise RuntimeError("training did not complete the declared decision budget")

    if gating_on and stalled_stage is None:
        manifest["bypassed_stages"] = validate_stage_completion(stage_verdicts, manifest["manual_stage_advances"], len(stages))
        manifest["curriculum_mastery_complete"] = not manifest["bypassed_stages"]
    status = "stalled" if stalled_stage is not None else "complete"
    checkpoint_name = "stalled" if stalled_stage is not None else "final"
    checkpoint(checkpoint_name)
    safe_close(train_env)
    model.logger.close()

    #STAGE TIMING/PROGRESS RECORD NEXT TO THE MODELS (FOR THE REPORT)
    write_json(out_dir / "train_summary.json", {
                "condition": condition.name,
                "run_seed": int(seed),
                "status": status,
                "stalled_stage": stalled_stage,
                "total_timesteps": int(model.num_timesteps),
                "total_decisions": int(model.num_timesteps),
                "total_physics_ticks": int(physics_ticks),
                "total_wall_time_s": round(time.monotonic() - run_start_time, 1),
                "stages": stage_times,
                "manual_stage_advances": manifest["manual_stage_advances"],
                "bypassed_stages": manifest["bypassed_stages"],
                "curriculum_mastery_complete": manifest["curriculum_mastery_complete"],
    })

    if training_source_hashes(source_files()) != manifest["training_source_sha256"]:
        raise RuntimeError("training source changed during run; results left incomplete to avoid false provenance")

    manifest["status"] = status
    manifest["stalled_stage"] = stalled_stage
    if gating_on:
        manifest["actual_stage_steps"] = [entry["stage_decisions"] for entry in stage_times]
        manifest["stage_step_boundaries"] = [entry["decisions_after"] for entry in stage_times]
        manifest["stage_checkpoint_names"] = {str(item["stage"]): item.get("checkpoint") or f"stage_{item['stage']}"
                                               for item in stage_verdicts if item["state"] == PASSED}
    manifest["artifact_sha256"] = {
        name: file_sha256(out_dir / name)
        for name in (f"{checkpoint_name}.zip", f"{checkpoint_name}_vecnormalize.pkl", "train_summary.json", "source_snapshot.zip")
    }
    if gating_on:
        manifest["artifact_sha256"]["stage_gate_history.json"] = file_sha256(out_dir / "stage_gate_history.json")
    manifest["artifact_sha256"].update({
        path.relative_to(out_dir).as_posix(): file_sha256(path)
        for path in out_dir.rglob("*") if path.is_file() and (
            path.suffix in {".zip", ".pkl"} or path.name.endswith("_training_state.json")
            or path.name == "resume_ancestry.json" or path.is_relative_to(diag_dir)
        )
    })
    manifest["train_log_sha256"] = file_sha256(log_path)
    manifest["train_episode_log_sha256"] = file_sha256(episode_log_path(log_path))
    manifest["actual_total_decisions"] = int(model.num_timesteps)
    manifest["actual_total_physics_ticks"] = int(physics_ticks)

    if not all(torch.isfinite(parameter).all().item() for parameter in model.policy.parameters()):
        raise RuntimeError("non-finite trained policy parameters; run left incomplete")

    write_json(out_dir / "run_manifest.json", manifest)

    if stalled_stage is not None:
        raise RuntimeError(f"stage {stalled_stage} STALLED at its optional ceiling; artifacts saved and no later stage started")

    print(
        f"[DONE] {condition.name} seed {seed} trained for "
        f"{model.num_timesteps} timesteps"
    )


#PARSE CLI ARGS
def parse_args():
    parser = argparse.ArgumentParser(description="PPO curriculum training (2x2 ablation grid)")
    parser.add_argument(
        "--condition",
        default="all",
        help="all, both (degenerate+full), or comma-separated condition names",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="scale every stage's timestep budget (use a small value for smoke runs)",
    )
    parser.add_argument("--n-envs", type=int, default=N_ENVS, help="parallel env copies")
    parser.add_argument(
        "--seeds",
        default=",".join(str(s) for s in TRAIN_SEEDS),
        help="comma-separated run seeds, one independent training run each",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="train even when the run's final model already exists",
    )
    parser.add_argument("--resume", default=None, metavar="latest|CHECKPOINT",
                        help="continue from an intact committed checkpoint; latest selects one per stopped run")
    parser.add_argument("--resume-stage", type=int, default=None,
                        help="with --resume and one condition/seed, manually enter a later stage; unmet/skipped gates are recorded BYPASSED")
    parser.add_argument("--resume-reason", default=None,
                        help="report explanation for a manual --resume-stage override")
    parser.add_argument("--init-from", default=None, metavar="MODELS_DIR",
                        help="warm start: load policy weights and normalisation from <MODELS_DIR>/ppo_<condition>_s<seed>/final.* (read-only); default from the preset's init_from")
    parser.add_argument(
        "--preset",
        default="compact",
        help="named experiment preset from config/ppo_defaults.PRESETS "
        "(compact = the reported study; phase2 = the wall training)",
    )

    parser.add_argument(
        "--stage-ceiling",
        type=int,
        default=0,
        help="optional safety valve: max policy decisions a gated stage may spend "
        "before it is recorded STALLED. 0 (the default) means UNBOUNDED - a stage "
        "advances only on validated success and never on a budget",
    )

    return parser.parse_args()


#TRAIN EVERY REQUESTED (CONDITION, SEED) RUN IN SEQUENCE, SKIPPING FINISHED
def main() -> None:
    args = parse_args()
    names = resolve_condition_names(args.condition)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    if not names or not seeds or len(set(names)) != len(names) or len(set(seeds)) != len(seeds):
        raise ValueError("conditions and seeds must be nonempty and unique")
    if args.resume and args.retrain:
        raise ValueError("--resume and --retrain are mutually exclusive")
    if args.resume and args.resume != "latest" and len(names) * len(seeds) != 1:
        raise ValueError("an explicit resume checkpoint requires one condition and seed")
    if args.resume_stage is not None and (not args.resume or len(names) * len(seeds) != 1):
        raise ValueError("--resume-stage requires --resume and exactly one condition/seed")
    if args.resume_reason and args.resume_stage is None:
        raise ValueError("--resume-reason requires --resume-stage")

    for name in names:
        get_condition(name)

    cfg_preset = get_preset(args.preset)
    preset_stages = Curriculum(cfg_preset["curriculum"]).stages

    for seed in seeds:
        validate_training_args(seed, args.scale, args.n_envs, preset_stages)

    os.makedirs(PPO_MODELS_DIR, exist_ok=True)

    for seed in seeds:
        for name in names:
            condition = get_condition(name)
            final_path = condition_dir(condition, seed) / "final.zip"

            marked_complete = (json.loads((final_path.parent / "run_manifest.json").read_text(encoding="utf-8")).get("status") == "complete"
                               if (final_path.parent / "run_manifest.json").is_file() else False)
            if final_path.is_file() and not args.retrain and args.resume_stage is None and (args.resume is None or (args.resume == "latest" and marked_complete)):
                expected = build_run_manifest(condition, final_path.parent, preset_stages,
                                              resolve_device(args.device), args.scale, seed, args.n_envs, args.preset,
                                              (int(args.stage_ceiling) or None))
                validate_complete_run(final_path.parent, expected, train_log_path(condition, seed), allow_bypassed=bool(args.resume))
                print(f"[SKIP] {name} s{seed}: complete matching artifacts verified (use --retrain to redo)")
                continue

            start = time.time()
            train(
                condition=condition,
                seed=seed,
                device=args.device,
                scale=args.scale,
                num_envs=args.n_envs,
                preset=args.preset,
                retrain=args.retrain,
                stage_ceiling=(int(args.stage_ceiling) or None),
                resume=args.resume,
                resume_stage=args.resume_stage,
                resume_reason=args.resume_reason,
                init_from=args.init_from,
            )
            print(f"[TIME] {name} s{seed}: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
