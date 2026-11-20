# -*- coding: utf-8 -*-
#NEWLY ADDED FOR CS5478 PROJECT
"""Committed checkpoint bundles for stopped-run evaluation and PPO continuation.

The JSON sidecar is committed last. A partially written bundle is never selected.
Simulator episodes restart on resume; policy/optimizer, normalization, scheduler,
decision counters and Python/NumPy/Torch random streams are restored.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch

BYPASSED = "BYPASSED"
SNAPSHOT_NAME = "source_snapshot.zip"


#SHA256 OF A FILE (OPTIONALLY OF ITS FIRST limit BYTES = LOG PREFIX)
def sha256(path: Path, limit: int | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        remaining = limit

        while remaining is None or remaining > 0:
            chunk = stream.read(1024 * 1024 if remaining is None else min(remaining, 1024 * 1024))

            if not chunk:
                break

            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)

        if remaining is not None and remaining > 0:
            raise ValueError(f"log is shorter than its checkpoint prefix: {path}")

    return digest.hexdigest()


#WRITE JSON VIA TEMP FILE + REPLACE SO A PARTIAL BUNDLE IS NEVER VISIBLE
def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


#NAME, BYTE COUNT & HASH OF EACH TRAINING LOG AT COMMIT TIME
def log_prefixes(paths: list[Path]) -> list[dict]:
    return [{"name": path.name, "bytes": path.stat().st_size,
             "sha256": sha256(path)} for path in paths]


#SAVE RNG STREAMS + HASHED SIDECAR LAST --> CHECKPOINT BUNDLE COMMITTED
def commit_state(model, run_dir: Path, name: str, state: dict) -> Path:
    """Model and paired normalizer must already have been saved successfully."""
    rng_path = run_dir / f"{name}_rng.pkl"
    temporary = rng_path.with_suffix(".pkl.tmp")
    rng = {"python": random.getstate(), "numpy": np.random.get_state(),
           "torch": torch.get_rng_state(),
           "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    action_rng = getattr(getattr(model, "action_space", None), "np_random", None)
    if action_rng is not None:
        rng["action_space"] = action_rng.bit_generator.state
    with temporary.open("wb") as stream:
        pickle.dump(rng, stream)
    temporary.replace(rng_path)
    names = (f"{name}.zip", f"{name}_vecnormalize.pkl", rng_path.name, SNAPSHOT_NAME)
    payload = {"checkpoint_schema": 1, "checkpoint": name,
               **state, "artifact_sha256": {item: sha256(run_dir / item) for item in names}}
    path = run_dir / f"{name}_training_state.json"
    atomic_json(path, payload)

    return path


#LOAD & VERIFY ONE COMMITTED BUNDLE (SCHEMA, NAME, HASH OF ALL ARTIFACTS; ONLY THE TRAINING-TIME SOURCE SNAPSHOT MAY BE ABSENT)
def read_checkpoint(model_path: Path) -> tuple[dict, Path]:
    model_path = Path(model_path).resolve()
    path = model_path.with_name(model_path.stem + "_training_state.json")

    if not path.is_file():
        raise ValueError(f"checkpoint has no committed training state: {path}")

    state = json.loads(path.read_text(encoding="utf-8"))

    if state.get("checkpoint_schema") != 1 or state.get("checkpoint") != model_path.stem:
        raise ValueError(f"invalid checkpoint state: {path}")

    required = {model_path.name, model_path.stem + "_vecnormalize.pkl", model_path.stem + "_rng.pkl"}
    hashes = state.get("artifact_sha256", {})

    if not required.issubset(hashes):
        raise ValueError(f"checkpoint bundle is missing artifact hashes: {path}")

    for name, digest in hashes.items():
        artifact = model_path.parent / name

        if not artifact.resolve().is_relative_to(model_path.parent):
            raise ValueError(f"checkpoint artifact escapes run directory: {name}")
        if name == SNAPSHOT_NAME and not artifact.is_file():
            continue
        if not artifact.is_file() or sha256(artifact) != digest:
            raise ValueError(f"missing or modified checkpoint artifact: {artifact}")

    return state, path


#RESOLVE "latest" (MOST DECISIONS, THEN NEWEST) OR AN EXPLICIT INTACT CHECKPOINT
def select_checkpoint(run_dir: Path, selector: str) -> tuple[Path, dict]:
    if selector != "latest":
        path = Path(selector)
        if path.parent == Path("."):
            path = run_dir / path
        if path.suffix != ".zip":
            path = path.with_suffix(".zip")
        path = path.resolve()

        if path.parent != run_dir.resolve():
            raise ValueError("checkpoint must belong to the selected condition/seed run directory")

        return path, read_checkpoint(path)[0]
    candidates = []

    for state_path in run_dir.glob("*_training_state.json"):
        model_path = state_path.with_name(state_path.name.removesuffix("_training_state.json") + ".zip")
        try:
            state, _ = read_checkpoint(model_path)
            candidates.append((int(state["total_decisions"]), state_path.stat().st_mtime_ns, model_path, state))
        except (ValueError, OSError, KeyError, json.JSONDecodeError):
            continue

    if not candidates:
        raise ValueError(f"no intact resumable checkpoint in {run_dir}")

    _, _, path, state = max(candidates, key=lambda item: item[:2])

    return path, state


#RESTORE PYTHON/NUMPY/TORCH/ACTION-SPACE RANDOM STREAMS FROM A BUNDLE
def restore_rng(run_dir: Path, name: str, model) -> None:
    with (run_dir / f"{name}_rng.pkl").open("rb") as stream:
        rng = pickle.load(stream)
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    if "action_space" in rng:
        model.action_space.np_random.bit_generator.state = rng["action_space"]


#MANIFEST KEYS THAT MUST MATCH ON RESUME (OUTPUT PATHS & OPTIONAL CEILING MAY DIFFER)
RESUME_MATCH_KEYS = (
    "schema_version", "condition", "run_seed", "stage_env_seeds", "reward_profile",
    "masked_slices", "use_lidar", "timestep_scale", "n_envs", "hyperparameters",
    "curriculum", "training_source_sha256", "packages", "curriculum_variant",
    "resolved_stage_settings", "reward_shaped", "obstacle_sensing", "blending_enabled",
    "blending_settings", "stage_gate_enabled", "stage_gate_settings", "decision_budget_mode",
    "reference_stage_steps", "reference_requested_stage_steps",
    "teacher", "imitation_enabled",
)


#REJECT RESUME ON CHANGED SETTINGS/SOURCES, BROKEN COUNTERS OR TAMPERED LOG PREFIXES
def validate_resume(state: dict, expected: dict, log_path: Path, run_dir: Path | None = None,
                    resume_stage: int | None = None) -> None:
    saved = state["training_manifest"]
    mismatches = [key for key in RESUME_MATCH_KEYS if saved.get(key) != expected.get(key)]

    if mismatches:
        raise ValueError(f"checkpoint cannot resume with changed training settings/sources: {mismatches}")

    if run_dir is not None:
        verdict_by_stage = {item["stage"]: item for item in state["stage_verdicts"]}
        checkpoints = ["initial", *(verdict_by_stage.get(item["stage"], {}).get("checkpoint") or f"stage_{item['stage']}"
                                    for item in state["stage_times"]
                                    if item["stage"] < state["stage_index"] and item.get("gate_state") != BYPASSED)]
        checkpoints.extend(item["checkpoint"] for item in state["stage_verdicts"]
                           if item.get("state") == BYPASSED and item.get("checkpoint"))

        for checkpoint in checkpoints:
            read_checkpoint(run_dir / (checkpoint + ".zip"))

    if not expected.get("stage_gate_enabled"):
        raise ValueError("checkpoint continuation is supported for success-gated presets")

    stage = int(state["stage_index"])
    decisions = int(state["stage_decisions"])
    scheduler = state["scheduler"]

    if resume_stage is not None and not stage < resume_stage < len(expected["curriculum"]):
        raise ValueError("--resume-stage must select a later stage within the curriculum")
    if (not 0 <= stage < len(expected["curriculum"]) or decisions < 0
            or scheduler["stage_index"] != stage
            or scheduler["assistance_enabled"] != bool(expected["blending_enabled"])
            or int(state["total_decisions"]) != decisions + sum(
                int(item["stage_decisions"]) for item in state["stage_times"] if item["stage"] < stage)):
        raise ValueError("inconsistent checkpoint curriculum/counters")
    if (scheduler["state"] not in {"ASSISTED", "CONSOLIDATING", "GATING", "PASSED", "STALLED"}
            or not 0 <= scheduler["beta"] <= saved["stage_gate_settings"]["beta_cap"][stage]
            or (not scheduler["assistance_enabled"] and scheduler["beta"] != 0.0)
            or (scheduler["state"] == "PASSED" and scheduler["beta"] != 0.0)):
        raise ValueError("inconsistent checkpoint assistance state")

    ceiling = expected.get("stage_ceiling")

    if resume_stage is None and scheduler["state"] != "PASSED" and ceiling is not None and ceiling <= decisions:
        raise ValueError("resume ceiling must exceed decisions already spent in this stage; lift/extend it")

    expected_logs = {log_path.name, log_path.stem + "_episodes.csv"}

    if len(state["log_prefixes"]) != 2 or {item["name"] for item in state["log_prefixes"]} != expected_logs:
        raise ValueError("checkpoint must identify both training log prefixes")

    for prefix in state["log_prefixes"]:
        path = log_path.parent / prefix["name"]

        if path.name not in {log_path.name, log_path.stem + "_episodes.csv"}:
            raise ValueError("checkpoint contains an unexpected training log")
        if sha256(path, prefix["bytes"]) != prefix["sha256"]:
            raise ValueError(f"training log prefix does not match checkpoint: {path}")


#MANUAL --resume-stage: RECORD ACTUAL COST & BYPASSED VERDICTS FOR SKIPPED STAGES
def advance_checkpoint_stage(state: dict, target: int, reason: str | None = None) -> dict:
    """Record actual work and honest verdicts for every manually skipped stage."""
    source = int(state["stage_index"])
    scheduler = state["scheduler"]
    manifest = state["training_manifest"]
    times = [item for item in state["stage_times"] if item["stage"] < source]
    verdicts = [item for item in state["stage_verdicts"] if item["stage"] < source]
    bypassed = []
    explanation = reason.strip() if reason and reason.strip() else "Manual curriculum advancement requested with --resume-stage"

    for stage in range(source, target):
        trained = stage == source
        passed = trained and scheduler["state"] == "PASSED"
        status = "PASSED" if passed else BYPASSED
        decisions = int(state["stage_decisions"]) if trained else 0
        if not passed:
            bypassed.append(stage)
        verdicts.append({
            **(scheduler if trained else {}), "stage": stage, "state": status,
            "decisions": decisions, "checks": scheduler["checks"] if trained else 0,
            "final_beta": scheduler["beta"] if trained else None,
            "gate_passed": bool(passed), "trained": bool(trained),
            "state_before_bypass": scheduler["state"] if trained and not passed else None,
            "checkpoint": state["checkpoint"] if trained else None,
            "history": list(scheduler["history"]) if trained else [],
            "manual_reason": explanation if not passed else None,
        })
        times.append({
            "stage": stage, "planned_steps": None,
            "reference_steps": manifest["reference_stage_steps"][stage],
            "stage_decisions": decisions, "gate_state": status,
            "timesteps_after": int(state["total_decisions"]), "decisions_after": int(state["total_decisions"]),
            "physics_ticks_after": int(state["physics_ticks"]),
            "stage_physics_ticks": int(state["stage_physics_ticks"]) if trained else 0,
            "env_seed": manifest["stage_env_seeds"][stage]["base"],
            "wall_time_s": round(state["stage_wall_time_s"], 1) if trained else 0.0,
            "trained": bool(trained), "manual_reason": explanation if not passed else None,
        })
    advance = {"from_stage": source, "to_stage": target, "checkpoint": state["checkpoint"],
               "total_decisions": int(state["total_decisions"]), "physics_ticks": int(state["physics_ticks"]),
               "source_gate_state": scheduler["state"], "source_beta": scheduler["beta"],
               "bypassed_stages": bypassed, "reason": explanation,
               "archived_run": state["resume_ancestry"][-1]["archived_run"]}

    return {**state, "stage_times": times, "stage_verdicts": verdicts,
            "manual_stage_advances": [*manifest.get("manual_stage_advances", []), advance]}


#CHECK EVERY STAGE IS PASSED OR EXPLICITLY BYPASSED; RETURN THE BYPASSED STAGES
def validate_stage_completion(verdicts: list[dict], advances: list[dict], num_stages: int) -> list[int]:
    """Accept documented manual bypasses without asserting that their gates passed."""
    covered = set()

    for advance in advances:
        source, target = advance["from_stage"], advance["to_stage"]

        if not 0 <= source < target < num_stages or not advance.get("reason"):
            raise ValueError("invalid manual curriculum advancement record")

        expected = set(range(source + (advance["source_gate_state"] == "PASSED"), target))

        if set(advance["bypassed_stages"]) != expected:
            raise ValueError("manual advancement does not account for every bypassed gate")

        covered.update(expected)

    if [item.get("stage") for item in verdicts] != list(range(num_stages)):
        raise ValueError("curriculum verdicts must account for every stage in order")

    bypassed = []

    for item in verdicts:
        if item["state"] == BYPASSED:
            if item["stage"] not in covered or item.get("gate_passed") is not False:
                raise ValueError("bypassed stage lacks an explicit manual advancement record")

            bypassed.append(item["stage"])
        elif item["state"] != "PASSED" or item.get("final_beta") != 0.0:
            raise ValueError("non-bypassed stage has not passed its unassisted gate")

    if covered != set(bypassed):
        raise ValueError("manual advancement records disagree with curriculum verdicts")

    return bypassed
