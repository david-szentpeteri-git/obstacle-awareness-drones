# -*- coding: utf-8 -*-
#NEWLY ADDED FOR CS5478 PROJECT
"""
Warm start (phase 2, 2026-10-15): a fresh run begins from the policy weights
and the observation/return normalisation statistics of a COMPLETE earlier run
(by default the compact final models), read-only. Optimizer state, decision
counters, curriculum and gate state start fresh; the source run's folder is
never written to. The manifest records what was loaded and its hashes.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

from stable_baselines3.common.save_util import load_from_zip_file

from training.checkpoint_io import sha256
from training.env_factory import unwrap_vec_normalize


#LOAD POLICY WEIGHTS + NORMALISATION FROM <source_dir>/ppo_<condition>_s<seed>/final.* INTO A FRESH MODEL
def load_warm_start(model, env, source_dir, condition_name: str, seed: int) -> dict:
    run_dir = Path(source_dir) / f"ppo_{condition_name}_s{seed}"
    manifest_path = run_dir / "run_manifest.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(f"warm-start source has no manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("status") != "complete":
        raise ValueError(f"warm-start source run is not complete ({manifest.get('status')}): {run_dir}")

    model_path = run_dir / "final.zip"
    norm_path = run_dir / "final_vecnormalize.pkl"
    hashes = manifest.get("artifact_sha256", {})
    digests = {}

    for path in (model_path, norm_path):
        digests[path.name] = sha256(path)

        if hashes.get(path.name) != digests[path.name]:
            raise ValueError(f"warm-start artifact missing or modified: {path}")

    data, params, _variables = load_from_zip_file(str(model_path), device=model.device)

    if data["observation_space"] != model.observation_space or data["action_space"] != model.action_space:
        raise ValueError("warm-start source spaces differ from the new run's spaces")

    model.policy.load_state_dict(params["policy"])
    vec_norm = unwrap_vec_normalize(env)

    with norm_path.open("rb") as stream:
        saved = pickle.load(stream)

    vec_norm.obs_rms = saved.obs_rms
    vec_norm.ret_rms = saved.ret_rms

    return {
        "source_run": str(run_dir.resolve()),
        "source_preset": manifest.get("preset"),
        "source_status": manifest.get("status"),
        "source_total_decisions": manifest.get("total_decisions"),
        "model": model_path.name,
        "model_sha256": digests[model_path.name],
        "normalization": norm_path.name,
        "normalization_sha256": digests[norm_path.name],
        "loaded": "policy weights + observation/return normalisation statistics; optimizer, counters and curriculum state fresh",
    }
