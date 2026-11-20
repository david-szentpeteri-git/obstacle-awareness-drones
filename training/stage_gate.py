# -*- coding: utf-8 -*-
#NEWLY ADDED FOR CS5478 PROJECT
"""
Current-blend validation batches; only beta-zero batches can pass a stage.

A gate is only meaningful if what it measures is the thing being claimed. Three
properties are enforced here rather than assumed:

  - CURRENT BLEND (revision 2). The batch runs the deterministic student action
    blended with the shared expert at the beta the student is training under.
    Judging student-only at beta 0.8 would never let beta move - the student
    alone scores near zero there. At beta 0 the blend is a strict pass-through,
    so the same batch IS the student-only gate that decides advancement.
  - FROZEN STATISTICS. The validation env copies the training observation
    statistics and runs with `training=False`, so a batch cannot move the
    normalizer it is being judged through.
  - SEEDS THAT ARE NEITHER TRAINED ON NOR TESTED ON. Validation seeds come from
    their own band (`CURRICULUM_VAL_SEED_*`), disjoint from training seeds and
    from the final held-out band, so passing a gate neither memorizes training
    worlds nor burns final test worlds.

Success is read from the env's `reached` info flag. That specific name matters:
an info key that is never set reads as a permanent 0% success rate, which would
silently stall every stage forever instead of failing loudly.
"""
from __future__ import annotations

from typing import List

import numpy as np

from config.blending import VALIDATION_DETERMINISTIC, VALIDATION_EPISODES
from config.experiment import Condition
from config.ppo_defaults import (
    CURRICULUM_VAL_SEED_BASE,
    CURRICULUM_VAL_SEED_HI,
    CURRICULUM_VAL_SEED_LO,
)
from training.env_factory import make_vec_env, safe_close
from training.teacher import make_teacher


#DETERMINISTIC VALIDATION SEEDS FOR ONE (RUN SEED, STAGE), OWN BAND
def validation_seeds(run_seed: int, stage_idx: int, n_eps: int) -> List[int]:
    if n_eps <= 0:
        raise ValueError("validation episode count must be positive")

    rng = np.random.default_rng(CURRICULUM_VAL_SEED_BASE + int(run_seed) * 1_000 + int(stage_idx))
    span = CURRICULUM_VAL_SEED_HI - CURRICULUM_VAL_SEED_LO

    return [int(s) + CURRICULUM_VAL_SEED_LO for s in rng.choice(span, size=n_eps, replace=False)]


#RUN ONE FROZEN CURRENT-BLEND BATCH (STUDENT-ONLY AT BETA ZERO)
def run_validation_batch(
    model,
    train_env,
    condition: Condition,
    cfg: dict,
    cfg_preset: dict,
    stage_idx: int,
    run_seed: int,
    stage_env_settings: dict,
    episodes: int = VALIDATION_EPISODES,
    beta: float = 0.0,
) -> dict:
    seeds = validation_seeds(run_seed, stage_idx, episodes)
    beta = float(beta)
    env = make_vec_env(
        condition=condition,
        world_scale=cfg["world_scale"],
        num_drones=cfg["num_drones"],
        num_static_obs=cfg["num_static_obs"],
        num_dynamic_obs=cfg["num_dynamic_obs"],
        episode_horizon=cfg["episode_horizon"],
        stage_lidar=cfg["stage_lidar"],
        num_envs=1,
        seed=seeds[0],
        action_repeat=cfg_preset["action_repeat"],
        blend_expert=make_teacher(cfg["world_scale"], kind=cfg_preset.get("teacher", "expert")) if beta > 0.0 else None,
        blend_beta=beta,
        **stage_env_settings,
    )

    try:
        #JUDGE THE POLICY THROUGH THE SAME INPUT SCALE IT TRAINS WITH, FROZEN
        _copy_norm_stats(train_env, env)
        reached = collided = timeout = 0
        obs = env.reset()

        #THE ENV IS SEEDED INTO THE VALIDATION BAND AT CONSTRUCTION AND THEN
        #AUTO-RESETS, SO ITS WORLD STREAM STAYS INSIDE THAT BAND - THE SAME WAY
        #A TRAINING STAGE DRAWS ITS WORLDS FROM ONE STAGE SEED. RESEEDING A
        #MID-CHAIN ENV PER EPISODE WOULD BYPASS THE OUTER WRAPPERS INSTEAD.
        for _ in range(int(episodes) * (cfg["episode_horizon"] + 10)):
            action, _ = model.predict(obs, deterministic=VALIDATION_DETERMINISTIC)
            obs, _, dones, infos = env.step(action)

            if not bool(np.asarray(dones).any()):
                continue

            info = infos[0] if isinstance(infos, (list, tuple)) else infos

            if info.get("reached"):
                reached += 1
            elif info.get("truncated") or info.get("TimeLimit.truncated"):
                timeout += 1
            else:
                collided += 1

            if reached + collided + timeout >= int(episodes):
                break

        total = max(reached + collided + timeout, 1)

        return {
            "episodes": reached + collided + timeout,
            "requested_episodes": episodes,
            "success_rate": reached / total,
            "collision_rate": collided / total,
            "timeout_rate": timeout / total,
            "deterministic": bool(VALIDATION_DETERMINISTIC),
            "validation_beta": beta,
            "student_only": beta <= 0.0,
            "seed_band": [CURRICULUM_VAL_SEED_LO, CURRICULUM_VAL_SEED_HI],
            "env_seed": int(seeds[0]),
        }
    finally:
        safe_close(env)


#CARRY TRAINING OBSERVATION STATISTICS INTO A FROZEN VALIDATION ENV
def _copy_norm_stats(train_env, val_env) -> None:
    source = _find_vec_normalize(train_env)
    target = _find_vec_normalize(val_env)

    if source is None or target is None:
        return

    target.obs_rms = source.obs_rms
    target.ret_rms = source.ret_rms
    target.training = False
    target.norm_reward = False


#WALK A VEC WRAPPER CHAIN FOR THE VecNormalize LAYER
def _find_vec_normalize(env):
    node = env

    for _ in range(12):
        if node is None:
            return None
        if type(node).__name__ == "VecNormalize":
            return node

        node = getattr(node, "venv", None)

    return None
