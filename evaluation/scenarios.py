# -*- coding: utf-8 -*-
#NEWLY ADDED FOR CS5478 PROJECT
"""
Evaluation scenarios: held-out worlds drawn from the training curriculum's own
generator (one scenario per curriculum stage, unseen seeds), plus the episode
reset shared with the wall-gap suite (evaluation/wall_gap.py).
"""
from __future__ import annotations

from copy import deepcopy
from typing import Dict

import numpy as np

from drone_env.defaults import POS_SLICE
from drone_env.curriculum import Curriculum


#BUILD ONE RANDOM EVAL SCENARIO PER CURRICULUM STAGE, DERIVED FROM THE
#CURRICULUM ITSELF SO THE EVAL DISTRIBUTION CANNOT DRIFT FROM TRAINING
def training_matched_scenarios(preset: dict) -> Dict[str, dict]:
    """
    Held-out worlds drawn from the SAME generator and the SAME stage settings
    the policy trained on - only the seeds differ (evaluation seed band).

    Every distribution-bearing field comes from the curriculum/preset rather
    than a hand-copied literal: world scale, static/dynamic counts and episode
    horizon from the stage tuple, and obstacle layout, blocker probability,
    size/speed/travel ranges and the obstacle clock from `stage_settings`.
    A hand-edited copy would be free to disagree with training; this cannot.
    """
    stages = Curriculum(preset["curriculum"]).stages
    stage_settings = preset.get("stage_settings")

    if not stage_settings or len(stage_settings) != len(stages):
        raise ValueError("training-matched evaluation needs one stage_settings entry per curriculum stage")

    scale_override = preset.get("eval_world_scale")
    scenarios: Dict[str, dict] = {}

    for idx, stage in enumerate(stages):
        world_scale, _, num_static, num_dynamic, _, horizon, stage_lidar = stage
        env_cfg = dict(
            world_scale=float(scale_override if scale_override is not None else world_scale),
            num_static_obs=int(num_static),
            num_dynamic_obs=int(num_dynamic),
            episode_horizon=int(horizon),
        )
        env_cfg.update(deepcopy(stage_settings[idx]))

        if "evaluation_obstacle_dt" in preset:
            env_cfg["obstacle_dt"] = preset["evaluation_obstacle_dt"]

        scenarios[f"stage{idx}_{int(num_static)}s{int(num_dynamic)}d"] = {
            "kind": "random",
            "stage_index": idx,
            "stage_lidar": bool(stage_lidar),
            "env_cfg": env_cfg,
        }

    return scenarios


#SCENARIO SUITE OF A PRESET (WHAT "--scenarios all" RESOLVES TO)
def resolve_scenarios(preset: dict) -> Dict[str, dict]:
    if preset.get("eval_suite") != "training_matched":
        raise ValueError("evaluation needs a preset whose eval_suite is 'training_matched'")

    return training_matched_scenarios(preset)


#TELEPORT THE (SINGLE) DRONE + ITS TARGET TO SAMPLED SPAWN/GOAL POINTS
def place_drone_and_target(env, spawn_pos: np.ndarray, goal_pos: np.ndarray) -> None:
    p = env._p
    did = env._drone_ids[0]

    env._call_physics(
        p.resetBasePositionAndOrientation, did, spawn_pos.tolist(), [0.0, 0.0, 0.0, 1.0]
    )
    env._call_physics(p.resetBaseVelocity, did, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    env.spawn_positions[0] = spawn_pos

    target = env._targets[0]
    target.start_pos = goal_pos
    body_id = getattr(target, "body_id", None)

    if body_id is not None:
        env._call_physics(
            p.resetBasePositionAndOrientation, body_id, goal_pos.tolist(), [0.0, 0.0, 0.0, 1.0]
        )

    #REFRESH AVIARY'S KINEMATIC CACHE SO THE FIRST OBS SEES THE TELEPORT
    refresh = getattr(env._env, "_updateAndStoreKinematicInformation", None)
    if refresh is not None:
        refresh()

    pos = env._env._getDroneStateVector(0)[POS_SLICE]
    env._prev_dists[0] = float(np.linalg.norm(target.start_pos - np.array(pos)))


#RESET AN ENV INTO ONE SCENARIO EPISODE (RANDOM OR WALL-GAP), RETURN FIRST OBS
def reset_scenario_episode(env, scenario: dict, seed: int):
    obs_dict, _ = env.reset(seed=seed)
    env.eval_layout = None

    if scenario["kind"] == "wall_gap":
        from evaluation.wall_gap import apply_wall_layout, layout_record, sample_wall_layout

        layout = sample_wall_layout(scenario["env_cfg"]["world_scale"], scenario["coverage"], np.random.default_rng(seed))
        apply_wall_layout(env, layout)
        place_drone_and_target(env, layout["spawn"], layout["goal"])
        env.eval_layout = layout_record(layout)
        #A GAP-AWARE REFERENCE CONTROLLER READS THE LAYOUT HERE, AS IN TRAINING
        env.current_wall_layout = layout
        obs_dict = env._build_obs_dict()

    return obs_dict
