# -*- coding: utf-8 -*-
#PREEXISTS, MODIFIED FOR CS5478 PROJECT (REWRITTEN: PROFILE-DRIVEN, SAME compute_rewards ENTRY POINT)
"""
Reward computation, driven by the RewardProfile carried on the environment
(see config/experiment.py). Every shaping term is weighted by the profile, so
the degenerate and full conditions run the exact same code path and differ
only in which coefficients are non-zero.
"""
from __future__ import annotations
from typing import Dict
import numpy as np

from drone_env.defaults import (
    POS_SLICE,
    FINAL_MILE_R,
    COHESION_PEN,
    SEPARATION_PEN,
)
from config.experiment import FULL_REWARD
from training.wrappers.lidar_layout import FRONT_RAYS, LEFT_RAYS, RIGHT_RAYS

#LIDAR RAY GROUPS USED FOR PROXIMITY SHAPING. RAYS ARE GOAL-RELATIVE, SO "FRONT"
#= TOWARD THE GOAL (FRONT + FRONT_UP + FRONT_DOWN); SIDES = ONE RAY EACH
FRONT_IDX = FRONT_RAYS
LEFT_IDX = LEFT_RAYS
RIGHT_IDX = RIGHT_RAYS

#DISTANCES (NORMALIZED LIDAR RETURNS) AT WHICH PROXIMITY SHAPING KICKS IN
FRONT_TRIGGER = 0.35
SIDE_TRIGGER = 0.25


#CLOSEST NORMALIZED LIDAR RETURN IN FRONT / LEFT / RIGHT RAY GROUPS
def _lidar_proximity(scan: np.ndarray) -> tuple[float, float, float]:
    scan = np.asarray(scan, dtype=np.float32).reshape(-1)
    front_min = float(np.min(scan[FRONT_IDX]))
    left_min = float(np.min(scan[LEFT_IDX]))
    right_min = float(np.min(scan[RIGHT_IDX]))

    return front_min, left_min, right_min


#PENALTY FOR CROWDING AN OBSTACLE (FRONT CHECKED FIRST, THEN SIDES)
def _proximity_pen(scan: np.ndarray, profile) -> float:
    front_min, left_min, right_min = _lidar_proximity(scan)

    if front_min < FRONT_TRIGGER:
        return -profile.near_front_obs_pen * (FRONT_TRIGGER - front_min) / FRONT_TRIGGER

    side_min = min(left_min, right_min)

    if side_min < SIDE_TRIGGER:
        return -profile.near_side_obs_pen * (SIDE_TRIGGER - side_min) / SIDE_TRIGGER

    return 0.0


#PER-AGENT REWARDS FOR ONE STEP, WEIGHTED BY ENV'S REWARD PROFILE
def compute_rewards(
    env, obs_dict, sim_actions: np.ndarray, step_state
) -> Dict[str, float]:

    profile = getattr(env, "reward_profile", FULL_REWARD)
    reward_dict: Dict[str, float] = {}

    #SWARM PENALTIES ARRIVE PRE-SCALED BY THE DEFAULTS CONSTANTS -> RESCALE TO PROFILE
    cohesion_scale = profile.cohesion_pen / COHESION_PEN if COHESION_PEN > 0 else 0.0
    sep_scale = profile.separation_pen / SEPARATION_PEN if SEPARATION_PEN > 0 else 0.0

    #PROXIMITY SHAPING NEEDS REAL RAYS + A NON-ZERO COEF
    use_proximity = bool(env.lidar) and (
        profile.near_front_obs_pen > 0.0 or profile.near_side_obs_pen > 0.0
    )

    for i, agent in enumerate(env.possible_agents):
        if i >= env.total_num_drones:
            reward_dict[agent] = 0.0
            continue

        #TERMINAL ONE-TIME OUTCOMES

        #COLLISIONS ALWAYS END THE EPISODE & ARE ALWAYS RECORDED FOR METRICS,
        #BUT ONLY FEED BACK AS REWARD WHEN THE PROFILE PENALIZES THEM
        if step_state.wall_collisions[i]:
            reward_dict[agent] = float(-profile.wall_collision_pen)
            continue
        if step_state.drone_collisions[i]:
            reward_dict[agent] = float(-profile.drone_collision_pen)
            continue
        if step_state.obstacle_collisions[i]:
            reward_dict[agent] = float(-profile.obstacle_collision_pen)
            continue

        if step_state.reached_now[i]:
            reward_dict[agent] = float(profile.goal_bonus)
            continue

        if env._dead[i] or env._reached_once[i] or env._despawned[i]:
            reward_dict[agent] = 0.0
            continue

        #GOAL GEOMETRY

        pos = obs_dict[agent][POS_SLICE]
        target = (
            env._targets[i].start_pos
            if env._targets is not None
            else np.zeros(3, dtype=np.float32)
        )

        to_target_vec = target - pos
        dist_to_target = float(np.linalg.norm(to_target_vec))
        to_target_norm = np.linalg.norm(to_target_vec)

        prev_dist_to_target = float(env._prev_dists[i])
        env._prev_dists[i] = dist_to_target

        if to_target_norm > 1e-6:
            target_dir = to_target_vec / to_target_norm
            alignment = float(np.dot(sim_actions[i][0:3], target_dir))
        else:
            target_dir = np.zeros(3, dtype=np.float32)
            alignment = 0.0

        vel = np.array(env._env._getDroneStateVector(i)[10:13], dtype=np.float32)
        speed = float(np.linalg.norm(vel))

        #PROGRESS, DISTANCE, TIME & ACTION COST

        if env._step_count == 0:
            r_progress = 0.0
        else:
            r_progress = profile.prog_coef * (prev_dist_to_target - dist_to_target)

        r_distance_pen = -profile.dist_pen * dist_to_target
        r_time = -profile.time_pen
        r_ctrl = -profile.ctrl_pen * float(np.linalg.norm(sim_actions[i]))

        #ALIGNMENT REWARD

        r_align = profile.align_coef * alignment if alignment > 0 else 0.0

        #FINAL-MILE SHAPING

        #SMALL BONUS GROWING AS DRONE CLOSES THE LAST METERS ON GOAL
        if dist_to_target < FINAL_MILE_R:
            r_final = profile.final_mile_coef * (
                1.0 - (dist_to_target / FINAL_MILE_R) ** 2
            )
        else:
            r_final = 0.0

        #TANGENTIAL VELOCITY PENALTY NEAR GOAL

        if dist_to_target < FINAL_MILE_R and to_target_norm > 1e-6 and speed > 1e-6:
            vel_parallel = np.dot(vel, target_dir) * target_dir
            tangential_speed = float(np.linalg.norm(vel - vel_parallel))
            r_tan = -profile.tangential_pen * tangential_speed
        else:
            r_tan = 0.0

        #OBSTACLE PROXIMITY SHAPING

        #OBSTACLE-ONLY SCAN, NOT THE OBSERVATION'S LIDAR SLICE: THE ARENA
        #WALLS ARE RAY-VISIBLE, AND THE TRIGGER RADIUS (~2.1 M) COVERS THE
        #WHOLE OF AN EARLY-CURRICULUM 4 M BOX, SO READING THE RAW SCAN
        #CHARGED THIS PENALTY EVERYWHERE AND MADE CRASHING THE CHEAPEST WAY
        #TO STOP LOSING REWARD. THE POLICY STILL SEES THE FULL SCAN.
        if use_proximity:
            r_near = _proximity_pen(env.obstacle_lidar_scan(i), profile)
        else:
            r_near = 0.0

        #COMBINE SHAPING TERMS

        r = (
            r_progress
            + r_distance_pen
            + r_time
            + r_ctrl
            + r_align
            + r_final
            + r_tan
            + r_near
            + cohesion_scale * step_state.cohesion_pen
            + sep_scale * step_state.sep_pen
        )

        #EXTRA PENALTIES

        if step_state.time_limit:
            r -= profile.trunc_pen

        reward_dict[agent] = float(r)

    return reward_dict
