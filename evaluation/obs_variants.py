# -*- coding: utf-8 -*-
#NEWLY ADDED FOR CS5478 PROJECT
"""
Observation variants fed to a trained model in place of its normal input
(evaluation only; training sources untouched). Every variant edits the RAW
24-D observation before the condition mask, so a blind model's input - and
therefore its action - cannot change.

  v1       rays see obstacles only: a ray whose first hit is the arena boundary
           (wall body or floor plane) reads 1.0 ("nothing"); other rays unchanged
  v2       rays see the boundary only: a ray whose first hit is an obstacle reads
           the analytic distance to the boundary along the same ray; the
           nearest-obstacle vector becomes the "no obstacle" placeholder
           (out of the trained input range - see v2b)
  v2b      as v2, but the nearest-obstacle vector becomes an in-range stand-in,
           a 1.2 m obstacle directly behind the drone (inside the stage-6 range
           of that input), so every input stays within what training produced
  behind2  nearest-obstacle vector = a 2 m obstacle directly behind the drone
           (opposite the horizontal goal bearing); rays unchanged
  scaled   nearest-obstacle vector = the placeholder direction at an in-range
           magnitude, [3, 0, 0]; rays unchanged

Raw observation layout: pos 0:3, vel 3:6, to_goal 6:9, to_obstacle 9:12,
to_centroid 12:15, to_teammate 15:18, rays 18:24 (fraction of the 6 m range).
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pybullet as p

from drone_env.defaults import LIDAR_RANGE
from evaluation.wall_gap import WALL_FACE
from training.wrappers.lidar_layout import goal_heading, ray_directions

PROBE_NAMES = ("v1", "v2", "v2b", "behind2", "scaled")
GOAL_SLICE = slice(6, 9)
OBS_VEC_SLICE = slice(9, 12)
RAY_SLICE = slice(18, 24)
#RAYS START THIS FAR ALONG THEIR DIRECTION FROM THE DRONE CENTRE (AS IN env_3d._lidar_scan)
RAY_START_OFFSET = 0.15
#"NO OBSTACLE" PLACEHOLDER OF THE NEAREST-OBSTACLE VECTOR (env_3d._nearest_obstacle_vector)
NO_OBSTACLE_PLACEHOLDER = np.array([LIDAR_RANGE * 2.0, 0.0, 0.0], dtype=np.float32)
BEHIND_DISTANCE = 2.0
#IN-RANGE STAND-IN FOR v2b (STAGE-6 NEAREST-OBSTACLE DISTANCES: MEDIAN 0.8 m, 95TH PERCENTILE 1.2-1.3 m)
BEHIND_DISTANCE_IN_RANGE = 1.2
SCALED_PLACEHOLDER = np.array([3.0, 0.0, 0.0], dtype=np.float32)
RAY_MATCH_TOL = 1e-4


#DRONE POSITION, GOAL, RAY DIRECTIONS & RAY STARTS EXACTLY AS THE ENV CASTS THEM
def ray_geometry(env):
    base_pos, _ = env._call_physics(p.getBasePositionAndOrientation, env._drone_ids[0])
    origin = np.array(base_pos, dtype=np.float32)
    goal = np.asarray(env._targets[0].start_pos, dtype=np.float32)
    dirs = ray_directions(goal - origin).astype(np.float32)
    ray_from = origin[None, :] + dirs * RAY_START_OFFSET

    return origin, goal, dirs, ray_from


#CAST THE SIX RAYS: HIT FRACTION (1 = NOTHING) & HIT BODY ID (-1 = NONE) PER RAY
def cast_rays(env, ray_from: np.ndarray, dirs: np.ndarray):
    ray_to = ray_from + dirs * LIDAR_RANGE
    results = env._call_physics(p.rayTestBatch, rayFromPositions=ray_from.tolist(), rayToPositions=ray_to.tolist())
    did = env._drone_ids[0]
    frac = np.ones(len(dirs), dtype=np.float32)
    body = np.full(len(dirs), -1, dtype=np.int64)

    for i, hit in enumerate(results):
        if int(hit[0]) >= 0 and int(hit[0]) != did:
            frac[i] = float(np.clip(hit[2], 0.0, 1.0))
            body[i] = int(hit[0])

    return frac, body


#BODY IDS OF OBSTACLES & OF THE ARENA BOUNDARY (WALL BOXES + FLOOR PLANE)
def body_classes(env):
    obstacles = {o.body_id for o in env._obstacles if getattr(o, "body_id", None) is not None}
    boundary = set(int(w) for w in env._wall_ids)
    plane = getattr(env._env, "PLANE_ID", None)

    if plane is not None:
        boundary.add(int(plane))

    return obstacles, boundary


#ANALYTIC RAY READING AGAINST THE BOUNDARY ALONE (INNER WALL FACES, FLOOR PLANE z=0, CEILING FACE)
def boundary_frac(world_scale: float, ray_from: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    face = float(world_scale) - WALL_FACE
    lo = np.array([-face, -face, 0.0])
    hi = np.array([face, face, face])
    out = np.ones(len(dirs), dtype=np.float32)

    for i in range(len(dirs)):
        t_exit = np.inf

        for k in range(3):
            d = float(dirs[i, k])

            if abs(d) < 1e-9:
                continue

            bound = hi[k] if d > 0 else lo[k]
            t = (bound - float(ray_from[i, k])) / d

            if t >= 0:
                t_exit = min(t_exit, t)

        out[i] = float(np.clip(t_exit / LIDAR_RANGE, 0.0, 1.0))

    return out


#REAL, OBSTACLE-ONLY (v1) & BOUNDARY-ONLY (v2) RAY READINGS FOR THE CURRENT STATE
def probe_rays(env):
    origin, goal, dirs, ray_from = ray_geometry(env)
    real, body = cast_rays(env, ray_from, dirs)
    obstacles, boundary = body_classes(env)
    hit_boundary = np.isin(body, list(boundary))
    hit_obstacle = np.isin(body, list(obstacles))
    v1 = np.where(hit_boundary, 1.0, real).astype(np.float32)
    v2 = np.where(hit_obstacle, boundary_frac(env.world_scale, ray_from, dirs), real).astype(np.float32)

    return dict(origin=origin, goal=goal, dirs=dirs, real=real, body=body, v1=v1, v2=v2,
                hit_boundary=hit_boundary, hit_obstacle=hit_obstacle)


#VARIANT OF ONE RAW OBSERVATION ROW FOR THE ENV'S CURRENT STATE
def variant_obs(env, obs_row: np.ndarray, variant: str) -> np.ndarray:
    obs = np.array(obs_row, dtype=np.float32, copy=True)

    if variant in ("v1", "v2", "v2b"):
        rays = probe_rays(env)
        mismatch = float(np.max(np.abs(rays["real"] - obs[RAY_SLICE])))

        if mismatch > RAY_MATCH_TOL:
            raise RuntimeError(f"re-cast rays differ from the observation's rays by {mismatch:.2e}")

        obs[RAY_SLICE] = rays["v2" if variant == "v2b" else variant]

        if variant == "v2":
            obs[OBS_VEC_SLICE] = NO_OBSTACLE_PLACEHOLDER
        elif variant == "v2b":
            obs[OBS_VEC_SLICE] = (-BEHIND_DISTANCE_IN_RANGE * goal_heading(obs[GOAL_SLICE])).astype(np.float32)
    elif variant == "behind2":
        obs[OBS_VEC_SLICE] = (-BEHIND_DISTANCE * goal_heading(obs[GOAL_SLICE])).astype(np.float32)
    elif variant == "scaled":
        obs[OBS_VEC_SLICE] = SCALED_PLACEHOLDER
    else:
        raise ValueError(f"unknown observation variant '{variant}' (expected one of {PROBE_NAMES})")

    return obs


#TRANSFORM HOOK FOR evaluate.run_episode: (env, obs_batch) -> obs_batch; None = NORMAL INPUT
def make_obs_transform(variant: Optional[str]) -> Optional[Callable]:
    if not variant or variant == "none":
        return None
    if variant not in PROBE_NAMES:
        raise ValueError(f"unknown observation variant '{variant}' (expected one of {PROBE_NAMES})")

    #APPLY THE VARIANT ROW BY ROW
    def transform(env, obs_batch):
        return np.stack([variant_obs(env, row, variant) for row in np.asarray(obs_batch)]).astype(np.float32)

    return transform
