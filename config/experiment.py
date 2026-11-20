# -*- coding: utf-8 -*-
#NEWLY ADDED FOR CS5478 PROJECT
"""
The four experiment conditions compared in this project (single drone).

2x2 ablation grid - reward axis x sensing axis:

                      | sensing ON (LiDAR + obstacle obs) | sensing OFF (blind)
  reward SHAPED       | full                              | reward_only
  reward GOAL-ONLY    | sensing_only                      | degenerate

All four share one algorithm (PPO), one curriculum, one policy architecture
and one goal-seeking reward stack (progress, alignment, distance/time/control
penalties, final-mile shaping, goal bonus, truncation penalty). Only two
things differ:
  - whether the OBSTACLE reward terms are active (collision penalties +
    LiDAR proximity shaping)
  - whether the policy can SEE obstacles (LiDAR returns + the
    vector-to-nearest-obstacle observation slice)

reward_only casts LiDAR rays so the reward can use them, but masks them (and
the obstacle slices) from the policy - reward knows, policy is blind.

Observation width is kept identical across conditions (ablated slices are
zeroed, not removed), so every policy uses the same network shape and any
measured gap is attributable to information/reward, not capacity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Tuple

from drone_env.defaults import (
    PROG_COEF,
    TIME_PEN,
    DIST_PEN,
    CNTRL_PEN,
    ALIGN_COEF,
    GOAL_BONUS,
    TRUNC_PEN,
    WALL_COLLISION_PEN,
    OBSTACLE_COLLISION_PEN,
    DRONE_COLLISION_PEN,
    FINAL_MILE_COEF,
    LIDAR_RAY_LAYOUT,
    OBS_SLICE,
    CENTROID_SLICE,
    TEAMMATE_SLICE,
)

#FULL BASE OBS WIDTH + LIDAR WIDTH (POLICY INPUT IS ALWAYS THIS WIDE)
BASE_OBS_DIM = 18
#SIX GOAL-RELATIVE RAYS (SEE training/wrappers/lidar_layout.py)
LIDAR_DIM = len(LIDAR_RAY_LAYOUT)
FULL_OBS_DIM = BASE_OBS_DIM + LIDAR_DIM

#OBS REGIONS CARRYING OBSTACLE INFORMATION (ZEROED FOR BLIND CONDITIONS)
LIDAR_SLICE = slice(BASE_OBS_DIM, FULL_OBS_DIM)
OBSTACLE_AWARE_SLICES = (OBS_SLICE, CENTROID_SLICE, TEAMMATE_SLICE)
BLIND_SLICES = OBSTACLE_AWARE_SLICES + (LIDAR_SLICE,)

#PROXIMITY SHAPING COEFS - DEFAULTS.PY LEAVES THESE AT 0, SO THE SHAPED
#PROFILE SETS ITS OWN NON-ZERO VALUES HERE (TUNABLE STARTING POINTS)
SHAPED_NEAR_FRONT_PEN = 2.0
SHAPED_NEAR_SIDE_PEN = 1.0

#GOAL-SEEKING TERMS SHARED BY EVERY PROFILE (SINGLE SOURCE SO THE TWO
#PROFILES CANNOT DRIFT APART ON THE NON-ABLATED STACK)
_GOAL_STACK = dict(
    goal_bonus=GOAL_BONUS,
    trunc_pen=TRUNC_PEN,
    prog_coef=PROG_COEF,
    align_coef=ALIGN_COEF,
    dist_pen=DIST_PEN,
    time_pen=TIME_PEN,
    ctrl_pen=CNTRL_PEN,
    final_mile_coef=FINAL_MILE_COEF,
    tangential_pen=0.15,
)


@dataclass(frozen=True)
class RewardProfile:
    name: str

    #TERMINAL OUTCOMES
    goal_bonus: float
    trunc_pen: float
    wall_collision_pen: float
    obstacle_collision_pen: float
    drone_collision_pen: float

    #GOAL-SEEKING SHAPING (IDENTICAL ACROSS PROFILES)
    prog_coef: float
    align_coef: float
    dist_pen: float
    time_pen: float
    ctrl_pen: float
    final_mile_coef: float
    tangential_pen: float

    #OBSTACLE / SWARM SHAPING (SWARM TERMS INERT - SINGLE-DRONE STUDY)
    near_front_obs_pen: float
    near_side_obs_pen: float
    cohesion_pen: float
    separation_pen: float


@dataclass(frozen=True)
class Condition:
    name: str
    reward: RewardProfile
    use_lidar: bool
    #SLICES ZEROED BEFORE THE OBS REACHES THE POLICY
    masked_slices: Tuple[slice, ...] = field(default_factory=tuple)
    #2X2 GRID AXES (FOR ANALYSIS / PLOT GROUPING)
    reward_shaped: bool = True
    obstacle_sensing: bool = True

    #OUTPUT FOLDER / FILE PREFIX FOR THIS CONDITION'S ARTIFACTS (SEEDLESS)
    @property
    def model_prefix(self) -> str:
        return f"ppo_{self.name}"


#APPLY AN EXPERIMENT PRESET'S REWARD OVERRIDES TO A CONDITION
def apply_preset_to_condition(condition: "Condition", preset_settings: dict) -> "Condition":
    """
    Single source of truth for preset-driven reward changes, used by BOTH
    training and evaluation. Evaluation re-derives the condition the same way
    and then checks it against the training manifest, so if these two ever
    disagreed the integrity check would reject the run - which is exactly what
    should happen, and why this lives in one place.

    `goal_stack_only` zeroes the shared goal-seeking extras. It is applied
    identically to all four conditions, so the ablated axes (collision
    penalties, proximity shaping, sensing) are untouched and the 2x2 design
    survives.
    """
    overrides = {}

    if preset_settings.get("goal_stack_only"):
        overrides.update(
            dist_pen=0.0,
            trunc_pen=0.0,
            final_mile_coef=0.0,
            align_coef=0.0,
            tangential_pen=0.0,
        )

    prox_scale = float(preset_settings.get("shaped_proximity_scale", 1.0))

    if prox_scale <= 0.0 or not math.isfinite(prox_scale):
        raise ValueError("shaped proximity scale must be finite and positive")

    if condition.reward_shaped and "shaped_proximity_scale" in preset_settings:
        overrides.update(
            near_front_obs_pen=SHAPED_NEAR_FRONT_PEN * prox_scale,
            near_side_obs_pen=SHAPED_NEAR_SIDE_PEN * prox_scale,
        )

    return replace(condition, reward=replace(condition.reward, **overrides)) if overrides else condition


#ONE TRAINING RUN = ONE (CONDITION, SEED) PAIR. EVERY ARTIFACT PATH CARRIES
#THE SEED, SO THE THREE RUNS PER CONDITION NEVER OVERWRITE EACH OTHER.
def run_name(condition_name: str, seed: int) -> str:
    return f"ppo_{condition_name}_s{int(seed)}"


#CONDITION NAME + SEED BACK OUT OF A RUN NAME (INVERSE OF run_name)
def split_run_name(name: str) -> tuple[str, int]:
    """
    Tolerates the pre-seed naming ("ppo_full", "full") by reporting seed 0,
    so a stale artifact folder from an earlier run cannot crash the analysis.
    """
    stem = name[4:] if name.startswith("ppo_") else name
    condition, sep, seed = stem.rpartition("_s")

    if not sep or not seed.isdigit():
        return stem, 0

    return condition, int(seed)


GOAL_ONLY_REWARD = RewardProfile(
    name="goal_only",
    #NO COLLISION FEEDBACK - COLLISIONS STILL END THE EPISODE & ARE STILL
    #RECORDED FOR METRICS, THE POLICY JUST NEVER LEARNS FROM THEM
    wall_collision_pen=0.0,
    obstacle_collision_pen=0.0,
    drone_collision_pen=0.0,
    near_front_obs_pen=0.0,
    near_side_obs_pen=0.0,
    cohesion_pen=0.0,
    separation_pen=0.0,
    **_GOAL_STACK,
)

SHAPED_REWARD = RewardProfile(
    name="shaped",
    wall_collision_pen=WALL_COLLISION_PEN,
    obstacle_collision_pen=OBSTACLE_COLLISION_PEN,
    drone_collision_pen=DRONE_COLLISION_PEN,
    near_front_obs_pen=SHAPED_NEAR_FRONT_PEN,
    near_side_obs_pen=SHAPED_NEAR_SIDE_PEN,
    cohesion_pen=0.0,
    separation_pen=0.0,
    **_GOAL_STACK,
)

#BACKWARD-COMPATIBLE ALIAS (ENV DEFAULTS TO THE FULL OBSTACLE-AWARE SET)
FULL_REWARD = SHAPED_REWARD

FULL = Condition(
    name="full",
    reward=SHAPED_REWARD,
    use_lidar=True,
    masked_slices=(),
    reward_shaped=True,
    obstacle_sensing=True,
)

REWARD_ONLY = Condition(
    name="reward_only",
    reward=SHAPED_REWARD,
    #RAYS CAST FOR THE REWARD'S PROXIMITY SHAPING, MASKED FROM THE POLICY
    use_lidar=True,
    masked_slices=BLIND_SLICES,
    reward_shaped=True,
    obstacle_sensing=False,
)

SENSING_ONLY = Condition(
    name="sensing_only",
    reward=GOAL_ONLY_REWARD,
    use_lidar=True,
    masked_slices=(),
    reward_shaped=False,
    obstacle_sensing=True,
)

DEGENERATE = Condition(
    name="degenerate",
    #RAYS ARE CAST ONLY SO THE SHARED FULL-INFORMATION EXPERT SEES THE SAME
    #THING IN EVERY CONDITION; THE POLICY NEVER SEES THEM (ALL MASKED)
    reward=GOAL_ONLY_REWARD,
    use_lidar=True,
    masked_slices=BLIND_SLICES,
    reward_shaped=False,
    obstacle_sensing=False,
)

CONDITIONS = {c.name: c for c in (FULL, REWARD_ONLY, SENSING_ONLY, DEGENERATE)}

#HEAD-TO-HEAD PAIR FIRST, ABLATION CELLS AFTER (CANONICAL RUN ORDER)
CONDITION_ORDER = ("degenerate", "full", "reward_only", "sensing_only")


#LOOK UP A CONDITION BY CLI NAME
def get_condition(name: str) -> Condition:
    key = name.strip().lower()

    if key not in CONDITIONS:
        valid = ", ".join(sorted(CONDITIONS))
        raise ValueError(f"unknown condition '{name}' (expected one of: {valid})")

    return CONDITIONS[key]


#RESOLVE A CLI CONDITION LIST (ALL, BOTH, OR COMMA-SEPARATED NAMES)
def resolve_condition_names(spec: str) -> list[str]:
    key = spec.strip().lower()

    if key == "all":
        return list(CONDITION_ORDER)
    if key == "both":
        return ["degenerate", "full"]

    return [n.strip() for n in key.split(",") if n.strip()]
