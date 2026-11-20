# -*- coding: utf-8 -*-
#PREEXISTS, MODIFIED FOR CS5478 PROJECT
"""
The following class contains the curriculum by which training is performed.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

#CURRICULUM VARIANT (ENV VAR DEFAULT; PRESETS SELECT EXPLICITLY). REPORTED RESULTS
#USE "compact" (REPORT SEC. 3-4) & "phase2" (SEC. 4.5); "default" = ORIGINAL
#5-STAGE LIST OF THE ENVIRONMENT, NOT USED BY ANY PRESET
CURRICULUM_VARIANT = os.environ.get("CS5478_CURRICULUM", "default")

#REPORTED STUDY (compact): 7 STAGES AT THE SMALLEST WORLD THE ENGINE ALLOWS.
#SPAWN HEIGHT IS uniform(SWARM_EDGE_MARGIN, scale - SWARM_EDGE_MARGIN) WITH A
#1 M MARGIN, SO BELOW 2.0 THE RANGE IS EMPTY. MEASURED AT 2.0 OVER 60 RESETS PER
#STAGE: EVERY OBSTACLE PLACED, 0/420 SPAWN-KILLS, NEAREST OBSTACLE >= 0.56 M AT
#SPAWN, <= 10% FLOOR COVER AT STAGE 6. BUDGETS (5TH FIELD, POLICY DECISIONS) ARE
#REFERENCE ONLY UNDER GATING; HORIZONS (6TH FIELD) ARE PHYSICS TICKS (10 S AT 240 HZ)
COMPACT_WORLD_SCALE = 2.0
COMPACT_STAGES = [
    (COMPACT_WORLD_SCALE, 1, 0, 0, 200_000, 2400, True),
    (COMPACT_WORLD_SCALE, 1, 1, 0, 100_000, 2400, True),
    (COMPACT_WORLD_SCALE, 1, 2, 0, 100_000, 2400, True),
    (COMPACT_WORLD_SCALE, 1, 3, 0, 100_000, 2400, True),
    (COMPACT_WORLD_SCALE, 1, 3, 1, 100_000, 2400, True),
    (COMPACT_WORLD_SCALE, 1, 5, 2, 100_000, 2400, True),
    (COMPACT_WORLD_SCALE, 1, 6, 3, 200_000, 2400, True),
]

#PHASE 2 (REPORTED, SEC. 4.5): WALL-GAP CURRICULUM AT SCALE 2. NO RANDOM OBSTACLES; THE ONE
#WALL PER WORLD COMES FROM THE PRESET'S stage_settings["wall_coverage"] (10 -> 30 ->
#50 -> 70% -> MIXED 10-70%). BUDGETS ARE REFERENCE ONLY UNDER GATING.
PHASE2_STAGES = [(COMPACT_WORLD_SCALE, 1, 0, 0, 200_000, 2400, True) for _ in range(5)]


class Curriculum:
    """
    PPO curriculum (single-drone 3D navigation, all four experiment conditions)

    Each stage tuple:
    (
        world_scale,
        num_drones,
        num_static_obstacles,
        num_dynamic_obstacles,
        stage_training_timesteps,
        episode_horizon,
        lidar_enabled
    )

    Notes:
    - stage_training_timesteps = how long to train for in that curriculum stage
    - episode_horizon = per-episode max number of steps before truncation
    - lidar_enabled = whether LiDAR rays may be cast in that stage (each
      condition further restricts this via config/experiment.py)
    - identical for every condition, so training curves are comparable
    """

    #BUILD THE STAGE LIST & START AT STAGE 0
    def __init__(self, variant: Optional[str] = None):
        self.variant = (variant or CURRICULUM_VARIANT).strip().lower()

        if self.variant not in ("default", "compact", "phase2"):
            raise ValueError(
                f"unknown curriculum variant '{self.variant}' "
                f"(expected 'default', 'compact' or 'phase2')"
            )

        if self.variant == "phase2":
            self.stages = list(PHASE2_STAGES)
            self.current_stage = 0
            return

        if self.variant == "compact":
            self.stages = list(COMPACT_STAGES)
            self.current_stage = 0
            return

        self.stages: List[Tuple[float, int, int, int, int, int, bool]] = [

            #ORIGINAL 5-STAGE CURRICULUM OF THE ENVIRONMENT
            #EMPTY SMALL WORLD - LEARN PURE GOAL-SEEKING
            (2, 1, 0, 0, 100_000, 1200, True),
            #ONE STATIC OBSTACLE
            (2, 1, 1, 0, 150_000, 1500, True),
            #CLUTTER
            (3, 1, 3, 0, 200_000, 1800, True),
            #FIRST MOVING OBSTACLE
            (3, 1, 3, 1, 150_000, 2000, True),
            #FINAL - BIGGER WORLD, DENSE STATIC + DYNAMIC MIX
            (4, 1, 4, 2, 200_000, 2400, True),

        ]
        self.current_stage = 0

    #RETURN PARAMS OF CURRENT STAGE
    def get_stage(self):
        return self.stages[self.current_stage]

    #ADVANCE TO NEXT STAGE IF AVAILABLE
    def next_stage(self):
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            print(f"Advancing to Curriculum Stage {self.current_stage}")
        else:
            print("Already at final stage.")
