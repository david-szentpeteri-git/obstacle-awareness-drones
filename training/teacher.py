# -*- coding: utf-8 -*-
#NEWLY ADDED FOR CS5478 PROJECT
"""
The one place that builds the expert teacher, so training rollouts and
validation batches are guaranteed to blend with an identical expert.

Teacher information (decided 2026-10-12). Every condition gets the SAME
full-information expert: it reads the raw, unmasked observation - nearest
obstacle vector and all six rays - even when the student it is helping is
blind. The students' own observations are unchanged and still follow their
condition's masks. This must be disclosed in the report: obstacle knowledge
enters blind-condition training through the teacher's demonstrations. A
per-condition teacher is not an option: a blind teacher reaches only
65% / 45% of goals at stages 4 / 6 and would dominate training at a
starting expert weight of 0.8.

Distances. The expert's radii were tuned for a 5 m world (react to obstacles
within 3.5 m, walls within 2.5 m, brake within 1.5 m of the goal). In a 2 m
world those radii exceed the arena half-width, so scaling them down looked
necessary - but it was benchmarked rather than assumed (2026-10-12, world
scale 2, six goal-relative rays, 20 episodes per stage, all 7 stages):
    original radii : 97.9% reached, 2.1% collided, 0% timeout (stage 6: 95%)
    scaled radii   : 96.4% reached, 3.6% collided, 0% timeout (stage 6: 85%)
The original radii are at least as good and more cautious in the dense stages,
so they are kept unchanged.

Teacher kinds (phase 2, 2026-10-15):
    expert     the environment's scripted expert as above
    gap_aware  the same expert that additionally knows the episode's wall
               layout (privileged): until it has passed the wall plane it flies
               at the centre of the largest opening, ignoring the panel as an
               "obstacle" (the opening is the way past it); afterwards it flies
               at the goal as the plain expert. Without a wall it IS the plain
               expert. The students never see this information.
"""
from __future__ import annotations

import numpy as np

from drone_env.expert_policy import ExpertPolicy
from drone_env.defaults import LIDAR_RANGE

#WORLD SCALE THE EXPERT'S RADII WERE TUNED FOR
TEACHER_REFERENCE_SCALE = 5.0

#THE EXPERT'S ORIGINAL RADII AT THE REFERENCE SCALE
REFERENCE_RADII = dict(
    avoid_radius=3.5,
    wall_margin=2.5,
    target_brake_dist=1.5,
    teammate_avoid_radius=2.0,
)

#KEEP THE ORIGINAL RADII (SCALE-2 BENCHMARK: 97.9% VS 96.4% WHEN SCALED)
TEACHER_SCALE_RADII = False
TEACHER_KINDS = ("expert", "gap_aware")
#WAYPOINT SITS THIS FAR PAST THE WALL PLANE, THROUGH THE OPENING
WAYPOINT_BEYOND_WALL = 0.35
WALL_THICKNESS = 0.1
NO_OBSTACLE = np.array([LIDAR_RANGE * 2.0, 0.0, 0.0], dtype=np.float32)


#RADII ACTUALLY USED AT A GIVEN WORLD SCALE
def teacher_radii(world_scale: float, scale_radii: bool = TEACHER_SCALE_RADII) -> dict:
    factor = float(world_scale) / TEACHER_REFERENCE_SCALE if scale_radii else 1.0

    return {name: round(value * factor, 6) for name, value in REFERENCE_RADII.items()}


#EXPERT THAT FLIES THROUGH THE EPISODE'S WALL OPENING (PRIVILEGED LAYOUT, READ FROM THE RAW ENV)
class GapAwareExpert(ExpertPolicy):

    #PLAIN EXPERT UNTIL AN ENV WITH A WALL LAYOUT IS ATTACHED
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.env = None

    #RAW ENV WHOSE current_wall_layout IS READ EVERY DECISION
    def attach_env(self, env) -> None:
        self.env = env

    #WAYPOINT THROUGH THE OPENING WHILE BEFORE THE WALL, ELSE THE GOAL
    def action(self, obs: np.ndarray) -> np.ndarray:
        layout = getattr(self.env, "current_wall_layout", None)

        if layout is None:
            return super().action(obs)

        obs = np.asarray(obs, dtype=np.float32)
        axis = "xy".index(layout["travel_axis"])
        toward = -float(layout["direction"])
        pos = obs[0:3]

        if (float(pos[axis]) - float(layout["wall_t"])) * toward > WALL_THICKNESS / 2 + 0.05:
            return super().action(obs)

        waypoint = np.array(layout["gap_point"], dtype=np.float32)
        waypoint[axis] += toward * WAYPOINT_BEYOND_WALL
        steered = obs.copy()
        steered[6:9] = waypoint - pos
        steered[9:12] = NO_OBSTACLE

        return super().action(steered)


#BUILD THE SHARED FULL-INFORMATION EXPERT FOR A WORLD SCALE
def make_teacher(world_scale: float, scale_radii: bool = TEACHER_SCALE_RADII, kind: str = "expert"):
    if kind not in TEACHER_KINDS:
        raise ValueError(f"unknown teacher kind '{kind}' (expected one of {TEACHER_KINDS})")

    cls = GapAwareExpert if kind == "gap_aware" else ExpertPolicy

    return cls(world_scale=float(world_scale), use_lidar=True, **teacher_radii(world_scale, scale_radii))


#TEACHER DESCRIPTION FOR THE RUN MANIFEST
def teacher_settings(world_scale: float, kind: str = "expert") -> dict:
    return {
        "kind": kind,
        "source": "drone_env/expert_policy.py::ExpertPolicy" + (" + training/teacher.py::GapAwareExpert" if kind == "gap_aware" else ""),
        "information": "full (unmasked observation, all six rays) for every condition"
                       + ("; plus the episode's wall layout (waypoint through the opening)" if kind == "gap_aware" else ""),
        "reference_scale": TEACHER_REFERENCE_SCALE,
        "scale_radii": TEACHER_SCALE_RADII,
        "radii": teacher_radii(world_scale),
        **({"waypoint_beyond_wall_m": WAYPOINT_BEYOND_WALL} if kind == "gap_aware" else {}),
    }
