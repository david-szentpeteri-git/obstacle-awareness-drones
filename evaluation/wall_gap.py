# -*- coding: utf-8 -*-
#NEWLY ADDED FOR CS5478 PROJECT
"""
Wall-gap evaluation suite (evaluation only; never used in training).

Each episode: the drone starts near one side of the arena, the goal sits near
the opposite side, and ONE flat wall panel stands across the passage between
them. Seen from the drone, the panel is a strip covering X% of the passage
cross-section:

  vertical   - floor-to-ceiling strip, gap(s) to the left and/or right
  horizontal - wall-to-wall strip, gap(s) above and/or below
  diagonal   - 45 degree band, gaps in the corners

Orientation, gap side(s), wall position along the way, travel axis/direction,
spawn and goal are all random per episode but fixed by the episode seed, so
every model faces the same worlds. Guarantees (asserted per layout):

  - the wall is ALWAYS placed (no silent skip)
  - wall face >= 1 m from the spawn and from the goal centre (no spawn-kill,
    goal never touches or sits behind the wall)
  - at least one opening fits a 0.4 m circle (~3x the drone's 0.13 m span);
    where X% would leave less, coverage is reduced to the largest passable
    value and the episode is flagged `wall_capped`

Coverage and opening are measured on a 2 cm grid over the flyable passage
cross-section (inside the arena wall faces, above the 0.1 m floor limit).
"""
from __future__ import annotations

from copy import deepcopy
from typing import Dict

import numpy as np
import pybullet as p
from scipy.ndimage import distance_transform_edt

from drone_env.createObjects import EnvObject

#SUITE SETTINGS
WALL_COVERAGES = (0.1, 0.3, 0.5, 0.7, 0.9)
WALL_SCALES = (2.0, 3.0)
WALL_THICKNESS = 0.1
MIN_OPENING = 0.4
WALL_CLEARANCE = 1.0
WALL_HORIZON_TICKS = 60 * 240
ORIENTATIONS = ("vertical", "horizontal", "diagonal")

#SPAWN/GOAL PLACEMENT (MATCHES TRAINING GOAL MARGIN 0.6)
SPAWN_BAND = (0.6, 1.2)
EDGE_MARGIN = 0.6
GOAL_MARGIN = 0.6

#FLYABLE CROSS-SECTION (ARENA WALL HALF-THICKNESS 0.05, FLOOR LIMIT z <= 0.1)
WALL_FACE = 0.05
FLOOR_LIMIT = 0.1
GRID_RES = 0.02
OPENING_TOL = 2 * GRID_RES
DIAG_TRIES = 64

#COLUMNS ADDED TO EVALUATION RECORDS FOR THIS SUITE
WALL_RECORD_FIELDS = [
    "world_scale", "wall_orientation", "wall_gap_mode", "wall_coverage_target", "wall_coverage",
    "wall_capped", "wall_opening_m", "wall_travel_axis", "wall_direction", "wall_t", "wall_diag_deg",
    "wall_cx", "wall_cy", "wall_cz", "wall_hx", "wall_hy", "wall_hz", "wall_roll", "wall_pitch",
]


#PANEL OBSTACLE: BOX WITH ARBITRARY HALF EXTENTS & ORIENTATION
class PanelObject(EnvObject):

    #STORE PANEL GEOMETRY & REGISTER "panel" SHAPE
    def __init__(self, center, half_ext, euler):
        super().__init__(np.asarray(center, dtype=np.float32), 1.0, list(euler))
        half = np.asarray(half_ext, dtype=np.float64)
        self.SHAPE_CONFIGS["panel"] = {
            "geom": p.GEOM_BOX,
            "collision_kwargs": lambda s: {"halfExtents": half},
            "visual_kwargs": lambda s: {"halfExtents": half},
            "use_orientation": True,
            "radius": float(np.linalg.norm(half)),
        }
        self.set_shape_spec("panel")


#ONE SCENARIO PER (SCALE, COVERAGE); EPISODE HORIZON 60 s
def wall_gap_scenarios(preset: dict) -> Dict[str, dict]:
    base_cfg = deepcopy(preset["stage_settings"][0]) if preset.get("stage_settings") else {}
    base_cfg["blocking_obstacle_prob"] = 0.0
    #THE SUITE PLACES ITS OWN WALL AT RESET; THE ENV'S TRAINING-TIME WALL HOOK STAYS OFF
    base_cfg.pop("wall_coverage", None)

    if "evaluation_obstacle_dt" in preset:
        base_cfg["obstacle_dt"] = preset["evaluation_obstacle_dt"]

    scenarios: Dict[str, dict] = {}

    for scale in WALL_SCALES:
        for cov in WALL_COVERAGES:
            env_cfg = dict(world_scale=float(scale), num_static_obs=0, num_dynamic_obs=0,
                           episode_horizon=WALL_HORIZON_TICKS, **deepcopy(base_cfg))
            scenarios[f"wall_s{int(scale)}_x{int(round(cov * 100)):02d}"] = {
                "kind": "wall_gap",
                "coverage": float(cov),
                "stage_lidar": True,
                "env_cfg": env_cfg,
            }

    return scenarios


#FLYABLE PASSAGE RECTANGLE (u = LATERAL, v = HEIGHT)
def passage_bounds(scale: float):
    return (-scale + WALL_FACE, scale - WALL_FACE), (FLOOR_LIMIT, scale - WALL_FACE)


#CELL-CENTRE GRID OVER PASSAGE
def passage_grid(scale: float):
    (u_lo, u_hi), (v_lo, v_hi) = passage_bounds(scale)
    u_pts = np.arange(u_lo + GRID_RES / 2, u_hi, GRID_RES)
    v_pts = np.arange(v_lo + GRID_RES / 2, v_hi, GRID_RES)

    return np.meshgrid(u_pts, v_pts, indexing="ij")


#COVERED FRACTION & LARGEST INSCRIBED OPENING DIAMETER OF BLOCKED MASK
def mask_metrics(blocked: np.ndarray):
    free = np.pad(~blocked, 1, constant_values=False)
    dist = distance_transform_edt(free)

    return float(blocked.mean()), float(2.0 * dist.max() * GRID_RES)


#GRID INDICES OF THE OPENING'S CENTRE (FREE CELL FARTHEST FROM ANY BLOCKED CELL OR EDGE)
def opening_centre(blocked: np.ndarray):
    free = np.pad(~blocked, 1, constant_values=False)
    dist = distance_transform_edt(free)[1:-1, 1:-1]

    return np.unravel_index(int(np.argmax(dist)), dist.shape)


#BLOCKED MASK OF AXIS-ALIGNED STRIP
def strip_mask(uu, vv, u_rng, v_rng):
    return (uu >= u_rng[0]) & (uu <= u_rng[1]) & (vv >= v_rng[0]) & (vv <= v_rng[1])


#BLOCKED MASK OF DIAGONAL BAND |n.p - c| <= w/2
def band_mask(uu, vv, normal, offset, width):
    return np.abs(normal[0] * uu + normal[1] * vv - offset) <= width / 2


#STRIP SPAN + GAP MODE ALONG ONE CROSS-SECTION AXIS (VERTICAL/HORIZONTAL)
def sample_strip(lo: float, hi: float, cov: float, rng: np.random.Generator):
    span = hi - lo
    need = MIN_OPENING + OPENING_TOL
    width = cov * span
    capped = span - width < need

    if capped:
        width = span - need

    gap = span - width
    mode = str(rng.choice(["flush_lo", "flush_hi", "interior"]))

    if capped and mode == "interior":
        mode = str(rng.choice(["flush_lo", "flush_hi"]))

    if mode == "flush_lo":
        start = lo
    elif mode == "flush_hi":
        start = hi - width
    else:
        #INTERIOR: BOTH GAPS > 0, LARGER GAP >= OPENING (UNION OF TWO RANGES)
        left_len = max(0.0, gap - need)
        right_len = max(0.0, gap - need)
        pick = rng.uniform(0.0, left_len + right_len)
        offset = pick if pick < left_len else need + (pick - left_len)
        offset = float(np.clip(offset, 1e-3, gap - 1e-3))
        start = lo + offset

    return start, start + width, mode, capped


#DIAGONAL BAND OFFSET/WIDTH FOR TARGET COVERAGE (CAPPED IF NO PASSABLE OFFSET)
def sample_band(uu, vv, normal, cov: float, rng: np.random.Generator):
    proj = normal[0] * uu + normal[1] * vv
    c_lo, c_hi = float(proj.min()), float(proj.max())
    need = MIN_OPENING + OPENING_TOL
    best = None

    for _ in range(DIAG_TRIES):
        offset = float(rng.uniform(c_lo, c_hi))
        lo_w, hi_w = 0.0, 2.0 * max(abs(c_hi - offset), abs(offset - c_lo)) + GRID_RES

        for _ in range(30):
            mid = (lo_w + hi_w) / 2

            if (np.abs(proj - offset) <= mid / 2).mean() < cov:
                lo_w = mid
            else:
                hi_w = mid

        width = hi_w
        cover, opening = mask_metrics(band_mask(uu, vv, normal, offset, width))

        if opening >= need:
            return offset, width, False

        if best is None or opening > best[2]:
            best = (offset, width, opening)

    #CAP: SHRINK BAND AT BEST OFFSET UNTIL OPENING FITS
    offset, hi_w, _ = best
    lo_w = 0.0

    for _ in range(30):
        mid = (lo_w + hi_w) / 2

        if mask_metrics(band_mask(uu, vv, normal, offset, mid))[1] >= need:
            lo_w = mid
        else:
            hi_w = mid

    return offset, lo_w, True


#MAP (t, u, v) VECTOR TO WORLD xyz FOR TRAVEL AXIS
def to_world(t_val, u_val, v_val, travel_axis: int) -> np.ndarray:
    out = np.zeros(3, dtype=np.float64)
    out[travel_axis] = t_val
    out[1 - travel_axis] = u_val
    out[2] = v_val

    return out


#SAMPLE ONE COMPLETE WALL-GAP LAYOUT (DETERMINISTIC GIVEN rng)
def sample_wall_layout(scale: float, cov: float, rng: np.random.Generator) -> dict:
    scale = float(scale)
    travel_axis = int(rng.integers(0, 2))
    direction = int(rng.choice([-1, 1]))
    orientation = str(rng.choice(ORIENTATIONS))

    #SPAWN NEAR + SIDE, GOAL NEAR - SIDE (IN TRAVEL COORDS BEFORE DIRECTION)
    spawn_t = scale - float(rng.uniform(*SPAWN_BAND))
    goal_t = -(scale - GOAL_MARGIN)
    spawn_uv = (float(rng.uniform(-scale + EDGE_MARGIN, scale - EDGE_MARGIN)),
                float(rng.uniform(EDGE_MARGIN, scale - EDGE_MARGIN)))
    goal_uv = (float(rng.uniform(-scale + EDGE_MARGIN, scale - EDGE_MARGIN)),
               float(rng.uniform(EDGE_MARGIN, scale - EDGE_MARGIN)))

    #WALL PLANE WITH >= CLEARANCE FROM BOTH (FACE DISTANCE)
    half_thick = WALL_THICKNESS / 2
    wall_lo = goal_t + WALL_CLEARANCE + half_thick
    wall_hi = spawn_t - WALL_CLEARANCE - half_thick

    if wall_hi < wall_lo:
        raise ValueError(f"no room for a wall at world_scale={scale}")

    wall_t = float(rng.uniform(wall_lo, wall_hi))

    (u_lo, u_hi), (v_lo, v_hi) = passage_bounds(scale)
    uu, vv = passage_grid(scale)
    diag_deg = float("nan")
    euler = [0.0, 0.0, 0.0]

    if orientation == "vertical":
        a_lo, a_hi, mode, capped = sample_strip(u_lo, u_hi, cov, rng)
        blocked = strip_mask(uu, vv, (a_lo, a_hi), (v_lo, v_hi))
        #FLUSH SIDE SEALED UP TO ARENA WALL CENTRE; FULL HEIGHT
        box_u = (-scale if mode == "flush_lo" else a_lo, scale if mode == "flush_hi" else a_hi)
        box_v = (0.0, scale)
    elif orientation == "horizontal":
        b_lo, b_hi, mode, capped = sample_strip(v_lo, v_hi, cov, rng)
        blocked = strip_mask(uu, vv, (u_lo, u_hi), (b_lo, b_hi))
        box_u = (-scale, scale)
        box_v = (0.0 if mode == "flush_lo" else b_lo, scale if mode == "flush_hi" else b_hi)
    else:
        diag_deg = float(rng.choice([45.0, 135.0]))
        theta = np.deg2rad(diag_deg)
        normal = np.array([np.cos(theta), np.sin(theta)])
        offset, width, capped = sample_band(uu, vv, normal, cov, rng)
        blocked = band_mask(uu, vv, normal, offset, width)
        mode = "corners"
        center_uv = np.array([(u_lo + u_hi) / 2, (v_lo + v_hi) / 2])
        center_uv = center_uv + (offset - normal @ center_uv) * normal
        length = 2.0 * float(np.hypot(u_hi - u_lo, v_hi - v_lo))

    cover, opening = mask_metrics(blocked)
    gap_i, gap_j = opening_centre(blocked)
    gap_point = to_world(wall_t, float(uu[gap_i, gap_j]), float(vv[gap_i, gap_j]), travel_axis)

    if orientation == "diagonal":
        center = to_world(wall_t, center_uv[0], center_uv[1], travel_axis)
        #LOCAL z --> BAND NORMAL; LOCAL LONG AXIS ALONG BAND
        if travel_axis == 0:
            half_ext = np.array([half_thick, length / 2, width / 2])
            euler = [float(theta - np.pi / 2), 0.0, 0.0]
        else:
            half_ext = np.array([length / 2, half_thick, width / 2])
            euler = [0.0, float(np.pi / 2 - theta), 0.0]
    else:
        center = to_world(wall_t, (box_u[0] + box_u[1]) / 2, (box_v[0] + box_v[1]) / 2, travel_axis)
        half_ext = to_world(half_thick, (box_u[1] - box_u[0]) / 2, (box_v[1] - box_v[0]) / 2, travel_axis)

    #DIRECTION FLIP: MIRROR TRAVEL COORDINATE (PANEL SYMMETRIC IN THICKNESS)
    center[travel_axis] *= direction
    gap_point[travel_axis] *= direction
    spawn = to_world(direction * spawn_t, *spawn_uv, travel_axis)
    goal = to_world(direction * goal_t, *goal_uv, travel_axis)

    layout = {
        "world_scale": scale,
        "orientation": orientation,
        "gap_mode": mode,
        "coverage_target": float(cov),
        "coverage": cover,
        "capped": bool(capped),
        "opening_m": opening,
        "travel_axis": "xy"[travel_axis],
        "direction": direction,
        "wall_t": direction * wall_t,
        "diag_deg": diag_deg,
        "center": center,
        "half_ext": np.asarray(half_ext, dtype=np.float64),
        "euler": euler,
        #CENTRE OF THE LARGEST OPENING, ON THE WALL PLANE (TEACHER WAYPOINT)
        "gap_point": gap_point.astype(np.float32),
        "spawn": spawn.astype(np.float32),
        "goal": goal.astype(np.float32),
    }
    check_layout(layout)

    return layout


#ASSERT LAYOUT GUARANTEES (WALL PRESENT, CLEARANCE, PASSABLE, ON-TARGET COVERAGE)
def check_layout(layout: dict) -> None:
    axis = "xy".index(layout["travel_axis"])
    scale = layout["world_scale"]
    face_gap = WALL_CLEARANCE - 1e-6

    if not np.all(layout["half_ext"] > 0):
        raise AssertionError("wall panel has no extent")
    if abs(layout["spawn"][axis] - layout["wall_t"]) - WALL_THICKNESS / 2 < face_gap:
        raise AssertionError("wall too close to spawn")
    if abs(layout["goal"][axis] - layout["wall_t"]) - WALL_THICKNESS / 2 < face_gap:
        raise AssertionError("wall too close to goal")
    if np.sign(layout["spawn"][axis] - layout["wall_t"]) == np.sign(layout["goal"][axis] - layout["wall_t"]):
        raise AssertionError("spawn and goal on the same side of the wall")
    if layout["opening_m"] < MIN_OPENING:
        raise AssertionError(f"opening {layout['opening_m']:.3f} m below {MIN_OPENING} m")
    if not layout["capped"] and abs(layout["coverage"] - layout["coverage_target"]) > 0.02:
        raise AssertionError(f"coverage {layout['coverage']:.3f} misses target {layout['coverage_target']}")
    if layout["capped"] and layout["coverage"] > layout["coverage_target"] + 0.02:
        raise AssertionError("capped layout covers more than requested")

    for pos in (layout["spawn"], layout["goal"]):
        if np.any(np.abs(pos[:2]) > scale - EDGE_MARGIN + 1e-6) or not (EDGE_MARGIN - 1e-6 <= pos[2] <= scale - EDGE_MARGIN + 1e-6):
            raise AssertionError("spawn/goal outside placement margins")


#REPLACE ENV OBSTACLES WITH LAYOUT PANEL
def apply_wall_layout(env, layout: dict) -> None:
    env.remove_all_obstacles()
    panel = PanelObject(layout["center"], layout["half_ext"], layout["euler"])
    panel.create("panel", physics_call=env._call_physics)
    env._obstacles = [panel]
    env.num_static_obstacles = 1
    env.num_dynamic_obstacles = 0
    env._rebuild_query_world_from_current_scene()


#FLAT RECORD COLUMNS FOR ONE LAYOUT
def layout_record(layout: dict) -> dict:
    center, half_ext, euler = layout["center"], layout["half_ext"], layout["euler"]

    return {
        "world_scale": layout["world_scale"],
        "wall_orientation": layout["orientation"],
        "wall_gap_mode": layout["gap_mode"],
        "wall_coverage_target": layout["coverage_target"],
        "wall_coverage": round(layout["coverage"], 4),
        "wall_capped": int(layout["capped"]),
        "wall_opening_m": round(layout["opening_m"], 3),
        "wall_travel_axis": layout["travel_axis"],
        "wall_direction": layout["direction"],
        "wall_t": round(layout["wall_t"], 4),
        "wall_diag_deg": layout["diag_deg"],
        "wall_cx": round(float(center[0]), 4),
        "wall_cy": round(float(center[1]), 4),
        "wall_cz": round(float(center[2]), 4),
        "wall_hx": round(float(half_ext[0]), 4),
        "wall_hy": round(float(half_ext[1]), 4),
        "wall_hz": round(float(half_ext[2]), 4),
        "wall_roll": round(float(euler[0]), 5),
        "wall_pitch": round(float(euler[1]), 5),
    }
