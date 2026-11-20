# -*- coding: utf-8 -*-
#NEWLY ADDED FOR CS5478 PROJECT
"""
Report figure 1: rendered snapshots of one stage-6 evaluation world and one
wall-gap world, drawn with the simulator's own software renderer (no window).

The worlds are real evaluation worlds (held-out seed band, same generators as
the evaluation), rendered after the reset and before any step. Drawn on top of
the scene, as visual-only bodies: the drone's six rays (red = hits an obstacle,
blue = reaches the arena boundary), a marker on the drone, and the straight
start-goal segment. The ceiling and the two walls nearest the camera are
removed for the view; nothing else is changed.

  .\\venv\\Scripts\\python.exe -B -m analysis.report_snapshots

Writes results/report/figures/env_stage6.png, env_wallgap.png and
env_snapshots.json (the seeds and world facts behind the two pictures).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pybullet as p
from PIL import Image

from drone_env.defaults import LIDAR_RANGE
from config.experiment import get_condition
from config.ppo_defaults import get_preset
from training.env_factory import safe_close
from evaluation.evaluate import make_scenario_env, scenario_episode_seeds
from evaluation.scenarios import reset_scenario_episode, resolve_scenarios
from evaluation.wall_gap import wall_gap_scenarios
from training.wrappers.lidar_layout import ray_directions

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results/report" / "figures"
#SAME RAY START OFFSET AS DroneSwarm3D._lidar_scan
RAY_START = 0.15
STAGE_SCENARIO, WALL_SCENARIO = "stage6_6s3d", "wall_s2_x50"

#COLOURS (REPORT ENCODING: BLUE = SEES, RED = OBSTACLE HIT; GREYS FOR THE ARENA)
RAY_HIT, RAY_FREE = (0.85, 0.15, 0.15, 1.0), (0.17, 0.47, 0.84, 1.0)
DRONE_MARK, STRAIGHT = (0.17, 0.47, 0.84, 1.0), (0.93, 0.72, 0.10, 1.0)
FLOOR, BACK_WALL, OBSTACLE = (0.92, 0.92, 0.90, 1.0), (0.82, 0.82, 0.80, 1.0), (0.45, 0.45, 0.43, 1.0)


#CAST THE DRONE'S RAYS AS THE ENV DOES; RETURN ORIGIN, GOAL AND (FROM, END, HIT-OBSTACLE) PER RAY
def cast_rays(env):
    did = env._drone_ids[0]
    pos, _ = env._call_physics(p.getBasePositionAndOrientation, did)
    origin = np.asarray(pos, dtype=np.float64)
    goal = np.asarray(env._targets[0].start_pos, dtype=np.float64)
    dirs = ray_directions((goal - origin).astype(np.float32)).astype(np.float64)
    ray_from = origin[None, :] + dirs * RAY_START
    ray_to = ray_from + dirs * LIDAR_RANGE
    results = env._call_physics(p.rayTestBatch, rayFromPositions=ray_from.tolist(), rayToPositions=ray_to.tolist())
    obstacle_ids = {o.body_id for o in env._obstacles if getattr(o, "body_id", None) is not None}
    rays = []

    for i, r in enumerate(results):
        body = int(r[0])
        if body < 0 or body == did:
            rays.append((ray_from[i], ray_to[i], False))
        else:
            rays.append((ray_from[i], np.asarray(r[3], dtype=np.float64), body in obstacle_ids))

    return origin, goal, rays


#VISUAL-ONLY CYLINDER FROM a TO b
def add_segment(env, a, b, radius, rgba):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    d = b - a
    length = float(np.linalg.norm(d))

    if length < 1e-6:
        return

    u = d / length
    axis = np.cross([0.0, 0.0, 1.0], u)
    s, c = float(np.linalg.norm(axis)), float(u[2])
    quat = ([0, 0, 0, 1] if c > 0 else [1, 0, 0, 0]) if s < 1e-8 else p.getQuaternionFromAxisAngle((axis / s).tolist(), float(np.arctan2(s, c)))
    shape = env._call_physics(p.createVisualShape, p.GEOM_CYLINDER, radius=radius, length=length, rgbaColor=list(rgba))
    env._call_physics(p.createMultiBody, baseMass=0, baseVisualShapeIndex=shape,
                      basePosition=((a + b) / 2).tolist(), baseOrientation=list(quat))


#VISUAL-ONLY SPHERE
def add_sphere(env, centre, radius, rgba):
    shape = env._call_physics(p.createVisualShape, p.GEOM_SPHERE, radius=radius, rgbaColor=list(rgba))
    env._call_physics(p.createMultiBody, baseMass=0, baseVisualShapeIndex=shape, basePosition=np.asarray(centre, dtype=float).tolist())


#OPEN THE ARENA TOWARDS THE CAMERA: DROP CEILING + THE TWO NEAR WALLS, SOFTEN THE REST
def open_arena(env, eye):
    keep = []

    for wid in env._wall_ids:
        pos, _ = env._call_physics(p.getBasePositionAndOrientation, wid)
        pos = np.asarray(pos)
        axis = int(np.argmax(np.abs(pos)))
        near = (axis == 2 and pos[2] > 0) or (axis < 2 and np.sign(pos[axis]) == np.sign(eye[axis]))

        if near:
            env._call_physics(p.removeBody, wid)
        else:
            env._call_physics(p.changeVisualShape, wid, -1, rgbaColor=list(FLOOR if axis == 2 else BACK_WALL))
            keep.append(wid)

    env._wall_ids = keep

    #OBSTACLES IN NEUTRAL GREY SO THAT A RED RAY (OBSTACLE HIT) STANDS OUT
    for obj in env._obstacles:
        if getattr(obj, "body_id", None) is not None:
            env._call_physics(p.changeVisualShape, obj.body_id, -1, rgbaColor=list(OBSTACLE))

    #THE AVIARY'S TEXTURED GROUND PLANE BECOMES A PLAIN LIGHT FLOOR
    for bid in range(env._call_physics(p.getNumBodies)):
        try:
            uid = env._call_physics(p.getBodyUniqueId, bid)
            name = env._call_physics(p.getBodyInfo, uid)[1].decode("utf-8", "ignore").lower()
            if "plane" in name:
                env._call_physics(p.changeVisualShape, uid, -1, rgbaColor=list(FLOOR), textureUniqueId=-1)
        except Exception:
            continue


#RENDER WITH THE SOFTWARE RENDERER AND CROP TO THE SCENE
def render(env, eye, target, width=1800, height=1350, fov=36.0):
    view = p.computeViewMatrix(list(eye), list(target), [0.0, 0.0, 1.0])
    proj = p.computeProjectionMatrixFOV(fov=fov, aspect=width / height, nearVal=0.1, farVal=50.0)
    img = env._call_physics(p.getCameraImage, width, height, viewMatrix=view, projectionMatrix=proj,
                            renderer=p.ER_TINY_RENDERER, shadow=0, lightDirection=[0.9, -0.5, 1.6],
                            lightAmbientCoeff=0.62, lightDiffuseCoeff=0.45, lightSpecularCoeff=0.03)
    rgba = np.reshape(np.asarray(img[2], dtype=np.uint8), (height, width, 4))[:, :, :3]
    background = rgba[0, 0]
    mask = np.any(rgba != background, axis=2)
    ys, xs = np.where(mask)
    pad = 24
    y0 = max(int(ys.min()) - pad, 0)
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad, width)
    #THE GROUND PLANE RUNS TO THE CAMERA; CUT THE EMPTY FOREGROUND TO A FIXED ASPECT
    y1 = min(y0 + int((x1 - x0) / 1.6), height)

    return Image.fromarray(rgba[y0:y1, x0:x1])


#DECORATE + RENDER ONE RESET WORLD
def snapshot(env, scale):
    origin, goal, rays = cast_rays(env)
    eye = np.array([2.6, -2.6, 2.3]) * scale
    target = np.array([0.0, 0.0, 0.4 * scale])
    open_arena(env, eye)
    add_segment(env, origin, goal, 0.009, STRAIGHT)

    for a, b, hit in rays:
        add_segment(env, a, b, 0.014, RAY_HIT if hit else RAY_FREE)
    add_sphere(env, origin, 0.07, DRONE_MARK)
    facts = {"drone": origin.round(3).tolist(), "goal": goal.round(3).tolist(),
             "rays_hitting_obstacle": int(sum(h for _, _, h in rays)), "obstacles": len(env._obstacles)}

    return render(env, eye, target), facts


#STAGE-6 WORLD: A HELD-OUT WORLD WITH THE BLOCKER ON THE LINE WHERE full_s0 REACHED AND degenerate_s0 DID NOT
def pick_stage_world(env, scenario, seeds):
    preferred = []
    records = ROOT / "results/compact_eval_200ep" / "eval" / "records.csv"

    if records.exists():
        df = pd.read_csv(records)
        df = df[(df.scenario == STAGE_SCENARIO) & (df.train_seed == 0)]
        won = set(df[(df.condition == "full") & (df.reached == 1)].seed)
        lost = set(df[(df.condition == "degenerate") & (df.reached == 0)].seed)
        preferred = [s for s in seeds if s in won and s in lost]

    for seed in preferred + [s for s in seeds if s not in preferred]:
        reset_scenario_episode(env, scenario, int(seed))

        if bool(getattr(env, "_blocking_obstacle_spawned", False)):
            return int(seed), seed in preferred

    raise RuntimeError("no stage-6 world with a blocker among the evaluation seeds")


#WALL WORLD: 50% HORIZONTAL WALL, STRAIGHT LINE BLOCKED, GAP-AWARE EXPERT SUCCEEDS
def pick_wall_seed(seeds):
    straight = pd.read_csv(ROOT / "results/reference" / "wall_gap_straight" / "records.csv")
    expert = pd.read_csv(ROOT / "results/reference" / "wall_gap_gap_expert" / "records.csv")
    m = straight.merge(expert, on=["scenario", "seed"], suffixes=("_s", "_e"))
    m = m[(m.scenario == WALL_SCENARIO) & (m.reached_s == 0) & (m.reached_e == 1) & (m.wall_orientation_s == "horizontal")]
    order = {s: i for i, s in enumerate(seeds)}
    m = m[m.seed.isin(order)].assign(order=lambda d: d.seed.map(order)).sort_values("order")

    if m.empty:
        raise RuntimeError("no suitable wall world among the evaluation seeds")

    return int(m.seed.iloc[0])


#RENDER THE TWO ENVIRONMENT SNAPSHOTS (STAGE-6 WORLD, WALL-GAP WORLD) FOR FIG. 1
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="compact")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    preset = get_preset(args.preset)
    condition = get_condition("full")
    facts = {"preset": args.preset}

    scenario = resolve_scenarios(preset)[STAGE_SCENARIO]
    env = make_scenario_env(condition, scenario, preset, preset["action_repeat"])
    try:
        seed, preferred = pick_stage_world(env, scenario, scenario_episode_seeds(STAGE_SCENARIO, 0, 200))
        image, info = snapshot(env, scenario["env_cfg"]["world_scale"])
        image.save(out / "env_stage6.png")
        facts["stage6"] = {"scenario": STAGE_SCENARIO, "seed": seed, "blocker_on_line": True,
                           "full_s0_reached_and_degenerate_s0_failed": bool(preferred), **info}
    finally:
        safe_close(env)

    scenario = wall_gap_scenarios(preset)[WALL_SCENARIO]
    env = make_scenario_env(condition, scenario, preset, preset["action_repeat"])
    try:
        seed = pick_wall_seed(scenario_episode_seeds(WALL_SCENARIO, 0, 200))
        reset_scenario_episode(env, scenario, seed)
        layout = dict(env.eval_layout)
        image, info = snapshot(env, scenario["env_cfg"]["world_scale"])
        image.save(out / "env_wallgap.png")
        facts["wall_gap"] = {"scenario": WALL_SCENARIO, "seed": seed, **info,
                             **{k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in layout.items()
                                if k in ("wall_orientation", "wall_gap_mode", "wall_coverage", "wall_opening_m")}}
    finally:
        safe_close(env)

    (out / "env_snapshots.json").write_text(json.dumps(facts, indent=2, default=float), encoding="utf-8")
    print(json.dumps(facts, indent=2, default=float))


if __name__ == "__main__":
    main()
