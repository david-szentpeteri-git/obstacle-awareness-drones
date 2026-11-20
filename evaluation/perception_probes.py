# -*- coding: utf-8 -*-
#NEWLY ADDED FOR CS5478 PROJECT
"""
Perception probes on the trained models (evaluation only; plans and
predictions in verification.md, "Perception experiments - plan").

  p2        replay the held-out worlds (reset only, no flight): blocker flag,
            straight-line obstruction, replay validated against the recorded
            start/goal positions
  p3-collect fly one model on held-out worlds and store every raw observation
  p3-analyse action change under blanked / shuffled obstacle inputs
  p4        wall-gap worlds: can any ray see the opening before the wall?
  p5-stats  how far outside its later inputs the "no obstacle" placeholder lies
  p5-traj   empty-world trajectories of one model: overshoot vs drift

P1 (v1 / v2 input variants) and the P5 substitutes run through
evaluation.evaluate --probe (see evaluation/obs_variants.py) and the parallel
launcher run_wall_gap_eval.py.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybullet as p
from stable_baselines3 import PPO

from drone_env.defaults import LIDAR_RANGE, POS_SLICE
from config.experiment import apply_preset_to_condition, get_condition
from config.ppo_defaults import get_preset
from training.env_factory import safe_close
from evaluation.evaluate import (
    load_vec_normalize, make_scenario_env, prepare_obs, resolve_artifacts, run_episode, scenario_episode_seeds,
)
from evaluation.obs_variants import (
    GOAL_SLICE, NO_OBSTACLE_PLACEHOLDER, OBS_VEC_SLICE, RAY_SLICE, RAY_START_OFFSET, body_classes, cast_rays,
    ray_geometry,
)
from evaluation.scenarios import reset_scenario_episode, resolve_scenarios
from evaluation.wall_gap import WALL_THICKNESS, sample_wall_layout, wall_gap_scenarios
from training.wrappers.lidar_layout import RAY_INDEX

RAY_NAMES = {idx: name for name, idx in RAY_INDEX.items()}
DRONE_HALF_SPAN = 0.07
NEAR_RAY_FRAC = 0.35
P3_VARIANTS = ("rays_blank", "obsvec_placeholder", "both_blank", "rays_shuffled", "goal_control")


#STRAIGHT START->GOAL SEGMENT OBSTRUCTED BY AN OBSTACLE BODY? (5 PARALLEL RAYS, DRONE HALF-SPAN)
def line_obstructed(env, start: np.ndarray, goal: np.ndarray):
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    axis = goal - start
    length = float(np.linalg.norm(axis))

    if length < 1e-6:
        return False, float("nan")

    unit = axis / length
    helper = np.array([0.0, 0.0, 1.0]) if abs(unit[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    side = np.cross(unit, helper)
    side /= np.linalg.norm(side)
    up = np.cross(unit, side)
    offsets = [np.zeros(3), DRONE_HALF_SPAN * side, -DRONE_HALF_SPAN * side, DRONE_HALF_SPAN * up, -DRONE_HALF_SPAN * up]
    froms = [(start + RAY_START_OFFSET * unit + o).tolist() for o in offsets]
    tos = [(goal + o).tolist() for o in offsets]
    results = env._call_physics(p.rayTestBatch, rayFromPositions=froms, rayToPositions=tos)
    obstacles, _ = body_classes(env)
    hits = [float(hit[2]) * (length - RAY_START_OFFSET) for hit in results if int(hit[0]) in obstacles]

    return bool(hits), (min(hits) if hits else float("nan"))


#DISTANCE FROM A POINT TO THE START->GOAL SEGMENT
def point_segment_distance(point, start, goal) -> float:
    axis = goal - start
    t = float(np.clip(np.dot(point - start, axis) / max(np.dot(axis, axis), 1e-12), 0.0, 1.0))

    return float(np.linalg.norm(point - (start + t * axis)))


#P2: REPLAY EVERY HELD-OUT WORLD AT RESET AND LABEL IT
def p2_replay(args) -> None:
    preset = get_preset(args.preset)
    scenarios = resolve_scenarios(preset)
    rows = list(csv.DictReader(open(args.records, encoding="utf-8")))
    worlds = {}

    for r in rows:
        key = (r["scenario"], int(r["seed"]))
        pose = np.array([float(r[k]) for k in ("start_x", "start_y", "start_z", "goal_x", "goal_y", "goal_z")])

        if key in worlds and np.max(np.abs(worlds[key] - pose)) > 1e-4:
            raise RuntimeError(f"recorded start/goal differ between models for world {key}")

        worlds[key] = pose

    env_cond = apply_preset_to_condition(get_condition("full"), preset)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["scenario", "seed", "blocker_requested", "blocker_spawned", "static_obstacles", "dynamic_obstacles",
              "line_obstructed", "first_obstacle_m", "min_centre_to_line_m", "straight_dist", "replay_error_m"]
    checked = 0

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()

        for name in sorted({k[0] for k in worlds}, key=lambda n: int(n[5])):
            scenario = scenarios[name]
            env = make_scenario_env(env_cond, scenario, preset, preset["action_repeat"])

            try:
                for seed in sorted(k[1] for k in worlds if k[0] == name):
                    reset_scenario_episode(env, scenario, seed)
                    spawn = np.asarray(env.spawn_positions[0], dtype=np.float64)
                    goal = np.asarray(env._targets[0].start_pos, dtype=np.float64)
                    err = float(np.max(np.abs(np.concatenate([spawn, goal]) - worlds[(name, seed)])))

                    if err > 1e-3:
                        raise RuntimeError(f"replay of {name} seed {seed} differs from the records by {err:.4f} m")

                    counts = getattr(env, "_episode_obstacle_counts", {}) or {}
                    obstructed, first_hit = line_obstructed(env, spawn, goal)
                    centres = [np.array(env._call_physics(p.getBasePositionAndOrientation, o.body_id)[0]) for o in env._obstacles]
                    min_centre = min((point_segment_distance(c, spawn, goal) for c in centres), default=float("nan"))
                    writer.writerow(dict(
                        scenario=name, seed=seed,
                        blocker_requested=int(bool(counts.get("blocking_obstacle_requested", False))),
                        blocker_spawned=int(bool(counts.get("blocking_obstacle_spawned", False))),
                        static_obstacles=len([o for o in env._obstacles if not hasattr(o, "axis")]),
                        dynamic_obstacles=len([o for o in env._obstacles if hasattr(o, "axis")]),
                        line_obstructed=int(obstructed), first_obstacle_m=round(first_hit, 3),
                        min_centre_to_line_m=round(min_centre, 3), straight_dist=round(float(np.linalg.norm(goal - spawn)), 3),
                        replay_error_m=round(err, 5),
                    ))
                    checked += 1
            finally:
                safe_close(env)

            print(f"  [p2] {name}: {checked} worlds replayed so far", flush=True)

    print(f"[SAVED] {out_path} ({checked} worlds, all replays matched the records)")


#LOAD ONE TRAINED RUN'S MODEL + NORMALISATION
def load_run(condition_name: str, train_seed: int, preset: dict):
    cond = apply_preset_to_condition(get_condition(condition_name), preset)
    model_path, norm_path = resolve_artifacts(cond, train_seed, preset["models_dir"], preset=preset)

    return cond, PPO.load(str(model_path), device="cpu"), load_vec_normalize(norm_path)


#P3 COLLECT: FLY ONE MODEL, STORE EVERY RAW OBSERVATION + OBSTACLE-ONLY SCAN
def p3_collect(args) -> None:
    preset = get_preset(args.preset)
    scenarios = resolve_scenarios(preset)
    cond, model, vec_norm = load_run(args.condition, args.train_seed, preset)
    names = [s.strip() for s in args.scenarios.split(",")]
    obs_rows, scans, scen_idx, ep_idx, step_idx, outcomes = [], [], [], [], [], []
    agent = None

    for si, name in enumerate(names):
        scenario = scenarios[name]
        env = make_scenario_env(cond, scenario, preset, preset["action_repeat"])
        agent = env.possible_agents[0]

        try:
            for ei, seed in enumerate(scenario_episode_seeds(name, 0, 200)[:args.worlds]):
                obs_dict = reset_scenario_episode(env, scenario, seed)
                ticks, step = 0, 0

                while True:
                    raw = np.array(obs_dict[agent], dtype=np.float32)
                    obs_rows.append(raw)
                    scans.append(np.array(env._lidar_obstacle_scans[0], dtype=np.float32))
                    scen_idx.append(si)
                    ep_idx.append(ei)
                    step_idx.append(step)
                    act, _ = model.predict(prepare_obs(raw[None, :], cond, vec_norm), deterministic=True)
                    obs_dict, _r, term, trunc, info = env.step({agent: act[0]})
                    ticks += int(info[agent].get("simulation_steps", 1))
                    step += 1

                    if term[agent] or trunc[agent] or ticks >= env.max_steps:
                        break

                outcomes.append((si, ei, int(env._reached_once[0]), int(env._wall_collision_once[0] or env._obstacle_collision_once[0])))
        finally:
            safe_close(env)

        print(f"  [p3] {args.condition} s{args.train_seed} {name}: {len(obs_rows)} observations", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, obs=np.stack(obs_rows), obstacle_scan=np.stack(scans), scenario=np.array(scen_idx), episode=np.array(ep_idx),
             step=np.array(step_idx), outcomes=np.array(outcomes), scenario_names=np.array(names))
    print(f"[SAVED] {out} ({len(obs_rows)} observations)")


#UNIT FLIGHT DIRECTIONS, SPEEDS & PAIRWISE DIRECTION CHANGE (DEGREES)
def direction_change(base: np.ndarray, other: np.ndarray):
    base, other = base.astype(np.float64), other.astype(np.float64)
    unit = lambda a: a[:, :3] / np.maximum(np.linalg.norm(a[:, :3], axis=1, keepdims=True), 1e-8)
    cosine = np.clip(np.sum(unit(base) * unit(other), axis=1), -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine))
    #IDENTICAL ACTIONS ARE A ZERO CHANGE (NOT FLOAT ROUNDING OF arccos NEAR 1)
    angle[np.all(np.abs(base - other) < 1e-6, axis=1)] = 0.0
    speed = lambda a: np.clip((a[:, 3] + 1.0) / 2.0, 0.0, 1.0)

    return angle, np.abs(speed(base) - speed(other))


#P3 ANALYSE: ACTION CHANGE UNDER PERTURBED OBSTACLE INPUTS, PER MODEL
def p3_analyse(args) -> None:
    preset = get_preset(args.preset)
    rng = np.random.default_rng(0)
    rows = []

    for npz_path in sorted(Path(args.collected).glob("*.npz")):
        condition_name, seed_tag = npz_path.stem.rsplit("_s", 1)
        train_seed = int(seed_tag)
        cond, model, vec_norm = load_run(condition_name, train_seed, preset)
        data = np.load(npz_path)
        raw, scan = data["obs"], data["obstacle_scan"]
        near = scan.min(axis=1) < NEAR_RAY_FRAC
        blind = len(cond.masked_slices) > 0
        base_norm = prepare_obs(raw, cond, vec_norm)
        base_act, _ = model.predict(base_norm, deterministic=True)
        variants = {}
        v = raw.copy(); v[:, RAY_SLICE] = 1.0; variants["rays_blank"] = v
        v = raw.copy(); v[:, OBS_VEC_SLICE] = NO_OBSTACLE_PLACEHOLDER; variants["obsvec_placeholder"] = v
        v = raw.copy(); v[:, RAY_SLICE] = 1.0; v[:, OBS_VEC_SLICE] = NO_OBSTACLE_PLACEHOLDER; variants["both_blank"] = v
        v = raw.copy(); v[:, RAY_SLICE] = raw[rng.permutation(len(raw)), RAY_SLICE]; variants["rays_shuffled"] = v
        #CONTROL: SHIFT THE GOAL VECTOR BY THE SAME NORMALISED DISTANCE AS "BOTH BLANK"
        both_norm = prepare_obs(variants["both_blank"], cond, vec_norm)
        delta = np.linalg.norm(both_norm - base_norm, axis=1)
        direction = rng.normal(size=(len(raw), 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        control = base_norm.copy()
        control[:, GOAL_SLICE] += delta[:, None] * direction
        control = np.clip(control, -float(vec_norm.clip_obs), float(vec_norm.clip_obs)).astype(np.float32)

        for variant in P3_VARIANTS:
            norm = control if variant == "goal_control" else prepare_obs(variants[variant], cond, vec_norm)
            act, _ = model.predict(norm, deterministic=True)
            angle, dspeed = direction_change(base_act, act)

            if blind and variant != "goal_control" and not np.allclose(act, base_act, atol=1e-6):
                raise RuntimeError(f"blind model {condition_name}_s{train_seed} reacted to masked inputs ({variant})")

            for split, mask in (("all", np.ones(len(raw), bool)), ("near", near), ("far", ~near)):
                if mask.sum() == 0:
                    continue

                rows.append(dict(condition=condition_name, train_seed=train_seed, variant=variant, split=split, n=int(mask.sum()),
                                 mean_angle_deg=round(float(angle[mask].mean()), 3), median_angle_deg=round(float(np.median(angle[mask])), 3),
                                 p90_angle_deg=round(float(np.percentile(angle[mask], 90)), 3),
                                 frac_angle_gt10=round(float((angle[mask] > 10).mean()), 4), mean_dspeed=round(float(dspeed[mask].mean()), 4),
                                 mean_norm_shift=round(float(np.linalg.norm(norm - base_norm, axis=1)[mask].mean()), 4)))

        print(f"  [p3] {condition_name} s{train_seed}: {len(raw)} observations, near {int(near.sum())}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[SAVED] {out}")


#P4: FIRST PROBE POINT ALONG THE STRAIGHT PATH FROM WHICH A RAY PASSES THE WALL PLANE
def gap_visibility(env, layout: dict, step: float = 0.1, stop: float = 0.3, margin: float = 0.05) -> dict:
    axis = "xy".index(layout["travel_axis"])
    spawn = np.asarray(layout["spawn"], dtype=np.float64)
    goal = np.asarray(layout["goal"], dtype=np.float64)
    unit = (goal - spawn) / np.linalg.norm(goal - spawn)
    toward = 1.0 if layout["wall_t"] > spawn[axis] else -1.0
    wall_face = float(layout["wall_t"]) - toward * WALL_THICKNESS / 2
    did = env._drone_ids[0]
    probes = 0
    k = 0

    while True:
        pos = spawn + k * step * unit
        dist_to_face = (wall_face - pos[axis]) * toward

        if dist_to_face < stop:
            break

        env._call_physics(p.resetBasePositionAndOrientation, did, pos.tolist(), [0.0, 0.0, 0.0, 1.0])
        env._call_physics(p.resetBaseVelocity, did, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        _origin, _goal, dirs, ray_from = ray_geometry(env)
        frac, _body = cast_rays(env, ray_from, dirs)
        probes += 1

        for i in range(len(dirs)):
            along = float(dirs[i, axis]) * toward

            if along <= 1e-6:
                continue

            t_face = (wall_face - float(ray_from[i, axis])) * toward / along

            if t_face < 0 or t_face > LIDAR_RANGE:
                continue
            if float(frac[i]) * LIDAR_RANGE > t_face + margin:
                return dict(visible=True, dist_to_wall=round(float(dist_to_face), 3), ray=RAY_NAMES[i], probes=probes)

        k += 1

    return dict(visible=False, dist_to_wall=float("nan"), ray="", probes=probes)


#P4: LABEL EVERY WALL-GAP WORLD WITH GAP VISIBILITY ALONG THE STRAIGHT PATH
def p4_gap_visibility(args) -> None:
    from analysis.report_figures import line_blocked

    preset = get_preset(args.preset)
    scenarios = wall_gap_scenarios(preset)
    env_cond = apply_preset_to_condition(get_condition("full"), preset)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["scenario", "seed", "world_scale", "wall_orientation", "wall_gap_mode", "wall_coverage", "line_blocked",
              "gap_visible", "first_visible_dist_to_wall", "first_visible_ray", "probes"]
    total = 0

    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()

        for name, scenario in scenarios.items():
            env = make_scenario_env(env_cond, scenario, preset, preset["action_repeat"])

            try:
                for seed in scenario_episode_seeds(name, 0, args.episodes):
                    reset_scenario_episode(env, scenario, seed)
                    layout = sample_wall_layout(scenario["env_cfg"]["world_scale"], scenario["coverage"], np.random.default_rng(seed))
                    rec = dict(env.eval_layout)
                    rec.update({f"start_{a}": float(v) for a, v in zip("xyz", layout["spawn"])})
                    rec.update({f"goal_{a}": float(v) for a, v in zip("xyz", layout["goal"])})
                    blocked = line_blocked(SimpleNamespace(**rec))
                    vis = gap_visibility(env, layout)
                    writer.writerow(dict(scenario=name, seed=seed, world_scale=rec["world_scale"], wall_orientation=rec["wall_orientation"],
                                         wall_gap_mode=rec["wall_gap_mode"], wall_coverage=rec["wall_coverage"], line_blocked=int(blocked),
                                         gap_visible=int(vis["visible"]), first_visible_dist_to_wall=vis["dist_to_wall"],
                                         first_visible_ray=vis["ray"], probes=vis["probes"]))
                    total += 1
            finally:
                safe_close(env)

            print(f"  [p4] {name}: {total} worlds labelled so far", flush=True)

    print(f"[SAVED] {out} ({total} worlds)")


#P5 STATS: NORMALISED PLACEHOLDER VS THE NEAREST-OBSTACLE INPUTS SEEN AT STAGE 6
def p5_stats(args) -> None:
    import pickle

    preset = get_preset(args.preset)
    rows = []

    for condition_name in ("sensing_only", "full"):
        for train_seed in range(3):
            run_dir = Path(preset["models_dir"]) / f"ppo_{condition_name}_s{train_seed}"
            vec_norm = pickle.load(open(run_dir / "final_vecnormalize.pkl", "rb"))
            mean, std = vec_norm.obs_rms.mean[OBS_VEC_SLICE], np.sqrt(vec_norm.obs_rms.var[OBS_VEC_SLICE])
            z_placeholder = (NO_OBSTACLE_PLACEHOLDER - mean) / std
            npz = Path(args.collected) / f"{condition_name}_s{train_seed}.npz"
            data = np.load(npz)
            names = list(data["scenario_names"])
            stage6 = data["obs"][data["scenario"] == names.index("stage6_6s3d" if "stage6_6s3d" in names else names[-1])]
            z_seen = (stage6[:, OBS_VEC_SLICE] - mean) / std
            dist_seen = np.linalg.norm(stage6[:, OBS_VEC_SLICE], axis=1)
            rows.append(dict(condition=condition_name, train_seed=train_seed, mean_x=round(float(mean[0]), 3), std_x=round(float(std[0]), 3),
                             placeholder_norm_x=round(float(z_placeholder[0]), 3), clip=float(vec_norm.clip_obs),
                             stage6_norm_x_p99=round(float(np.percentile(z_seen[:, 0], 99)), 3), stage6_norm_x_max=round(float(z_seen[:, 0].max()), 3),
                             frac_stage6_above_placeholder=round(float((z_seen[:, 0] >= z_placeholder[0]).mean()), 4),
                             stage6_obstacle_dist_median=round(float(np.median(dist_seen)), 3), stage6_obstacle_dist_p95=round(float(np.percentile(dist_seen, 95)), 3),
                             stage6_obstacle_dist_max=round(float(dist_seen.max()), 3)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[SAVED] {out}")


#P5 TRAJ: EMPTY-WORLD EPISODES OF ONE MODEL, CRASH GEOMETRY (OVERSHOOT VS DRIFT)
def p5_traj(args) -> None:
    preset = get_preset(args.preset)
    scenarios = resolve_scenarios(preset)
    scenario = scenarios[args.scenario]
    cond, model, vec_norm = load_run(args.condition, args.train_seed, preset)
    env = make_scenario_env(cond, scenario, preset, preset["action_repeat"])
    scale = float(scenario["env_cfg"]["world_scale"])
    summary = []

    try:
        for seed in scenario_episode_seeds(args.scenario, 0, 200)[:args.worlds]:
            record, traj = run_episode(env, model, cond, vec_norm, scenario, seed)
            goal = np.array([record[f"goal_{a}"] for a in "xyz"])
            dist = np.linalg.norm(traj - goal, axis=1)
            closest_idx = int(np.argmin(dist))
            end = traj[-1]
            #NEAREST ARENA FACE AT THE FINAL POSITION
            face_dist = {"floor": float(end[2]), "ceiling": float(scale - end[2]), "side": float(scale - max(abs(end[0]), abs(end[1])))}
            boundary = min(face_dist, key=face_dist.get)
            summary.append(dict(seed=seed, reached=record["reached"], collided=record["collided"], timeout=record["timeout"],
                                closest_to_goal_m=round(float(dist[closest_idx]), 3), closest_at_frac=round(closest_idx / max(len(traj) - 1, 1), 3),
                                crash_boundary=boundary if record["collided"] else "", end_z=round(float(end[2]), 3),
                                overshoot=int(record["collided"] and dist[closest_idx] < 1.0 and closest_idx < len(traj) - 1),
                                steps=record["steps"], straight_dist=record["straight_dist"]))
    finally:
        safe_close(env)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(condition=args.condition, train_seed=args.train_seed, scenario=args.scenario, episodes=summary), indent=2), encoding="utf-8")
    crashed = [s for s in summary if s["collided"]]
    print(f"[SAVED] {out}: {len(summary)} episodes, {sum(s['reached'] for s in summary)} reached, {len(crashed)} crashed "
          f"(overshoot {sum(s['overshoot'] for s in crashed)}; boundaries {sorted({s['crash_boundary'] for s in crashed})}), "
          f"{sum(s['timeout'] for s in summary)} timeouts")


#CLI
def main() -> None:
    parser = argparse.ArgumentParser(description="Perception probes on the trained models")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ("p2", "p3-collect", "p3-analyse", "p4", "p5-stats", "p5-traj"):
        sp = sub.add_parser(name)
        sp.add_argument("--preset", default="compact")
        sp.add_argument("--out", required=True)
        if name == "p2":
            sp.add_argument("--records", default="results/compact_eval_200ep/eval/records.csv")
        if name == "p3-collect":
            sp.add_argument("--condition", required=True)
            sp.add_argument("--train-seed", type=int, required=True)
            sp.add_argument("--scenarios", default="stage3_3s0d,stage6_6s3d")
            sp.add_argument("--worlds", type=int, default=50)
        if name in ("p3-analyse", "p5-stats"):
            sp.add_argument("--collected", required=True, help="folder of <condition>_s<seed>.npz from p3-collect")
        if name == "p4":
            sp.add_argument("--episodes", type=int, default=200)
        if name == "p5-traj":
            sp.add_argument("--condition", required=True)
            sp.add_argument("--train-seed", type=int, required=True)
            sp.add_argument("--scenario", default="stage0_0s0d")
            sp.add_argument("--worlds", type=int, default=40)

    args = parser.parse_args()
    {"p2": p2_replay, "p3-collect": p3_collect, "p3-analyse": p3_analyse, "p4": p4_gap_visibility,
     "p5-stats": p5_stats, "p5-traj": p5_traj}[args.cmd](args)


if __name__ == "__main__":
    main()
