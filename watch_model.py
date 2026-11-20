# -*- coding: utf-8 -*-
#NEWLY ADDED FOR CS5478 PROJECT
"""
Watch a trained policy fly in the PyBullet window.

    python watch_model.py                                      full, seed 0, final model, stage-6 worlds
    python watch_model.py --condition degenerate --seed 1 --stage 3
    python watch_model.py --preset phase2 --checkpoint stage_2 --wall 0.5
    python watch_model.py --controller expert --wall 0.7       the scripted teacher instead of a model
    python watch_model.py --no-gui --episodes 20               same episodes without a window (setup check)

The worlds are the evaluation worlds: same generators, same seed band and the
same episode runner as evaluation/evaluate.py, so what the window shows is what
the records measure. The six rays are drawn from the drone every few decisions,
red where they hit an obstacle or the arena boundary. Flight runs in real time
(--speed changes the factor) and the window closes after the last episode.
"""
from __future__ import annotations

import argparse

import pybullet as p
from stable_baselines3 import PPO

from config.experiment import CONDITION_ORDER, apply_preset_to_condition, get_condition
from config.ppo_defaults import get_preset
from evaluation.evaluate import load_vec_normalize, make_scenario_env, resolve_artifacts, run_episode, scenario_episode_seeds
from evaluation.reference_eval import CONTROLLERS, make_controller
from evaluation.scenarios import resolve_scenarios
from evaluation.wall_gap import WALL_COVERAGES, WALL_SCALES, wall_gap_scenarios
from training.env_factory import safe_close


#PARSE CLI ARGS
def parse_args():
    parser = argparse.ArgumentParser(description="Watch a trained policy (or a scripted controller) fly in the PyBullet window")
    parser.add_argument("--preset", default="compact", choices=["compact", "phase2"],
                        help="which models: compact = the reported study, phase2 = after the wall training")
    parser.add_argument("--condition", default="full", choices=CONDITION_ORDER)
    parser.add_argument("--seed", type=int, default=0, help="training seed of the run (0, 1 or 2)")
    parser.add_argument("--checkpoint", default=None, help="checkpoint instead of the final model (phase2: stage_2 or stalled; default stage_2)")
    parser.add_argument("--controller", default=None, choices=CONTROLLERS, help="fly a scripted controller instead of a learned model")
    parser.add_argument("--stage", type=int, default=6, help="curriculum stage whose unseen worlds to fly (0 = empty arena, 6 = densest)")
    parser.add_argument("--wall", type=float, default=None, choices=WALL_COVERAGES, help="wall-gap worlds at this coverage instead of a stage")
    parser.add_argument("--scale", type=float, default=2.0, choices=WALL_SCALES, help="arena scale of the wall-gap worlds (2 = training size, 3 = larger)")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--world-seed", type=int, default=0, help="base seed of the evaluation worlds (0 = the reported ones)")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed factor (1 = real time)")
    parser.add_argument("--no-gui", action="store_true", help="no window; fly the episodes and print the outcomes only")

    return parser.parse_args()


#SCENARIO TO FLY: THE UNSEEN WORLDS OF ONE CURRICULUM STAGE, OR ONE WALL-GAP SETTING
def pick_scenario(args, preset):
    if args.wall is None:
        suite = resolve_scenarios(get_preset("compact"))
        name = next((key for key, scenario in suite.items() if scenario["stage_index"] == args.stage), None)

        if name is None:
            raise SystemExit(f"stage {args.stage} does not exist; choose 0 to {len(suite) - 1}")

        return name, suite[name]

    name = f"wall_s{int(args.scale)}_x{int(round(args.wall * 100)):02d}"

    return name, wall_gap_scenarios(preset)[name]


#LEARNED MODEL OR SCRIPTED CONTROLLER, WITH THE CONDITION ITS INPUT IS BUILT FOR
def load_policy(args, preset, world_scale):
    if args.controller:
        condition = apply_preset_to_condition(get_condition("full"), preset)

        return make_controller(args.controller, world_scale), None, condition, f"scripted {args.controller}"

    condition = apply_preset_to_condition(get_condition(args.condition), preset)
    checkpoint = args.checkpoint or ("stage_2" if args.preset == "phase2" else None)
    model_path, norm_path = resolve_artifacts(condition, args.seed, preset["models_dir"], preset=preset, checkpoint=checkpoint)
    label = f"{args.preset} {args.condition} seed {args.seed} ({checkpoint or 'final'})"

    return PPO.load(str(model_path), device="cpu"), load_vec_normalize(norm_path), condition, label


#POINT THE CAMERA AT THE ARENA & HIDE THE SIDE PANELS
def set_camera(env, world_scale):
    env._call_gui(p.configureDebugVisualizer, p.COV_ENABLE_GUI, 0)
    env._call_gui(p.resetDebugVisualizerCamera, cameraDistance=2.4 * world_scale, cameraYaw=50, cameraPitch=-35,
                  cameraTargetPosition=[0.0, 0.0, world_scale / 2])


#ONE EPISODE'S OUTCOME IN WORDS
def outcome(record):
    if record["reached"]:
        return f"reached the goal in {record['time_to_goal_s']:.1f} s"
    if record["collided"]:
        return f"hit {'an obstacle' if record['obstacle_hit'] else 'the arena boundary'} after {record['time_s']:.1f} s"

    return f"timed out after {record['time_s']:.1f} s"


#FLY THE EPISODES & PRINT WHAT HAPPENED
def main():
    args = parse_args()
    preset = get_preset(args.preset)
    name, scenario = pick_scenario(args, preset)
    world_scale = scenario["env_cfg"]["world_scale"]
    policy, vec_norm, condition, label = load_policy(args, preset, world_scale)
    env = make_scenario_env(condition, scenario, preset, preset["action_repeat"], None if args.no_gui else "human")
    env.show_rays = True
    env.pace_seconds = 0.0 if args.no_gui else preset["action_repeat"] / float(env._env.CTRL_FREQ) / args.speed
    scripted = getattr(policy, "policy", None)

    if hasattr(scripted, "attach_env"):
        scripted.attach_env(env)
    if not args.no_gui:
        set_camera(env, world_scale)

    seeds = scenario_episode_seeds(name, args.world_seed, args.episodes)
    reached = 0
    print(f"{label} on {name}, {args.episodes} episodes", flush=True)

    try:
        for idx, seed in enumerate(seeds, start=1):
            record, _traj = run_episode(env, policy, condition, vec_norm, scenario, seed)
            reached += record["reached"]
            print(f"  episode {idx}/{args.episodes}, world {seed}: {outcome(record)}", flush=True)
    finally:
        safe_close(env)

    print(f"reached the goal in {reached} of {args.episodes} episodes")


if __name__ == "__main__":
    main()
