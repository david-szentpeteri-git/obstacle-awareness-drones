# -*- coding: utf-8 -*-
#NEWLY ADDED FOR CS5478 PROJECT
"""
Environment construction shared by training and evaluation, so all four
conditions are always built through one code path:

    DroneSwarm3D (PettingZoo ParallelEnv)
      -> pettingzoo_env_to_vec_env_v1
      -> concat_vec_envs_v1            (N_ENVS copies in parallel)
      -> SB3VecResetTupleFix           (gymnasium/SB3 API bridge)
      -> MaskObservations              (condition's ablated slices -> 0)
      -> VecMonitor
      -> VecNormalize                  (obs only)
      -> HandleInactiveDrones          (re-zero inactive slots post-normalize)
"""
from __future__ import annotations

from typing import Optional, Tuple

import supersuit as ss
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor

from config.experiment import Condition, FULL_OBS_DIM
from config.ppo_defaults import CLIP_OBS, GAMMA, N_ENVS
from drone_env.env_3d import DroneSwarm3D
from training.wrappers.action_repeat import RepeatedActions
from drone_env.env_reset_fix import SB3VecResetTupleFix
from training.wrappers.expert_blend import VecExpertBlend
from drone_env.handle_inactive_drones import HandleInactiveDrones
from training.wrappers.obs_mask import MaskObservations

#CURRICULUM STAGE TUPLE LAYOUT
StageCfg = Tuple[float, int, int, int, int, int, bool]


#UNPACK A CURRICULUM STAGE INTO A NAMED DICT
def stage_to_kwargs(stage: StageCfg) -> dict:
    world_scale, num_drones, num_static, num_dynamic, steps, horizon, lidar = stage

    return {
        "world_scale": float(world_scale),
        "num_drones": int(num_drones),
        "num_static_obs": int(num_static),
        "num_dynamic_obs": int(num_dynamic),
        "stage_steps": int(steps),
        "episode_horizon": int(horizon),
        "stage_lidar": bool(lidar),
    }


#RAW PETTINGZOO ENV CARRYING THIS CONDITION'S REWARD PROFILE & LIDAR SETTING
def make_raw_env(
    condition: Condition,
    world_scale: float,
    num_drones: int,
    num_static_obs: int,
    num_dynamic_obs: int,
    episode_horizon: int,
    stage_lidar: bool = True,
    render_mode: Optional[str] = None,
    seed: Optional[int] = None,
    action_repeat: int = 1,
    blocking_obstacle_prob: float = 0.0,
    obstacle_layout: str = "legacy",
    obstacle_size_range=(0.8, 1.2),
    dynamic_speed_range=(0.25, 0.5),
    dynamic_travel_range=(0.35, 0.75),
    obstacle_dt: Optional[float] = None,
    wall_coverage=None,
) -> DroneSwarm3D:

    #RAY CASTING FOLLOWS THE CONDITION/STAGE FLAGS. BLIND STUDENTS' RAYS ARE
    #MASKED LATER; THEIR RAW RAYS REMAIN AVAILABLE TO THE SHARED TEACHER.
    lidar_enabled = bool(stage_lidar) and condition.use_lidar

    #CAPACITY = ACTIVE COUNT (SINGLE-DRONE CURRICULUM -> NO WASTED VEC SLOTS)
    return RepeatedActions(
        world_scale=world_scale,
        num_drones_per_swarm=num_drones,
        num_static_obstacles=num_static_obs,
        num_dynamic_obstacles=num_dynamic_obs,
        max_steps=episode_horizon,
        max_drones_per_swarm=num_drones,
        lidar=lidar_enabled,
        render_mode=render_mode,
        seed=seed,
        reward_profile=condition.reward,
        action_repeat=action_repeat,
        blocking_obstacle_prob=blocking_obstacle_prob,
        obstacle_layout=obstacle_layout,
        obstacle_size_range=obstacle_size_range,
        dynamic_speed_range=dynamic_speed_range,
        dynamic_travel_range=dynamic_travel_range,
        obstacle_dt=obstacle_dt,
        wall_coverage=wall_coverage,
    )


#WALK A WRAPPER CHAIN DOWN TO SUPERSUIT'S CONCAT LAYER
def unwrap_concat_vec_env(env):
    while env is not None:
        if type(env).__name__ == "ConcatVecEnv":
            return env

        env = getattr(env, "venv", None)

    return None


#SEED EVERY ENV COPY ONTO ITS OWN REPRODUCIBLE WORLD STREAM
def seed_vec_env(vec, seed: int) -> bool:
    """
    SB3's usual route (``PPO(seed=...)`` -> ``VecEnv.seed()``) raises
    AttributeError on this stack: SuperSuit's ConcatVecEnv is a gymnasium
    VectorEnv and gymnasium dropped ``seed()``. Its ``reset(seed=s)`` does
    work, and hands copy i the seed ``s + i``, so that is the one path that
    seeds all copies onto distinct, reproducible streams.

    DroneSwarm3D.reset only rebuilds its RNG when a seed is passed, so this
    single seeded reset fixes the stream for the whole stage: SB3's own
    seedless reset at ``learn()`` start keeps it and simply advances it,
    which is what keeps episodes varied instead of replayed.
    """
    concat = unwrap_concat_vec_env(vec)

    if concat is None:
        raise RuntimeError("no ConcatVecEnv in wrapper chain; refusing an unseeded run")

    concat.reset(seed=int(seed))

    return True


#REFUSE TO RUN IF THE "PARALLEL" COPIES SHARE ONE PHYSICS WORLD
def assert_envs_are_independent(vec, num_envs: int) -> None:
    """
    SuperSuit builds its extra env copies with cloudpickle. That duplicates the
    Python object graph, but a PyBullet connection is process-global state
    identified by an integer client id, so every copy silently inherits the
    SAME client - and therefore the same world and the same drone body. With
    num_envs=8 that produced eight views of ONE simulation, each applying its
    own velocity command to the same drone, while per-copy episode bookkeeping
    carried on as if they were independent. It inflated the logged goal-reach
    rate roughly 30 points above the truth and corrupted every transition PPO
    learned from.

    Unpickling never calls __init__, so DroneSwarm3D never gets a fresh
    aviary, and its own _pybullet_client_alive/_rebuild_aviary recovery never
    fires because client 0 genuinely is alive - just not exclusively.

    This check turns that silent corruption into a loud failure.
    """
    if num_envs <= 1:
        return

    concat = unwrap_concat_vec_env(vec)

    if concat is None:
        raise RuntimeError("cannot inspect vector copies for independent physics clients")

    ids = []

    for sub in getattr(concat, "vec_envs", []):
        raw = getattr(sub, "par_env", sub)
        cid = getattr(raw, "_client_id", None)

        if cid is not None:
            ids.append(cid)

    if len(ids) != num_envs or len(set(ids)) != len(ids):
        raise RuntimeError(
            f"env copies do not own {num_envs} distinct PyBullet clients: {ids}. "
            "Train with --n-envs 1 and use run_experiment.py --parallel "
            "for independent training processes."
        )


#FULL SB3-READY VEC ENV FOR ONE CURRICULUM STAGE
def make_vec_env(
    condition: Condition,
    world_scale: float,
    num_drones: int,
    num_static_obs: int,
    num_dynamic_obs: int,
    episode_horizon: int,
    stage_lidar: bool = True,
    num_envs: int = N_ENVS,
    render_mode: Optional[str] = None,
    seed: Optional[int] = None,
    action_repeat: int = 1,
    blocking_obstacle_prob: float = 0.0,
    obstacle_layout: str = "legacy",
    obstacle_size_range=(0.8, 1.2),
    dynamic_speed_range=(0.25, 0.5),
    dynamic_travel_range=(0.35, 0.75),
    obstacle_dt: Optional[float] = None,
    blend_expert=None,
    blend_beta: float = 0.0,
    wall_coverage=None,
):
    if num_envs != 1 or num_drones != 1:
        raise ValueError("the verified training stack requires one environment and one drone")

    env = make_raw_env(
        condition=condition,
        world_scale=world_scale,
        num_drones=num_drones,
        num_static_obs=num_static_obs,
        num_dynamic_obs=num_dynamic_obs,
        episode_horizon=episode_horizon,
        stage_lidar=stage_lidar,
        render_mode=render_mode,
        seed=seed,
        action_repeat=action_repeat,
        blocking_obstacle_prob=blocking_obstacle_prob,
        obstacle_layout=obstacle_layout,
        obstacle_size_range=obstacle_size_range,
        dynamic_speed_range=dynamic_speed_range,
        dynamic_travel_range=dynamic_travel_range,
        obstacle_dt=obstacle_dt,
        wall_coverage=wall_coverage,
    )

    #A GAP-AWARE TEACHER READS THE EPISODE'S WALL LAYOUT STRAIGHT FROM THE RAW ENV
    if blend_expert is not None and hasattr(blend_expert, "attach_env"):
        blend_expert.attach_env(env)

    try:
        vec = ss.pettingzoo_env_to_vec_env_v1(env)
        vec = ss.concat_vec_envs_v1(vec, num_envs, base_class="stable_baselines3")

        #COPIES MUST OWN SEPARATE PHYSICS WORLDS
        assert_envs_are_independent(vec, num_envs)

        #DIRECT CONCAT RESET SEEDS ALL COPIES
        if seed is not None:
            seed_vec_env(vec, seed)
    except Exception:
        safe_close(env)
        raise

    vec = SB3VecResetTupleFix(vec)

    #BELOW THE MASK & NORMALIZATION: THE SHARED TEACHER READS THE RAW, UNMASKED
    #OBSERVATION IN METRES IN EVERY CONDITION (FULL-INFORMATION EXPERT). THE
    #STUDENT IS STILL MASKED BY THE NEXT WRAPPER EXACTLY AS ITS CONDITION SAYS.
    if blend_expert is not None:
        vec = VecExpertBlend(vec, blend_expert, beta=blend_beta)

    vec = MaskObservations(vec, condition.masked_slices, FULL_OBS_DIM)
    vec = VecMonitor(vec)

    vec = VecNormalize(
        vec,
        gamma=GAMMA,
        norm_obs=True,
        norm_reward=False,
        clip_obs=CLIP_OBS,
    )

    vec = HandleInactiveDrones(
        vec,
        active_drones=num_drones,
        max_drones=num_drones,
    )

    return vec


#WALK THE WRAPPER CHAIN FOR THE VECNORMALIZE LAYER
def unwrap_vec_normalize(env) -> Optional[VecNormalize]:
    while env is not None:
        if isinstance(env, VecNormalize):
            return env

        env = getattr(env, "venv", None)

    return None


#CARRY RUNNING OBS STATISTICS ACROSS A CURRICULUM STAGE SWAP
def copy_vec_normalize_stats(src_env, dst_env) -> None:
    src_vn = unwrap_vec_normalize(src_env)
    dst_vn = unwrap_vec_normalize(dst_env)

    if src_vn is None or dst_vn is None:
        return

    dst_vn.obs_rms = src_vn.obs_rms
    dst_vn.ret_rms = src_vn.ret_rms


#CLOSE AN ENV WITHOUT LETTING TEARDOWN ERRORS KILL TRAINING
def safe_close(env) -> None:
    if env is None:
        return

    try:
        env.close()
    except Exception:
        pass
