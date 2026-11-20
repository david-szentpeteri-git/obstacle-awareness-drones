# -*- coding: utf-8 -*-
#PREEXISTS, MODIFIED FOR CS5478 PROJECT
from __future__ import annotations

from typing import Dict, Optional, Tuple, List

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from drone_env.createObjects import Target, EnvObject, Dynamic
from drone_env.collisionQuery import CollisionQueryWorld
from drone_env.step_logic import (
    freeze_inactive_actions,
    convert_actions_for_simulator,
    compute_step_state,
    compute_terminations,
    build_info_dict,
)
from drone_env.rewards import compute_rewards
from training.wrappers.lidar_layout import LIDAR_DIM, ray_directions
from drone_env.stats_trackers import EpisodeStatsTracker

from drone_env.defaults import *
from config.experiment import FULL_REWARD


class ObstaclePlacementError(RuntimeError):
    """Requested obstacle geometry could not fit within bounded attempts."""


class DroneSwarm3D(ParallelEnv):
    metadata = {"name": "drone_swarms_3d"}

    # ----------------------------- Initialise Custom 3D Environment -----------------------------

    def __init__(
        self,
        world_scale: float,
        num_drones_per_swarm: int,
        num_static_obstacles: int,
        num_dynamic_obstacles: int,
        max_steps: int,
        max_drones_per_swarm: Optional[int] = None,
        num_swarms: int = 1,
        lidar: bool = False,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        reward_profile=None,
        stats_csv: Optional[str] = None,
        blocking_obstacle_prob: float = 0.0,
        obstacle_layout: str = "legacy",
        obstacle_size_range: Tuple[float, float] = (0.8, 1.2),
        dynamic_speed_range: Tuple[float, float] = (0.25, 0.5),
        dynamic_travel_range: Tuple[float, float] = (0.35, 0.75),
        obstacle_dt: Optional[float] = None,
        wall_coverage=None,
    ):
        # Probability that an episode spawns exactly ONE obstacle on the
        # straight line between drone and goal, instead of the fixed
        # num_static_obstacles count. 0.0 keeps the original behaviour.
        # Used by the single-stage learnability curriculum, where the point is
        # a task that is mostly open flying with an occasional real blocker.
        if not isinstance(blocking_obstacle_prob, (int, float)) or isinstance(blocking_obstacle_prob, bool):
            raise ValueError("blocking_obstacle_prob must be a number")
        if not (0.0 <= float(blocking_obstacle_prob) <= 1.0):
            raise ValueError("blocking_obstacle_prob must be within [0, 1]")

        self.blocking_obstacle_prob = float(blocking_obstacle_prob)
        #PHASE-2 WALL: FRACTION OF THE PASSAGE CROSS-SECTION ONE PANEL COVERS PER EPISODE
        #(FLOAT, OR [lo, hi] DRAWN UNIFORMLY PER EPISODE); None = NO WALL, ORIGINAL BEHAVIOUR
        self.wall_coverage = self._validate_wall_coverage(wall_coverage)
        self.current_wall_layout = None
        self._init_obstacle_settings(
            obstacle_layout, obstacle_size_range, dynamic_speed_range,
            dynamic_travel_range, obstacle_dt,
        )
        self.stats_csv = stats_csv
        self._init_config(
            world_scale=world_scale,
            num_drones_per_swarm=num_drones_per_swarm,
            num_static_obstacles=num_static_obstacles,
            num_dynamic_obstacles=num_dynamic_obstacles,
            max_steps=max_steps,
            max_drones_per_swarm=max_drones_per_swarm,
            num_swarms=num_swarms,
            lidar=lidar,
            render_mode=render_mode,
            reward_profile=reward_profile,
        )

        self._init_rng(seed)
        self._init_agents()
        self._init_episode_state()
        self._init_simulator()
        self._init_query_world()
        self._init_spaces()
        self._init_logging()

    #VALIDATE OPT-IN CURRICULUM GEOMETRY & PHYSICS CLOCK
    def _init_obstacle_settings(self, layout, size_range, speed_range, travel_range, dt):
        if layout not in ("legacy", "curriculum"):
            raise ValueError("obstacle_layout must be 'legacy' or 'curriculum'")

        self.obstacle_layout = layout
        ranges = {
            "obstacle_size_range": size_range,
            "dynamic_speed_range": speed_range,
            "dynamic_travel_range": travel_range,
        }

        for name, bounds in ranges.items():
            if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                raise ValueError(f"{name} must contain two positive ordered values")

            if any(isinstance(val, bool) or not isinstance(val, (int, float)) for val in bounds):
                raise ValueError(f"{name} must contain two positive ordered values")

            low, high = map(float, bounds)

            if not np.isfinite([low, high]).all() or low <= 0 or high < low:
                raise ValueError(f"{name} must contain two positive ordered values")

            setattr(self, name, (low, high))

        resolved_dt = (1.0 / 240.0 if layout == "curriculum" else 1.0 / 60.0) if dt is None else dt

        if isinstance(resolved_dt, bool) or not isinstance(resolved_dt, (int, float)):
            raise ValueError("obstacle_dt must be positive and finite")

        if not np.isfinite(resolved_dt) or resolved_dt <= 0:
            raise ValueError("obstacle_dt must be positive and finite")

        self.obstacle_dt = float(resolved_dt)

    def _init_config(
        self,
        world_scale: float,
        num_drones_per_swarm: int,
        num_static_obstacles: int,
        num_dynamic_obstacles: int,
        max_steps: int,
        max_drones_per_swarm: Optional[int],
        num_swarms: int,
        lidar: bool,
        render_mode: Optional[str],
        reward_profile=None,
    ):
        """Store environment configuration."""
        self.world_scale = float(world_scale)
        self.num_swarms = int(num_swarms)

        # which reward terms are active; defaults to the full obstacle-aware set
        self.reward_profile = reward_profile if reward_profile is not None else FULL_REWARD

        self._p = None

        # active drones for the current curriculum stage
        self.num_drones_per_swarm = int(num_drones_per_swarm)

        # fixed capacity across all curriculum stages
        if max_drones_per_swarm is None:
            max_drones_per_swarm = num_drones_per_swarm
        self.max_drones_per_swarm = int(max_drones_per_swarm)

        if self.max_drones_per_swarm < self.num_drones_per_swarm:
            raise ValueError(
                f"max_drones_per_swarm ({self.max_drones_per_swarm}) "
                f"must be >= num_drones_per_swarm ({self.num_drones_per_swarm})"
            )

        # active count this stage
        # ensure at least one drone in environment
        self.total_num_drones = max(1, self.num_drones_per_swarm * self.num_swarms)

        # fixed count exposed to PettingZoo/SB3
        self.max_total_num_drones = max(1, self.max_drones_per_swarm * self.num_swarms)

        # Spawn position of every drone
        self.spawn_positions = np.empty((0, 3))

        # Dict with {swarm: [position of drone]}
        self.drones_in_swarm = {}

        self.num_static_obstacles = int(num_static_obstacles)
        self.num_dynamic_obstacles = int(num_dynamic_obstacles)
        self.max_steps = int(max_steps)
        self.lidar = bool(lidar)
        self.render_mode = render_mode

    def _init_rng(self, seed: Optional[int]) -> None:
        """Create per-environment random number generator."""
        self.rng = np.random.default_rng(seed)

    def _init_agents(self) -> None:
        """Initialize PettingZoo agents."""
        self.possible_agents = [
            f"drone_{i}" for i in range(self.max_total_num_drones)
        ]  # fixed list of all drone names in env
        self.agents = self.possible_agents[
            :
        ]  # current active list in episode (used to control dead / despawned drones)

    def _init_episode_state(self) -> None:
        """Initialise per-episode bookkeeping."""
        # boolean array representing death status of each drone
        self._dead = np.zeros(self.max_total_num_drones, dtype=np.bool_)

        # boolean array of target reached memory: once reached, stays True until reset
        self._reached_once = np.zeros((self.max_total_num_drones,), dtype=np.bool_)
        self._despawned = np.zeros((self.max_total_num_drones,), dtype=np.bool_)

        # target for each drone; self._targets[i] belongs to drone i
        self._targets: Optional[list[Target]] = None

        # distance of each drone to its target in previous step (used for reward computation)
        self._prev_dists = np.zeros(self.max_total_num_drones, dtype=np.float32)
        # reward value of each drone in previous step
        self._last_reward = np.zeros(self.max_total_num_drones, dtype=np.float32)

        self._wall_collision_once = np.zeros(
            (self.max_total_num_drones,), dtype=np.bool_
        )
        self._obstacle_collision_once = np.zeros(
            (self.max_total_num_drones,), dtype=np.bool_
        )
        self._drone_collision_once = np.zeros(
            (self.max_total_num_drones,), dtype=np.bool_
        )

        self._step_count = 0

    def _init_simulator(self) -> None:
        """Create drone simulator and PyBullet handles."""
        # list of drone's initial position
        self.spawn_positions = np.empty((0, 3), dtype=np.float32)
        self.drones_in_swarm = {}

        for i in range(self.num_swarms):
            edge, center, drones = self.generate_drone_swarm_spawn()
            self.drones_in_swarm[i] = {"edge": edge, "center": center, "drones": drones}
            self.spawn_positions = np.vstack([self.spawn_positions, drones])

        self.spawn_positions = self.spawn_positions.astype(np.float32)

        n_missing = self.max_total_num_drones - self.spawn_positions.shape[0]
        if n_missing > 0:
            dummy_positions = []
            for k in range(n_missing):
                dummy_positions.append([1000.0 + 10.0 * k, 1000.0, -100.0])
            dummy_positions = np.array(dummy_positions, dtype=np.float32)
            # pad rest with dummy positions far away so aviary gets exactly max_total_num_drones initial positions
            self.spawn_positions = np.vstack([self.spawn_positions, dummy_positions])

        self._create_aviary()
        self._targets = self._make_targets()

    def _init_spaces(self) -> None:
        """Create action and observation spaces."""

        # Action: (vx_dir, vy_dir, vz_dir, speed)
        self._action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        # base observations
        # pos(3), vel(3), to_target(3), to_nearest_obstacle(3), to_centroid(3), to_nearest_teammate(3) = 18
        self._base_obs_dim = 18

        # LiDAR observations
        self._lidar_dim = LIDAR_DIM if self.lidar else 0

        # Per-drone LiDAR scan with only real obstacle hits kept; walls, the
        # floor and the target read as "no hit". Consumed by the reward's
        # proximity shaping - see the note in _lidar_scan.
        self._lidar_obstacle_scans: Dict[int, np.ndarray] = {}

        # total observations
        obs_dim = self._base_obs_dim + self._lidar_dim

        self._obs_space = spaces.Box(
            low=np.full((obs_dim,), -1e9, dtype=np.float32),
            high=np.full((obs_dim,), 1e9, dtype=np.float32),
            dtype=np.float32,
        )

    def _init_logging(self) -> None:
        """Create CSV log file if a path was given (None disables logging)."""

        self.stats = EpisodeStatsTracker(self.stats_csv) if self.stats_csv else None

    def _init_pybullet_handles(self) -> None:
        self._gui_client_id = None
        try:
            import pybullet as p

            self._p = p
            self._client_id = None

            for attr in ("_CLIENT", "CLIENT", "client", "_client"):
                if hasattr(self._env, attr):
                    self._client_id = getattr(self._env, attr)
                    break

            if self.render_mode == "human":
                self._gui_client_id = self._client_id

            try:
                self._call_physics(p.setGravity, 0, 0, -9.8)
            except Exception:
                pass

            self._drone_ids = []

            for attr in ("DRONE_IDS", "drone_ids", "_drone_ids"):
                if hasattr(self._env, attr):
                    try:
                        self._drone_ids = list(getattr(self._env, attr))
                        break
                    except Exception:
                        pass

        except Exception:
            pass

    # ----------------------------- PYBULLET CLIENT FUNCTIONS -----------------------------

    def _connected_ids(self) -> List[int]:
        """Scan and return a list of all active PyBullet Client IDs (simulation connections)"""
        if not hasattr(self, "_p") or self._p is None:
            return []
        p = self._p

        ids: List[int] = []

        for i in range(64):
            try:
                if p.isConnected(i):
                    ids.append(int(i))
            except Exception:
                continue

        return ids

    def _pcid_for_physics(self) -> Optional[int]:
        if self._p is None:
            return None

        for attr in ("_CLIENT", "CLIENT", "client", "_client"):
            if hasattr(self._env, attr):
                cid = getattr(self._env, attr)
                try:
                    if cid is not None and self._p.isConnected(cid):
                        return int(cid)
                except Exception:
                    pass

        return None

    def _call_physics(self, fn, *args, **kwargs):
        if self._p is None:
            raise RuntimeError("PyBullet not available")

        pcid = self._pcid_for_physics()
        if pcid is None:
            raise RuntimeError("No active PyBullet physics client found")

        try:
            return fn(*args, **kwargs, physicsClientId=pcid)
        except Exception as e:
            print(
                f"[PHYSICS ERROR] fn={fn.__name__} pcid={pcid} "
                f"args={args} kwargs={kwargs} error={repr(e)}"
            )
            raise

    def _call_gui(self, fn, *args, **kwargs):
        if self._p is None:
            return None
        pcid = self._gui_client_id  # use stored ID, not max()
        try:
            if pcid is not None and "physicsClientId" not in kwargs:
                return fn(*args, **kwargs, physicsClientId=pcid)
            return fn(*args, **kwargs)
        except Exception:
            return None

    def _refresh_drone_ids_from_pybullet(self) -> None:
        """Check PyBullet simulation itself to find the drones - useful if env does not give it."""
        if self._p is None:
            return
        p = self._p

        try:
            num_bodies = self._call_physics(p.getNumBodies)
        except Exception:
            return

        ids: List[int] = []
        for i in range(num_bodies):
            try:
                body_id = self._call_physics(p.getBodyUniqueId, i)
                info = self._call_physics(p.getBodyInfo, body_id)
                name = info[1].decode("utf-8", errors="ignore").lower()
                if (
                    ("crazyflie" in name)
                    or (name.startswith("cf"))
                    or ("drone" in name)
                ):
                    ids.append(int(body_id))
            except Exception:
                continue

        if len(ids) == 0:
            ids = []
            for i in range(num_bodies):
                try:
                    ids.append(int(self._call_physics(p.getBodyUniqueId, i)))
                except Exception:
                    pass

        self._drone_ids = ids

    def _pybullet_client_alive(self) -> bool:
        """Return True if the VelocityAviary's physics client is still connected."""
        try:
            import pybullet as p
        except Exception:
            return False

        cid = getattr(self._env, "CLIENT", None)
        if cid is None:
            cid = getattr(self._env, "_CLIENT", None)

        if cid is None:
            return False

        try:
            return bool(p.isConnected(cid))
        except Exception:
            return False

    def _rebuild_aviary(self) -> None:
        """Recreate VelocityAviary from scratch (fresh PyBullet connection)."""
        try:
            self._env.close()
        except Exception:
            pass

        self._create_aviary()

    #CLEAR LOW-LEVEL PID MEMORY (LAST RPY, INTEGRAL ERRS, CTRL COUNTER) - AVIARY
    #RESET LEAVES IT INTACT --> PREV EPISODE ATTITUDE HISTORY STEERS NEW ONE
    def _reset_controllers(self) -> None:
        for ctrl in getattr(self._env, "ctrl", None) or []:
            ctrl.reset()

    def _ensure_live_aviary(self) -> None:
        """
        Make sure the underlying VelocityAviary still has a live PyBullet connection.
        Rebuild it if needed, then refresh handles.
        """
        if getattr(self, "_env", None) is None or not self._pybullet_client_alive():
            self._rebuild_aviary()
            self._init_pybullet_handles()
            self._refresh_drone_ids_from_pybullet()

    def _create_aviary(self):
        """Create VelocityAviary from scratch."""
        self._env = VelocityAviary(
            drone_model=DroneModel.CF2X,
            num_drones=self.max_total_num_drones,
            initial_xyzs=self.spawn_positions,
            physics=Physics.PYB,
            gui=(self.render_mode == "human"),
            record=False,
            user_debug_gui=False,
        )
        self._env.SPEED_LIMIT = 3.0

        self._p = None
        self._client_id = None
        self._drone_ids = []
        self._obstacles = []
        self._lidar_line_ids = {}
        self._wall_ids = []

        self._init_pybullet_handles()
        self._refresh_drone_ids_from_pybullet()

    # ----------------------------- (DE)SPAWN DRONES, OBSTACLES AND TARGETS -----------------------------

    def generate_drone_swarm_spawn(self):
        max_attempts = 1000
        edge = 0

        for i in range(max_attempts):
            edge = self.rng.integers(0, 4)
            center = self.generate_swarm_center_xyz(edge, SWARM_EDGE_MARGIN)

            if self.is_far_from_other_swarms(center):
                break
            else:
                continue

        positions = []

        while len(positions) < self.num_drones_per_swarm:
            x = self.rng.uniform(center[0] - SWARM_RADIUS, center[0] + SWARM_RADIUS)
            y = self.rng.uniform(center[1] - SWARM_RADIUS, center[1] + SWARM_RADIUS)
            z = self.rng.uniform(center[2] - SWARM_RADIUS, center[2] + SWARM_RADIUS)

            candidate = np.array([x, y, z])

            # enforce min separation
            valid = True
            for p in positions:
                if np.linalg.norm(candidate - p) < SWARM_MIN_SEPARATION:
                    valid = False
                    break

            if valid:
                positions.append(candidate)

        return edge, center, np.array(positions, dtype=np.float32)

    def generate_edge_xyz(self, edge, margin):
        """Generate a random position on a given world edge."""
        from drone_env.createWorld import generate_edge_xyz as _generate_edge_xyz
        return _generate_edge_xyz(edge, self.world_scale, margin, rng=self.rng)

    def generate_swarm_center_xyz(self, edge, margin):
        """ Generates the swarm center """
        return self.generate_edge_xyz(edge, margin)

    def generate_target_xyz(self, edge, margin):
        """ Generate the spawn position of target, opposite to the swarms center. """
        opposite_edge = {0: 1, 1: 0, 2: 3, 3: 2}[edge]
        return self.generate_edge_xyz(opposite_edge, margin)

    def _spawn_walls(self) -> None:
        if self._p is None:
            return

        s = self.world_scale

        positions = [
            ("z_pos_wall", [0, 0, s]),  # ceiling
            ("z_pos_wall", [0, 0, -s]),  # floor
            ("x_pos_wall", [s, 0, 0]),  # front
            ("x_pos_wall", [-s, 0, 0]),  # back
            ("y_pos_wall", [0, s, 0]),  # right
            ("y_pos_wall", [0, -s, 0]),  # left
        ]

        objects = [(key, EnvObject(pos, s, [0, 0, 0])) for key, pos in positions]
        self._wall_ids = [
            obj.create(key, self._call_physics, color=(0.5, 0.5, 0.5, 1.0))
            for key, obj in objects
        ]

    #WALL COVERAGE OPTION: None, A FRACTION IN (0, 1), OR AN ORDERED PAIR OF FRACTIONS
    @staticmethod
    def _validate_wall_coverage(value):
        if value is None:
            return None

        values = list(value) if isinstance(value, (tuple, list)) else [value]

        if len(values) not in (1, 2) or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
            raise ValueError("wall_coverage must be a fraction in (0, 1) or an ordered pair of such fractions")
        if any(not np.isfinite(v) or not 0.0 < float(v) < 1.0 for v in values) or (len(values) == 2 and values[0] >= values[1]):
            raise ValueError("wall_coverage must be a fraction in (0, 1) or an ordered pair of such fractions")

        return float(values[0]) if len(values) == 1 else (float(values[0]), float(values[1]))

    #PHASE-2 WALL: ONE PANEL ACROSS THE PASSAGE, DRONE & GOAL MOVED TO THE LAYOUT'S POSITIONS
    def _apply_episode_wall(self) -> None:
        self.current_wall_layout = None

        if self.wall_coverage is None or self._p is None:
            return

        from evaluation.wall_gap import PanelObject, sample_wall_layout

        coverage = (self.wall_coverage if isinstance(self.wall_coverage, float)
                    else float(self.rng.uniform(*self.wall_coverage)))
        layout = sample_wall_layout(self.world_scale, coverage, self.rng)

        #THE PANEL IS THE EPISODE'S ONLY OBSTACLE (WALL STAGES CARRY NO RANDOM OBSTACLES)
        self.remove_all_obstacles()
        panel = PanelObject(layout["center"], layout["half_ext"], layout["euler"])
        panel.create("panel", physics_call=self._call_physics)
        self._obstacles = [panel]
        self._rebuild_query_world_from_current_scene()

        did = self._drone_ids[0]
        self._call_physics(self._p.resetBasePositionAndOrientation, did, layout["spawn"].tolist(), [0.0, 0.0, 0.0, 1.0])
        self._call_physics(self._p.resetBaseVelocity, did, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        self.spawn_positions[0] = layout["spawn"]
        target = self._targets[0]
        target.start_pos = layout["goal"]
        body_id = getattr(target, "body_id", None)

        if body_id is not None:
            self._call_physics(self._p.resetBasePositionAndOrientation, body_id, layout["goal"].tolist(), [0.0, 0.0, 0.0, 1.0])

        #KINEMATIC CACHE MUST SEE THE TELEPORT BEFORE THE FIRST OBSERVATION
        self._env._updateAndStoreKinematicInformation()
        self.current_wall_layout = layout

    def make_valid_target(
        self, swarm_edge, color, min_dist=0.5, max_attempts=1000
    ) -> Target:
        """
        Make a valid target for all drones, i.e. a target that does not collide with the drone itself, other obstacles
        or other drones.
        """
        for _ in range(max_attempts):
            t = self.generate_target_xyz(swarm_edge, TARGET_EDGE_MARGIN)

            drone_positions = self.spawn_positions[:self.total_num_drones]
            possible_target = Target(color, t, TARGET_RADIUS, physics_call= self._call_physics, call_gui=self._call_gui)
            if possible_target.isValidPosition(drone_positions, self._obstacles, min_dist):
                possible_target.spawn_target()
                return possible_target
            else:
                continue

        backup = self.generate_xyz()

        return Target(
            color,
            backup,
            TARGET_RADIUS,
            physics_call=self._call_physics,
            call_gui=self._call_gui,
        )

    def _make_targets(self) -> list[Target]:
        """Make a target for all drones."""
        targets = []

        if self.num_swarms == 1:
            swarm_edge = self.drones_in_swarm[0]["edge"]
            t = self.make_valid_target(swarm_edge, COLORS[0])
            # repeat shared target for all drones
            targets = [t] * self.total_num_drones
            return targets
        else:
            for i in range(self.num_swarms):
                swarm_centroid = self.drones_in_swarm[i]["center"]
                swarm_edge = self.drones_in_swarm[i]["edge"]

                t = self.make_valid_target(swarm_edge, COLORS[i])

                # List that contain the target per drone
                drone_targets = [t] * self.num_drones_per_swarm
                targets.extend(drone_targets)

            return targets

    def _spawn_obstacles(self):
        """Spawn all the obstacles."""

        self._obstacles = []

        if self.obstacle_layout == "curriculum":
            self._spawn_curriculum_obstacles()

            return

        # Stochastic single blocker mode: draw once per episode, and when it
        # fires place the obstacle on the drone->goal line rather than
        # anywhere in the arena. Recorded on the env so the callbacks can
        # report the realised rate against the configured probability.
        self._blocking_obstacle_spawned = False
        num_static = self.num_static_obstacles

        if self.blocking_obstacle_prob > 0.0:
            self._blocking_obstacle_spawned = bool(
                self.rng.uniform(0.0, 1.0) < self.blocking_obstacle_prob
            )
            num_static = 1 if self._blocking_obstacle_spawned else 0

        if num_static + self.num_dynamic_obstacles <= 0 or self._p is None:
            return

        # A max number of attempts for creating an obstacle
        swarm_list = list(self.drones_in_swarm.keys())
        max_attempts = 100000

        for _ in range(self.num_dynamic_obstacles):
            axis = self.rng.choice([0, 1, 2])
            shape = self.rng.choice(SUPPORTED_SHAPES)
            self._spawn_dynamic_obstacles(max_attempts, shape, axis)
            self._rebuild_query_world_from_current_scene()
        for _ in range(num_static):
            shape = self.rng.choice(SUPPORTED_SHAPES)
            swarm_list = self._spawn_static_obstacles(
                max_attempts,
                shape,
                swarm_list,
                force_blocking=self.blocking_obstacle_prob > 0.0,
            )

    #SPAWN EXACT COUNTS WITH OPTIONAL FIRST STATIC BLOCKER
    def _spawn_curriculum_obstacles(self):
        self._blocking_obstacle_requested = bool(
            self.num_static_obstacles > 0
            and self.rng.random() < self.blocking_obstacle_prob
        )
        if self.num_static_obstacles < 0 or self.num_dynamic_obstacles < 0:
            raise ValueError("curriculum obstacle counts must be nonnegative")

        if self._p is None:
            raise RuntimeError("curriculum obstacle placement requires an active physics client")

        for scene_attempt in range(10):
            self._blocking_obstacle_spawned = False

            try:
                self._spawn_curriculum_scene()
                break
            except ObstaclePlacementError:
                self.remove_all_obstacles()

                if scene_attempt == 9:
                    raise

        self._episode_obstacle_counts = {
            "static_obstacles": self.num_static_obstacles,
            "dynamic_obstacles": self.num_dynamic_obstacles,
            "blocking_obstacle_requested": self._blocking_obstacle_requested,
            "blocking_obstacle_spawned": self._blocking_obstacle_spawned,
            "obstacle_scene_attempts": scene_attempt + 1,
        }
        self._rebuild_query_world_from_current_scene()

    #RETRY COMPLETE PACKING WITHOUT CHANGING REQUESTED COUNTS OR BLOCKER DRAW
    def _spawn_curriculum_scene(self):
        if self._blocking_obstacle_requested:
            self._blocking_obstacle_spawned = self._spawn_curriculum_object(
                dynamic=False, blocking=True
            )

        for _ in range(self.num_dynamic_obstacles):
            self._spawn_curriculum_object(dynamic=True)

        remaining_static = self.num_static_obstacles - int(self._blocking_obstacle_spawned)

        for _ in range(remaining_static):
            self._spawn_curriculum_object(dynamic=False)

    #PLACE ONE OBJECT INSIDE WORLD WITH SAFE SPAWN & SWEPT VOLUME
    def _spawn_curriculum_object(self, dynamic, blocking=False):
        max_attempts = 2000

        for _ in range(max_attempts):
            size = float(self.rng.uniform(*self.obstacle_size_range))
            shape = str(self.rng.choice(SUPPORTED_SHAPES))
            ori = self.rng.uniform(-np.pi, np.pi, size=3).tolist()

            if blocking:
                candidate = self.generate_blocking_xyz(
                    self.spawn_positions[0], self._targets[0].start_pos, margin=0.0
                )
            else:
                candidate = self.generate_xyz()

            if dynamic:
                axis = int(self.rng.integers(0, 3))
                travel = float(self.rng.uniform(*self.dynamic_travel_range))
                obj = Dynamic(
                    axis, candidate[axis] - travel, candidate[axis] + travel,
                    float(self.rng.uniform(*self.dynamic_speed_range)),
                    int(self.rng.choice([-1, 1])), candidate, size, ori,
                )
            else:
                obj = EnvObject(candidate, size, ori)

            obj.set_shape_spec(shape)
            center, half_extents = self._curriculum_swept_bounds(obj)
            lower = center - half_extents
            upper = center + half_extents
            world_lower = np.array([-self.world_scale + 0.1, -self.world_scale + 0.1, 0.1])
            world_upper = np.full(3, self.world_scale - 0.1)

            if np.any(lower <= world_lower) or np.any(upper >= world_upper):
                continue

            if not self._curriculum_spawn_clearance(obj):
                continue

            blocked = False

            for other in self._obstacles:
                other_center, other_extents = self._curriculum_swept_bounds(other)

                if np.all(np.abs(center - other_center) <= half_extents + other_extents + 0.05):
                    blocked = True
                    break

            if blocked:
                continue

            obj.create(shape, physics_call=self._call_physics)
            self._obstacles.append(obj)

            return True

        if blocking:
            return False

        label = "blocking static" if blocking else ("dynamic" if dynamic else "static")
        raise ObstaclePlacementError(
            f"unable to place requested {label} obstacle after {max_attempts} attempts "
            f"in world_scale={self.world_scale}; refusing an underfilled curriculum scene"
        )

    #BOUND FULL ROTATED GEOMETRY & DYNAMIC CENTER TRAVEL
    def _curriculum_swept_bounds(self, obj):
        kwargs = obj.collision_shape_spec["collision_kwargs"]

        if "halfExtents" in kwargs:
            radius = float(np.linalg.norm(kwargs["halfExtents"]))
        elif "height" in kwargs:
            radius = float(np.hypot(kwargs["radius"], kwargs["height"] / 2.0))
        else:
            radius = float(kwargs["radius"])

        center = np.array(obj.start_pos, dtype=np.float64, copy=True)
        extents = np.full(3, radius, dtype=np.float64)

        if isinstance(obj, Dynamic):
            center[obj.axis] = (obj.min + obj.max) / 2.0
            extents[obj.axis] += (obj.max - obj.min) / 2.0

        return center, extents

    #KEEP FULL DYNAMIC PATH CLEAR OF INITIAL DRONE & TARGET POSITIONS
    def _curriculum_spawn_clearance(self, obj):
        positions = list(self.spawn_positions[:self.total_num_drones])
        positions.extend(target.start_pos for target in self._targets)

        for pos in positions:
            closest = np.array(obj.start_pos, dtype=np.float64, copy=True)

            if isinstance(obj, Dynamic):
                closest[obj.axis] = np.clip(pos[obj.axis], obj.min, obj.max)

            if np.linalg.norm(np.asarray(pos) - closest) < MIN_SPAWN_DIST_FROM_DRONE:
                return False

        return True

    def _spawn_static_obstacles(
        self, max_attempts, shape, swarm_list, force_blocking: bool = False
    ) -> None:
        """Spawn the static obstacles.

        force_blocking makes every placement sit on the drone->goal line
        instead of taking the original 50/50 blocking-vs-anywhere draw.
        """
        chosen_swarm = None
        blocking_obstacles_chance = 0.0 if force_blocking else self.rng.uniform(0.0, 1.0)

        for _ in range(max_attempts):
            if blocking_obstacles_chance < 0.5 and len(swarm_list) >= 1:
                chosen_swarm = self.rng.choice(swarm_list)
                obj = self._spawn_blocking_obstacles(chosen_swarm)
            else:
                candidate = self.generate_xyz()
                ori = [
                    float(self.rng.uniform(-np.pi, np.pi)),
                    float(self.rng.uniform(-np.pi, np.pi)),
                    float(self.rng.uniform(-np.pi, np.pi)),
                ]
                size = float(self.rng.uniform(0.8, 1.2))

                obj = EnvObject(candidate, size, ori)

            if self.in_swept_area(obj):
                continue
            body = obj.create(shape, physics_call=self._call_physics)

            # Check if the obstacle is overlapping with any other obstacle or too close to drones or target
            if self.obstacle_overlap(obj) or not obj.isValidPosition(
                self.spawn_positions, self._targets, MIN_SPAWN_DIST_FROM_DRONE
            ):
                self._call_physics(self._p.removeBody, body)
                continue

            if chosen_swarm is not None:
                swarm_list.remove(chosen_swarm)
            self._obstacles.append(obj)
            break
        return swarm_list

    def in_swept_area(self, object):
        for obs in self._obstacles:
            if isinstance(obs, Dynamic) and object.is_in_swept_area(obs):
                return True
        return False

    def _spawn_dynamic_obstacles(self, max_attempts, shape, axis) -> None:
        """Spawn the dynamic obstacles."""
        for _ in range(max_attempts):
            candidate = self.generate_xyz()

            displacement = self.rng.uniform(1, 3)
            min_pos = candidate[axis] - displacement
            max_pos = candidate[axis] + displacement

            # Check if obstacle is going beyond world borders, if so generate another obstacle
            if self.out_of_bounds(max_pos, min_pos, axis):
                continue

            # Generate other attributes with some variability
            ori = [
                float(self.rng.uniform(-np.pi, np.pi)),
                float(self.rng.uniform(-np.pi, np.pi)),
                float(self.rng.uniform(-np.pi, np.pi)),
            ]
            size = float(self.rng.uniform(0.8, 1.2))
            speed = self.rng.uniform(0.5, 0.8)
            direction = self.rng.choice([-1,1])

            obj = Dynamic(
                axis, min_pos, max_pos, speed, direction, candidate, size, ori
            )
            obj.set_shape_spec(shape)

            body = obj.create(shape, physics_call=self._call_physics)

            # Check if the obstacle is overlapping with any other obstacle and is far enough from targets and drones
            if self.obstacle_overlap(obj) or not obj.isValidPosition(
                self.spawn_positions, self._targets, MIN_SPAWN_DIST_FROM_DRONE
            ):
                self._call_physics(self._p.removeBody, body)
                continue
            self._obstacles.append(obj)
            break

    def _spawn_blocking_obstacles(self, swarm_num) -> EnvObject:
        # A max number of attempts for creating an obstacle

        swarm_center = self.drones_in_swarm[swarm_num]["center"]
        target_pos = self._targets[swarm_num * self.num_drones_per_swarm].start_pos

        candidate = self.generate_blocking_xyz(swarm_center, target_pos)
        ori = [
            float(self.rng.uniform(-np.pi, np.pi)),
            float(self.rng.uniform(-np.pi, np.pi)),
            float(self.rng.uniform(-np.pi, np.pi)),
        ]
        size = float(self.rng.uniform(1.2, 1.5))

        obj = EnvObject(candidate, size, ori)
        return obj

    def _update_obstacles(self) -> None:
        """Update the dynamic obstacles."""
        if self._p is None or len(self._obstacles) == 0:
            return

        t = self.obstacle_dt

        for obstacle in self._obstacles:
            if isinstance(obstacle, Dynamic):
                obstacle.update_obstacle(t)

    def obstacle_overlap(self, object):
        """Helper function to see if the obstacle being created overlap with other obstalces."""
        for obs in self._obstacles:
            if object.isOverlapping(obs):
                return True

        return False

    def out_of_bounds(self, max_pos, min_pos, axis):
        """Helper function used to see if entities are going out of bounds."""
        if axis != 2:
            if max_pos >= self.world_scale or min_pos <= -self.world_scale:
                return True
            else:
                return False
        else:
            if max_pos >= self.world_scale or min_pos <= 0:
                return True
            else:
                return False

    def _init_query_world(self):
        self._qworld = CollisionQueryWorld()

    def _rebuild_query_world_from_current_scene(self):
        self._qworld.clear()

        for obs in self._obstacles:
            if isinstance(obs, Dynamic):
                self._qworld.add_dynamic_object(id(obs), obs)
            else:
                self._qworld.add_static_object(
                    obs
                )

    def candidate_collides_over_time(
        self, candidate, horizon_steps, dt=1.0 / 60.0, margin=0.0
    ):
        temp_body = self._qworld.add_temp_object(candidate)

        try:
            for step in range(horizon_steps + 1):
                t = step * dt
                cand_pos, cand_quat = candidate.pose_at_time(t)
                self._qworld.set_body_pose(temp_body, cand_pos, cand_quat)

                for obs in self._obstacles:
                    if isinstance(obs, Dynamic):
                        body_id = self._qworld.dynamic_bodies[id(obs)]
                        pos, quat = obs.pose_at_time(t)
                        self._qworld.set_body_pose(body_id, pos, quat)

                for obs in self._obstacles:
                    if isinstance(obs, Dynamic):
                        body_id = self._qworld.dynamic_bodies[id(obs)]
                        if self._qworld.bodies_collide(
                            temp_body, body_id, margin=margin
                        ):
                            return True
            return False
        finally:
            self._qworld.remove_body(temp_body)

    def drone_out_of_bounds(self, pos: np.ndarray) -> bool:
        """Helper function to see if drones are going out of bounds (considered as a collision)."""
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])

        return (
            x <= -self.world_scale
            or x >= self.world_scale
            or y <= -self.world_scale
            or y >= self.world_scale
            or z <= 0.1
            or z >= self.world_scale
        )

    def generate_xyz(self):
        """Generate the starting position."""
        x = self.rng.uniform(-self.world_scale, self.world_scale)
        y = self.rng.uniform(-self.world_scale, self.world_scale)
        z = self.rng.uniform(0, self.world_scale)

        return np.array([x, y, z], dtype=np.float32)

    def generate_blocking_xyz(
        self, swarm_center: np.ndarray, target_pos: np.ndarray, margin: float = 0.1
    ) -> np.ndarray:
        """Creates an obstacles that blocks the straight line from a center to the target"""
        # Vector from swarm to target
        line_vec = target_pos - swarm_center

        # Pick a random fraction along the line (0=start, 1=end)
        t = self.rng.uniform(0.2, 0.8)  # avoid spawning exactly on swarm or target
        point_on_line = swarm_center + t * line_vec

        # Optional: add a small random perpendicular offset so obstacle is not exactly on line
        # Generate a random vector
        random_offset = self.rng.uniform(-margin, margin, size=3)

        # Project offset perpendicular to the line
        line_dir = line_vec / (
            np.linalg.norm(line_vec) + 1e-8
        )  # avoid division by zero
        perp_offset = random_offset - np.dot(random_offset, line_dir) * line_dir

        # Final candidate position
        candidate = point_on_line + perp_offset

        # Clip to world bounds if needed
        candidate = np.clip(
            candidate,
            [-self.world_scale, -self.world_scale, 0],
            [self.world_scale, self.world_scale, self.world_scale],
        )

        return candidate.astype(np.float32)

    def is_far_from_other_swarms(self, candidate_center):
        if not self.drones_in_swarm:
            return True
        else:
            for swarm_info in self.drones_in_swarm.values():
                d = np.linalg.norm(
                    np.array(candidate_center) - np.array(swarm_info["center"])
                )
                if d <= SWARM_RADIUS:
                    return False
        return True

    def despawn_drone_in_gui(self, drone_idx: int) -> None:
        """Helper function to despawn a drone if it reached the target."""
        if self._p is None or not (0 <= drone_idx < len(self._drone_ids)):
            return

        p = self._p
        body_id = self._drone_ids[drone_idx]

        # "despawning" happens by moving drone far away
        far_position = [1000.0 + 10.0 * drone_idx, 1000.0, -100.0]
        far_orientation = [0.0, 0.0, 0.0, 1.0]

        try:
            self._call_physics(
                p.resetBasePositionAndOrientation,
                body_id,
                far_position,
                far_orientation,
            )
        except:
            pass

        try:
            self._call_physics(
                p.resetBaseVelocity,
                body_id,
                [0.0, 0.0, 0.0],  # make drone stop moving at far position
                [0.0, 0.0, 0.0],
            )
        except Exception:
            pass

        # disable collisions against everything
        try:
            self._call_physics(p.setCollisionFilterGroupMask, body_id, -1, 0, 0)
        except Exception:
            pass

        # make drone invisible on gui
        try:
            num_joints = self._call_physics(p.getNumJoints, body_id)
        except Exception:
            num_joints = 0

        for link_idx in range(-1, num_joints):
            try:
                self._call_physics(
                    p.changeVisualShape,
                    body_id,
                    link_idx,
                    rgbaColor=[1.0, 1.0, 1.0, 0.0],
                )
            except Exception:
                pass

        # remove lidar lines for despawned drone
        for lid in self._lidar_line_ids.get(drone_idx, []):
            self._call_gui(p.removeUserDebugItem, lid)
        self._lidar_line_ids[drone_idx] = []

    def despawn_newly_reached_drones(self, just_reached: np.ndarray) -> None:
        """Helper function to despawn a drone if it reached the target."""
        for i in range(len(just_reached)):
            if bool(just_reached[i] and not bool(self._despawned[i])):
                self.despawn_drone_in_gui(i)
                self._despawned[i] = True

    def despawn_newly_collided_drones(self, collisions: np.ndarray) -> None:
        """Helper function to despawn a drone if it collided with an obstacle and drone."""
        for i in range(len(collisions)):
            if bool(collisions[i] and not bool(self._despawned[i])):
                self.despawn_drone_in_gui(i)
                self._despawned[i] = True

    # ----------------------------- OBSERVATION / ENVIRONMENT RETRIEVAL METHODS -----------------------------

    def _build_obs_dict(self) -> Dict[str, np.ndarray]:
        """Build a dictionary containing every drone's observation space vector"""

        obs_dict: Dict[str, np.ndarray] = {}

        # compute drone positions and swarm centre
        positions = self.active_drone_positions()
        centroid = self._swarm_centroid(positions)

        for i, a in enumerate(self.possible_agents):
            # check if agent already despawned
            if self._despawned[i]:
                obs_dict[a] = np.zeros(
                    (self._base_obs_dim + self._lidar_dim,), dtype=np.float32
                )
                continue

            # if drone not active, give it observation zero and skip the rest
            if i >= self.total_num_drones:
                obs_dict[a] = np.zeros(
                    (self._base_obs_dim + self._lidar_dim,), dtype=np.float32
                )
                continue

            # get raw state vector from simulator, and extract position and velocity
            state = self._env._getDroneStateVector(i)
            pos = np.array(state[POS_SLICE], dtype=np.float32)

            vel = np.array(state[10:13], dtype=np.float32)

            # compute distance to assigned target
            target = self._targets[i].start_pos
            to_target = (target - pos).astype(np.float32)

            # compute distance to nearest obstacle, swarm centre and drone
            to_obs = self._nearest_obstacle_vector(pos)
            to_centroid = (centroid - pos).astype(np.float32)
            to_nearest_teammate = self._nearest_active_teammate_vector(i)

            # form the base observation vector
            base = np.concatenate(
                [pos, vel, to_target, to_obs, to_centroid, to_nearest_teammate], axis=0
            ).astype(np.float32)

            # add lidar observation to base vector
            if self.lidar:
                scan = self._lidar_scan(i, pos)
            else:
                # Neutral LiDAR values: 1.0 means "no hit within range" for all rays
                scan = np.ones((self._lidar_dim,), dtype=np.float32)

            obs_dict[a] = np.concatenate([base, scan], axis=0).astype(np.float32)

        return obs_dict

    def active_drone_mask(self) -> np.ndarray:
        """Drones that should still influence the ongoing episodes (i.e. not dead or despawned)."""
        mask = np.zeros(self.max_total_num_drones, dtype=np.bool_)
        mask[: self.total_num_drones] = True
        mask &= ~(self._dead | self._reached_once | self._despawned)
        return mask

    def active_drone_indices(self) -> list[int]:
        """Retrieve all active drone indices."""
        return [i for i, active in enumerate(self.active_drone_mask()) if active]

    def active_drone_positions(self) -> np.ndarray:
        """Retrieve all active drone positions."""
        idx = self.active_drone_indices()

        if len(idx) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        positions = np.zeros((len(idx), 3), dtype=np.float32)
        for k, i in enumerate(idx):
            state = self._env._getDroneStateVector(int(i))
            positions[k] = np.array(state[POS_SLICE], dtype=np.float32)

        return positions

    @staticmethod
    def _swarm_centroid(positions: np.ndarray) -> np.ndarray:
        """Compute the swarm centroid."""
        if positions.shape[0] == 0:
            return np.zeros(3, dtype=np.float32)

        return positions.mean(axis=0).astype(np.float32)

    def _nearest_active_teammate_vector(self, i: int) -> np.ndarray:
        """Retrieve the vector to the nearest teammate to drone i."""
        if self._dead[i] or self._reached_once[i] or self._despawned[i]:
            return np.zeros(3, dtype=np.float32)

        state_i = self._env._getDroneStateVector(i)
        pos_i = np.array(state_i[POS_SLICE], dtype=np.float32)

        candidate_neighbours = []
        for j in range(self.total_num_drones):
            if j == i:
                continue
            if self._dead[j] or self._reached_once[j] or self._despawned[j]:
                continue
            state_j = self._env._getDroneStateVector(j)
            pos_j = np.array(state_j[POS_SLICE], dtype=np.float32)
            candidate_neighbours.append(pos_j)

        if len(candidate_neighbours) == 0:
            return np.zeros(3, dtype=np.float32)

        candidate_neighbours = np.array(candidate_neighbours, dtype=np.float32)
        d = candidate_neighbours - pos_i[None, :]
        dist = np.linalg.norm(d, axis=1) + 1e-8

        # return vector distance to nearest drone
        return d[int(np.argmin(dist))].astype(np.float32)

    def _nearest_obstacle_vector(self, drone_pos: np.ndarray) -> np.ndarray:
        """Retrieve the vector to the nearest obstacle."""

        if self._p is None or len(self._obstacles) == 0:
            # if there are no obstacles, set the distance really far away
            return np.array([LIDAR_RANGE * 2.0, 0.0, 0.0], dtype=np.float32)

        p = self._p
        obs_pos = []
        for object in self._obstacles:
            try:
                pos, _ = self._call_physics(
                    p.getBasePositionAndOrientation, object.body_id
                )
                obs_pos.append(pos)
            except Exception:
                pass

        if len(obs_pos) == 0:
            return np.zeros(3, dtype=np.float32)

        obs_pos = np.array(obs_pos, dtype=np.float32)
        d = obs_pos - drone_pos[None, :]
        dist = np.linalg.norm(d, axis=1) + 1e-8
        idx = int(np.argmin(dist))

        return d[idx].astype(np.float32)

    def check_collisions_per_drone(self):
        """
        Returns:
            collided[i] is True if drone i collided with anything:
            - obstacle
            - another drone
            - world boundary (wall)

        Also returns the specific collision types
        """
        wall_collisions = np.zeros(self.total_num_drones, dtype=np.bool_)
        obstacle_collisions = np.zeros(self.total_num_drones, dtype=np.bool_)
        drone_collisions = np.zeros(self.total_num_drones, dtype=np.bool_)

        if self._p is None or len(self._drone_ids) == 0:
            collided = wall_collisions | obstacle_collisions | drone_collisions
            return collided, wall_collisions, obstacle_collisions, drone_collisions

        p = self._p
        active_ids = self._drone_ids[: self.total_num_drones]

        thresh_obstacle = 0.01
        thresh_drone = 0.01
        thresh_wall = 0.02

        # drone vs wall
        for i, did in enumerate(active_ids):
            if self._dead[i] or self._reached_once[i] or self._despawned[i]:
                continue

            # 1) direct contact with wall bodies
            for wid in self._wall_ids:
                try:
                    pts = self._call_physics(p.getContactPoints, bodyA=did, bodyB=wid)
                    if pts and len(pts) > 0:
                        wall_collisions[i] = True
                        break
                except Exception:
                    continue

                try:
                    pts = self._call_physics(
                        p.getClosestPoints,
                        bodyA=did,
                        bodyB=wid,
                        distance=thresh_wall,
                    )
                    if pts and len(pts) > 0:
                        wall_collisions[i] = True
                        break
                except Exception:
                    continue

            # 2) fallback numeric boundary check
            if not wall_collisions[i]:
                try:
                    state = self._env._getDroneStateVector(i)
                    pos_i = np.array(state[POS_SLICE], dtype=np.float32)
                    if self.drone_out_of_bounds(pos_i):
                        wall_collisions[i] = True
                except Exception:
                    pass

        # drone vs obstacle
        if len(self._obstacles) > 0:
            for i, did in enumerate(active_ids):
                if self._dead[i] or self._reached_once[i] or self._despawned[i]:
                    continue

                for object in self._obstacles:
                    try:
                        pts = self._call_physics(
                            p.getClosestPoints,
                            bodyA=did,
                            bodyB=object.body_id,
                            distance=thresh_obstacle,
                        )
                        if pts and len(pts) > 0:
                            obstacle_collisions[i] = True
                            break
                    except Exception:
                        continue

        # drone vs drone
        for i, did1 in enumerate(active_ids):
            if self._dead[i] or self._reached_once[i] or self._despawned[i]:
                continue

            for j, did2 in enumerate(active_ids[i + 1 :], start=i + 1):
                if self._dead[j] or self._reached_once[j] or self._despawned[j]:
                    continue
                try:
                    pts = self._call_physics(
                        p.getClosestPoints,
                        bodyA=did1,
                        bodyB=did2,
                        distance=thresh_drone,
                    )
                    if pts and len(pts) > 0:
                        drone_collisions[i] = True
                        drone_collisions[j] = True
                except Exception:
                    continue

        collided = wall_collisions | obstacle_collisions | drone_collisions
        return collided, wall_collisions, obstacle_collisions, drone_collisions

    @staticmethod
    def _pairwise_sep_stats(positions: np.ndarray) -> Tuple[float, float, bool, bool]:
        """
        Check every unique drone pair and retrieve swarm drone stats.
        Computes:
          max_d   = maximum pairwise distance
          min_d   = minimum pairwise distance
          too_far = any pair exceeds swarm_max_sep
          too_close = any pair below swarm_min_sep
        """
        n = positions.shape[0]
        if n <= 1:
            return 0.0, 1e9, False, False

        max_d = 0.0
        min_d = 1e9
        too_far = False
        too_close = False

        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(positions[i] - positions[j]))
                max_d = max(max_d, d)
                min_d = min(min_d, d)
                if d > DEFAULT_SWARM_MAX_SEP:
                    too_far = True
                if d < DEFAULT_SWARM_MIN_SEP:
                    too_close = True

        return max_d, min_d, too_far, too_close

    # ----------------------------- LIDAR METHODS -----------------------------

    def _rotmat_from_quat(self, qxyzw: np.ndarray) -> np.ndarray:
        x, y, z, w = [float(v) for v in qxyzw]
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        return np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float32,
        )

    def obstacle_lidar_scan(self, drone_idx: int) -> np.ndarray:
        """
        LiDAR returns with wall/floor/target hits removed, for the reward's
        proximity shaping. Defaults to all-ones ("nothing nearby") so a drone
        whose scan was never computed this step cannot be charged a penalty
        from a stale array.
        """
        scan = self._lidar_obstacle_scans.get(drone_idx)

        if scan is None or scan.shape[0] != self._lidar_dim:
            return np.ones((self._lidar_dim,), dtype=np.float32)

        return scan

    def _lidar_scan(self, drone_idx: int, pos_unused: np.ndarray) -> np.ndarray:
        # Cleared up front so that EVERY early return below leaves the
        # obstacle-only scan at "nothing nearby" rather than at last step's
        # values; the successful path overwrites it with the real returns.
        self._lidar_obstacle_scans[drone_idx] = np.ones(
            (self._lidar_dim,), dtype=np.float32
        )

        if (not self.lidar) or (self._p is None) or (self._lidar_dim <= 0):
            return np.ones((self._lidar_dim,), dtype=np.float32)

        p = self._p

        did = (
            self._drone_ids[drone_idx]
            if (0 <= drone_idx < len(self._drone_ids))
            else None
        )
        if did is None:
            return np.ones((self._lidar_dim,), dtype=np.float32)

        try:
            base_pos, _ = self._call_physics(p.getBasePositionAndOrientation, did)
            origin = np.array(base_pos, dtype=np.float32)
        except Exception:
            return np.ones((self._lidar_dim,), dtype=np.float32)

        #GOAL-RELATIVE RAYS (DRONE NEVER YAWS, SO BODY FRAME = FIXED WORLD AXES)
        goal = (
            np.asarray(self._targets[drone_idx].start_pos, dtype=np.float32)
            if drone_idx < len(self._targets)
            else origin + np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )
        dirs_world = ray_directions(goal - origin).astype(np.float32)

        # move ray start slightly outside the drone body along each ray direction
        LIDAR_START_OFFSET = 0.15
        ray_from = origin[None, :] + dirs_world * LIDAR_START_OFFSET
        ray_to = ray_from + dirs_world * LIDAR_RANGE

        try:
            results = self._call_physics(
                p.rayTestBatch,
                rayFromPositions=ray_from.tolist(),
                rayToPositions=ray_to.tolist(),
            )
        except Exception:
            return np.ones((dirs_world.shape[0],), dtype=np.float32)

        hit_frac = np.ones((dirs_world.shape[0],), dtype=np.float32)
        hit_pos = ray_to.copy()

        # Rays that hit an actual obstacle body, as opposed to a wall, the
        # floor, the target or the drone itself. The reward's proximity
        # shaping reads this variant while the policy keeps the full scan:
        # the arena walls are ray-visible, and since the proximity term fires
        # below a normalised return of 0.35 (~2.1 m of the 6 m range), in the
        # early curriculum's 4 m box EVERY position is inside the trigger
        # radius. That charged the shaped conditions ~-1.6 of a possible
        # -2.0 per step no matter where they flew, which swamped the one-off
        # +250 goal bonus and made ending the episode early (i.e. crashing)
        # the reward-maximising move. It also meant the ablated term was not
        # really about obstacles, which is what the study varies.
        obstacle_ids = {
            bid
            for bid in (getattr(o, "body_id", None) for o in self._obstacles)
            if bid is not None
        }
        obstacle_frac = np.ones((dirs_world.shape[0],), dtype=np.float32)

        for i, r in enumerate(results):
            try:
                hit_body = int(r[0])
                hf = float(r[2])
                hp = r[3]

                # ignore self-hit if it still happens
                if hit_body == did:
                    hit_frac[i] = 1.0
                    hit_pos[i] = ray_to[i]
                else:
                    hit_frac[i] = np.clip(hf, 0.0, 1.0)
                    hit_pos[i] = np.array(hp, dtype=np.float32)

                    if hit_body in obstacle_ids:
                        obstacle_frac[i] = hit_frac[i]
            except Exception:
                pass

        self._lidar_obstacle_scans[drone_idx] = obstacle_frac.astype(np.float32)

        if LIDAR_NOISE_STD > 0.0:
            hit_frac = np.clip(
                hit_frac
                + self.rng.normal(0.0, LIDAR_NOISE_STD, size=hit_frac.shape).astype(
                    np.float32
                ),
                0.0,
                1.0,
            )

        show_rays = False
        if self.render_mode == "human":
            show_rays = LIDAR_SHOW_RAYS_TRAIN or bool(getattr(self, "show_rays", False))

        if show_rays and self._step_count % 50 == 0:
            old = self._lidar_line_ids.get(drone_idx, [])
            for lid in old:
                self._call_gui(p.removeUserDebugItem, lid)

            new_ids: List[int] = []
            for i in range(dirs_world.shape[0]):
                endp = hit_pos[i] if hit_frac[i] < 0.999 else ray_to[i]
                col = [1.0, 0.2, 0.2] if hit_frac[i] < 0.999 else [0.2, 0.6, 1.0]
                lid = self._call_gui(
                    p.addUserDebugLine,
                    lineFromXYZ=ray_from[i].tolist(),
                    lineToXYZ=endp.tolist(),
                    lineColorRGB=col,
                    lineWidth=1.0,
                    lifeTime=0,
                )
                if lid is not None:
                    new_ids.append(int(lid))
            self._lidar_line_ids[drone_idx] = new_ids

        return hit_frac.astype(np.float32)

    # ----------------------------- STEP FUNCTION -----------------------------
    def step(self, actions: Dict[str, np.ndarray], allow_rebuild: bool = True):
        """
        One environment step.

        Flow:
        1. Update world (moving obstacles, counters, etc.)
        2. Freeze drones that already reached the goal.
        3. Convert policy actions into simulator velocity commands.
        4. Step the physics.
        5. Build observations.
        6. Compute the swarm stats and termination flags (global).
        7. Compute per-agent rewards.
        8. Build the info dict.

        Args:
            actions: ndarray, shape (n, 4);
                    (vx_dir, vy_dir, vz_dir, speed)

        Returns:
            obs_dict, reward_dict, term_dict, trunc_dict, info_dict
        """

        # 1. Update the world
        self._step_count += 1
        self._update_obstacles()

        # 2. Freeze drones that already reached the goal:
        freeze_inactive_actions(self, actions)

        # force inactive drones to do nothing
        for i, agent in enumerate(self.possible_agents):
            if i >= self.total_num_drones:
                actions[agent] = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)

        # 3. Convert normalised policy actions into simulator (VelocityAviary) velocity commands
        sim_actions = convert_actions_for_simulator(self, actions)

        self._last_sim_actions = sim_actions.copy()

        # 4. Ensure the simulator connection is alive before stepping
        if allow_rebuild:
            self._ensure_live_aviary()

            try:
                self._env.step(sim_actions)
            except Exception as e:
                # If PyBullet died mid-episode, rebuild once and hard-reset this env
                msg = str(e)
                if "Not connected to physics server" in msg:
                    self._rebuild_aviary()
                    obs_dict, info_dict = self.reset()

                    reward_dict = {agent: 0.0 for agent in self.agents}
                    term_dict = {agent: False for agent in self.agents}
                    trunc_dict = {agent: True for agent in self.agents}

                    for agent in self.agents:
                        if agent in info_dict and isinstance(info_dict[agent], dict):
                            info_dict[agent]["physics_server_recovered"] = True

                    return obs_dict, reward_dict, term_dict, trunc_dict, info_dict
            
                raise

        elif not allow_rebuild:
            try:
                self._env.step(sim_actions)
            except Exception as e:
                msg = str(e)
                if "Not connected to physics server" in msg or not self._pybullet_client_alive():
                    if not allow_rebuild:
                        obs_dict = {
                            agent: np.zeros(self.observation_space(agent).shape, dtype=np.float32)
                            for agent in self.agents
                        }
                        reward_dict = {agent: 0.0 for agent in self.agents}
                        term_dict = {agent: False for agent in self.agents}
                        trunc_dict = {agent: True for agent in self.agents}
                        info_dict = {agent: {"physics_server_recovered": True} for agent in self.agents}
                        return obs_dict, reward_dict, term_dict, trunc_dict, info_dict
                    self._rebuild_aviary()
                    obs_dict, info_dict = self.reset()
                    reward_dict = {agent: 0.0 for agent in self.agents}
                    term_dict = {agent: False for agent in self.agents}
                    trunc_dict = {agent: True for agent in self.agents}
                    for agent in self.agents:
                        if agent in info_dict and isinstance(info_dict[agent], dict):
                            info_dict[agent]["physics_server_recovered"] = True
                    return obs_dict, reward_dict, term_dict, trunc_dict, info_dict

                raise

        # 5. Build the observation dictionary
        obs_dict = self._build_obs_dict()

        # 6. Global episode state (collision, swarm cohesion, time limit)
        step_state = compute_step_state(self, obs_dict)

        # 7. Compute rewards (per agent)
        reward_dict = compute_rewards(self, obs_dict, sim_actions, step_state)
        term_dict, trunc_dict = compute_terminations(self, step_state)

        # 8. Info dict (for callbacks & logging)
        info_dict = build_info_dict(self, step_state)

        if self.obstacle_layout == "curriculum":
            for info in info_dict.values():
                info.update(self._episode_obstacle_counts)

        #PRESERVE TERMINAL POSES THROUGH OBSERVATION, REWARD & METRIC COLLECTION
        self.despawn_newly_collided_drones(step_state.collisions)
        self.despawn_newly_reached_drones(step_state.reached_now)

        self.agents = self.possible_agents[:]

        if self.stats is not None:
            self.stats.record_step(
                reward_dict,
                self.possible_agents,
                self.total_num_drones,
                step_state,
                self.active_drone_mask(),
            )

        return obs_dict, reward_dict, term_dict, trunc_dict, info_dict

    # ----------------------------- RESET ENVIRONMENT -----------------------------

    def write_ep_to_csv(self):
        """Write episode stats to csv"""
        if self.stats is not None:
            self.stats.write_episode(self._step_count)

    def reset(self, *, seed=None, options=None):
        """Reset the environment. Save episode stats in csv file for debugging."""

        if self._step_count > 0:
            self.write_ep_to_csv()

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._step_count = 0
        self.agents = self.possible_agents[:]

        self._set_rendering_enabled(False)

        try:
            # remove episode-specific GUI/custom bodies from previous episode
            self.clear_episode_artifacts()

            # fresh episode spawn positions
            self.spawn_positions = np.empty((0, 3), dtype=np.float32)
            self.drones_in_swarm = {}

            for i in range(self.num_swarms):
                edge, center, drones = self.generate_drone_swarm_spawn()
                self.drones_in_swarm[i] = {
                    "edge": edge,
                    "center": center,
                    "drones": drones,
                }
                self.spawn_positions = np.vstack([self.spawn_positions, drones])

            self.spawn_positions = self.spawn_positions.astype(np.float32)

            n_missing = self.max_total_num_drones - self.spawn_positions.shape[0]
            if n_missing > 0:
                dummy_positions = []
                for k in range(n_missing):
                    dummy_positions.append([1000.0 + 10.0 * k, 1000.0, -100.0])
                dummy_positions = np.array(dummy_positions, dtype=np.float32)
                self.spawn_positions = np.vstack(
                    [self.spawn_positions, dummy_positions]
                )

            # if underlying VelocityAviary client is disconnected, rebuild it
            if not self._pybullet_client_alive():
                self._rebuild_aviary()

            self._env.reset()

            #FRESH CONTROLLER STATE - AVIARY RESET DOES NOT TOUCH IT
            self._reset_controllers()

            self._init_pybullet_handles()
            self._refresh_drone_ids_from_pybullet()

            self.restore_drone_visuals_and_collisions()

            # Move drones to their new spawn position.
            # Routed through _call_physics like every other physics call in
            # this file: the bare pybullet module functions default to
            # physicsClientId=0, so these two silently targeted the wrong
            # client whenever this env did not happen to own client 0 (e.g.
            # while a previous curriculum stage's env was still connected),
            # and raised "Not connected to physics server" once client 0 was
            # disconnected.
            for i, pos in enumerate(self.spawn_positions):
                self._call_physics(
                    self._p.resetBasePositionAndOrientation,
                    self._drone_ids[i],
                    pos,  # new spawn position
                    [0, 0, 0, 1],  # neutral orientation (quaternion)
                )
                self._call_physics(
                    self._p.resetBaseVelocity,
                    self._drone_ids[i],
                    linearVelocity=[0, 0, 0],
                    angularVelocity=[0, 0, 0],
                )

            #SYNC CACHED STATE WITH NEW SPAWNS BEFORE OBSERVATIONS & PROGRESS BASELINE
            self._env._updateAndStoreKinematicInformation()

            # reset episode bookkeeping
            self._dead[:] = False
            self._reached_once[:] = False
            self._despawned[:] = False

            if self.stats is not None:
                self.stats.reset_episode()

            self._prev_dists[:] = 0.0
            self._last_reward[:] = 0.0
            self._status = np.zeros(self.max_total_num_drones, dtype=np.int32)

            self._wall_collision_once[:] = False
            self._obstacle_collision_once[:] = False
            self._drone_collision_once[:] = False

            self._rebuild_query_world_from_current_scene()
            self._targets = self._make_targets()
            self._spawn_obstacles()
            self._spawn_walls()
            self._apply_episode_wall()

            obs0 = self._build_obs_dict()

            for i in range(self.total_num_drones):
                pos = obs0[self.possible_agents[i]][POS_SLICE]
                tgt = self._targets[i]
                self._prev_dists[i] = float(np.linalg.norm(tgt.start_pos - pos))

            info_dict = {a: {} for a in self.agents}

            if self.obstacle_layout == "curriculum":
                for info in info_dict.values():
                    info.update(self._episode_obstacle_counts)
        finally:
            self._set_rendering_enabled(True)

        return obs0, info_dict

    def _set_rendering_enabled(self, enabled: bool) -> None:
        """Helper to temporarily enable/disable rendering"""
        if self._p is None or self.render_mode != "human":
            return
        try:
            self._call_gui(
                self._p.configureDebugVisualizer,
                self._p.COV_ENABLE_RENDERING,
                1 if enabled else 0,
            )
        except Exception:
            pass

    def clear_episode_artifacts(self) -> None:
        """Helper for resetting episode artifacts."""
        self._clear_debug_visuals()
        self.remove_all_obstacles()
        self.remove_targets()
        self.remove_walls()
        # reset spawn positions
        self.spawn_positions = np.empty((0, 3))

    def remove_all_obstacles(self) -> None:
        """Helper for removing obstacles from current episode environment."""
        if self._p is None:
            self._obstacles = []
            return

        for obj in self._obstacles:
            try:
                if getattr(obj, "body_id", None) is not None:
                    self._call_physics(self._p.removeBody, obj.body_id)
            except Exception:
                pass

        self._obstacles = []

    def remove_walls(self) -> None:
        """Helper for removing walls from current episode environment."""
        for body_id in self._wall_ids:
            try:
                self._call_physics(self._p.removeBody, body_id)
            except Exception:
                pass
        self._wall_ids = []

    def remove_targets(self) -> None:
        """Helper for removing targets from current episode environment."""
        if self._p is None or not self._targets:
            self._targets = []
            return

        seen = set()
        for tgt in self._targets:
            if tgt is None:
                continue

            body_id = getattr(tgt, "body_id", None)
            if body_id is None or body_id in seen:
                continue

            seen.add(body_id)
            try:
                self._call_physics(self._p.removeBody, body_id)
            except Exception:
                pass

        self._targets = []

    def restore_drone_visuals_and_collisions(self) -> None:
        """Helper for restoring visuals and collisions from previously despawned drones."""
        if self._p is None:
            return

        p = self._p

        for i, body_id in enumerate(self._drone_ids[: self.max_total_num_drones]):
            swarm_idx = min(i // max(1, self.max_drones_per_swarm), len(COLORS) - 1)
            color_for_drone = COLORS[swarm_idx]
            try:
                self._call_physics(p.setCollisionFilterGroupMask, body_id, -1, 1, 1)
            except Exception:
                pass

            try:
                num_joints = self._call_physics(p.getNumJoints, body_id)
            except Exception:
                num_joints = 0

            for link_idx in range(-1, num_joints):
                try:
                    self._call_physics(
                        p.changeVisualShape,
                        body_id,
                        link_idx,
                        rgbaColor=color_for_drone,
                    )
                except Exception:
                    pass

    def _clear_debug_visuals(self) -> None:
        if self._p is None or self.render_mode != "human":
            self._lidar_line_ids = {}
            return

        p = self._p

        for _, line_ids in list(self._lidar_line_ids.items()):
            for lid in line_ids:
                self._call_gui(p.removeUserDebugItem, lid)
        self._lidar_line_ids = {}

    def observation_space(self, agent):
        """Retrieve the observation space."""
        return self._obs_space

    def action_space(self, agent):
        """Retrieve the action space"""
        return self._action_space

    def render(self):
        return None

    def close(self):
        try:
            self._clear_debug_visuals()
        except Exception:
            pass

        if getattr(self, "_qworld", None) is not None:
            self._qworld.close()
            self._qworld = None

        try:
            if getattr(self, "_env", None) is not None:
                self._env.close()
        except Exception:
            pass
        finally:
            self._env = None
            self._p = None
            self._client_id = None
            self._gui_client_id = None
            self._drone_ids = []
