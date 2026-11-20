# -*- coding: utf-8 -*-
#NEWLY ADDED FOR CS5478 PROJECT
"""
Rollout-level statistics for the PPO runs. Reads the flattened per-drone-slot
info dicts produced by wrappers/step_logic.build_info_dict, records
reach/collision rates to TensorBoard AND appends one CSV row per rollout so
learning curves for all four conditions can be rebuilt offline on identical
axes (results/train_logs/<condition>.csv).
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

CSV_FIELDS = [
    "condition",
    "seed",
    "stage",
    "timesteps",
    "decisions",
    "physics_ticks",
    "stage_physics_ticks",
    "rollout_physics_ticks",
    "wall_time_s",
    "stage_wall_time_s",
    "episodes",
    "reaches",
    "wall_hits",
    "obstacle_hits",
    "drone_hits",
    "collisions",
    "timeouts",
    "reach_rate",
    "collision_rate",
    "timeout_rate",
    "ep_rew_mean",
    "ep_len_mean",
    "ep_len_mean_decisions",
    "completed_ep_len_mean_ticks",
]

EPISODE_FIELDS = [
    "condition", "seed", "stage", "episode", "decisions_after", "physics_ticks_after",
    "episode_decisions", "episode_physics_ticks", "raw_return", "reached", "collided",
    "timeout", "wall_hit", "obstacle_hit", "drone_hit", "final_distance", "wall_time_s",
    "static_obstacles", "dynamic_obstacles", "blocking_obstacle_requested",
    "blocking_obstacle_spawned", "obstacle_scene_attempts",
]


class RolloutStatsCallback(BaseCallback):
    """
    Counts goal reaches and collisions per rollout, records them under
    rollout_stats/* alongside the usual PPO diagnostics, and appends the same
    numbers to a per-condition CSV for offline plotting.
    """

    #SET UP TALLIES FOR ONE CURRICULUM STAGE
    def __init__(
        self,
        stage_idx: int = 0,
        condition_name: str = "",
        seed: int = 0,
        csv_path: Optional[str] = None,
        print_every: int = 10,
        verbose: int = 0,
        run_start_time: Optional[float] = None,
        physics_ticks_before: int = 0,
    ):
        super().__init__(verbose)

        self.stage_idx = int(stage_idx)
        self.condition_name = str(condition_name)
        self.seed = int(seed)
        self.csv_path = Path(csv_path) if csv_path else None
        self.print_every = int(print_every)
        self._rollout_num = 0
        self._start_time = time.monotonic()
        self._run_start_time = run_start_time if run_start_time is not None else self._start_time
        self.physics_ticks = int(physics_ticks_before)
        self.stage_physics_ticks = 0
        self.stage_episodes = 0
        self.episode_csv_path = self.csv_path.with_name(self.csv_path.stem + "_episodes.csv") if self.csv_path else None
        self._reset_counters()

        if self.csv_path is not None:
            self._ensure_csv()

    #CREATE CSV WITH HEADER IF MISSING
    def _ensure_csv(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=CSV_FIELDS).writeheader()

        if not self.episode_csv_path.exists():
            with self.episode_csv_path.open("w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=EPISODE_FIELDS).writeheader()

    #CLEAR PER-ROLLOUT TALLIES
    def _reset_counters(self) -> None:
        self.active_steps = 0
        self.reaches = 0
        self.wall_hits = 0
        self.obstacle_hits = 0
        self.drone_hits = 0
        self.timeouts = 0
        self.collisions = 0
        self.episodes = 0
        self.rollout_physics_ticks = 0
        self.completed_episode_ticks = []

    #START A FRESH TALLY EACH ROLLOUT
    def _on_rollout_start(self) -> None:
        self._reset_counters()

    #TALLY REACH/COLLISION/TIMEOUT EVENTS ACROSS ALL DRONE SLOTS
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for idx, info in enumerate(infos):
            if not isinstance(info, dict):
                continue

            ticks = int(info.get("simulation_steps", 1))
            self.physics_ticks += ticks
            self.stage_physics_ticks += ticks
            self.rollout_physics_ticks += ticks

            if info.get("active", False):
                self.active_steps += 1

            if idx < len(dones) and dones[idx]:
                self.episodes += 1
                reached = bool(info.get("reached_now", False))
                collided = any(bool(info.get(key, False)) for key in (
                    "wall_collision_now", "obstacle_collision_now", "drone_collision_now",
                ))
                self.reaches += int(reached)
                self.wall_hits += int(info.get("wall_collision_now", False))
                self.obstacle_hits += int(info.get("obstacle_collision_now", False))
                self.drone_hits += int(info.get("drone_collision_now", False))
                self.collisions += int(collided and not reached)

                if info.get("truncated", False) and not reached and not collided:
                    self.timeouts += 1

                self.stage_episodes += 1
                terminal_ticks = int(info.get("step", 0))
                self.completed_episode_ticks.append(terminal_ticks)

                if self.episode_csv_path is not None:
                    episode = info.get("episode", {})
                    row = {
                        "condition": self.condition_name, "seed": self.seed, "stage": self.stage_idx,
                        "episode": self.stage_episodes, "decisions_after": int(self.num_timesteps),
                        "physics_ticks_after": self.physics_ticks,
                        "episode_decisions": episode.get("l", ""), "episode_physics_ticks": terminal_ticks,
                        "raw_return": episode.get("r", ""), "reached": int(reached),
                        "collided": int(collided and not reached),
                        "timeout": int(info.get("truncated", False) and not reached and not collided),
                        "wall_hit": int(info.get("wall_collision_now", False)),
                        "obstacle_hit": int(info.get("obstacle_collision_now", False)),
                        "drone_hit": int(info.get("drone_collision_now", False)),
                        "final_distance": info.get("dist_to_target", ""),
                        "wall_time_s": round(time.monotonic() - self._run_start_time, 3),
                        "static_obstacles": info.get("static_obstacles", ""),
                        "dynamic_obstacles": info.get("dynamic_obstacles", ""),
                        "blocking_obstacle_requested": int(bool(info["blocking_obstacle_requested"])) if "blocking_obstacle_requested" in info else "",
                        "blocking_obstacle_spawned": int(bool(info["blocking_obstacle_spawned"])) if "blocking_obstacle_spawned" in info else "",
                        "obstacle_scene_attempts": info.get("obstacle_scene_attempts", ""),
                    }
                    with self.episode_csv_path.open("a", newline="", encoding="utf-8") as fh:
                        csv.DictWriter(fh, fieldnames=EPISODE_FIELDS).writerow(row)

        return True

    #MEAN EPISODE REWARD/LENGTH FROM SB3'S ROLLING BUFFER (NAN IF EMPTY)
    def _ep_info_means(self) -> tuple[float, float]:
        buf = getattr(self.model, "ep_info_buffer", None)

        if not buf:
            return float("nan"), float("nan")

        rews = [e["r"] for e in buf if "r" in e]
        lens = [e["l"] for e in buf if "l" in e]

        return (
            float(np.mean(rews)) if rews else float("nan"),
            float(np.mean(lens)) if lens else float("nan"),
        )

    #RECORD RATES TO TENSORBOARD + CSV & OPTIONALLY PRINT A PROGRESS LINE
    def _on_rollout_end(self) -> None:
        self._rollout_num += 1
        episodes = max(1, self.episodes)
        collisions = self.collisions
        ep_rew_mean, ep_len_mean = self._ep_info_means()

        self.logger.record("rollout_stats/stage", self.stage_idx)
        self.logger.record("rollout_stats/reaches_per_episode", self.reaches / episodes)
        self.logger.record("rollout_stats/collisions_per_episode", collisions / episodes)
        self.logger.record("rollout_stats/timeout_rate", self.timeouts / episodes)
        self.logger.record("rollout_stats/episodes", self.episodes)
        self.logger.record("rollout_stats/physics_ticks", self.physics_ticks)
        self.logger.record("rollout_stats/stage_physics_ticks", self.stage_physics_ticks)
        self.logger.record("rollout_stats/decisions", self.num_timesteps)

        if self.csv_path is not None:
            row = {
                "condition": self.condition_name,
                "seed": self.seed,
                "stage": self.stage_idx,
                "timesteps": int(self.num_timesteps),
                "decisions": int(self.num_timesteps),
                "physics_ticks": self.physics_ticks,
                "stage_physics_ticks": self.stage_physics_ticks,
                "rollout_physics_ticks": self.rollout_physics_ticks,
                "wall_time_s": round(time.monotonic() - self._run_start_time, 1),
                "stage_wall_time_s": round(time.monotonic() - self._start_time, 1),
                "episodes": self.episodes,
                "reaches": self.reaches,
                "wall_hits": self.wall_hits,
                "obstacle_hits": self.obstacle_hits,
                "drone_hits": self.drone_hits,
                "collisions": collisions,
                "timeouts": self.timeouts,
                "reach_rate": round(self.reaches / episodes, 4),
                "collision_rate": round(collisions / episodes, 4),
                "timeout_rate": round(self.timeouts / episodes, 4),
                "ep_rew_mean": round(ep_rew_mean, 3) if np.isfinite(ep_rew_mean) else "",
                "ep_len_mean": round(ep_len_mean, 1) if np.isfinite(ep_len_mean) else "",
                "ep_len_mean_decisions": round(ep_len_mean, 1) if np.isfinite(ep_len_mean) else "",
                "completed_ep_len_mean_ticks": round(float(np.mean(self.completed_episode_ticks)), 1) if self.completed_episode_ticks else "",
            }

            with open(self.csv_path, "a", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=CSV_FIELDS).writerow(row)

        if self.verbose > 0 and self._rollout_num % self.print_every == 0:
            print(
                f"[{self.condition_name} s{self.seed} | stage {self.stage_idx} | "
                f"{self.num_timesteps} steps] "
                f"episodes={self.episodes} reaches={self.reaches} "
                f"collisions={collisions} (wall={self.wall_hits} "
                f"obs={self.obstacle_hits} drone={self.drone_hits}) "
                f"timeouts={self.timeouts}"
            )
