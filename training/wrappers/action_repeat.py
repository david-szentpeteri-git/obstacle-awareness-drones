#NEWLY ADDED FOR CS5478 PROJECT
"""Hold policy actions across physics ticks while retaining tick-level outcomes.

Rewards sum without within-decision discounting. PPO gamma therefore applies
per decision. Physics horizons and info['step'] still count individual ticks.
"""
from __future__ import annotations

import time

import numpy as np

from drone_env.env_3d import DroneSwarm3D


class RepeatedActions(DroneSwarm3D):
    #WALL-CLOCK SECONDS PER DECISION WHEN A VIEWER WATCHES (0 = AS FAST AS PHYSICS RUNS)
    pace_seconds = 0.0

    #VALIDATE ACTION HOLD LENGTH BEFORE ALLOCATING PHYSICS CLIENTS
    def __init__(self, *args, action_repeat=1, **kwargs):
        if isinstance(action_repeat, bool) or not isinstance(action_repeat, (int, np.integer)) or action_repeat < 1:
            raise ValueError("action_repeat must be a positive integer")

        self.action_repeat = int(action_repeat)
        super().__init__(*args, **kwargs)

        if self.action_repeat > 1 and self.max_total_num_drones != 1:
            self.close()
            raise ValueError("action repetition requires exactly one drone")

    #SUM TICK REWARDS & STOP HELD ACTION AT FIRST TERMINAL TRANSITION
    def step(self, actions, allow_rebuild=True):
        started = time.perf_counter()
        rewards_total = {agent: 0.0 for agent in self.possible_agents}
        record_metrics = bool(getattr(self, "record_eval_metrics", False))
        positions = {agent: [] for agent in self.possible_agents}
        velocities = {agent: [] for agent in self.possible_agents}
        min_clearance = {agent: float("inf") for agent in self.possible_agents}

        for tick in range(self.action_repeat):
            tick_actions = {agent: np.array(action, copy=True) for agent, action in actions.items()}
            obs, rewards, terms, truncs, infos = super().step(tick_actions, allow_rebuild=allow_rebuild)

            for agent, reward in rewards.items():
                rewards_total[agent] += float(reward)

                if record_metrics:
                    info = infos[agent]
                    positions[agent].append(np.asarray(info["position"], dtype=np.float32).copy())
                    velocities[agent].append(np.asarray(info["velocity"], dtype=np.float32).copy())
                    min_clearance[agent] = min(min_clearance[agent], float(info["min_obstacle_clearance"]))

            if any(terms.values()) or any(truncs.values()):
                break

        for agent, info in infos.items():
            info["simulation_steps"] = tick + 1

            if record_metrics:
                info["min_obstacle_clearance"] = min_clearance[agent]
                info["substep_positions"] = np.asarray(positions[agent], dtype=np.float32)
                info["substep_velocities"] = np.asarray(velocities[agent], dtype=np.float32)

        if self.pace_seconds > 0:
            time.sleep(max(0.0, self.pace_seconds - (time.perf_counter() - started)))

        return obs, rewards_total, terms, truncs, infos
