# -*- coding: utf-8 -*-
#NEWLY ADDED FOR CS5478 PROJECT
"""
Expert action blending for PPO: the env executes `beta*expert + (1-beta)*student`
while PPO's rollout buffer keeps the STUDENT's sampled action and its matching
log probability.

Why this is an env wrapper and not a policy subclass. SB3's on-policy collector
samples an action, records that action together with the log probability it just
computed for it, and only then hands the action to the env. Nothing downstream
re-reads what the env actually executed. Blending on the env side therefore
leaves PPO's likelihood ratios self-consistent for free. A policy-side override
of the forward path cannot do this: with an on-policy collector it makes the
stored log probability belong to a different action than the stored one, which
silently corrupts the ratios.

Teacher information (decided 2026-10-12). The wrapper sits BELOW the observation
mask (and below normalization), so the teacher reads the raw, unmasked
observation in metres - nearest-obstacle vector and all six rays - in EVERY
condition, including the blind ones. The student's observation is masked above
this wrapper exactly as its condition requires; only the teacher is privileged.
The report must disclose that obstacle knowledge reaches blind-condition
training through the teacher's demonstrations.

Imitation. Every assisted step also publishes the expert's action in the step
info (`expert_action`) and in `last_expert_actions`, aligned with the
observation the student acted on, so a behaviour-cloning term can train the
student toward it. At beta 0 nothing is published: the student stands alone.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper


class VecExpertBlend(VecEnvWrapper):
    #BLEND EXECUTED ACTIONS WITH AN EXPERT WHILE PPO KEEPS STUDENT ACTIONS
    def __init__(self, venv: VecEnv, expert, beta: float = 0.0, action_low: float = -1.0, action_high: float = 1.0):
        super().__init__(venv)

        if expert is None:
            raise ValueError("expert blending needs an expert policy")

        self.expert = expert
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self._last_obs: Optional[np.ndarray] = None
        self._pending_expert: Optional[np.ndarray] = None
        self.last_expert_actions: Optional[np.ndarray] = None
        self._beta = 0.0
        self.set_beta(beta)
        self.reset_diagnostics()

    #EXPERT WEIGHT IN [0, 1]; 0 = STUDENT ONLY. CHANGED ONLY AT ROLLOUT EDGES
    def set_beta(self, beta: float) -> None:
        beta = float(beta)

        if not np.isfinite(beta) or beta < 0.0 or beta > 1.0:
            raise ValueError("blend weight beta must be finite and within [0, 1]")

        self._beta = beta

    #CURRENT EXPERT WEIGHT
    @property
    def beta(self) -> float:
        return self._beta

    #ZERO THE RUNNING STUDENT/EXPERT/EXECUTED ACTION DIAGNOSTICS
    def reset_diagnostics(self) -> None:
        self._diag = dict(
            steps=0,
            assisted_steps=0,
            student_norm=0.0,
            expert_norm=0.0,
            exec_norm=0.0,
            dir_shift_cos=0.0,
            speed_delta=0.0,
        )

    #MEAN STUDENT/EXPERT/EXECUTED ACTION STATISTICS SINCE THE LAST RESET
    def diagnostics(self) -> dict:
        steps = max(self._diag["steps"], 1)

        return {
            "blend_steps": self._diag["steps"],
            "blend_assisted_steps": self._diag["assisted_steps"],
            "blend_beta": self._beta,
            "blend_student_action_norm": self._diag["student_norm"] / steps,
            "blend_expert_action_norm": self._diag["expert_norm"] / steps,
            "blend_executed_action_norm": self._diag["exec_norm"] / steps,
            "blend_direction_shift_cos": self._diag["dir_shift_cos"] / steps,
            "blend_speed_delta": self._diag["speed_delta"] / steps,
        }

    #RESET & CACHE THE RAW OBS THE EXPERT READS AT THE NEXT STEP
    def reset(self):
        obs = self.venv.reset()
        self._last_obs = np.asarray(obs, dtype=np.float32).copy()

        return obs

    #UNIT-DIRECTION BLEND (SPEED LINEAR): BETA = THE EXPERT'S REAL SHARE OF THE STEERING
    def blend(self, student: np.ndarray, expert: np.ndarray, beta: float) -> np.ndarray:
        """
        The simulator normalises the first three action components before flying
        them, so a linear mix lets whichever direction has the larger magnitude
        win. An untrained student's direction output is near zero, so under a
        linear mix the expert steered almost completely even at beta 0.2 - an
        untrained student + expert reached 100% of stage-0 goals at beta
        0.8, 0.6, 0.4 AND 0.2, which would have stripped all help within hours
        without the student learning anything. Mixing the two UNIT directions
        makes beta the expert's actual share of where the drone flies (measured
        with an untrained student: 100% / 70% / 8% / 0% at beta 0.8 / 0.6 /
        0.4 / 0.2). Speed is a scalar and is mixed linearly.
        """
        student = np.clip(np.asarray(student, dtype=np.float32), self.action_low, self.action_high)
        expert = np.asarray(expert, dtype=np.float32)
        executed = np.empty_like(student)

        for idx in range(student.shape[0]):
            executed[idx, :3] = beta * self._unit_dir(expert[idx]) + (1.0 - beta) * self._unit_dir(student[idx])

        executed[:, 3] = beta * expert[:, 3] + (1.0 - beta) * student[:, 3]

        return np.clip(executed, self.action_low, self.action_high).astype(np.float32)

    #UNIT DIRECTION THE SIMULATOR WILL ACTUALLY FLY FOR AN ACTION ROW (ZERO IF ~0)
    @staticmethod
    def _unit_dir(action_row: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(action_row[0:3]))

        if norm <= 1e-8:
            return np.zeros(3, dtype=np.float32)

        return (action_row[0:3] / norm).astype(np.float32)

    #HAND THE SIMULATOR THE BLENDED ACTION; PPO'S BUFFERED STUDENT ACTION STAYS UNTOUCHED
    def step_async(self, actions: np.ndarray) -> None:
        student = np.asarray(actions, dtype=np.float32)

        #beta = 0 IS A STRICT PASS-THROUGH: NO EXPERT CALL, NO MUTATION
        if self._beta <= 0.0 or self._last_obs is None:
            self._pending_expert = None
            self._record(student, student, student, assisted=False)
            self.venv.step_async(actions)

            return

        expert = np.zeros_like(student)

        for idx in range(student.shape[0]):
            expert[idx] = np.asarray(self.expert.action(self._last_obs[idx]), dtype=np.float32)

        executed = self.blend(student, expert, self._beta)
        self._pending_expert = expert.copy()
        self._record(student, expert, executed, assisted=True)

        #THE STUDENT ARRAY PPO ALREADY BUFFERED IS LEFT UNTOUCHED - ONLY THE
        #COPY HANDED TO THE SIMULATOR CARRIES THE BLEND
        self.venv.step_async(executed)

    #FORWARD STEP RESULTS; PUBLISH EXPERT ACTION & BETA IN info FOR IMITATION
    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self._last_obs = np.asarray(obs, dtype=np.float32).copy()
        self.last_expert_actions = self._pending_expert

        #EXPERT ACTION FOR THE STATE THE STUDENT JUST ACTED ON (IMITATION TARGET)
        if self._pending_expert is not None:
            for idx, info in enumerate(infos):
                info["expert_action"] = self._pending_expert[idx].copy()
                info["blend_beta"] = self._beta

        return obs, rewards, dones, infos

    #ACCUMULATE PER-STEP ACTION DIAGNOSTICS (ASSISTANCE ACTUALLY APPLIED)
    def _record(self, student: np.ndarray, expert: np.ndarray, executed: np.ndarray, assisted: bool) -> None:
        rows = student.shape[0]
        self._diag["steps"] += rows

        if assisted:
            self._diag["assisted_steps"] += rows

        self._diag["student_norm"] += float(np.linalg.norm(student, axis=1).sum())
        self._diag["expert_norm"] += float(np.linalg.norm(expert, axis=1).sum())
        self._diag["exec_norm"] += float(np.linalg.norm(executed, axis=1).sum())

        for idx in range(rows):
            self._diag["dir_shift_cos"] += float(np.dot(self._unit_dir(student[idx]), self._unit_dir(executed[idx])))
            self._diag["speed_delta"] += float(abs(executed[idx, 3] - student[idx, 3]))
