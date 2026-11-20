# -*- coding: utf-8 -*-
#NEWLY ADDED FOR CS5478 PROJECT
"""
Constant reward scaling for the PPO optimizer only.

This sits OUTSIDE the VecMonitor in the training stack, which is the whole
point: the monitor keeps recording unscaled episode returns, so the learning
CSVs and TensorBoard stay comparable with every other run, while PPO's value
function and advantages see the smaller numbers.

The scale reduces value-target magnitudes for this task's terminal rewards
(+250 at the goal and -50 on collision). A positive constant preserves the
ordering of task returns. PPO's advantage normalization, value and entropy
weights, clipping and Adam make its optimization response different from a
learning-rate change; the scale is an explicit experiment setting.
"""
from __future__ import annotations

import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper


class ScaleReward(VecEnvWrapper):
    #VALIDATE UP FRONT - A ZERO OR NEGATIVE SCALE SILENTLY DESTROYS TRAINING
    def __init__(self, venv, scale: float) -> None:
        scale = float(scale)

        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("reward scale must be finite and positive")

        super().__init__(venv)
        self.scale = scale

    #PASS OBSERVATIONS THROUGH UNCHANGED
    def reset(self):
        return self.venv.reset()

    #SCALE ONLY THE REWARD HANDED TO THE OPTIMIZER
    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()

        return obs, np.asarray(rewards, dtype=np.float32) * self.scale, dones, infos
