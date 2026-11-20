# -*- coding: utf-8 -*-
#NEWLY ADDED FOR CS5478 PROJECT
"""
Observation shaping for the degenerate-vs-full comparison.

Blind students have their obstacle/centroid/teammate and LiDAR slices zeroed.
Revision 2 casts raw rays for the shared full-information teacher below this
wrapper. The vector is padded to the full width if necessary, so every
condition presents the same input shape and network capacity to its student.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnvWrapper


#ZERO MASKED SLICES & PAD OUT TO TARGET WIDTH (WORKS ON 1-D OR BATCHED OBS)
def apply_obs_mask(
    obs: np.ndarray,
    masked_slices: Sequence[slice],
    target_dim: int,
) -> np.ndarray:
    arr = np.array(obs, dtype=np.float32, copy=True)
    single = arr.ndim == 1

    if single:
        arr = arr[None, :]

    cur_dim = arr.shape[-1]

    if cur_dim > target_dim:
        raise ValueError(f"obs width {cur_dim} exceeds target width {target_dim}")

    if cur_dim < target_dim:
        pad = np.zeros((arr.shape[0], target_dim - cur_dim), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)

    for sl in masked_slices:
        arr[:, sl] = 0.0

    return arr[0] if single else arr


class MaskObservations(VecEnvWrapper):
    """
    Applies apply_obs_mask to every observation leaving the vec env, and
    advertises the padded width as the observation space so SB3 builds a
    policy of the right shape.
    """

    #ADVERTISE PADDED WIDTH SO SB3 BUILDS A POLICY OF THE RIGHT SHAPE
    def __init__(
        self,
        venv,
        masked_slices: Sequence[slice],
        target_dim: int,
    ) -> None:
        obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(target_dim),),
            dtype=np.float32,
        )
        super().__init__(venv, observation_space=obs_space)

        self.masked_slices = tuple(masked_slices)
        self.target_dim = int(target_dim)

    #MASK + PAD ONE OBS BATCH
    def _transform(self, obs: np.ndarray) -> np.ndarray:
        return apply_obs_mask(obs, self.masked_slices, self.target_dim)

    #RESET & SHAPE THE FIRST OBS
    def reset(self):
        return self._transform(self.venv.reset())

    #SHAPE EVERY OBS LEAVING THE VEC ENV, TERMINAL OBS INCLUDED
    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()

        #TERMINAL OBS IS READ BY SB3 FOR BOOTSTRAPPING -> MASK IT TOO
        for info in infos:
            if isinstance(info, dict) and "terminal_observation" in info:
                info["terminal_observation"] = self._transform(
                    info["terminal_observation"]
                )

        return self._transform(obs), rewards, dones, infos
