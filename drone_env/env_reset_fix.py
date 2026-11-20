# -*- coding: utf-8 -*-
#PREEXISTS, MODIFIED FOR CS5478 PROJECT
from __future__ import annotations

from typing import Any, Optional
import numpy as np

from stable_baselines3.common.vec_env import VecEnvWrapper


#TRUE WHEN EVERY SUPERSUIT SUB-ENV HOLDS EXACTLY ONE AGENT
def _subenvs_are_single_agent(venv) -> bool:
    """
    Walk down to SuperSuit's ConcatVecEnv and check that each of its
    MarkovVectorEnv children wraps a single-agent par_env. That is the
    condition under which a done flag always corresponds to a sub-env that
    has already auto-reset itself (see the note on step_wait).
    """
    env = venv

    while env is not None:
        if type(env).__name__ == "ConcatVecEnv":
            subs = getattr(env, "vec_envs", None)

            if not subs:
                return False

            return all(getattr(s, "num_envs", 0) == 1 for s in subs)

        env = getattr(env, "venv", None)

    return False


class SB3VecResetTupleFix(VecEnvWrapper):
    """
    Fixes Gymnasium/SuperSuit incompatibilities with SB3:

    1) reset() returns (obs, info) -> SB3 expects obs only
    2) step_wait() sometimes returns (obs, rewards, terminations, truncations, infos)
       -> SB3 expects (obs, rewards, dones, infos)

    Stores reset-info in self.last_reset_info.
    """

    def __init__(self, venv):
        super().__init__(venv)
        self.last_reset_info: Optional[Any] = None

        #SINGLE-AGENT SUB-ENVS AUTO-RESET THEMSELVES, SO THE LEGACY MANUAL
        #RESET BELOW IS BOTH REDUNDANT AND DESTRUCTIVE AND IS SKIPPED
        self._skip_manual_reset = _subenvs_are_single_agent(venv)

    def reset(self, **kwargs):
        out = self.venv.reset(**kwargs)

        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
            self.last_reset_info = info
            return obs

        self.last_reset_info = None

        return out

    def step_async(self, actions) -> None:
        self.venv.step_async(actions)

    def step_wait(self):
        out = self.venv.step_wait()

        if isinstance(out, tuple) and len(out) == 5:
            obs, rewards, terminations, truncations, infos = out

            terminations = np.asarray(terminations, dtype=np.bool_)
            truncations = np.asarray(truncations, dtype=np.bool_)

            if terminations.ndim == 0:
                terminations = terminations.reshape(1)
            if truncations.ndim == 0:
                truncations = truncations.reshape(1)

            dones = np.logical_or(terminations, truncations).astype(
                np.bool_, copy=False
            )

            if infos is not None:
                for i in range(min(len(infos), len(dones))):
                    if isinstance(infos[i], dict):
                        infos[i]["terminated"] = bool(terminations[i])
                        infos[i]["truncated"] = bool(truncations[i])
                        infos[i]["TimeLimit.truncated"] = bool(
                            truncations[i] and not terminations[i]
                        )

            if dones.any():
                #LEGACY PATH, KEPT ONLY FOR MULTI-AGENT SUB-ENVS.
                #
                #SuperSuit's MarkovVectorEnv.step already resets its own
                #par_env as soon as that env is done, and returns the new
                #episode's first observation while stashing the final one in
                #info["terminal_observation"]. When each sub-env holds one
                #agent, a done flag therefore ALWAYS refers to a sub-env that
                #has already auto-reset, and this extra ConcatVecEnv.reset()
                #would reset EVERY copy:
                #  - correctness: the copies that were NOT done have their
                #    in-progress episodes silently restarted while SB3 is
                #    still told done=False, so it bootstraps across the
                #    discontinuity and keeps a stale observation for them.
                #  - speed: with 8 copies, one finished episode triggered 8
                #    full world rebuilds (resetSimulation, drone URDF reload,
                #    obstacle respawn with retry loops, walls, targets, query
                #    world). Measured ~15x slowdown once obstacles appear.
                if self._skip_manual_reset:
                    return obs, rewards, dones, infos

                for i in range(len(dones)):
                    if dones[i] and isinstance(infos[i], dict):
                        infos[i]["terminal_observation"] = obs[i].copy()

                new_obs = self.venv.reset()
                if isinstance(new_obs, tuple):
                    new_obs, _ = new_obs

                for i in range(len(dones)):
                    if dones[i]:
                        obs[i] = new_obs[i]

            return obs, rewards, dones, infos

        if isinstance(out, tuple) and len(out) == 4:
            obs, rewards, dones, infos = out

            dones = np.asarray(dones, dtype=np.bool_)
            rewards = np.asarray(rewards, dtype=np.float32)

            if dones.ndim == 0:
                dones = dones.reshape(1)
            if rewards.ndim == 0:
                rewards = rewards.reshape(1)

            if infos is not None:
                for i in range(min(len(infos), len(dones))):
                    if isinstance(infos[i], dict):
                        infos[i]["terminated"] = bool(infos[i].get("terminated", False))

                        if "truncated" in infos[i]:
                            infos[i]["truncated"] = bool(infos[i]["truncated"])
                        elif "TimeLimit.truncated" in infos[i]:
                            infos[i]["truncated"] = bool(
                                infos[i]["TimeLimit.truncated"]
                            )
                        else:
                            infos[i]["truncated"] = False

                        infos[i]["TimeLimit.truncated"] = bool(
                            infos[i]["truncated"] and not infos[i]["terminated"]
                        )

            if dones.any():
                #SEE THE NOTE ON THE 5-TUPLE BRANCH ABOVE
                if self._skip_manual_reset:
                    return obs, rewards, dones, infos

                for i in range(len(dones)):
                    if dones[i] and isinstance(infos[i], dict):
                        infos[i]["terminal_observation"] = obs[i].copy()

                new_obs = self.venv.reset()
                if isinstance(new_obs, tuple):
                    new_obs, _ = new_obs

                for i in range(len(dones)):
                    if dones[i]:
                        obs[i] = new_obs[i]

            assert dones.dtype == np.bool_, f"dones dtype is {dones.dtype}"
            assert dones.shape == (
                self.num_envs,
            ), f"dones shape is {dones.shape}, expected {(self.num_envs,)}"

            return obs, rewards, dones, infos
