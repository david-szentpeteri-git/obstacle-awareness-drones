# -*- coding: utf-8 -*-
#NEWLY ADDED FOR CS5478 PROJECT
"""
PPO plus a behaviour-cloning pass toward the blended expert.

Why. PPO only reinforces the student's OWN sampled actions; the expert's part of
the blend merely changes outcomes. In the first assisted run the student reached
the goal in up to 44% of assisted training episodes but only 2-8% on its own -
it was riding the expert, not learning from it. This class adds the direct
signal: after each normal PPO update, the student's mean action is regressed
toward the expert's action on the states the student actually visited.

How it stays out of PPO's way.
  - The PPO update runs first, untouched, on the rollout it was collected from,
    so its likelihood ratios are exactly as SB3 computes them.
  - The imitation pass runs afterwards on the same rollout, as a separate
    optimizer phase. Its weight is `imitation_weight`, which the training loop
    sets to IMITATION_COEF * current beta - so imitation fades out exactly as the
    expert is withdrawn and is zero whenever the student must stand alone.
  - Targets come from the blend wrapper's `expert_action` step info, recorded by
    `ExpertActionRecorder` at the buffer position SB3 is about to write - SB3
    calls `callback.on_step()` before `rollout_buffer.add()`, so they align with
    the observation the student acted on. Steps without a target (beta 0) are
    simply masked out.
  - The loss is MSE on the policy's UNCLIPPED mean action. That also pulls the
    mean back inside the [-1, 1] action box, where the previous runs had 58-78%
    of mean components saturated and clipped.
"""
from __future__ import annotations

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class ImitationPPO(PPO):
    #PPO WITH A WEIGHTED BEHAVIOUR-CLONING PASS AFTER EVERY UPDATE
    def __init__(self, *args, imitation_epochs: int = 1, imitation_batch_size: int = 256, **kwargs):
        self.imitation_epochs = int(imitation_epochs)
        self.imitation_batch_size = int(imitation_batch_size)
        self.imitation_weight = 0.0
        super().__init__(*args, **kwargs)

    #ALLOCATE THE EXPERT-TARGET BUFFER ALONGSIDE SB3'S ROLLOUT BUFFER
    def _setup_model(self) -> None:
        super()._setup_model()
        self._reset_expert_buffer()

    #ZERO TARGETS & VALIDITY MASK, SHAPED LIKE THE ROLLOUT BUFFER
    def _reset_expert_buffer(self) -> None:
        act_dim = int(np.prod(self.action_space.shape))
        self._expert_targets = np.zeros((self.n_steps, self.n_envs, act_dim), dtype=np.float32)
        self._expert_valid = np.zeros((self.n_steps, self.n_envs), dtype=bool)

    #BUFFERS ARE PER-ROLLOUT SCRATCH, NOT MODEL STATE
    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["_expert_targets", "_expert_valid"]

    #STORE THIS STEP'S EXPERT ACTIONS AT THE POSITION SB3 IS ABOUT TO WRITE
    def record_expert(self, infos) -> None:
        pos = int(self.rollout_buffer.pos)

        if pos >= self.n_steps:
            return

        for idx, info in enumerate(infos):
            target = info.get("expert_action") if isinstance(info, dict) else None

            if target is None:
                self._expert_valid[pos, idx] = False

                continue

            self._expert_targets[pos, idx] = np.asarray(target, dtype=np.float32)
            self._expert_valid[pos, idx] = True

    #INVALIDATE THE LAST ROLLOUT'S TARGETS BEFORE SB3 COLLECTS THE NEXT ONE
    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps) -> bool:
        self._expert_valid[:] = False

        return super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)

    #PPO UPDATE FIRST (UNTOUCHED), THEN THE WEIGHTED IMITATION PASS
    def train(self) -> None:
        super().train()
        self._imitation_update()

    #ONE WEIGHTED MSE PASS FROM THE STUDENT'S MEAN ACTION TO THE EXPERT'S ACTION
    def _imitation_update(self) -> None:
        weight = float(self.imitation_weight)
        valid = self._expert_valid

        #SB3 FLATTENS THE BUFFER ENV-MAJOR (swapaxes(0, 1)) WHEN TRAINING STARTS
        if self.rollout_buffer.generator_ready:
            valid = valid.swapaxes(0, 1).reshape(-1)
            targets = self._expert_targets.swapaxes(0, 1).reshape(-1, self._expert_targets.shape[-1])
            observations = self.rollout_buffer.observations.reshape(valid.shape[0], -1)
        else:
            targets = self._expert_targets.reshape(-1, self._expert_targets.shape[-1])
            observations = self.rollout_buffer.observations.reshape(-1, *self.observation_space.shape)
            valid = valid.reshape(-1)

        samples = int(valid.sum())
        self.logger.record("train/imitation_weight", weight)
        self.logger.record("train/imitation_samples", samples)

        if weight <= 0.0 or self.imitation_epochs <= 0 or samples == 0:
            self.logger.record("train/imitation_loss", 0.0)

            return

        obs_t = th.as_tensor(observations[valid], dtype=th.float32, device=self.device)
        tgt_t = th.as_tensor(targets[valid], dtype=th.float32, device=self.device)
        self.policy.set_training_mode(True)
        losses = []

        for _ in range(self.imitation_epochs):
            order = th.randperm(samples, device=self.device)

            for start in range(0, samples, self.imitation_batch_size):
                idx = order[start:start + self.imitation_batch_size]
                mean = self.policy.get_distribution(obs_t[idx]).distribution.mean
                loss = weight * F.mse_loss(mean, tgt_t[idx])
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                losses.append(float(loss.item()))

        self.logger.record("train/imitation_loss", float(np.mean(losses)))


class ExpertActionRecorder(BaseCallback):
    #COPY EACH STEP'S EXPERT ACTIONS INTO THE IMITATION BUFFER
    def _on_step(self) -> bool:
        if isinstance(self.model, ImitationPPO):
            self.model.record_expert(self.locals.get("infos", []))

        return True
