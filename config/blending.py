# -*- coding: utf-8 -*-
#NEWLY ADDED FOR CS5478 PROJECT
"""
Frozen parameters for the adaptive expert-blending curriculum.

These values are SELECTED, not validated. They are recorded here, and written
into every run manifest, so a run is reproducible from its artifacts and so the
report can state what the protocol was rather than what it was meant to be.

Protocol (revision 2, 2026-10-12). Every stage, including stage 0, starts with
the expert at `BETA_START` (0.8) of the executed action, capped at `BETA_CAP`.
Every `CHECK_INTERVAL_DECISIONS` policy decisions a frozen, deterministic
validation batch runs WITH THE CURRENT BLEND - the student's action mixed with
the expert's at the weight it is currently training under. Success judged
student-only would never let beta move: at beta 0.8 the student alone scores
near zero. If that success is at least `SUCCESS_HIGH` on `PERSISTENCE_Y`
consecutive checks, beta falls by `BETA_DECREMENT`; if it is at most
`SUCCESS_LOW` on `PERSISTENCE_Z` consecutive checks, beta rises by
`BETA_INCREMENT` (never above the cap); otherwise beta holds. Any change resets
both streaks and starts a `COOLDOWN_CHECKS` cooldown, which stops oscillation.
At beta 0 the blend is the student alone, so the same validation becomes the
unassisted gate: success strictly above the advance threshold on
`CONSECUTIVE_CHECKS_TO_ADVANCE` consecutive checks passes the stage, and
`PERSISTENCE_Z` checks at or below `SUCCESS_LOW` send assistance back up by one
`BETA_INCREMENT`. Training is unbounded by default; an explicitly requested
stage ceiling stops a run as STALLED rather than advancing it without passing.

Imitation. The student also takes `IMITATION_EPOCHS` behaviour-cloning passes
per rollout toward the expert's action on the states it visited, weighted by
`IMITATION_COEF * beta`, so imitation fades out exactly as assistance does and
is zero once the student must stand alone. PPO alone only reinforces the
student's own sampled actions; the last assisted run showed a 28-42 point gap
between assisted and student-only success, i.e. the student was not absorbing
the expert.

Persistence is counted in validation checks and policy decisions, never in wall
clock, so a slow machine cannot change the curriculum.
"""
from __future__ import annotations

#SUCCESS GATE TO LEAVE A STAGE AT BETA 0 (STRICTLY GREATER THAN)
ADVANCE_THRESHOLD_STAGE0 = 0.40
ADVANCE_THRESHOLD_LATER = 0.40

#HYSTERESIS BAND ON SUCCESS WITH CURRENT HELP (LOW < HIGH); BETWEEN THEM BETA HOLDS
SUCCESS_LOW = 0.30
SUCCESS_HIGH = 0.50

#HELP COMES OFF IN LARGE STEPS, GOES BACK ON IN SMALLER ONES
BETA_DECREMENT = 0.2
BETA_INCREMENT = 0.1

#EVERY STAGE STARTS WITH HEAVY ASSISTANCE AND IS CAPPED THERE
BETA_START = (0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8)
BETA_CAP = (0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8)

#VALIDATION BATCH: BIG ENOUGH THAT A 40% GATE IS NOT NOISE
VALIDATION_EPISODES = 200
CONSECUTIVE_CHECKS_TO_ADVANCE = 2

#DECISIONS BETWEEN CHECKS = 40 PPO ROLLOUTS AT n_steps 512, SO SEVERAL UPDATES
#ALWAYS SEPARATE TWO BETA ADJUSTMENTS
CHECK_INTERVAL_DECISIONS = 20_480

#CONSECUTIVE CHECKS BEFORE BETA MOVES (Y = DECREASE, Z = INCREASE)
PERSISTENCE_Y = 2
PERSISTENCE_Z = 3

#CHECKS TO WAIT AFTER ANY BETA CHANGE BEFORE THE NEXT ONE
COOLDOWN_CHECKS = 1

#NO EXTRA UNASSISTED WAITING PERIOD: THE 2-CONSECUTIVE-CHECK GATE STARTS AT BETA 0
UNASSISTED_CONSOLIDATION_DECISIONS = 0

#EXAMPLE OPT-IN CEILING USED BY REGRESSIONS; TRAINING HAS NO DEFAULT CEILING
MAX_STAGE_DECISIONS = 400_000

#VALIDATION: DETERMINISTIC STUDENT ACTION, BLENDED AT THE CURRENT BETA
VALIDATION_DETERMINISTIC = True
VALIDATION_USES_CURRENT_BETA = True

#BEHAVIOUR CLONING TOWARD THE EXPERT; EFFECTIVE WEIGHT = COEF * CURRENT BETA
IMITATION_COEF = 1.0
IMITATION_EPOCHS = 1
IMITATION_BATCH_SIZE = 256


#OPTIONAL STAGE CEILING ROUNDED DOWN TO WHOLE ROLLOUTS (NONE/0 = UNBOUNDED)
def resolve_stage_ceiling(requested: int | None, rollout_steps: int) -> int | None:
    """A maximum must leave room for whole rollouts without exceeding it."""
    if requested is None or requested == 0:
        return None
    if requested < 0 or rollout_steps <= 0:
        raise ValueError("stage ceiling must be nonnegative and rollout size positive")

    effective = int(requested) // int(rollout_steps) * int(rollout_steps)

    if effective < rollout_steps:
        raise ValueError("stage ceiling must permit at least one complete PPO rollout")

    return effective


#EVERY FROZEN PARAMETER AS ONE DICT FOR THE RUN MANIFEST
def blending_settings() -> dict:
    return {
        "protocol_revision": 2,
        "blend_mode": "unit_direction",
        "advance_threshold_stage0": ADVANCE_THRESHOLD_STAGE0,
        "advance_threshold_later": ADVANCE_THRESHOLD_LATER,
        "success_low": SUCCESS_LOW,
        "success_high": SUCCESS_HIGH,
        "beta_decrement": BETA_DECREMENT,
        "beta_increment": BETA_INCREMENT,
        "beta_start": list(BETA_START),
        "beta_cap": list(BETA_CAP),
        "validation_episodes": VALIDATION_EPISODES,
        "consecutive_checks_to_advance": CONSECUTIVE_CHECKS_TO_ADVANCE,
        "check_interval_decisions": CHECK_INTERVAL_DECISIONS,
        "persistence_y": PERSISTENCE_Y,
        "persistence_z": PERSISTENCE_Z,
        "cooldown_checks": COOLDOWN_CHECKS,
        "unassisted_consolidation_decisions": UNASSISTED_CONSOLIDATION_DECISIONS,
        "default_stage_ceiling": None,
        "validation_deterministic": VALIDATION_DETERMINISTIC,
        "validation_uses_current_beta": VALIDATION_USES_CURRENT_BETA,
        "imitation_coef": IMITATION_COEF,
        "imitation_epochs": IMITATION_EPOCHS,
        "imitation_batch_size": IMITATION_BATCH_SIZE,
    }


#REJECT AN INTERNALLY INCONSISTENT PARAMETER SET BEFORE ANY COMPUTE IS SPENT
def validate_blending_settings(num_stages: int) -> None:
    if not 0.0 <= SUCCESS_LOW < SUCCESS_HIGH <= 1.0:
        raise ValueError("blending thresholds must satisfy 0 <= SUCCESS_LOW < SUCCESS_HIGH <= 1")
    if not 0.0 < BETA_DECREMENT <= 1.0 or not 0.0 < BETA_INCREMENT <= 1.0:
        raise ValueError("beta steps must be in (0, 1]")
    if len(BETA_START) < num_stages or len(BETA_CAP) < num_stages:
        raise ValueError("BETA_START and BETA_CAP need at least one entry per curriculum stage")
    if not SUCCESS_LOW < ADVANCE_THRESHOLD_STAGE0 < SUCCESS_HIGH or not SUCCESS_LOW < ADVANCE_THRESHOLD_LATER < SUCCESS_HIGH:
        raise ValueError("advance thresholds must sit inside the hysteresis band")
    if VALIDATION_EPISODES < 50 or CONSECUTIVE_CHECKS_TO_ADVANCE < 1:
        raise ValueError("validation batches must be adequately sized and repeated")
    if PERSISTENCE_Y < 1 or PERSISTENCE_Z < 1 or COOLDOWN_CHECKS < 0:
        raise ValueError("persistence counters must be positive and cooldown nonnegative")
    if UNASSISTED_CONSOLIDATION_DECISIONS < 0:
        raise ValueError("consolidation decisions must be nonnegative")
    if IMITATION_COEF < 0.0 or IMITATION_EPOCHS < 0 or IMITATION_BATCH_SIZE < 1:
        raise ValueError("imitation settings must be nonnegative with a positive batch size")

    for stage, (start, cap) in enumerate(zip(BETA_START, BETA_CAP)):
        if not 0.0 <= start <= cap <= 1.0:
            raise ValueError(f"stage {stage}: need 0 <= BETA_START <= BETA_CAP <= 1")
