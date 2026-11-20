# -*- coding: utf-8 -*-
#NEWLY ADDED FOR CS5478 PROJECT
"""
Success-driven expert-assistance scheduler with hysteresis.

A time-only linear decay would lower the expert weight on elapsed timesteps
regardless of whether the student had learned anything. Here the weight only
moves on validated success measured
WITH THE CURRENT BLEND, and a stage is only left once beta is 0 - when that same
measurement is the student alone - and the student passes.

One `StageScheduler` owns one curriculum stage. Training calls `observe()` once
per validation check; everything else - the weight, the stage verdict, the full
audit trail - is derived from those checks. Every decision this object makes is
appended to `history` with the reason that caused it, because an assisted run
whose assistance schedule is not reconstructible afterwards is not evidence.

States: ASSISTED (beta may move) -> CONSOLIDATING (beta 0, optional unassisted
training) -> GATING (counting consecutive passing checks) -> PASSED. Help comes
off in `BETA_DECREMENT` steps and goes back on in `BETA_INCREMENT` steps.
Advancement is driven by validated success alone; a stage is STALLED only if an
optional ceiling is set and reached. Persistently low success at beta 0 sends
the stage back to ASSISTED by one increment.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import List, Optional

from config.blending import (
    ADVANCE_THRESHOLD_LATER,
    ADVANCE_THRESHOLD_STAGE0,
    BETA_CAP,
    BETA_DECREMENT,
    BETA_INCREMENT,
    BETA_START,
    CONSECUTIVE_CHECKS_TO_ADVANCE,
    COOLDOWN_CHECKS,
    PERSISTENCE_Y,
    PERSISTENCE_Z,
    SUCCESS_HIGH,
    SUCCESS_LOW,
    UNASSISTED_CONSOLIDATION_DECISIONS,
)

ASSISTED = "ASSISTED"
CONSOLIDATING = "CONSOLIDATING"
GATING = "GATING"
PASSED = "PASSED"
STALLED = "STALLED"


@dataclass
class StageScheduler:
    stage_index: int
    #None = UNBOUNDED: THE STAGE ADVANCES ON VALIDATED SUCCESS AND NOTHING
    #ELSE. A NUMBER IS AN OPT-IN SAFETY VALVE THAT RECORDS A STALL INSTEAD.
    ceiling: Optional[int] = None
    assistance_enabled: bool = True
    beta: float = 0.0
    state: str = ASSISTED
    low_streak: int = 0
    high_streak: int = 0
    pass_streak: int = 0
    cooldown: int = 0
    checks: int = 0
    decisions_at_beta_zero: Optional[int] = None
    history: List[dict] = field(default_factory=list)

    #INITIALIZE WEIGHT/STATE FROM THE FROZEN PER-STAGE PARAMETERS
    def __post_init__(self) -> None:
        self.beta = float(BETA_START[self.stage_index]) if self.assistance_enabled else 0.0
        self.state = CONSOLIDATING if self.beta <= 0.0 else ASSISTED

        if self.beta <= 0.0:
            self.decisions_at_beta_zero = 0

        self._log("init", 0, None, f"stage {self.stage_index} starts at beta {self.beta:.3f}")

    #STAGE-0 OR LATER-STAGE UNASSISTED PASS MARK
    @property
    def advance_threshold(self) -> float:
        return ADVANCE_THRESHOLD_STAGE0 if self.stage_index == 0 else ADVANCE_THRESHOLD_LATER

    #PASSED OR STALLED: NO FURTHER OBSERVATION CHANGES THIS STAGE
    @property
    def finished(self) -> bool:
        return self.state in (PASSED, STALLED)

    #COUNTERS, WEIGHT & FULL TRANSITION HISTORY AS ONE DICT (FOR CHECKPOINTS)
    def snapshot(self) -> dict:
        """Save the current counters, weight and full transition history."""
        return asdict(self)

    #REBUILD FROM A SNAPSHOT WITHOUT REINITIALISING BETA; A LIFTED CEILING UN-STALLS
    @classmethod
    def restore(cls, saved: dict, ceiling: Optional[int] = None) -> "StageScheduler":
        """Restore persistence/cooldown exactly, without reinitializing beta."""
        scheduler = cls(saved["stage_index"], ceiling=ceiling,
                        assistance_enabled=saved["assistance_enabled"])

        for name in scheduler.__dataclass_fields__:
            if name != "ceiling":
                setattr(scheduler, name, saved[name])
        if scheduler.state == STALLED:
            scheduler.state = next(event["state"] for event in reversed(scheduler.history)
                                   if event["state"] != STALLED)
            scheduler._log("resume", saved["history"][-1]["decisions"], None,
                           "optional ceiling lifted/extended; persistence counters retained")

        return scheduler

    #APPEND ONE AUDITABLE SCHEDULER EVENT
    def _log(self, event: str, decisions: int, success: Optional[float], reason: str) -> None:
        self.history.append(
            {
                "event": event,
                "stage": self.stage_index,
                "check": self.checks,
                "decisions": int(decisions),
                "success": None if success is None else round(float(success), 6),
                "beta": round(float(self.beta), 6),
                "state": self.state,
                "reason": reason,
            }
        )

    #ZERO SUCCESS WINDOW & PERSISTENCE COUNTERS, START COOLDOWN
    def _reset_after_change(self) -> None:
        self.low_streak = 0
        self.high_streak = 0
        self.pass_streak = 0
        self.cooldown = COOLDOWN_CHECKS

    #APPLY ONE VALIDATION RESULT; RETURN THE WEIGHT TO TRAIN THE NEXT BLOCK AT
    def observe(self, success_rate: float, decisions: int) -> float:
        """
        `success_rate` must come from a frozen-policy, deterministic validation
        batch blended at the CURRENT beta, on curriculum-validation seeds - never
        from training rollouts (exploratory) and never from final test seeds
        (would leak). At beta 0 that is the student alone.
        """
        if self.finished:
            return self.beta

        success = float(success_rate)

        if not math.isfinite(success) or not 0.0 <= success <= 1.0:
            raise ValueError("validation success must be finite and within [0, 1]")
        if decisions < 0:
            raise ValueError("stage decisions must be nonnegative")

        self.checks += 1

        if self.state == ASSISTED:
            self._step_assisted(success, decisions)
        elif self._collapsed(success, decisions):
            pass
        elif self.state == CONSOLIDATING:
            self._step_consolidating(success, decisions)
        elif self.state == GATING:
            self._step_gating(success, decisions)

        #AN OPT-IN CEILING IS A HARD STOP: RECORD A STALL, NEVER AUTO-ADVANCE.
        #WITHOUT ONE THE STAGE SIMPLY KEEPS TRAINING UNTIL ITS GATE OPENS.
        if not self.finished and self.ceiling is not None and decisions >= self.ceiling:
            self.state = STALLED
            self._log("stalled", decisions, success,
                      f"stage ceiling {self.ceiling} reached without passing the unassisted gate")

        return self.beta

    #BETA > 0: MOVE ONLY ON PERSISTENT EVIDENCE, THEN COOL DOWN
    def _step_assisted(self, success: float, decisions: int) -> None:
        if self.cooldown > 0:
            self.cooldown -= 1
            self._log("cooldown", decisions, success, f"holding beta; {self.cooldown} checks of cooldown left")

            return

        if success >= SUCCESS_HIGH:
            self.high_streak += 1
            self.low_streak = 0

            if self.high_streak >= PERSISTENCE_Y:
                self.beta = max(0.0, round(self.beta - BETA_DECREMENT, 10))
                self._reset_after_change()
                self._log("beta_down", decisions, success,
                          f"success >= {SUCCESS_HIGH} on {PERSISTENCE_Y} consecutive checks")

                if self.beta <= 0.0:
                    self.state = CONSOLIDATING
                    self.decisions_at_beta_zero = int(decisions)
                    self._log("consolidate", decisions, success,
                              f"beta reached 0; {UNASSISTED_CONSOLIDATION_DECISIONS} unassisted decisions required")

                return

            self._log("hold", decisions, success, f"high streak {self.high_streak}/{PERSISTENCE_Y}")

            return

        if success <= SUCCESS_LOW:
            self.low_streak += 1
            self.high_streak = 0

            if self.low_streak >= PERSISTENCE_Z:
                capped = min(BETA_CAP[self.stage_index], round(self.beta + BETA_INCREMENT, 10))
                reason = (f"success <= {SUCCESS_LOW} on {PERSISTENCE_Z} consecutive checks"
                          if capped > self.beta else
                          f"success <= {SUCCESS_LOW} but beta already at stage cap {BETA_CAP[self.stage_index]}")
                self.beta = capped
                self._reset_after_change()
                self._log("beta_up", decisions, success, reason)

                return

            self._log("hold", decisions, success, f"low streak {self.low_streak}/{PERSISTENCE_Z}")

            return

        self.low_streak = 0
        self.high_streak = 0
        self._log("hold", decisions, success, f"success inside hysteresis band ({SUCCESS_LOW}, {SUCCESS_HIGH})")

    #BETA == 0: TRAIN UNASSISTED FOR THE CONSOLIDATION PERIOD BEFORE GATING
    def _step_consolidating(self, success: float, decisions: int) -> None:
        elapsed = int(decisions) - int(self.decisions_at_beta_zero or 0)

        if elapsed < UNASSISTED_CONSOLIDATION_DECISIONS:
            self._log("consolidating", decisions, success,
                      f"{elapsed}/{UNASSISTED_CONSOLIDATION_DECISIONS} unassisted decisions")

            return

        self.state = GATING
        self._log("gate_open", decisions, success, "consolidation complete; counting passing checks")
        self._step_gating(success, decisions)

    #BETA == 0 AND CONSOLIDATED: ADVANCE ONLY ON REPEATED UNASSISTED PASSES
    def _step_gating(self, success: float, decisions: int) -> None:
        if success > self.advance_threshold:
            self.pass_streak += 1

            if self.pass_streak >= CONSECUTIVE_CHECKS_TO_ADVANCE:
                self.state = PASSED
                self._log("passed", decisions, success,
                          f"unassisted success > {self.advance_threshold} on "
                          f"{CONSECUTIVE_CHECKS_TO_ADVANCE} consecutive checks")

                return

            self._log("gate_progress", decisions, success,
                      f"pass streak {self.pass_streak}/{CONSECUTIVE_CHECKS_TO_ADVANCE}")

            return

        self.pass_streak = 0
        self._log("gate_reset", decisions, success, f"success <= {self.advance_threshold}; streak reset")

    #PERSISTENTLY LOW UNASSISTED SUCCESS AT BETA 0 LETS ASSISTANCE RETURN
    def _collapsed(self, success: float, decisions: int) -> bool:
        if not self.assistance_enabled:
            return False

        if self.cooldown > 0:
            self.cooldown -= 1
            self.low_streak = 0
            return False
        #SAME "LOW" DEFINITION AS THE ASSISTED PHASE: AT OR BELOW SUCCESS_LOW
        if success > SUCCESS_LOW:
            self.low_streak = 0
            return False
        self.low_streak += 1
        self.pass_streak = 0
        if self.low_streak < PERSISTENCE_Z:
            self._log("collapse_pending", decisions, success,
                      f"unassisted low streak {self.low_streak}/{PERSISTENCE_Z}")
            return False

        restart = min(BETA_CAP[self.stage_index], BETA_INCREMENT)

        if restart <= 0.0:
            return False

        self.beta = restart
        self.state = ASSISTED
        self.pass_streak = 0
        self.decisions_at_beta_zero = None
        self._reset_after_change()
        self._log("collapse", decisions, success,
                  f"unassisted success <= {SUCCESS_LOW} on {PERSISTENCE_Z} checks; "
                  f"assistance increased by one increment to {self.beta:.3f}")

        return True
