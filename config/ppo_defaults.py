# -*- coding: utf-8 -*-
#NEWLY ADDED FOR CS5478 PROJECT
"""
PPO hyperparameters, parallel-environment count and output paths, identical
for all four experiment conditions - only the reward/observation config in
config/experiment.py differs between them.

The two environment variables read below redirect every output path, so a
smoke run can write to throwaway folders instead of the real ones.
"""
import os
from copy import deepcopy

#PPO HYPERPARAMS
GAMMA = 0.99
GAE_LAMBDA = 0.95
LEARNING_RATE = 3e-4

#ROLLOUT LENGTH PER ENV SLOT. ONE BUFFER = N_STEPS * N_ENVS (4096 SAMPLES PER
#UPDATE WITH THE SINGLE REAL ENV, SEE N_ENVS BELOW); PRESETS OVERRIDE TO 512
N_STEPS = 4096
BATCH_SIZE = 256

#PASSES OVER EACH ROLLOUT BUFFER
N_EPOCHS = 10
CLIP_RANGE = 0.2

#EARLY-STOP AN UPDATE WHOSE APPROX KL EXCEEDS THIS (CLIPPING ALONE DOES NOT
#BOUND UPDATE SIZE; UNBOUNDED UPDATES DESTABILISE THE WIDE-INPUT SENSING RUNS)
TARGET_KL = 0.03

#SMALL BONUS, KEEPS EXPLORATION ALIVE THROUGH THE CURRICULUM
ENTROPY_COEF = 0.005
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5

NET_ARCH = dict(pi=[512, 256, 128], vf=[512, 256, 128])

#ENV COPIES PER TRAINING PROCESS - MUST STAY 1. SUPERSUIT CLONES COPIES WITH
#cloudpickle, WHICH DUPLICATES THE PYTHON OBJECT BUT NOT THE PYBULLET CONNECTION
#(PROCESS-GLOBAL, KEYED BY CLIENT ID) --> ALL COPIES SHARE ONE WORLD & ONE DRONE.
#env_factory.assert_envs_are_independent REFUSES SUCH A STACK. COPIES ALSO STEP
#SEQUENTIALLY (concat_vec_envs_v1 WITH num_cpus=0); PARALLELISM = SEPARATE
#CONDITION/SEED PROCESSES (run_experiment.py --parallel)
N_ENVS = 1

#TORCH INTRA-OP THREADS (0 = LEAVE TORCH'S OWN DEFAULT ALONE). SET THIS WHEN
#SEVERAL TRAINING PROCESSES RUN AT ONCE, OR THEY ALL GRAB EVERY CORE AND
#OVERSUBSCRIBE THE MACHINE.
TORCH_NUM_THREADS = int(os.environ.get("CS5478_TORCH_THREADS", "0"))

#OBSERVATION NORMALIZATION CLIP
CLIP_OBS = 5.0

#SEEDS
#TRAINING AND EVALUATION DRAW FROM DISJOINT, DOCUMENTED SEED NAMESPACES, SO
#"EVALUATED ON SEEDS NEVER SEEN IN TRAINING" IS A CHECKABLE STATEMENT RATHER
#THAN AN ASSUMPTION. ANY TRAINING SEED IS < 100_000; EVERY EVALUATION EPISODE
#SEED IS >= EVAL_SEED_LO.

#RUN SEEDS: ONE INDEPENDENT TRAINING RUN PER CONDITION PER SEED
TRAIN_SEEDS = (0, 1, 2)

#ENV COPY i OF CURRICULUM STAGE s IN RUN r IS SEEDED AT
#TRAIN_SEED_BASE + r * RUN_SEED_STRIDE + s * STAGE_SEED_STRIDE + i
#STRIDES ARE WIDE ENOUGH THAT NO TWO (RUN, STAGE, COPY) TRIPLES COLLIDE
#FOR UP TO 100 ENV COPIES, 100 STAGES AND 9 RUN SEEDS.
TRAIN_SEED_BASE = 1_000
RUN_SEED_STRIDE = 10_000
STAGE_SEED_STRIDE = 100

#EVALUATION EPISODE SEEDS ARE DRAWN FROM A BAND FAR ABOVE EVERY TRAINING SEED
EVAL_SEED_LO = 10_000_000
EVAL_SEED_HI = 2**31 - 1

#OFFSETS THE PER-SCENARIO EVALUATION SEED STREAM (CLI --seed ADDS TO THIS)
EVAL_SEED_BASE = 900_000_000

#PRESET KEYS THAT DEFINE WHAT A POLICY LEARNED AND THE CONTROL RATE IT EXPECTS.
#EVALUATION CHECKS A TRAINING MANIFEST AGAINST THESE ONLY: OUTPUT DIRECTORIES
#AND EVALUATION-SIDE KNOBS (SUITE, EVAL SCALE, EVAL CLOCK) CANNOT CHANGE WHAT A
#SAVED MODEL IS, SO COMPARING THEM WOULD REJECT EVERY EARLIER RUN THE MOMENT AN
#EVALUATION-ONLY SETTING CHANGED - WITHOUT CATCHING ANY REAL MISMATCH.
TRAINING_SEMANTIC_PRESET_KEYS = (
    "curriculum",
    "action_repeat",
    "reward_scale",
    "blocking_obstacle_prob",
    "n_steps",
    "goal_stack_only",
    "shaped_proximity_scale",
    "stage_settings",
    "teacher",
    "blending",
    "imitation",
    "learning_rate",
)


#SUBSET OF A PRESET THAT MUST MATCH BETWEEN TRAINING AND EVALUATION
def training_semantics(preset: dict) -> dict:
    return {key: preset[key] for key in TRAINING_SEMANTIC_PRESET_KEYS if key in preset}


#CURRICULUM-GATE VALIDATION SEEDS SIT IN THEIR OWN BAND BETWEEN TRAINING AND
#EVALUATION, SO A POLICY PROMOTED BY A GATE WAS NEVER GATED ON A FINAL TEST
#WORLD. THE THREE BANDS PARTITION CLEANLY: TRAINING < 100_000, VALIDATION IN
#[2_000_000, 9_999_999], FINAL TEST >= EVAL_SEED_LO (10_000_000).
CURRICULUM_VAL_SEED_LO = 2_000_000
CURRICULUM_VAL_SEED_HI = 9_999_999
CURRICULUM_VAL_SEED_BASE = 3_000_000


#EXPERIMENT PRESETS
#A PRESET IS A NAMED BUNDLE OF NON-DEFAULT TRAINING SETTINGS. ONE NAME ON THE
#COMMAND LINE THEN FIXES THE WHOLE CONFIGURATION, AND EVERY RESOLVED VALUE IS
#WRITTEN INTO THE RUN MANIFEST - SO AN EXPERIMENT IS REPRODUCIBLE FROM THE
#SAVED ARTIFACTS, NOT FROM REMEMBERING A SHELL COMMAND.

PRESETS = {
    #THE REPORTED STUDY (REPORT SEC. 3-4): SCALE 2, 6 GOAL-RELATIVE RAYS, PPO +
    #IMITATION, HELP FROM 0.8 WITHDRAWN ON VALIDATED SUCCESS, SHARED TEACHER.
    #action_repeat 10 = ONE DECISION PER 10 PHYSICS TICKS (24 DECISIONS/S, PHYSICS
    #& COLLISIONS AT 240 HZ, GAMMA PER DECISION); goal_stack_only ZEROES dist_pen,
    #trunc_pen, final_mile_coef, align_coef & tangential_pen IDENTICALLY FOR ALL 4
    #CONDITIONS (ABLATED AXES UNTOUCHED); reward_scale OPTIMIZER-SIDE ONLY
    #(training/wrappers/reward_scale.py); n_steps 512 DECISIONS PER ROLLOUT
    "compact": {
        "curriculum": "compact",
        "eval_suite": "training_matched",
        "stage_gate": True,
        "blending": True,
        "imitation": True,
        "learning_rate": 6e-4,
        "models_dir": "models/compact",
        "results_dir": "results/compact",
        "eval_world_scale": 2.0,
        "evaluation_obstacle_dt": 1.0 / 240.0,
        "action_repeat": 10,
        "reward_scale": 0.01,
        "blocking_obstacle_prob": 0.0,
        "n_steps": 512,
        "goal_stack_only": True,
        "shaped_proximity_scale": 0.01,
        "stage_settings": [
            {
                "obstacle_layout": "curriculum",
                "blocking_obstacle_prob": block_prob,
                "obstacle_size_range": [0.8, 1.2],
                "dynamic_speed_range": speed_range,
                "dynamic_travel_range": [0.35, 0.75],
                "obstacle_dt": 1.0 / 240.0,
            }
            for block_prob, speed_range in (
                (0.0, [0.2, 0.35]),
                (0.3, [0.2, 0.35]),
                (0.3, [0.2, 0.35]),
                (0.4, [0.2, 0.35]),
                (0.4, [0.2, 0.35]),
                (0.5, [0.25, 0.45]),
                (0.5, [0.3, 0.6]),
            )
        ],
    },
    #PHASE 2 (REPORT SEC. 4.5): SAME ALGORITHM AS compact (PPO + IMITATION + GATED EXPERT HELP),
    #WARM-STARTED FROM THE compact FINAL MODELS, ONE WALL PER WORLD, GAP-AWARE TEACHER.
    #WRITES ONLY TO ITS OWN DIRECTORIES; THE compact RUNS ARE READ, NEVER TOUCHED.
    "phase2": {
        "curriculum": "phase2",
        "eval_suite": "training_matched",
        "stage_gate": True,
        "blending": True,
        "imitation": True,
        "learning_rate": 6e-4,
        "teacher": "gap_aware",
        "init_from": "models/compact",
        "models_dir": "models/phase2",
        "results_dir": "results/phase2",
        "eval_world_scale": 2.0,
        "evaluation_obstacle_dt": 1.0 / 240.0,
        "action_repeat": 10,
        "reward_scale": 0.01,
        "blocking_obstacle_prob": 0.0,
        "n_steps": 512,
        "goal_stack_only": True,
        "shaped_proximity_scale": 0.01,
        "stage_settings": [
            {
                "obstacle_layout": "curriculum",
                "blocking_obstacle_prob": 0.0,
                "obstacle_size_range": [0.8, 1.2],
                "dynamic_speed_range": [0.2, 0.35],
                "dynamic_travel_range": [0.35, 0.75],
                "obstacle_dt": 1.0 / 240.0,
                "wall_coverage": coverage,
            }
            for coverage in (0.1, 0.3, 0.5, 0.7, [0.1, 0.7])
        ],
    },
}


#RESOLVE A PRESET NAME INTO ITS SETTINGS DICT
def get_preset(name: str) -> dict:
    key = (name or "compact").strip().lower()

    if key not in PRESETS:
        valid = ", ".join(sorted(PRESETS))
        raise ValueError(f"unknown preset '{name}' (expected one of: {valid})")

    return deepcopy(PRESETS[key])


#SEED FOR ONE TRAINING ENV COPY - SEE THE NAMESPACE NOTE ABOVE
def train_env_seed(run_seed: int, stage_idx: int) -> int:
    """Base seed handed to a stage's vec env; copy i then uses base + i."""
    return (
        TRAIN_SEED_BASE
        + int(run_seed) * RUN_SEED_STRIDE
        + int(stage_idx) * STAGE_SEED_STRIDE
    )

#MODEL / RESULT PATHS (ENV VARS LET A SMOKE RUN ISOLATE ITS OUTPUTS)
PPO_MODELS_DIR = os.environ.get("CS5478_MODELS_DIR", "models/default")
TB_LOG_DIR = f"{PPO_MODELS_DIR}/tb_logs"

RESULTS_DIR = os.environ.get("CS5478_RESULTS_DIR", "results/default")
TRAIN_LOG_DIR = f"{RESULTS_DIR}/train_logs"
EVAL_RESULTS_DIR = f"{RESULTS_DIR}/eval"
FIG_DIR = f"{RESULTS_DIR}/figures"
TABLE_DIR = f"{RESULTS_DIR}/tables"

#MID-STAGE SNAPSHOT INTERVAL
CHECKPOINT_EVERY_STEPS = 50_000

#EVAL EPISODES PER CONDITION PER SCENARIO (CLI-OVERRIDABLE)
EVAL_EPISODES_RANDOM = 40
