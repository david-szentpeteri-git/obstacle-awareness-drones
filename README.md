# Does Obstacle-Awareness Actually Matter?

Code, trained models and evaluation records behind the report *Does Obstacle-Awareness Actually Matter? A Controlled 2x2 Study of Obstacle Sensing and Reward Shaping for Learned Quadrotor Navigation* (CS4278/CS5478 Intelligent Robots: Algorithms and Systems, National University of Singapore, final project, Team 50, David Szentpeteri, A0357858E).

One quadrotor learns to reach a goal among static and moving obstacles with PPO. Four conditions differ only in whether the policy can see obstacles (six rays plus a nearest-obstacle vector, or zeros) and whether its reward penalises collisions and proximity. All four share the same network, curriculum and scripted teacher. Every number in the report comes from the models in `models/` evaluated on the worlds recorded in `results/`, and every table and figure regenerates from those records with the commands below.

## What is in the repository

```
watch_model.py           NEW   fly a trained model (or the scripted teacher) in the simulator window
run_experiment.py        NEW   train, evaluate and analyse the study
run_wall_gap_eval.py     NEW   parallel evaluation: wall-gap worlds, perception probes, phase-2 models
config/                  NEW   what is compared: the four conditions and reward profiles (experiment.py),
                               PPO hyperparameters, seed bands and the two presets (ppo_defaults.py),
                               expert-help schedule and imitation parameters (blending.py)
drone_env/               OLD   the 3D drone environment the project builds on, written before the course
                               project (13 files: 8 modified here, 5 unchanged; see PROVENANCE.md):
                               PettingZoo environment over gym-pybullet-drones, vector-env bridge, step
                               logic, obstacle and world generation, physical constants, reward function,
                               curriculum stage lists, scripted reactive controller (the teacher)
training/                NEW   PPO + imitation + success-gated expert help: train_ppo.py (the loop),
                               stage_gate.py, blend_scheduler.py, imitation_ppo.py, checkpoint_io.py,
                               warm_start.py, teacher.py, env_factory.py, callbacks/stats_cb.py,
                               wrappers/ (action_repeat, expert_blend, obs_mask, reward_scale, lidar_layout)
evaluation/              NEW   evaluate.py (paired episodes, metrics), scenarios.py (held-out worlds),
                               wall_gap.py (wall-with-gap worlds), reference_eval.py (expert and
                               straight-flight rows), obs_variants.py and perception_probes.py (probes)
analysis/                NEW   analyze.py (tables and statistics of one results folder), report_figures.py,
                               report_snapshots.py (Fig. 1), perception_report.py, phase2_report.py
models/                        the trained models (models/README.md): compact/ = the 12 final policies of
                               the study; phase2/ = each phase-2 run's end-of-50%-stage and last checkpoint
results/                       per-episode records of every reported evaluation, probe outputs, derived
                               tables, and the report's figures and tables as generated (results/report/)
PROVENANCE.md                  one row per Python file: path, OLD or NEW, role
requirements.txt               pinned dependencies (gym-pybullet-drones at a fixed commit)
```

Every `.py` file states its status on its first lines: `#NEWLY ADDED FOR CS5478 PROJECT`, `#PREEXISTS, MODIFIED FOR CS5478 PROJECT` or `#PREEXISTS - 3D DRONE ENVIRONMENT CODE WRITTEN BEFORE THE CS5478 COURSE PROJECT`. The pipeline defects listed in the report's Appendix A were found in the pre-existing environment code and fixed here. The simulator itself, gym-pybullet-drones, is installed unmodified from the commit pinned in `requirements.txt`. Only the new files follow the project's code style (CAPS comments above code, one summary comment per function); pre-existing files keep their original style.

## Setup

Python 3.11 and git (pip clones gym-pybullet-drones from GitHub at a pinned commit), CPU only (the policy is a small MLP; PyBullet dominates the run time). The commands below are PowerShell, as the study was run on Windows; on Linux or macOS create the venv with `python3.11 -m venv venv` and use `venv/bin/python` with the same arguments. The setup was checked from an empty environment on 20 November 2026.

```powershell
#FRESH VENV (py -3.11 PICKS PYTHON 3.11 WHEN SEVERAL VERSIONS ARE INSTALLED) + PINNED DEPENDENCIES
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run everything from this folder. Library modules are launched as `python -m <folder>.<module>`; the three scripts in the root are run directly.

## Watch a model fly

```powershell
#BEST CONDITION (SEES OBSTACLES, SHAPED REWARD), SEED 0, DENSEST UNSEEN WORLDS
.\venv\Scripts\python.exe -B watch_model.py
#BLIND MODEL: EMPTY ARENA, THEN STAGE-3 WORLDS
.\venv\Scripts\python.exe -B watch_model.py --condition degenerate --seed 1 --stage 0
.\venv\Scripts\python.exe -B watch_model.py --condition degenerate --seed 1 --stage 3
#WALL AT 50% COVERAGE: MODEL AFTER WALL TRAINING, THEN SCRIPTED TEACHERS
.\venv\Scripts\python.exe -B watch_model.py --preset phase2 --condition full --checkpoint stage_2 --wall 0.5
.\venv\Scripts\python.exe -B watch_model.py --controller expert --wall 0.5
.\venv\Scripts\python.exe -B watch_model.py --controller gap_expert --wall 0.7
#NO WINDOW: SAME EPISODES, OUTCOMES PRINTED (QUICK SETUP CHECK)
.\venv\Scripts\python.exe -B watch_model.py --no-gui --episodes 10
```

A PyBullet window opens with the arena, the obstacles, the green goal and the drone; the six rays are redrawn every few decisions, red where they hit something. Flight runs in real time (`--speed 0.5` halves it), one line per episode reports the outcome, and the window closes after the last episode. The worlds are the evaluation worlds of the report (`--world-seed` picks another set), so the outcomes match the records. `--stage` selects the curriculum stage of the unseen worlds (0 = empty arena, 6 = six static and three moving obstacles); `--wall` and `--scale` select a wall-gap world instead (coverage 0.1 to 0.9, arena scale 2 = training size or 3 = larger). Conditions: `full` (sees obstacles, shaped reward), `sensing_only` (sees, goal-only reward), `reward_only` (blind, shaped), `degenerate` (blind, goal-only).

## Reproduce the reported numbers, in order

The shipped models let steps 2 to 5 and 7 run without step 1, and the evaluation commands of step 6 run without the phase-2 training. The shipped records let every table and figure command (`analysis.perception_report` in step 5, `analysis.phase2_report` in step 6, both commands of step 7) run as it is; on 20 November 2026 all 112 shipped tables and figures regenerated byte for byte from them. The launchers refuse to overwrite records, so to re-run an evaluation delete or rename its shipped folder first, or give a fresh `--results-dir`.

### 1. Train the study (12 runs, about 20 h with 12 processes)

```powershell
#PPO + IMITATION, SUCCESS-GATED 7-STAGE CURRICULUM, 4 CONDITIONS x 3 SEEDS --> models/compact/, results/compact/
#RUNS ALREADY MARKED COMPLETE IN models/compact/ (SHIPPED ONES) ARE SKIPPED; --retrain TRAINS FROM SCRATCH
.\venv\Scripts\python.exe -B -u run_experiment.py --preset compact --train-only --conditions all --seeds 0,1,2 --parallel 12 --torch-threads 1 --n-envs 1 --device cpu
#STOPPED RUN: SAME COMMAND + --resume latest CONTINUES FROM LAST COMMITTED CHECKPOINT
```

Do not edit the files under `config/`, `drone_env/` or `training/` while a run is in progress: each run re-hashes them at the end and refuses to finish if they changed. A short check of the training path is `run_experiment.py --smoke --preset compact --train-only --conditions full --seeds 0 --stage-ceiling 1024 --parallel 1 --torch-threads 1`: it trains one run for 1,024 decisions into `models/smoke/` and `results/smoke/` and then stops with "stage 0 STALLED at its optional ceiling", which is the intended end of a run that hits its ceiling before the first validation check.

### 2. Held-out evaluation (report Sec. 4.1 and 4.2, Table 3, Fig. 2)

```powershell
#STUDENT-ONLY, DETERMINISTIC, 200 UNSEEN WORLDS PER CURRICULUM STAGE --> results/compact_eval_200ep/
.\venv\Scripts\python.exe -B -u run_experiment.py --preset compact --skip-train --skip-analyze --conditions all --seeds 0,1,2 --episodes-random 200 --parallel 4 --torch-threads 1 --results-dir results/compact_eval_200ep
#STANDARD TABLES & FIGURES OF ONE RESULTS FOLDER (eval_summary.csv, paired_tests.csv, results.json, ...)
$env:CS5478_RESULTS_DIR = "results/compact_eval_200ep"
.\venv\Scripts\python.exe -B -m analysis.analyze
Remove-Item Env:CS5478_RESULTS_DIR
```

### 3. Wall-gap generalisation test (Sec. 4.3, Fig. 3 and 4)

```powershell
#ALL 12 FINAL MODELS ON SAME 2,000 WALL-GAP WORLDS (10-90% COVERAGE, 2 SCALES) --> results/compact_wall_gap/
.\venv\Scripts\python.exe -B -u run_wall_gap_eval.py --episodes 200 --results-dir results/compact_wall_gap
```

### 4. Scripted references on the same worlds (grey ticks and lines in every figure)

```powershell
#SHARED TEACHER (expert) & SAME CONTROLLER WITH AVOIDANCE OFF (straight), BOTH SUITES
foreach ($c in "expert", "straight") { foreach ($s in "preset", "wall_gap") {
  .\venv\Scripts\python.exe -B -u -m evaluation.reference_eval --controller $c --suite $s --episodes-random 200 --out-dir "results/reference/${s}_$c"
} }
```

### 5. Perception probes and sampled-action evaluation (Sec. 4.1 and 4.4, Table 4)

```powershell
#OBSERVATION VARIANTS FED TO TRAINED SENSING MODELS (v1 = OBSTACLES ONLY, v2b = BOUNDARY ONLY, behind2/scaled = IN-RANGE "NO OBSTACLE" PLACEHOLDER)
.\venv\Scripts\python.exe -B -u run_wall_gap_eval.py --suite preset --probe v1 --conditions sensing_only,full --results-dir results/perception/p1_v1_preset
.\venv\Scripts\python.exe -B -u run_wall_gap_eval.py --suite preset --probe v2b --conditions sensing_only,full --results-dir results/perception/p1_v2b_preset
.\venv\Scripts\python.exe -B -u run_wall_gap_eval.py --suite wall_gap --probe v2b --conditions sensing_only,full --results-dir results/perception/p1_v2b_wall_gap
.\venv\Scripts\python.exe -B -u run_wall_gap_eval.py --suite preset --scenarios stage0_0s0d --probe behind2 --conditions sensing_only,full --results-dir results/perception/p5_behind2
.\venv\Scripts\python.exe -B -u run_wall_gap_eval.py --suite preset --scenarios stage0_0s0d --probe scaled --conditions sensing_only,full --results-dir results/perception/p5_scaled
#ACTIONS SAMPLED FROM POLICY (NOT MEAN ACTION), ALL 12 MODELS
.\venv\Scripts\python.exe -B -u run_wall_gap_eval.py --suite preset --sample-actions --results-dir results/perception/sampled_preset
#RESET-ONLY & STORED-OBSERVATION PROBES (p3-collect ONCE PER CONDITION x SEED), THEN TABLES --> results/perception/tables/
.\venv\Scripts\python.exe -B -m evaluation.perception_probes p2 --out results/perception/p2/worlds.csv
.\venv\Scripts\python.exe -B -m evaluation.perception_probes p3-collect --condition full --train-seed 0 --out results/perception/p3/collected/full_s0.npz
.\venv\Scripts\python.exe -B -m evaluation.perception_probes p3-analyse --collected results/perception/p3/collected --out results/perception/p3/sensitivity.csv
.\venv\Scripts\python.exe -B -m evaluation.perception_probes p4 --out results/perception/p4/gap_visibility.csv
.\venv\Scripts\python.exe -B -m evaluation.perception_probes p5-stats --collected results/perception/p3/collected --out results/perception/p5/placeholder_stats.csv
.\venv\Scripts\python.exe -B -m evaluation.perception_probes p5-traj --condition full --train-seed 2 --out results/perception/p5/traj_full_s2.json
.\venv\Scripts\python.exe -B -m analysis.perception_report --part all
```

### 6. Phase 2: training with walls (Sec. 4.5, Table 5, Fig. 5)

```powershell
#WARM START FROM models/compact/, ONE WALL PER WORLD, GAP-AWARE TEACHER, 30-CHECK STAGE CEILING --> models/phase2/
#models/phase2/ ALREADY HOLDS ALL RUNS' stage_2 & stalled CHECKPOINTS --> EVALUATION COMMANDS BELOW RUN WITHOUT THIS TRAINING
.\venv\Scripts\python.exe -B -u run_experiment.py --preset phase2 --train-only --conditions all --seeds 0,1,2 --parallel 12 --torch-threads 1 --n-envs 1 --device cpu --stage-ceiling 614400
#SAME WORLDS AS STEPS 2-3; MODEL AT END OF 50% STAGE (stage_2) & LAST MODEL (latest); FOLDER NAMES = WHAT analysis.phase2_report READS
.\venv\Scripts\python.exe -B -u run_wall_gap_eval.py --preset phase2 --suite wall_gap --checkpoint stage_2 --conditions degenerate,reward_only --episodes 200 --results-dir results/phase2_wall_gap_stage2_blind
.\venv\Scripts\python.exe -B -u run_wall_gap_eval.py --preset phase2 --suite wall_gap --checkpoint stage_2 --conditions sensing_only,full --episodes 200 --results-dir results/phase2_wall_gap_stage2_sensing
.\venv\Scripts\python.exe -B -u run_wall_gap_eval.py --preset phase2 --suite wall_gap --checkpoint latest --conditions sensing_only,full --episodes 200 --results-dir results/phase2_wall_gap_latest_sensing
.\venv\Scripts\python.exe -B -u run_wall_gap_eval.py --preset phase2 --suite preset --worlds-preset compact --checkpoint stage_2 --conditions degenerate,reward_only --episodes 200 --results-dir results/phase2_training_like_stage2_blind
.\venv\Scripts\python.exe -B -u run_wall_gap_eval.py --preset phase2 --suite preset --worlds-preset compact --checkpoint stage_2 --conditions sensing_only,full --episodes 200 --results-dir results/phase2_training_like_stage2_sensing
.\venv\Scripts\python.exe -B -u run_wall_gap_eval.py --preset phase2 --suite preset --worlds-preset compact --checkpoint latest --conditions sensing_only,full --episodes 200 --results-dir results/phase2_training_like_latest_sensing
#GAP-AWARE TEACHER ON WALL-GAP WORLDS, THEN PHASE-2 TABLES --> results/phase2_tables/
.\venv\Scripts\python.exe -B -u -m evaluation.reference_eval --controller gap_expert --suite wall_gap --episodes-random 200 --out-dir results/reference/wall_gap_gap_expert
.\venv\Scripts\python.exe -B -m analysis.phase2_report
```

`--checkpoint latest` resolves to each run's `stalled` bundle (the last model). The blind runs stall at the 50% stage, where `stage_2` and `latest` hold the same weights, so their `stage_2` records serve both; five of the six sensing runs stall at the 70% stage (`sensing_only` seed 2 at the 50% stage).

### 7. Report figures and tables

```powershell
#FIG. 1: TWO RENDERED EVALUATION WORLDS (SOFTWARE RENDERER, NO WINDOW) --> results/report/figures/env_*.png
.\venv\Scripts\python.exe -B -m analysis.report_snapshots
#FIG. 2-5 & REPORT TABLES FROM STEPS 2-6 --> results/report/figures/, results/report/tables/
.\venv\Scripts\python.exe -B -m analysis.report_figures
```

## Shipped records and manifests

Each `records.csv` under `results/` holds one row per evaluation episode (scenario, condition, training seed, world seed, outcome, steps, path and clearance metrics). The manifests and checkpoint states under `models/` and `results/reference/` are the files written when training and evaluation ran. They name the folders the code had at that time (`rl_train/`, `rl_models/`, `archive/train/expert_blender.py`), before it was reorganised into the layout above (`training/`, `models/`, `drone_env/expert_policy.py`), and are kept unedited because the loaders verify their hashes.

## Not in the repository

The regression tests and diagnostic scripts used during development, the per-check and intermediate-stage training checkpoints, each run's training-time source snapshot, the models and the code paths of the superseded trials of the report's Table 2, and the evaluation logs and per-run copies of the records.
