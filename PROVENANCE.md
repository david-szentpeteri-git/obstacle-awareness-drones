# Provenance of the code

44 Python files: 31 new for this project, 8 pre-existing and modified here, 5 pre-existing and unchanged. OLD means the 3D drone environment this project builds on (`drone_env/`), written before the course project; everything under `config/`, `training/`, `evaluation/` and `analysis/` and the three scripts in the root are NEW. Each file repeats its status on its first lines (`#NEWLY ADDED FOR CS5478 PROJECT`, `#PREEXISTS, MODIFIED FOR CS5478 PROJECT`, `#PREEXISTS - 3D DRONE ENVIRONMENT CODE WRITTEN BEFORE THE CS5478 COURSE PROJECT`).

| Path | Status | Role |
|---|---|---|
| `run_experiment.py` | NEW | entry point: train, evaluate, analyse |
| `run_wall_gap_eval.py` | NEW | entry point: wall-gap, probe and phase-2 evaluation |
| `watch_model.py` | NEW | entry point: fly a trained model or a scripted controller in the simulator window |
| `config/experiment.py` | NEW | experiment definition: the four conditions and their reward profiles |
| `config/ppo_defaults.py` | NEW | experiment definition: PPO hyperparameters, seed bands, presets |
| `config/blending.py` | NEW | experiment definition: expert-help schedule and imitation parameters |
| `drone_env/env_3d.py` | OLD, modified for the project | environment: PettingZoo env over gym-pybullet-drones (physics, obstacles, rays, episodes) |
| `drone_env/env_reset_fix.py` | OLD, modified for the project | environment: vector-env bridge to Stable-Baselines3 |
| `drone_env/step_logic.py` | OLD, modified for the project | environment: action conversion, terminations, info dict |
| `drone_env/step_types.py` | OLD, unchanged | environment: step-state dataclass |
| `drone_env/stats_trackers.py` | OLD, unchanged | environment: per-episode statistics |
| `drone_env/handle_inactive_drones.py` | OLD, unchanged | environment: zeroes inactive drone slots |
| `drone_env/collisionQuery.py` | OLD, modified for the project | environment: offline collision query used when spawning |
| `drone_env/rewards.py` | OLD, modified for the project | environment: reward function, driven by the condition's profile |
| `drone_env/createObjects.py` | OLD, modified for the project | environment: obstacle and target geometry |
| `drone_env/createWorld.py` | OLD, unchanged | environment: world-edge position sampling |
| `drone_env/defaults.py` | OLD, unchanged | environment: physical and reward constants |
| `drone_env/curriculum.py` | OLD, modified for the project | environment: curriculum stage lists |
| `drone_env/expert_policy.py` | OLD, modified for the project | environment: scripted reactive controller, used as the teacher |
| `training/train_ppo.py` | NEW | training: PPO loop with stage gates, expert help, imitation, checkpoints |
| `training/stage_gate.py` | NEW | training: deterministic validation batches for the stage gate |
| `training/blend_scheduler.py` | NEW | training: success-driven expert-help schedule with hysteresis |
| `training/imitation_ppo.py` | NEW | training: PPO with a behaviour-cloning pass toward the expert |
| `training/checkpoint_io.py` | NEW | training: committed, resumable checkpoint bundles |
| `training/warm_start.py` | NEW | training: phase-2 warm start from the phase-1 models |
| `training/teacher.py` | NEW | training: builds the shared teacher and its gap-aware variant |
| `training/env_factory.py` | NEW | training: environment stack, seeding, independence check |
| `training/callbacks/stats_cb.py` | NEW | training: per-rollout statistics and learning-curve logs |
| `training/wrappers/action_repeat.py` | NEW | training: one decision per 10 physics ticks |
| `training/wrappers/expert_blend.py` | NEW | training: env-side blend of student and expert actions |
| `training/wrappers/obs_mask.py` | NEW | training: observation masking for the blind conditions |
| `training/wrappers/reward_scale.py` | NEW | training: optimizer-side reward scaling |
| `training/wrappers/lidar_layout.py` | NEW | training: the six goal-relative rays |
| `evaluation/evaluate.py` | NEW | evaluation: paired episodes, metrics, per-episode records |
| `evaluation/scenarios.py` | NEW | evaluation: held-out scenario suite of a preset and the shared episode reset |
| `evaluation/wall_gap.py` | NEW | evaluation: wall-with-gap world generator |
| `evaluation/reference_eval.py` | NEW | evaluation: scripted expert and straight-flight reference rows |
| `evaluation/obs_variants.py` | NEW | evaluation: observation variants for the perception probes |
| `evaluation/perception_probes.py` | NEW | evaluation: reset-only and stored-observation probes |
| `analysis/analyze.py` | NEW | analysis: tables, figures and statistics of one results folder |
| `analysis/report_figures.py` | NEW | analysis: the report's figures and tables |
| `analysis/report_snapshots.py` | NEW | analysis: the rendered worlds of Fig. 1 |
| `analysis/perception_report.py` | NEW | analysis: perception-probe tables |
| `analysis/phase2_report.py` | NEW | analysis: phase-2 tables |

Also in the repository: `README.md`, `requirements.txt`, the trained models under `models/` (see `models/README.md`), the per-episode evaluation records and probe outputs under `results/`, and the report's figures and tables as generated under `results/report/`.
