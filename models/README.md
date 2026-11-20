# Trained models

Every number in the report comes from the models in this folder. `watch_model.py` in the repository root flies any of them in the simulator window; the README's "Reproduce the reported numbers" section lists the evaluation commands that read them.

## compact/ppo_<condition>_s<seed>/ (the reported study, report Sec. 3 to 4)

One folder per run: 4 conditions (`degenerate`, `sensing_only`, `reward_only`, `full`) x 3 training seeds (0, 1, 2).

| File | What it is |
|---|---|
| `final.zip` | the final policy (Stable-Baselines3 PPO), after all seven curriculum stages |
| `final_vecnormalize.pkl` | the observation normalisation the policy was trained with; loaded with the policy |
| `run_manifest.json` | training provenance: condition, seed, reward profile, preset, environment seeds, hashes of the artifacts (verified before any evaluation) |
| `stage_gate_history.json` | the curriculum record: every validation check, the expert-help level in force and the pass decisions |
| `diagnostics/<id>/progress.csv` | the training log (update size, action spread, learning rate), report Sec. 4.1 |

## phase2/ppo_<condition>_s<seed>/ (training with walls, report Sec. 4.5)

The same 12 runs continued on wall worlds, warm-started from `compact/`. Two checkpoints per run, each a bundle of `<name>.zip`, `<name>_vecnormalize.pkl`, `<name>_rng.pkl` and `<name>_training_state.json`, hash-checked on load:

| Checkpoint | What it is |
|---|---|
| `stage_2` | the model at the end of the 50% coverage stage, the one Table 5 compares with the phase-1 models |
| `stalled` | each run's last model (`--checkpoint latest` resolves to it), the dotted lines of Fig. 5 |

The six blind runs and `sensing_only` seed 2 stopped at the 50% stage, so their two bundles hold the same weights; the other five sensing runs went on to the 70% stage and stopped there. `run_manifest.json` and `stage_gate_history.json` are the run's training record.

## Notes

The manifests and checkpoint states are the files written when training ran, kept unedited because the loaders verify their hashes. They name the folders the code had at that time (`rl_train/`, `rl_models/`, `archive/train/expert_blender.py`), before it was reorganised into the current layout (`training/`, `models/`, `drone_env/expert_policy.py`).

Not included: the per-check and intermediate-stage checkpoints of these runs, each run's training-time source snapshot (the loader skips its hash check when it is absent), and the models of the superseded trials of report Table 2.
