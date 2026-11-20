# -*- coding: utf-8 -*-
#NEWLY ADDED FOR CS5478 PROJECT
"""
Turns the raw experiment outputs into every figure and table the report needs:

  <results_dir>/train_logs/*.csv  -> learning curves + per-stage training table
  <results_dir>/eval/records.csv  -> metric bars, 2x2 ablation grid, failure
                                     breakdown, clearance CDF, summary tables

Everything lands in <results_dir>/figures/ and <results_dir>/tables/, plus one
<results_dir>/results.json master file. Inputs that do not exist yet are skipped
with a note, so partial runs can still be analyzed. <results_dir> is the
CS5478_RESULTS_DIR environment variable (run_experiment.py sets it).

    python -m analysis.analyze
"""
from __future__ import annotations

import json
import hashlib
import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from config.experiment import CONDITION_ORDER, CONDITIONS, split_run_name
from config.ppo_defaults import (
    EVAL_RESULTS_DIR,
    FIG_DIR,
    PPO_MODELS_DIR,
    RESULTS_DIR,
    TABLE_DIR,
    TRAIN_LOG_DIR,
)

COND_COLORS = {
    "full": "#2F6FED",
    "reward_only": "#7A4FD0",
    "sensing_only": "#1FA97A",
    "degenerate": "#E07A2E",
}
OUTCOME_COLORS = {"reached": "#1FA97A", "collided": "#D64545", "timeout": "#E0A32E"}

#METRICS PLOTTED AS GROUPED BARS PER SCENARIO (RATE FLAG -> WILSON CI)
BAR_METRICS = [
    ("success_rate", "Success rate", True),
    ("collision_free_rate", "Collision-free rate", True),
    ("wall_hit_rate", "Wall-hit rate", True),
    ("timeout_rate", "Timeout rate", True),
    ("path_ratio", "Path ratio (1.0 = straight line)", False),
    ("mean_time_to_goal_s", "Mean time-to-goal (s, reached episodes)", False),
    ("mean_min_clearance", "Mean min obstacle clearance (m)", False),
]

#SAMPLE-EFFICIENCY THRESHOLD (FIRST TIMESTEP WITH ROLLING REACH RATE >= THIS)
REACH_THRESHOLD = 0.8

#THE HEAD-TO-HEAD PAIR THE PROPOSAL IS BUILT AROUND, PLUS THE TWO ABLATION
#CELLS - EACH TESTED AGAINST THE SAME BASELINE ON SHARED EPISODE SEEDS
PAIRED_COMPARISONS = [
    ("degenerate", "full"),
    ("degenerate", "reward_only"),
    ("degenerate", "sensing_only"),
    ("full", "reward_only"),
    ("full", "sensing_only"),
]


#WILSON 95% INTERVAL FOR A RATE - LO/HI, NOT A SYMMETRIC HALF-WIDTH
def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson rather than the normal approximation: at p = 0 or p = 1 the normal
    interval collapses to zero width, which is exactly where this experiment
    expects the degenerate policy to sit. A 0/40 result should still show
    honest uncertainty.
    """
    if n <= 0:
        return (float("nan"), float("nan"))

    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(max(p * (1.0 - p) / n + z * z / (4 * n * n), 0.0)) / denom

    return (max(0.0, center - half), min(1.0, center + half))


#WILSON HALF-WIDTHS AS MATPLOTLIB ASYMMETRIC YERR
def wilson_yerr(p: float, n: int) -> tuple[float, float]:
    if n <= 0 or not np.isfinite(p):
        return (0.0, 0.0)

    lo, hi = wilson_interval(int(round(p * n)), n)

    return (max(0.0, p - lo), max(0.0, hi - p))


#EXACT McNEMAR TEST ON PAIRED BINARY OUTCOMES (TWO-SIDED)
def mcnemar_exact(a_wins: int, b_wins: int) -> float:
    """
    Two-sided exact binomial p-value on the discordant pairs only - the
    standard paired test for "did this change the outcome on the same task".
    Returns 1.0 when there is nothing to discriminate.
    """
    n = a_wins + b_wins

    if n == 0:
        return 1.0

    return float(binomtest(a_wins, n, p=0.5).pvalue)


#CONDITIONS PRESENT IN A DATAFRAME, IN CANONICAL ORDER
def ordered_conditions(names) -> list[str]:
    present = set(names)

    return [c for c in CONDITION_ORDER if c in present]


#SCENARIOS PRESENT IN A DATAFRAME, SORTED BY NAME (STAGE ORDER FOR THE CURRICULUM SUITE)
def ordered_scenarios(names) -> list[str]:
    return sorted(set(names))


#================================ TRAINING ANALYSIS ================================

#LOAD EVERY RUN'S PER-ROLLOUT TRAINING LOG INTO ONE DATAFRAME
def load_train_logs() -> pd.DataFrame:
    frames = []

    for path in sorted(Path(TRAIN_LOG_DIR).glob("*.csv")):
        if path.stem.endswith("_episodes"):
            continue

        df = pd.read_csv(path)

        if not len(df):
            continue

        #FILENAME IS <condition>_s<seed>; THE CSV ALSO CARRIES BOTH COLUMNS,
        #SO FALL BACK TO THE NAME ONLY FOR LOGS WRITTEN BEFORE THE SEED WORK
        stem_cond, stem_seed = split_run_name(path.stem)

        if "condition" not in df.columns:
            df["condition"] = stem_cond
        if "seed" not in df.columns:
            df["seed"] = stem_seed

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce").fillna(0).astype(int)

    return out


#RUN DIRECTORIES ON DISK, NEWEST NAMING FIRST
def run_dirs() -> list[Path]:
    root = Path(PPO_MODELS_DIR)

    if not root.is_dir():
        return []

    return sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("ppo_"))


#STAGE BOUNDARY TIMESTEPS FROM ANY RUN MANIFEST
def load_stage_boundaries() -> list[int]:
    for d in run_dirs():
        path = d / "run_manifest.json"

        if path.is_file():
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh).get("stage_step_boundaries", [])

    return []


#EVERY RUN'S MANIFEST + TRAIN SUMMARY, FOLDED INTO THE RESULTS JSON
def collect_run_metadata() -> dict:
    """
    Copying every run's manifest and training summary into results.json makes
    a results folder self-sufficient for writing the report.
    """
    runs = {}

    for d in run_dirs():
        entry = {}

        for key, fname in (("manifest", "run_manifest.json"), ("train_summary", "train_summary.json")):
            path = d / fname

            if path.is_file():
                with open(path, "r", encoding="utf-8") as fh:
                    entry[key] = json.load(fh)

        if entry:
            runs[d.name] = entry

    return runs


#EVALUATION CONFIGURATION & MODEL HASHES FROM ALL WORKERS
def collect_eval_metadata() -> dict:
    root = Path(EVAL_RESULTS_DIR)
    manifests = {}

    for path in sorted(root.glob("**/eval_manifest.json")):
        manifests[str(path.relative_to(root))] = json.loads(path.read_text(encoding="utf-8"))

    return manifests


#ONE CONDITION'S SEED RUNS RESAMPLED ONTO A SHARED TIMESTEP GRID
def seed_band(sub: pd.DataFrame, col: str, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Each seed logs at its own rollout boundaries, so the runs are interpolated
    onto one grid before averaging - otherwise the mean would be taken over
    whichever seeds happened to log at that x.
    """
    curves = []

    for _seed, run in sub.groupby("seed"):
        run = run.sort_values("timesteps")
        vals = pd.to_numeric(run[col], errors="coerce")
        #LIGHT SMOOTHING PER RUN, BEFORE ANY CROSS-SEED AVERAGING
        smooth = vals.rolling(10, min_periods=1).mean().to_numpy(dtype=float)
        x = run["timesteps"].to_numpy(dtype=float)
        ok = np.isfinite(smooth)

        if ok.sum() >= 2:
            curves.append(np.interp(grid, x[ok], smooth[ok], left=np.nan, right=np.nan))

    if not curves:
        return np.full_like(grid, np.nan, dtype=float), np.zeros_like(grid, dtype=float)

    stack = np.vstack(curves)

    #GRID POINTS BEYOND A RUN'S LAST ROLLOUT ARE ALL-NAN COLUMNS; THAT IS
    #EXPECTED, NOT A PROBLEM, SO THE EMPTY-SLICE WARNINGS ARE SILENCED
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        return np.nanmean(stack, axis=0), np.nanstd(stack, axis=0)


#LEARNING-CURVE FIGURE (REWARD, REACH, COLLISION, TIMEOUT VS TIMESTEPS)
def plot_learning_curves(train_df: pd.DataFrame, out_dir: Path) -> None:
    panels = [
        #THE REWARD FUNCTIONS DIFFER BETWEEN CONDITIONS (THE GOAL-ONLY
        #PROFILES CARRY NO COLLISION OR PROXIMITY TERMS), SO THIS PANEL IS
        #ONLY READABLE WITHIN A CONDITION - NEVER ACROSS THEM
        ("ep_rew_mean", "Mean episode reward\n(NOT comparable across conditions - reward differs)"),
        ("reach_rate", "Goal-reach rate"),
        ("collision_rate", "Collision rate"),
        ("timeout_rate", "Timeout rate"),
    ]
    boundaries = load_stage_boundaries()
    n_seeds = train_df.groupby("condition")["seed"].nunique().max()
    conditions = ordered_conditions(train_df["condition"].unique())

    hi = float(pd.to_numeric(train_df["timesteps"], errors="coerce").max())
    grid = np.linspace(0.0, hi, 400)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6), sharex=True)

    for ax, (col, label) in zip(axes.flat, panels):
        for name in conditions:
            sub = train_df[train_df["condition"] == name]
            mean, std = seed_band(sub, col, grid)

            ax.plot(grid, mean, label=name, color=COND_COLORS[name], lw=1.6)
            #SHADED BAND = +/- 1 SD ACROSS SEEDS, THE RUN-TO-RUN SPREAD
            ax.fill_between(
                grid, mean - std, mean + std, color=COND_COLORS[name], alpha=0.15, lw=0
            )

        for b in boundaries[:-1]:
            ax.axvline(b, color="#999999", ls=":", lw=0.8)

        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.3)

    axes[0, 0].legend(fontsize=9)
    axes[1, 0].set_xlabel("Policy decisions")
    axes[1, 1].set_xlabel("Policy decisions")
    fig.suptitle(
        f"Training dynamics - mean +/- 1 SD over {int(n_seeds)} seeds "
        f"(curriculum stage boundaries dotted)"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curves.png", dpi=150)
    plt.close(fig)


#FIRST TIMESTEP AT WHICH A RUN'S ROLLING REACH RATE CROSSES THE THRESHOLD
def steps_to_threshold(run: pd.DataFrame) -> int:
    run = run.sort_values("timesteps")
    rolling = (
        pd.to_numeric(run["reach_rate"], errors="coerce")
        .rolling(10, min_periods=3)
        .mean()
    )
    crossed = run["timesteps"][rolling >= REACH_THRESHOLD]

    return int(crossed.iloc[0]) if len(crossed) else -1


#PER-STAGE TABLE (ONE ROW PER CONDITION x SEED x STAGE)
def train_tables(train_df: pd.DataFrame, table_dir: Path) -> dict:
    rows = []

    for name in ordered_conditions(train_df["condition"].unique()):
        cond_df = train_df[train_df["condition"] == name]

        for seed, sub in cond_df.groupby("seed"):
            for stage, stage_df in sub.groupby("stage"):
                tail = stage_df.sort_values("timesteps").tail(10)
                denom = max(1, int(tail["episodes"].sum()))
                rows.append(
                    {
                        "condition": name,
                        "seed": int(seed),
                        "stage": int(stage),
                        "timesteps_end": int(stage_df["timesteps"].max()),
                        "episodes": int(stage_df["episodes"].sum()),
                        "final_reach_rate": round(float(tail["reaches"].sum() / denom), 3),
                        "final_collision_rate": round(float(tail["collisions"].sum() / denom if "collisions" in tail else (tail["collision_rate"] * tail["episodes"]).sum() / denom), 3),
                        "final_timeout_rate": round(float(tail["timeouts"].sum() / denom), 3),
                        "final_ep_rew_mean": round(float(pd.to_numeric(tail["ep_rew_mean"], errors="coerce").mean()), 1),
                        "wall_time_s": round(float(stage_df["stage_wall_time_s" if "stage_wall_time_s" in stage_df else "wall_time_s"].max()), 1),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(table_dir / "training_by_stage.csv", index=False)
    (table_dir / "training_by_stage.md").write_text(
        df.to_markdown(index=False), encoding="utf-8"
    )

    return {"training_by_stage": rows}


#SAMPLE EFFICIENCY: STEPS TO THE REACH THRESHOLD, MEAN +/- SD OVER SEEDS
def sample_efficiency_table(train_df: pd.DataFrame, table_dir: Path) -> dict:
    rows = []

    for name in ordered_conditions(train_df["condition"].unique()):
        cond_df = train_df[train_df["condition"] == name]
        per_seed = {}

        for seed, sub in cond_df.groupby("seed"):
            per_seed[int(seed)] = steps_to_threshold(sub)

        reached = [v for v in per_seed.values() if v > 0]
        #FINAL-STAGE PERFORMANCE, AVERAGED OVER THE LAST 10 ROLLOUTS OF EACH RUN
        finals = []

        for _seed, sub in cond_df.groupby("seed"):
            last_stage = sub[sub["stage"] == sub["stage"].max()].tail(10)
            finals.append(float(pd.to_numeric(last_stage["reach_rate"]).mean()))

        rows.append(
            {
                "condition": name,
                "seeds": len(per_seed),
                "runs_reaching_threshold": len(reached),
                f"steps_to_reach_{int(REACH_THRESHOLD * 100)}pct_mean": (
                    round(float(np.mean(reached)), 0) if reached else -1
                ),
                f"steps_to_reach_{int(REACH_THRESHOLD * 100)}pct_sd": (
                    round(float(np.std(reached)), 0) if len(reached) > 1 else 0.0
                ),
                "per_seed_steps": json.dumps(per_seed),
                "final_stage_reach_rate_mean": round(float(np.nanmean(finals)), 3),
                "final_stage_reach_rate_sd": round(float(np.nanstd(finals)), 3),
                "total_wall_time_s": round(float(cond_df.groupby("seed" if "stage_wall_time_s" in cond_df else ["seed", "stage"])["wall_time_s"].max().sum()), 1),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(table_dir / "sample_efficiency.csv", index=False)
    (table_dir / "sample_efficiency.md").write_text(
        df.to_markdown(index=False), encoding="utf-8"
    )

    return {
        "sample_efficiency": rows,
        "sample_efficiency_scope": "First training rollout threshold crossing, conditional on the changing curriculum; not held-out sample efficiency. Means exclude runs that never cross (-1); inspect runs_reaching_threshold and per_seed_steps.",
    }


#=============================== EVALUATION ANALYSIS ===============================

#LOAD PER-EPISODE EVAL RECORDS
def load_eval_records() -> pd.DataFrame:
    path = Path(EVAL_RESULTS_DIR) / "records.csv"

    if not path.is_file():
        return pd.DataFrame()

    records = pd.read_csv(path)
    key = ["scenario", "condition", "train_seed", "seed"]

    if records.duplicated(key).any():
        raise ValueError("duplicate evaluation episode keys; refusing ambiguous report statistics")

    for col in ("reached", "collided", "wall_hit", "obstacle_hit", "drone_hit", "timeout"):
        if not records[col].isin([0, 1]).all():
            raise ValueError(f"invalid binary outcome in evaluation column {col}")

    if not (records[["reached", "collided", "timeout"]].sum(axis=1) == 1).all():
        raise ValueError("evaluation outcomes must partition into reached, collided and timeout")

    return records


#AGGREGATE ONE SLICE OF EVAL RECORDS INTO THE HEADLINE METRIC SET
def aggregate_records(sub: pd.DataFrame) -> dict:
    n = len(sub)
    succ = int(sub["reached"].sum())
    coll_free = int((1 - sub["collided"]).sum())
    s_lo, s_hi = wilson_interval(succ, n)
    c_lo, c_hi = wilson_interval(coll_free, n)

    return {
        "episodes": n,
        "success_rate": sub["reached"].mean(),
        "success_ci_lo": s_lo,
        "success_ci_hi": s_hi,
        "collision_free_rate": 1.0 - sub["collided"].mean(),
        "collision_free_ci_lo": c_lo,
        "collision_free_ci_hi": c_hi,
        "wall_hit_rate": sub["wall_hit"].mean(),
        "obstacle_hit_rate": sub["obstacle_hit"].mean(),
        "timeout_rate": sub["timeout"].mean(),
        "mean_steps": sub["steps"].mean(),
        "mean_time_to_goal_s": sub["time_to_goal_s"].mean(),
        "path_ratio": sub["path_ratio"].mean(),
        "mean_min_clearance": sub["min_clearance"].mean(),
    }


#AGGREGATE EVAL RECORDS PER (SCENARIO, CONDITION), POOLED OVER RUN SEEDS
def eval_summary_frame(records: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (scenario, condition), sub in records.groupby(["scenario", "condition"]):
        rows.append({"scenario": scenario, "condition": condition, **aggregate_records(sub)})

    return pd.DataFrame(rows)


#SAME METRICS BROKEN OUT PER RUN SEED, PLUS THE ACROSS-SEED SPREAD
def eval_by_seed_frame(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "train_seed" not in records.columns:
        return pd.DataFrame(), pd.DataFrame()

    rows = []

    for (scenario, condition, seed), sub in records.groupby(
        ["scenario", "condition", "train_seed"]
    ):
        rows.append(
            {
                "scenario": scenario,
                "condition": condition,
                "train_seed": int(seed),
                **aggregate_records(sub),
            }
        )

    per_seed = pd.DataFrame(rows)

    if not len(per_seed):
        return per_seed, pd.DataFrame()

    #SPREAD ACROSS SEEDS - THE NUMBER THE REPORT QUOTES AS "+/-"
    metrics = [
        "success_rate",
        "collision_free_rate",
        "wall_hit_rate",
        "timeout_rate",
        "path_ratio",
        "mean_time_to_goal_s",
        "mean_min_clearance",
    ]
    spread = (
        per_seed.groupby(["scenario", "condition"])[metrics]
        .agg(["mean", "std", "min", "max"])
        .round(4)
        .reset_index()
    )
    spread.columns = [
        c[0] if not c[1] else f"{c[0]}_{c[1]}" for c in spread.columns.to_flat_index()
    ]

    return per_seed, spread


#GROUPED-BAR FIGURE FOR ONE METRIC ACROSS SCENARIOS & CONDITIONS
def plot_metric_bars(summary: pd.DataFrame, metric: str, label: str, is_rate: bool, out_dir: Path) -> None:
    scenarios = ordered_scenarios(summary["scenario"])
    conditions = ordered_conditions(summary["condition"].unique())

    if not scenarios or not conditions:
        return

    x = np.arange(len(scenarios))
    width = 0.8 / len(conditions)
    fig, ax = plt.subplots(figsize=(11, 4.5))

    for i, name in enumerate(conditions):
        vals, lo_err, hi_err = [], [], []

        for s in scenarios:
            row = summary[(summary["scenario"] == s) & (summary["condition"] == name)]
            v = float(row[metric].iloc[0]) if len(row) else np.nan
            vals.append(v)

            if is_rate and len(row):
                lo, hi = wilson_yerr(v, int(row["episodes"].iloc[0]))
            else:
                lo, hi = 0.0, 0.0

            lo_err.append(lo)
            hi_err.append(hi)

        ax.bar(
            x + (i - (len(conditions) - 1) / 2) * width,
            vals,
            width,
            #WILSON, SO A 0/N OR N/N CELL STILL CARRIES AN HONEST INTERVAL
            yerr=np.array([lo_err, hi_err]) if is_rate else None,
            capsize=2,
            label=name,
            color=COND_COLORS[name],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel(label)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncols=len(conditions), fontsize=9)

    if is_rate:
        ax.set_ylim(0, 1.05)
        ax.set_title("Pooled episode Wilson intervals (descriptive); see per-seed spread", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_dir / f"eval_{metric}.png", dpi=150)
    plt.close(fig)


#PAIRED SIGNIFICANCE TESTS ON SHARED EPISODE SEEDS
def paired_tests(records: pd.DataFrame, table_dir: Path) -> dict:
    """
    Test shared episode seeds separately for each trained model pair. Reusing
    the same worlds across training seeds does not create independent trials;
    these conditional tests do not establish an effect across training runs.
    Holm adjustment covers all reported scenario/model/metric comparisons.
    """
    if "train_seed" not in records.columns:
        return {}

    rows = []
    outcomes = [("reached", "success"), ("collision_free", "collision_free")]

    work = records.copy()
    work["collision_free"] = 1 - work["collided"]

    key = ["scenario", "condition", "train_seed", "seed"]

    if work.duplicated(key).any():
        raise ValueError("duplicate paired episode keys")

    for (scenario, train_seed), scen in work.groupby(["scenario", "train_seed"]):

        for base, other in PAIRED_COMPARISONS:
            if base not in set(scen["condition"]) or other not in set(scen["condition"]):
                continue

            left = scen[scen["condition"] == base].set_index("seed")
            right = scen[scen["condition"] == other].set_index("seed")
            shared = left.index.intersection(right.index)

            if not len(shared):
                continue

            for col, label in outcomes:
                a = left.loc[shared, col].to_numpy(dtype=int)
                b = right.loc[shared, col].to_numpy(dtype=int)

                #DISCORDANT PAIRS: ONLY ONE OF THE TWO SUCCEEDED
                base_only = int(np.sum((a == 1) & (b == 0)))
                other_only = int(np.sum((a == 0) & (b == 1)))
                p = mcnemar_exact(base_only, other_only)

                rows.append(
                    {
                        "scenario": scenario,
                        "train_seed": int(train_seed),
                        "metric": label,
                        "baseline": base,
                        "comparison": other,
                        "paired_episodes": int(len(shared)),
                        "unpaired_episodes": int(len(left) + len(right) - 2 * len(shared)),
                        "rate": round(float(a.mean()), 4),
                        "comparison_rate": round(float(b.mean()), 4),
                        "delta": round(float(b.mean() - a.mean()), 4),
                        "baseline_only_wins": base_only,
                        "comparison_only_wins": other_only,
                        "mcnemar_p": p,
                    }
                )

    if not rows:
        return {}

    df = pd.DataFrame(rows)
    order = np.argsort(df["mcnemar_p"].to_numpy())
    adjusted = np.minimum(1.0, np.maximum.accumulate(df["mcnemar_p"].to_numpy()[order] * np.arange(len(df), 0, -1)))
    df["mcnemar_holm_p"] = np.nan
    df.loc[order, "mcnemar_holm_p"] = adjusted
    df["significant_at_0.05"] = df["mcnemar_holm_p"] < 0.05
    df.to_csv(table_dir / "paired_tests.csv", index=False)
    (table_dir / "paired_tests.md").write_text(df.to_markdown(index=False), encoding="utf-8")

    return {
        "paired_tests": df.to_dict(orient="records"),
        "paired_test_scope": "McNemar tests are conditional on each trained model pair and scenario, with Holm adjustment across all rows. Shared worlds across training seeds are not pooled as independent trials; these p-values do not measure training-seed uncertainty or prove an algorithm-level effect.",
    }


#THE SHORTLIST OF NUMBERS THE REPORT'S CLAIMS REST ON
def headline_numbers(records: pd.DataFrame) -> dict:
    """
    One block holding the figures that answer the proposal's question
    directly, so they need not be re-derived by hand from the CSVs.
    """
    work = records.copy()
    work["collision_free"] = 1 - work["collided"]

    pooled = {}

    for name in ordered_conditions(work["condition"].unique()):
        sub = work[work["condition"] == name]
        pooled[name] = {k: (None if pd.isna(v) else v) for k, v in aggregate_records(sub).items()}

    #MACRO AVERAGE GIVES EVERY SCENARIO EQUAL WEIGHT; THE POOLED FIGURE
    #ABOVE WEIGHTS BY EPISODE COUNT
    macro = {}

    for name in ordered_conditions(work["condition"].unique()):
        sub = work[work["condition"] == name]
        per_scen = sub.groupby("scenario")[["reached", "collision_free", "wall_hit", "timeout"]].mean()
        macro[name] = {
            "success_rate": round(float(per_scen["reached"].mean()), 4),
            "collision_free_rate": round(float(per_scen["collision_free"].mean()), 4),
            "wall_hit_rate": round(float(per_scen["wall_hit"].mean()), 4),
            "timeout_rate": round(float(per_scen["timeout"].mean()), 4),
        }

    deltas = {}

    for base, other in PAIRED_COMPARISONS:
        if base in pooled and other in pooled:
            deltas[f"{other}_minus_{base}"] = {
                "success_rate": round(pooled[other]["success_rate"] - pooled[base]["success_rate"], 4),
                "collision_free_rate": round(
                    pooled[other]["collision_free_rate"] - pooled[base]["collision_free_rate"], 4
                ),
            }

    return {
        "headline": {
            "episodes_total": int(len(work)),
            "run_seeds": sorted(int(s) for s in set(work["train_seed"]))
            if "train_seed" in work.columns
            else [],
            "scenarios": ordered_scenarios(work["scenario"]),
            "pooled_by_condition": pooled,
            "macro_avg_by_condition": macro,
            "deltas": deltas,
            "uncertainty_scope": "Pooled Wilson intervals are descriptive episode-level summaries conditional on this suite. Worlds are reused across training seeds; use eval_seed_spread for variation across trained runs. Three training seeds give limited evidence for generalization across training runs.",
        }
    }


#2X2 ABLATION GRID (REWARD AXIS X SENSING AXIS), POOLED OVER SCENARIOS
def plot_ablation_grid(records: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    grid = np.full((2, 2), np.nan)

    for name, sub in records.groupby("condition"):
        cond = CONDITIONS[name]
        #ROW 0 = SHAPED REWARD, COL 0 = SENSING ON
        r = 0 if cond.reward_shaped else 1
        c = 0 if cond.obstacle_sensing else 1
        succ = sub["reached"].mean()
        cfr = 1.0 - sub["collided"].mean()
        grid[r, c] = succ
        ax.text(
            c,
            r,
            f"{name}\nsuccess {succ:.1%}\ncoll-free {cfr:.1%}",
            ha="center",
            va="center",
            fontsize=11,
        )

    ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks([0, 1], ["sensing ON", "sensing OFF"])
    ax.set_yticks([0, 1], ["reward SHAPED", "reward GOAL-ONLY"])
    #POOLED OVER EPISODES & SCENARIOS; THE PER-SCENARIO EQUAL-WEIGHT VIEW IS
    #THE macro_avg_by_condition BLOCK IN results.json
    ax.set_title(
        "2x2 ablation grid - success rate\n(all scenarios and run seeds pooled by episode)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "ablation_grid.png", dpi=150)
    plt.close(fig)


#STACKED OUTCOME BREAKDOWN (REACHED / OBSTACLE / WALL / TIMEOUT) PER SCENARIO
def plot_failure_breakdown(records: pd.DataFrame, out_dir: Path) -> None:
    scenarios = ordered_scenarios(records["scenario"])
    conditions = ordered_conditions(records["condition"].unique())

    if not scenarios or not conditions:
        return

    work = records.copy()
    work["obstacle_hit"] = work["collided"] & work["obstacle_hit"]
    work["wall_hit"] = work["collided"] & (1 - work["obstacle_hit"]) & work["wall_hit"]
    work["other_collision"] = work["collided"] & (1 - work["obstacle_hit"]) & (1 - work["wall_hit"])
    parts = [
        ("reached", "reached", "#1FA97A"),
        ("obstacle_hit", "obstacle hit", "#D64545"),
        ("wall_hit", "wall hit", "#8A4A4A"),
        ("other_collision", "other collision", "#965CAC"),
        ("timeout", "timeout", "#E0A32E"),
    ]
    fig, axes = plt.subplots(1, len(scenarios), figsize=(2.6 * len(scenarios), 4.2), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, s in zip(axes, scenarios):
        bottoms = np.zeros(len(conditions))

        for col, label, color in parts:
            vals = []

            for name in conditions:
                sub = work[(work["scenario"] == s) & (work["condition"] == name)]
                vals.append(sub[col].mean() if len(sub) else 0.0)

            ax.bar(range(len(conditions)), vals, bottom=bottoms, color=color, label=label, width=0.7)
            bottoms += np.array(vals)

        ax.set_title(s, fontsize=10)
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1.15)

    axes[0].set_ylabel("Fraction of episodes")
    axes[0].legend(fontsize=8)
    fig.suptitle("Episode outcomes (simultaneous obstacle + wall hits assigned to obstacle)")
    fig.tight_layout()
    fig.savefig(out_dir / "failure_breakdown.png", dpi=150)
    plt.close(fig)


#CDF OF PER-EPISODE MIN OBSTACLE CLEARANCE PER CONDITION
def plot_clearance_cdf(records: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for name in ordered_conditions(records["condition"].unique()):
        vals = records[records["condition"] == name]["min_clearance"].dropna().sort_values()

        if not len(vals):
            continue

        ax.plot(
            vals,
            np.arange(1, len(vals) + 1) / len(vals),
            label=name,
            color=COND_COLORS[name],
            lw=1.8,
        )

    ax.set_xlabel("Per-episode min obstacle clearance (m)")
    ax.set_ylabel("CDF")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "clearance_cdf.png", dpi=150)
    plt.close(fig)


#=================================== ENTRY POINT ===================================

#BUILD EVERY FIGURE/TABLE AVAILABLE FROM THE CURRENT RESULTS ON DISK
def main() -> None:
    fig_dir = Path(FIG_DIR)
    table_dir = Path(TABLE_DIR)
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    master: dict = {}

    train_df = load_train_logs()
    if len(train_df):
        plot_learning_curves(train_df, fig_dir)
        master.update(train_tables(train_df, table_dir))
        master.update(sample_efficiency_table(train_df, table_dir))
        print(
            f"[OK] training analysis ({sorted(set(train_df['condition']))}, "
            f"seeds {sorted(set(train_df['seed']))})"
        )
    else:
        print("[SKIP] no training logs in", TRAIN_LOG_DIR)

    records = load_eval_records()
    if len(records):
        summary = eval_summary_frame(records)
        summary.to_csv(table_dir / "eval_summary.csv", index=False)
        (table_dir / "eval_summary.md").write_text(
            summary.round(3).to_markdown(index=False), encoding="utf-8"
        )

        per_seed, spread = eval_by_seed_frame(records)
        if len(per_seed):
            per_seed.to_csv(table_dir / "eval_by_seed.csv", index=False)
            spread.to_csv(table_dir / "eval_seed_spread.csv", index=False)
            (table_dir / "eval_seed_spread.md").write_text(
                spread.to_markdown(index=False), encoding="utf-8"
            )
            master["eval_by_seed"] = per_seed.round(4).to_dict(orient="records")
            master["eval_seed_spread"] = spread.to_dict(orient="records")

        for metric, label, is_rate in BAR_METRICS:
            plot_metric_bars(summary, metric, label, is_rate, fig_dir)

        plot_ablation_grid(records, fig_dir)
        plot_failure_breakdown(records, fig_dir)
        plot_clearance_cdf(records, fig_dir)

        master["eval_summary"] = summary.round(4).to_dict(orient="records")
        master.update(paired_tests(records, table_dir))
        master.update(headline_numbers(records))
        print(f"[OK] eval analysis ({len(records)} episode records)")
    else:
        print("[SKIP] no eval records in", EVAL_RESULTS_DIR)

    #HYPERPARAMETERS, SEEDS & STAGE TIMINGS COPIED OUT OF THE MODEL FOLDERS SO
    #THE RESULTS FOLDER ALONE IS ENOUGH TO WRITE THE REPORT
    runs = collect_run_metadata()
    if runs:
        master["runs"] = runs
        print(f"[OK] run metadata embedded ({len(runs)} runs)")

    eval_metadata = collect_eval_metadata()

    if eval_metadata:
        master["evaluation_manifests"] = eval_metadata
        master["evaluation_complete"] = all(meta.get("status") == "complete" for meta in eval_metadata.values())

    master["analysis_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    with open(Path(RESULTS_DIR) / "results.json", "w", encoding="utf-8") as fh:
        json.dump(master, fh, indent=2)

    print(f"[SAVED] figures -> {fig_dir} | tables -> {table_dir} | master -> {RESULTS_DIR}/results.json")


if __name__ == "__main__":
    main()
