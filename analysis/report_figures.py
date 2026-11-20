# -*- coding: utf-8 -*-
#NEWLY ADDED FOR CS5478 PROJECT
"""
Report figures and tables for the final trial (E7 held-out evaluation, E8
wall-gap evaluation, scripted reference rows, training diagnostics).

Reads only saved artifacts; writes PNG figures + CSV/Markdown tables to
results/report/ (or --out-dir). Reference rows (results/reference/) are
included when present.

  .\\venv\\Scripts\\python.exe -B -m analysis.report_figures

Encoding (all figures): colour = perception (blue = sees obstacles, orange =
blind), hatch / dashed line = shaped reward, greys = scripted references.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

#PAPER SIZING: FIGURES DRAWN AT ~7 IN & SHOWN AT 5.5 IN (NEURIPS TEXT WIDTH)
plt.rcParams.update({"font.size": 8.5, "legend.fontsize": 7.5, "axes.titlesize": 9,
                     "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix"})
import numpy as np
import pandas as pd
from scipy.stats import binomtest

#ENCODING (VALIDATED PAIR: BLUE/ORANGE, CVD dE 24.7)
SEES, BLIND = "#2a78d6", "#eb6834"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
EXPERT, STRAIGHT = "#3b3a37", "#a3a29c"
CONDS = ["degenerate", "sensing_only", "reward_only", "full"]
STYLE = {
    "degenerate": dict(color=BLIND, hatch="", dash="-", label="degenerate (blind, goal-only)"),
    "sensing_only": dict(color=SEES, hatch="", dash="-", label="sensing_only (sees, goal-only)"),
    "reward_only": dict(color=BLIND, hatch="///", dash="--", label="reward_only (blind, shaped)"),
    "full": dict(color=SEES, hatch="///", dash="--", label="full (sees, shaped)"),
}
PAIRS = [("sensing_only", "degenerate", "goal-only pair"), ("full", "reward_only", "shaped pair")]
STAGE_LABELS = ["0\n(0,0)", "1\n(1,0)", "2\n(2,0)", "3\n(3,0)", "4\n(3,1)", "5\n(5,2)", "6\n(6,3)"]
COVERAGES = [10, 30, 50, 70, 90]
INFLATE = 0.07


#RECESSIVE AXES (LIGHT GRID, NO TOP/RIGHT SPINES)
def style_axes(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)

    ax.tick_params(colors=INK_2, labelsize=8)


#EXACT McNEMAR p (TWO-SIDED BINOMIAL ON DISCORDANT PAIRS)
def mcnemar(only_a: int, only_b: int) -> float:
    total = only_a + only_b

    return 1.0 if total == 0 else float(binomtest(only_a, total, 0.5).pvalue)


#HOLM-ADJUSTED p-VALUES
def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0

    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[idx]))
        adj[idx] = running

    return adj.tolist()


#PER-SEED SUCCESS (%) TABLE: SCENARIO x CONDITION x SEED
def seed_success(rec: pd.DataFrame) -> pd.DataFrame:
    return rec.groupby(["scenario", "condition", "train_seed"])["reached"].mean().mul(100).rename("success").reset_index()


#PAIRED GAPS PER SCENARIO x PAIR x SEED WITH EXACT McNEMAR + HOLM
def paired_gaps(rec: pd.DataFrame) -> pd.DataFrame:
    rows = []
    wide = rec.pivot_table(index=["scenario", "train_seed", "seed"], columns="condition", values="reached")

    for (scen, seed), grp in wide.groupby(level=[0, 1]):
        for sees, blind, label in PAIRS:
            only_sees = int(((grp[sees] == 1) & (grp[blind] == 0)).sum())
            only_blind = int(((grp[blind] == 1) & (grp[sees] == 0)).sum())
            rows.append(dict(scenario=scen, pair=label, train_seed=seed,
                             gap=100 * (grp[sees].mean() - grp[blind].mean()),
                             only_sees=only_sees, only_blind=only_blind, p=mcnemar(only_sees, only_blind)))

    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"].tolist())

    return out


#ROTATION MATRIX FROM LOGGED PANEL EULER (ROLL OR PITCH ONLY)
def rot(roll: float, pitch: float) -> np.ndarray:
    rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])

    return ry @ rx


#DOES STRAIGHT START->GOAL SEGMENT HIT PANEL (INFLATED BY DRONE HALF-SPAN)?
def line_blocked(row) -> bool:
    center = np.array([row.wall_cx, row.wall_cy, row.wall_cz])
    half = np.array([row.wall_hx, row.wall_hy, row.wall_hz]) + INFLATE
    mat = rot(row.wall_roll, row.wall_pitch)
    p0 = mat.T @ (np.array([row.start_x, row.start_y, row.start_z]) - center)
    p1 = mat.T @ (np.array([row.goal_x, row.goal_y, row.goal_z]) - center)
    delta = p1 - p0
    t_lo, t_hi = 0.0, 1.0

    for k in range(3):
        if abs(delta[k]) < 1e-12:
            if abs(p0[k]) > half[k]:
                return False

            continue

        a, b = (-half[k] - p0[k]) / delta[k], (half[k] - p0[k]) / delta[k]
        t_lo, t_hi = max(t_lo, min(a, b)), min(t_hi, max(a, b))

        if t_lo > t_hi:
            return False

    return True


#LOAD FINISHED REFERENCE RECORDS FOR A SUITE (MANIFEST WRITTEN AT COMPLETION; EMPTY IF NONE)
def load_refs(ref_dir: Path, suite: str) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in glob.glob(str(ref_dir / f"{suite}_*" / "records.csv"))
              if (Path(path).parent / "manifest.json").exists()]

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


#WRITE CSV + MARKDOWN TABLE
def save_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path.with_suffix(".csv"), index=False)
    path.with_suffix(".md").write_text(df.to_markdown(index=False, floatfmt=".1f"), encoding="utf-8")


#FIG: HELD-OUT SUCCESS PER STAGE (BARS = MEAN OF SEEDS, DOTS = SEEDS, GREY TICKS = REFERENCES)
def fig_heldout_success(per_seed: pd.DataFrame, refs: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    stages = sorted(per_seed["scenario"].unique(), key=lambda s: int(s[5]))
    width = 0.19

    for j, cond in enumerate(CONDS):
        st = STYLE[cond]
        means = [per_seed[(per_seed.scenario == s) & (per_seed.condition == cond)]["success"].mean() for s in stages]
        xs = np.arange(len(stages)) + (j - 1.5) * (width + 0.01)
        ax.bar(xs, means, width, color=st["color"], hatch=st["hatch"], edgecolor="white", linewidth=0.6, label=st["label"])

        for x, scen in zip(xs, stages):
            vals = per_seed[(per_seed.scenario == scen) & (per_seed.condition == cond)]["success"]
            ax.scatter([x] * len(vals), vals, s=10, color=INK, zorder=3, linewidths=0)

    for name, colour, marker in (("straight", STRAIGHT, "_"), ("expert", EXPERT, "_")):
        if len(refs) and name in set(refs["condition"]):
            vals = [100 * refs[(refs.scenario == s) & (refs.condition == name)]["reached"].mean() for s in stages]
            ax.scatter(np.arange(len(stages)), vals, marker=marker, s=900, linewidths=2.2, color=colour, zorder=4,
                       label=f"{name} (scripted reference)")

    ax.set_xticks(np.arange(len(stages)), STAGE_LABELS[:len(stages)])
    ax.set_xlabel("curriculum stage (static, moving obstacles)", color=INK_2)
    ax.set_ylabel("success (%)", color=INK_2)
    ax.set_ylim(0, 105)
    style_axes(ax)
    ax.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.3), markerscale=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


#FIG: PAIRED GAP (SEES - BLIND) PER STAGE/SETTING, ONE PANEL PER PAIR
def fig_gaps(gaps: pd.DataFrame, order: list[str], labels: list[str], title: str, xlabel: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)

    for ax, (_, _, label) in zip(axes, PAIRS):
        sub = gaps[gaps["pair"] == label]
        means = [sub[sub.scenario == s]["gap"].mean() for s in order]
        xs = np.arange(len(order))
        ax.axhline(0, color=INK_2, linewidth=0.8)
        ax.bar(xs, means, 0.55, color=GRID, edgecolor=INK_2, linewidth=0.6)

        for x, scen in zip(xs, order):
            seeds = sub[sub.scenario == scen]
            sig = seeds["p_holm"] < 0.05
            ax.scatter([x] * int((~sig).sum()), seeds.loc[~sig, "gap"], s=16, facecolors="white", edgecolors=INK, zorder=3)
            ax.scatter([x] * int(sig.sum()), seeds.loc[sig, "gap"], s=16, color=INK, zorder=3)

        ax.set_xticks(xs, labels)
        ax.set_title(f"{label}: sees minus blind", color=INK, loc="left")
        ax.set_xlabel(xlabel, color=INK_2)
        style_axes(ax)

    axes[0].set_ylabel("success gap (percentage points)", color=INK_2)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


#FIG: WALL-GAP SUCCESS VS COVERAGE, ONE PANEL PER SCALE
def fig_wall_coverage(per_seed: pd.DataFrame, refs: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)

    for ax, scale in zip(axes, (2, 3)):
        scens = [f"wall_s{scale}_x{c:02d}" for c in COVERAGES]

        for cond in CONDS:
            st = STYLE[cond]
            means = [per_seed[(per_seed.scenario == s) & (per_seed.condition == cond)]["success"].mean() for s in scens]
            ax.plot(COVERAGES, means, linestyle=st["dash"], color=st["color"], linewidth=2, marker="o", markersize=5, label=st["label"])

        for name, colour, dash in (("expert", EXPERT, "-"), ("straight", STRAIGHT, ":")):
            if len(refs) and name in set(refs["condition"]):
                vals = [100 * refs[(refs.scenario == s) & (refs.condition == name)]["reached"].mean() for s in scens]
                ax.plot(COVERAGES, vals, linestyle=dash, color=colour, linewidth=1.6, marker="s", markersize=4, label=f"{name} (scripted reference)")

        ax.set_xticks(COVERAGES, [f"{c}%" if c < 90 else "90%*" for c in COVERAGES])
        ax.set_xlabel("wall coverage of the passage", color=INK_2)
        ax.set_title(f"world scale {scale}" + (" (training size)" if scale == 2 else " (larger than training)"), color=INK, loc="left")
        ax.set_ylim(0, 102)
        style_axes(ax)

    axes[0].set_ylabel("success (%)", color=INK_2)
    axes[1].legend(fontsize=6.8, frameon=False, loc="upper right")
    fig.text(0.01, 0.005, "*90% reduced where needed to keep a 0.4 m opening (actual mean 84.6% at scale 2, 88.5% at scale 3)", fontsize=7.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


#FIG: GROUPED BARS (CONDITIONS) OVER CATEGORY GROUPS
def fig_grouped(table: pd.DataFrame, groups: list[str], group_col: str, title: str, out: Path, refs: dict | None = None, ymax: float = 105) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    width = 0.19

    for j, cond in enumerate(CONDS):
        st = STYLE[cond]
        vals = [table[(table[group_col] == g) & (table.condition == cond)]["success"].mean() for g in groups]
        xs = np.arange(len(groups)) + (j - 1.5) * (width + 0.01)
        ax.bar(xs, vals, width, color=st["color"], hatch=st["hatch"], edgecolor="white", linewidth=0.6, label=st["label"])

        for x, v in zip(xs, vals):
            ax.text(x, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=6.5, color=INK_2)

    for name, (colour, vals) in (refs or {}).items():
        ax.scatter(np.arange(len(groups)), vals, marker="_", s=900, linewidths=2.2, color=colour, zorder=4, label=f"{name} (scripted reference)")

    ax.set_xticks(np.arange(len(groups)), groups)
    ax.set_ylabel("success (%)", color=INK_2)
    ax.set_ylim(0, ymax)
    style_axes(ax)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), markerscale=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


#FIG + TABLE: TRAINING DIAGNOSTICS PER STAGE (KL, ACTION STD)
def diagnostics(models_dir: Path, fig_dir: Path, tab_dir: Path) -> pd.DataFrame:
    rows = []

    for cond in CONDS:
        for seed in range(3):
            path = glob.glob(str(models_dir / f"ppo_{cond}_s{seed}" / "diagnostics" / "*" / "progress.csv"))[0]
            data = pd.read_csv(path).dropna(subset=["train/approx_kl"])

            for stage, grp in data.groupby("rollout_stats/stage"):
                rows.append(dict(condition=cond, train_seed=seed, stage=int(stage),
                                 approx_kl=grp["train/approx_kl"].median(), clip_fraction=grp["train/clip_fraction"].median(),
                                 action_std=grp["train/std"].median(),
                                 imitation_loss=grp.loc[grp["train/imitation_weight"] > 0, "train/imitation_loss"].median(),
                                 explained_variance=grp["train/explained_variance"].median()))

    diag = pd.DataFrame(rows)
    summary = diag.groupby(["condition", "stage"])[["approx_kl", "clip_fraction", "action_std", "imitation_loss", "explained_variance"]].mean().reset_index()
    save_table(summary.round(4), tab_dir / "training_diagnostics_by_stage")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    for ax, metric, label in ((axes[0], "approx_kl", "median approx. KL per update"), (axes[1], "action_std", "median policy action std")):
        for cond in CONDS:
            st = STYLE[cond]
            sub = summary[summary.condition == cond]
            ax.plot(sub["stage"], sub[metric], linestyle=st["dash"], color=st["color"], linewidth=2, marker="o", markersize=4, label=st["label"])

        ax.set_xlabel("curriculum stage", color=INK_2)
        ax.set_title(label, color=INK, loc="left")
        style_axes(ax)

    axes[0].axhline(0.045, color=INK_2, linewidth=0.8, linestyle=":")
    axes[0].text(0.05, 0.0475, "early-stop cap (1.5 x target_kl = 0.045)", fontsize=6.5, color=INK_2)
    axes[0].set_ylim(0, 0.06)
    axes[1].set_ylim(0, 0.85)
    axes[1].legend(fontsize=6.8, frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "training_diagnostics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return summary


#FIG: PHASE 2 - SUCCESS BEFORE PHASE 2, AT THE END OF THE 50% STAGE AND AT THE FINAL MODEL (ROWS = PAIRS)
def fig_phase2(tables: Path, refs_wall: pd.DataFrame, refs_held: pd.DataFrame, out: Path) -> None:
    wall = {key: pd.read_csv(tables / f"wall_gap_success_{key}.csv", index_col=0) for key in ("phase1", "stage2", "latest")}
    held = {key: pd.read_csv(tables / f"training_like_before_after_{key}.csv") for key in ("stage2", "latest")}
    stages = sorted(held["stage2"]["scenario"].unique(), key=lambda s: int(s[5]))
    checkpoints = (("phase1", "before phase 2", dict(linewidth=1.1, alpha=0.55, linestyle="-", marker="o", markerfacecolor="white", markersize=4)),
                   ("stage2", "end of the 50% stage", dict(linewidth=2.2, linestyle="-", marker="o", markersize=5)),
                   ("latest", "final model", dict(linewidth=1.4, linestyle=":", marker="x", markersize=5)))
    wall_refs = (("expert", EXPERT, "-", "expert (scripted)"), ("gap_expert", EXPERT, "--", "gap-aware teacher"), ("straight", STRAIGHT, ":", "straight flight"))

    #SUCCESS SERIES FOR ONE CONDITION AT ONE CHECKPOINT
    def held_series(cond, key):
        table, column = (held["stage2"], "before") if key == "phase1" else (held[key], "after")
        sub = table[table.condition == cond].set_index("scenario")

        return [float(sub.loc[s, column]) for s in stages]

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9), sharey=True)

    for row, (sees, blind, label) in enumerate(PAIRS):
        for col, scale in enumerate((2, 3)):
            ax = axes[row, col]
            scens = [f"wall_s{scale}_x{c:02d}" for c in COVERAGES]

            for cond in (blind, sees):
                for key, _, kw in checkpoints:
                    ax.plot(COVERAGES, [float(wall[key].loc[s, cond]) for s in scens], color=STYLE[cond]["color"], **kw)

            for name, colour, dash, _ in wall_refs:
                if len(refs_wall) and name in set(refs_wall["condition"]):
                    vals = [100 * refs_wall[(refs_wall.scenario == s) & (refs_wall.condition == name)]["reached"].mean() for s in scens]
                    ax.plot(COVERAGES, vals, linestyle=dash, color=colour, linewidth=1.0)

            ax.set_xticks(COVERAGES, [f"{c}%" if c < 90 else "90%*" for c in COVERAGES])

            if row == 1:
                ax.set_xlabel("wall coverage", color=INK_2)

        ax = axes[row, 2]

        for cond in (blind, sees):
            for key, _, kw in checkpoints:
                ax.plot(np.arange(len(stages)), held_series(cond, key), color=STYLE[cond]["color"], **kw)

        for name, colour, dash, _ in wall_refs:
            if name != "gap_expert" and len(refs_held) and name in set(refs_held["condition"]):
                vals = [100 * refs_held[(refs_held.scenario == s) & (refs_held.condition == name)]["reached"].mean() for s in stages]
                ax.plot(np.arange(len(stages)), vals, linestyle=dash, color=colour, linewidth=1.0)

        ax.set_xticks(np.arange(len(stages)), [str(i) for i in range(len(stages))])

        if row == 1:
            ax.set_xlabel("curriculum stage", color=INK_2)

        axes[row, 0].set_ylabel(f"{label}: success (%)", color=INK_2)

    for ax, title in zip(axes[0], ("wall-gap worlds, scale 2 (training size)", "wall-gap worlds, scale 3", "unseen training-like worlds")):
        ax.set_title(title, color=INK, loc="left", fontsize=8.5)

    for ax in axes.ravel():
        ax.set_ylim(0, 102)
        style_axes(ax)

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=BLIND, linewidth=2.2, label="blind (degenerate / reward_only)"),
               Line2D([], [], color=SEES, linewidth=2.2, label="sees obstacles (sensing_only / full)")]
    handles += [Line2D([], [], color=INK, label=name, **kw) for _, name, kw in checkpoints]
    handles += [Line2D([], [], color=colour, linestyle=dash, linewidth=1.0, label=name) for _, colour, dash, name in wall_refs]
    fig.legend(handles=handles, ncol=4, fontsize=6.8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.text(0.01, 0.075, "*90% reduced where needed to keep a 0.4 m opening", fontsize=6.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


#BUILD EVERYTHING
def main() -> None:
    parser = argparse.ArgumentParser(description="Report figures and tables")
    parser.add_argument("--heldout", default="results/compact_eval_200ep/eval/records.csv")
    parser.add_argument("--wall", default="results/compact_wall_gap/eval/records.csv")
    parser.add_argument("--refs", default="results/reference")
    parser.add_argument("--models-dir", default="models/compact")
    parser.add_argument("--out-dir", default="results/report")
    parser.add_argument("--phase2-tables", default="results/phase2_tables", help="output of analysis.phase2_report; the phase-2 figure is skipped when absent")
    args = parser.parse_args()

    out = Path(args.out_dir)
    fig_dir, tab_dir = out / "figures", out / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    ref_dir = Path(args.refs)

    #HELD-OUT (TRAINING-MATCHED) EVALUATION
    held = pd.read_csv(args.heldout)
    held_refs = load_refs(ref_dir, "preset")
    held_seed = seed_success(held)
    held_gaps = paired_gaps(held)
    stages = sorted(held["scenario"].unique(), key=lambda s: int(s[5]))
    save_table(held_seed.pivot_table(index=["scenario", "train_seed"], columns="condition", values="success").reset_index(), tab_dir / "heldout_success_by_seed")
    save_table(held_gaps, tab_dir / "heldout_paired_gaps")

    if len(held_refs):
        save_table(held_refs.groupby(["scenario", "condition"]).agg(success=("reached", "mean"), crash=("collided", "mean"), timeout=("timeout", "mean"))
                   .mul(100).reset_index(), tab_dir / "heldout_reference_rows")

    fig_heldout_success(held_seed, held_refs, fig_dir / "heldout_success_by_stage.png")
    fig_gaps(held_gaps, stages, [str(i) for i in range(len(stages))], "Unseen training-like worlds: paired success gap per stage",
             "curriculum stage", fig_dir / "heldout_gap_by_stage.png")

    #WALL-GAP EVALUATION
    wall = pd.read_csv(args.wall)
    wall_refs = load_refs(ref_dir, "wall_gap")
    blocked = {key: line_blocked(row) for key, row in wall.drop_duplicates(["scenario", "seed"]).set_index(["scenario", "seed"]).iterrows()}
    wall["line"] = [("blocked" if blocked[(s, e)] else "clear") for s, e in zip(wall["scenario"], wall["seed"])]
    wall_seed = seed_success(wall)
    wall_gaps = paired_gaps(wall)
    wall_order = [f"wall_s{s}_x{c:02d}" for s in (2, 3) for c in COVERAGES]
    save_table(wall_seed.pivot_table(index=["scenario", "train_seed"], columns="condition", values="success").reset_index(), tab_dir / "wallgap_success_by_seed")
    save_table(wall_gaps, tab_dir / "wallgap_paired_gaps")
    fig_wall_coverage(wall_seed, wall_refs, fig_dir / "wallgap_success_vs_coverage.png")
    fig_gaps(wall_gaps, wall_order, [f"s{s}\n{c}%" for s in (2, 3) for c in COVERAGES], "Wall-gap worlds: paired success gap per setting",
             "world scale / wall coverage", fig_dir / "wallgap_gap_by_setting.png")

    #PHASE 2 (ONLY ONCE ITS TABLES EXIST)
    tables = Path(args.phase2_tables)

    if (tables / "wall_gap_success_stage2.csv").exists():
        fig_phase2(tables, wall_refs, held_refs, fig_dir / "phase2_success.png")

    #CLEAR VS BLOCKED STRAIGHT LINE
    wall["group"] = "scale " + wall["world_scale"].astype(int).astype(str) + ", line " + wall["line"]
    clear_tab = wall.groupby(["group", "condition"])["reached"].mean().mul(100).rename("success").reset_index()
    groups = ["scale 2, line clear", "scale 2, line blocked", "scale 3, line clear", "scale 3, line blocked"]
    save_table(clear_tab, tab_dir / "wallgap_clear_vs_blocked")
    ref_marks = {}

    if len(wall_refs):
        wall_refs["line"] = [("blocked" if blocked[(s, e)] else "clear") for s, e in zip(wall_refs["scenario"], wall_refs["seed"])]
        wall_refs["group"] = "scale " + wall_refs["world_scale"].astype(int).astype(str) + ", line " + wall_refs["line"]

        for name, colour in (("straight", STRAIGHT), ("expert", EXPERT)):
            sub = wall_refs[wall_refs.condition == name]

            if len(sub):
                ref_marks[name] = (colour, [100 * sub[sub.group == g]["reached"].mean() for g in groups])

    fig_grouped(clear_tab, groups, "group", "Wall-gap worlds: success when the straight start-goal line is clear vs blocked by the wall",
                fig_dir / "wallgap_clear_vs_blocked.png", ref_marks)

    #ORIENTATION (BLOCKED-LINE WORLDS ONLY)
    blk = wall[wall["line"] == "blocked"]
    orient_tab = blk.groupby(["wall_orientation", "condition"])["reached"].mean().mul(100).rename("success").reset_index()
    save_table(orient_tab, tab_dir / "wallgap_blocked_by_orientation")
    fig_grouped(orient_tab, ["vertical", "horizontal", "diagonal"], "wall_orientation",
                "Wall-gap worlds with the straight line blocked: success by wall orientation", fig_dir / "wallgap_blocked_by_orientation.png", ymax=40)

    if len(wall_refs):
        save_table(wall_refs.groupby(["scenario", "condition"]).agg(success=("reached", "mean"), crash=("collided", "mean"), timeout=("timeout", "mean"))
                   .mul(100).reset_index(), tab_dir / "wallgap_reference_rows")

    #TRAINING DIAGNOSTICS
    diagnostics(Path(args.models_dir), fig_dir, tab_dir)
    print(f"[SAVED] figures -> {fig_dir} | tables -> {tab_dir}")


if __name__ == "__main__":
    main()
