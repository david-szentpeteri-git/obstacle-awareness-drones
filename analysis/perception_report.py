# -*- coding: utf-8 -*-
#NEWLY ADDED FOR CS5478 PROJECT
"""
Tables for the perception probes (read-only over saved records):

  p2       blocker / straight-line splits of the held-out results
  p4       gap-visibility splits of the wall-gap results
  p5       empty-world failures: placeholder statistics, substitutes, trajectories
  p1       walls hidden from the rays (v1 / v2) vs normal input vs blind partner
  sampled  sampled vs deterministic actions on the held-out worlds

  .\\venv\\Scripts\\python.exe -B -m analysis.perception_report --part all

Writes CSV + Markdown tables to results/perception/tables/ and prints a
summary. Paired tests: exact McNemar on worlds where only one model
succeeded (same worlds for every model), per training seed; pooled rows
merge seeds and are supporting evidence only.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

CONDS = ["degenerate", "sensing_only", "reward_only", "full"]
PAIRS = [("sensing_only", "degenerate", "goal-only"), ("full", "reward_only", "shaped")]
DENSE = ["stage3_3s0d", "stage4_3s1d", "stage5_5s2d", "stage6_6s3d"]
OUT = Path("results/perception/tables")


#EXACT McNEMAR p ON DISCORDANT PAIRS
def mcnemar(only_a: int, only_b: int) -> float:
    total = only_a + only_b

    return 1.0 if total == 0 else float(binomtest(only_a, total, 0.5).pvalue)


#PAIRED COMPARISON OF TWO MODEL SETS ON THE SAME WORLDS: SUCCESS, GAP, DISCORDANT, p
def paired(a: pd.DataFrame, b: pd.DataFrame, keys=("scenario", "seed", "train_seed")) -> dict:
    merged = a.merge(b, on=list(keys), suffixes=("_a", "_b"))
    only_a = int(((merged.reached_a == 1) & (merged.reached_b == 0)).sum())
    only_b = int(((merged.reached_b == 1) & (merged.reached_a == 0)).sum())

    return dict(n=len(merged), success_a=100 * merged.reached_a.mean(), success_b=100 * merged.reached_b.mean(),
                gap=100 * (merged.reached_a.mean() - merged.reached_b.mean()), only_a=only_a, only_b=only_b, p=mcnemar(only_a, only_b))


#WRITE CSV + MARKDOWN
def save(df: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{name}.csv", index=False)
    (OUT / f"{name}.md").write_text(df.to_markdown(index=False, floatfmt=".1f"), encoding="utf-8")
    print(f"\n== {name} ==\n{df.to_string(index=False)}")


#RECORDS OF ONE RESULTS FOLDER (MERGED OR PARTS)
def load_records(folder: str) -> pd.DataFrame:
    merged = Path(folder) / "eval" / "records.csv"

    if merged.exists():
        return pd.read_csv(merged)

    parts = sorted(glob.glob(str(Path(folder) / "eval" / "_part_*" / "records.csv")))

    return pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)


#SCRIPTED REFERENCE RECORDS FOR A SUITE
def load_refs(suite: str) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in glob.glob(f"results/reference/{suite}_*/records.csv")]

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


#SUCCESS / HIT RATES PER GROUP
def rates(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    out = df.groupby(by).agg(n=("reached", "size"), success=("reached", "mean"), obstacle_hit=("obstacle_hit", "mean"),
                             boundary_hit=("wall_hit", "mean"), timeout=("timeout", "mean")).reset_index()

    for col in ("success", "obstacle_hit", "boundary_hit", "timeout"):
        out[col] = 100 * out[col]

    return out


#P2: BLOCKER / LINE SPLITS
def part_p2() -> None:
    worlds = pd.read_csv("results/perception/p2/worlds.csv")
    held = load_records("results/compact_eval_200ep").merge(worlds, on=["scenario", "seed"])
    refs = load_refs("preset").merge(worlds, on=["scenario", "seed"])
    worlds["stage"] = worlds.scenario.str[5].astype(int)
    save(worlds.groupby("stage").agg(worlds=("seed", "size"), blocker_requested=("blocker_requested", "mean"),
                                     blocker_spawned=("blocker_spawned", "mean"), line_obstructed=("line_obstructed", "mean"),
                                     replay_error_max_m=("replay_error_m", "max")).mul([1, 100, 100, 100, 1]).reset_index(), "p2_world_labels")
    held["line"] = np.where(held.line_obstructed == 1, "obstructed", "clear")
    held["blocker"] = np.where(held.blocker_spawned == 1, "blocker", "no_blocker")
    refs["line"] = np.where(refs.line_obstructed == 1, "obstructed", "clear")
    dense = held[held.scenario.isin(DENSE)]
    save(rates(dense, ["condition", "line"]).round(1), "p2_success_by_line_stages3to6")
    save(rates(dense, ["condition", "blocker"]).round(1), "p2_success_by_blocker_stages3to6")
    save(rates(held[held.scenario != "stage0_0s0d"], ["scenario", "condition", "line"]).round(1), "p2_success_by_line_per_stage")
    save(rates(refs[refs.scenario.isin(DENSE)], ["condition", "line"]).round(1), "p2_reference_by_line_stages3to6")
    rows = []

    for sees, blind, label in PAIRS:
        for line in ("clear", "obstructed"):
            for scope, frame in (("stages 3-6", dense), *[(s, held[held.scenario == s]) for s in DENSE]):
                sub = frame[frame.line == line]
                res = paired(sub[sub.condition == sees], sub[sub.condition == blind])
                res.update(pair=label, line=line, scope=scope)
                rows.append(res)

    save(pd.DataFrame(rows)[["pair", "line", "scope", "n", "success_a", "success_b", "gap", "only_a", "only_b", "p"]].round(4), "p2_paired_gaps_by_line")


#P4: GAP VISIBILITY SPLITS (BLOCKED-LINE WORLDS)
def part_p4() -> None:
    vis = pd.read_csv("results/perception/p4/gap_visibility.csv")
    wall = load_records("results/compact_wall_gap").merge(vis[["scenario", "seed", "line_blocked", "gap_visible", "first_visible_dist_to_wall", "first_visible_ray"]], on=["scenario", "seed"])
    refs = load_refs("wall_gap").merge(vis[["scenario", "seed", "line_blocked", "gap_visible"]], on=["scenario", "seed"])
    blocked_worlds = vis[vis.line_blocked == 1]
    save(blocked_worlds.groupby("wall_orientation").agg(worlds=("seed", "size"), gap_visible=("gap_visible", "mean"),
                                                        first_visible_dist_median=("first_visible_dist_to_wall", "median"))
         .assign(gap_visible=lambda d: 100 * d.gap_visible).reset_index().round(2), "p4_visibility_by_orientation")
    save(blocked_worlds[blocked_worlds.gap_visible == 1].groupby(["wall_orientation", "first_visible_ray"]).size().rename("worlds").reset_index(), "p4_first_visible_ray")
    blk = wall[wall.line_blocked == 1].copy()
    blk["visible"] = np.where(blk.gap_visible == 1, "visible", "not_visible")
    save(rates(blk, ["wall_orientation", "visible", "condition"]).round(1), "p4_success_by_visibility")
    rblk = refs[refs.line_blocked == 1].copy()
    rblk["visible"] = np.where(rblk.gap_visible == 1, "visible", "not_visible")
    save(rates(rblk, ["wall_orientation", "visible", "condition"]).round(1), "p4_reference_by_visibility")
    rows = []

    for sees, blind, label in PAIRS:
        for visible in ("visible", "not_visible"):
            for orient in ("all", "vertical", "horizontal", "diagonal"):
                sub = blk[(blk.visible == visible) & ((blk.wall_orientation == orient) if orient != "all" else True)]
                res = paired(sub[sub.condition == sees], sub[sub.condition == blind])
                res.update(pair=label, visible=visible, orientation=orient)
                rows.append(res)

    save(pd.DataFrame(rows)[["pair", "orientation", "visible", "n", "success_a", "success_b", "gap", "only_a", "only_b", "p"]].round(4), "p4_paired_gaps_by_visibility")


#P5: PLACEHOLDER STATISTICS, SUBSTITUTES, TRAJECTORIES
def part_p5() -> None:
    stats = pd.read_csv("results/perception/p5/placeholder_stats.csv")
    save(stats, "p5_placeholder_stats")
    held = load_records("results/compact_eval_200ep")
    stage0 = held[held.scenario == "stage0_0s0d"]
    rows = []

    for cond in ("sensing_only", "full"):
        partner = "degenerate" if cond == "sensing_only" else "reward_only"

        for seed in range(3):
            base = stage0[(stage0.condition == cond) & (stage0.train_seed == seed)]
            row = dict(condition=cond, train_seed=seed, normal=100 * base.reached.mean(), normal_boundary_hit=100 * base.wall_hit.mean(),
                       normal_timeout=100 * base.timeout.mean(), blind_partner=100 * stage0[(stage0.condition == partner) & (stage0.train_seed == seed)].reached.mean())

            for variant in ("behind2", "scaled"):
                sub = load_records(f"results/perception/p5_{variant}")
                sub = sub[(sub.condition == cond) & (sub.train_seed == seed)]
                res = paired(sub, base)
                row[variant] = 100 * sub.reached.mean()
                row[f"{variant}_vs_normal_p"] = res["p"]
                row[f"{variant}_timeout"] = 100 * sub.timeout.mean()

            rows.append(row)

    save(pd.DataFrame(rows).round(3), "p5_substitutes_stage0")
    rows = []

    for path in sorted(glob.glob("results/perception/p5/traj_*.json")):
        data = json.load(open(path, encoding="utf-8"))
        eps = pd.DataFrame(data["episodes"])
        crashed = eps[eps.collided == 1]
        rows.append(dict(model=f"{data['condition']}_s{data['train_seed']}", episodes=len(eps), reached=int(eps.reached.sum()), crashed=len(crashed),
                         timeout=int(eps.timeout.sum()), overshoot=int(crashed.overshoot.sum()) if len(crashed) else 0,
                         crash_floor=int((crashed.crash_boundary == "floor").sum()), crash_ceiling=int((crashed.crash_boundary == "ceiling").sum()),
                         crash_side=int((crashed.crash_boundary == "side").sum()),
                         closest_to_goal_median_m=float(crashed.closest_to_goal_m.median()) if len(crashed) else float("nan"),
                         reached_closest_median_m=float(eps[eps.reached == 1].closest_to_goal_m.median()) if eps.reached.sum() else float("nan")))

    save(pd.DataFrame(rows).round(3), "p5_trajectories_stage0")


#P1: WALLS HIDDEN FROM THE RAYS
def part_p1() -> None:
    for suite, base_dir, scope_name, scope in (("preset", "results/compact_eval_200ep", "stages 3-6", lambda d: d.scenario.isin(DENSE)),
                                               ("wall_gap", "results/compact_wall_gap", "all settings", lambda d: pd.Series(True, index=d.index))):
        normal = load_records(base_dir)
        #ONLY COMPLETED (MERGED) VARIANT RUNS; v1 = OUT-OF-RANGE CONTROL, v2b = IN-RANGE BOUNDARY-ONLY INPUT
        variants = {v: load_records(f"results/perception/p1_{v}_{suite}") for v in ("v1", "v2", "v2b")
                    if (Path(f"results/perception/p1_{v}_{suite}") / "eval" / "records.csv").exists()}

        if suite == "wall_gap":
            vis = pd.read_csv("results/perception/p4/gap_visibility.csv")[["scenario", "seed", "line_blocked"]]
            normal = normal.merge(vis, on=["scenario", "seed"])
            variants = {v: d.merge(vis, on=["scenario", "seed"]) for v, d in variants.items()}
            scopes = [("all settings", lambda d: pd.Series(True, index=d.index)), ("line blocked", lambda d: d.line_blocked == 1)]
        else:
            scopes = [(scope_name, scope)]

        rows = []

        for scope_label, selector in scopes:
            for sees, blind, label in PAIRS:
                blind_df = normal[(normal.condition == blind) & selector(normal)]
                sets = {"normal": normal[(normal.condition == sees) & selector(normal)]}
                sets.update({v: d[(d.condition == sees) & selector(d)] for v, d in variants.items()})

                for name, df in sets.items():
                    vs_blind = paired(df, blind_df)
                    vs_normal = paired(df, sets["normal"]) if name != "normal" else dict(gap=0.0, p=1.0)
                    rows.append(dict(suite=suite, scope=scope_label, pair=label, input=name, n=len(df), success=100 * df.reached.mean(),
                                     obstacle_hit=100 * df.obstacle_hit.mean(), boundary_hit=100 * df.wall_hit.mean(), timeout=100 * df.timeout.mean(),
                                     blind_success=vs_blind["success_b"], advantage_over_blind=vs_blind["gap"], p_vs_blind=vs_blind["p"],
                                     change_vs_normal=vs_normal["gap"], p_vs_normal=vs_normal["p"]))

        save(pd.DataFrame(rows).round(4), f"p1_{suite}")
        per_seed = []

        for scope_label, selector in scopes:
            for sees, blind, label in PAIRS:
                for seed in range(3):
                    blind_df = normal[(normal.condition == blind) & (normal.train_seed == seed) & selector(normal)]

                    for name, d in (("normal", normal), *variants.items()):
                        df = d[(d.condition == sees) & (d.train_seed == seed) & selector(d)]
                        res = paired(df, blind_df)
                        per_seed.append(dict(suite=suite, scope=scope_label, pair=label, train_seed=seed, input=name, success=res["success_a"],
                                             blind=res["success_b"], advantage=res["gap"], p=res["p"]))

        save(pd.DataFrame(per_seed).round(4), f"p1_{suite}_per_seed")


#SAMPLED VS DETERMINISTIC ACTIONS
def part_sampled() -> None:
    det = load_records("results/compact_eval_200ep")
    smp = load_records("results/perception/sampled_preset")
    det["stage"] = det.scenario.str[5].astype(int)
    smp["stage"] = smp.scenario.str[5].astype(int)
    both = pd.concat([det.assign(mode="deterministic"), smp.assign(mode="sampled")], ignore_index=True)
    save(rates(both, ["mode", "condition", "stage"]).round(1), "sampled_by_stage")
    save(rates(both[both.stage >= 3], ["mode", "condition"]).round(1), "sampled_stages3to6")
    rows = []

    for mode, frame in (("deterministic", det), ("sampled", smp)):
        for sees, blind, label in PAIRS:
            for scope, sub in (("stages 3-6", frame[frame.stage >= 3]), *[(f"stage {s}", frame[frame.stage == s]) for s in range(7)]):
                res = paired(sub[sub.condition == sees], sub[sub.condition == blind])
                res.update(mode=mode, pair=label, scope=scope)
                rows.append(res)

    save(pd.DataFrame(rows)[["mode", "pair", "scope", "n", "success_a", "success_b", "gap", "only_a", "only_b", "p"]].round(4), "sampled_paired_gaps")


#RUN THE SELECTED TABLE PARTS (p1/p2/p4/p5/sampled OR ALL)
def main() -> None:
    parser = argparse.ArgumentParser(description="Perception probe tables")
    parser.add_argument("--part", default="all", choices=["all", "p2", "p4", "p5", "p1", "sampled"])
    args = parser.parse_args()
    parts = {"p2": part_p2, "p4": part_p4, "p5": part_p5, "p1": part_p1, "sampled": part_sampled}

    for name, fn in parts.items():
        if args.part in ("all", name):
            fn()


if __name__ == "__main__":
    main()
