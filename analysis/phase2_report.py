# -*- coding: utf-8 -*-
#NEWLY ADDED FOR CS5478 PROJECT
"""
Phase-2 (E10) evaluation tables. Reads finished evaluation records only.

Inputs (all on the same worlds, paired by scenario + episode seed):
  * phase-1 models   : results/compact_wall_gap (E8), results/compact_eval_200ep (E7)
  * phase-2 models   : results/phase2_*_{stage2,latest}_{blind,sensing}
  * scripted refs    : results/reference/wall_gap_{straight,gap_expert}

Outputs -> results/phase2_tables/: markdown + csv tables and summary.json.

Tests follow E8: exact McNemar per training seed and scenario for the two
sensing contrasts (goal-only: degenerate -> sensing_only, shaped:
reward_only -> full), Holm across all rows of one table. Pooled numbers
are descriptive only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.analyze import mcnemar_exact

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ["degenerate", "reward_only", "sensing_only", "full"]
BLIND = ["degenerate", "reward_only"]
SENSING = ["sensing_only", "full"]
CONTRASTS = [("goal-only", "degenerate", "sensing_only"), ("shaped", "reward_only", "full")]
KEY = ["scenario", "condition", "train_seed", "seed"]


#LOAD ONE MERGED RECORDS FILE
def load(path: Path) -> pd.DataFrame:
    path = path / "eval" / "records.csv" if path.is_dir() else path

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if df.duplicated(KEY).any():
        raise ValueError(f"duplicate episode keys in {path}")

    return df


#PHASE-2 RECORDS FOR ONE SUITE AND CHECKPOINT. BLIND RUNS STALLED AT STAGE 2,
#WHERE stage_2 AND latest ARE THE SAME WEIGHTS, SO THEIR stage_2 RECORDS SERVE BOTH.
def phase2(suite: str, checkpoint: str) -> pd.DataFrame:
    blind = load(ROOT / f"results/phase2_{suite}_stage2_blind")
    sensing = load(ROOT / f"results/phase2_{suite}_{checkpoint}_sensing")

    return pd.concat([blind, sensing], ignore_index=True)


#SUCCESS (%) BY SCENARIO x CONDITION, POOLED OVER TRAINING SEEDS
def success_table(df: pd.DataFrame) -> pd.DataFrame:
    t = df.groupby(["scenario", "condition"]).reached.mean().unstack() * 100

    return t[[c for c in CONDITIONS if c in t.columns]].round(1)


#PER-SEED SUCCESS RANGE "lo-hi" BY SCENARIO x CONDITION
def seed_range_table(df: pd.DataFrame) -> pd.DataFrame:
    t = df.groupby(["scenario", "condition", "train_seed"]).reached.mean().mul(100)
    r = t.groupby(["scenario", "condition"]).agg(["min", "max"])
    out = (r["min"].round(1).astype(str) + "-" + r["max"].round(1).astype(str)).unstack()

    return out[[c for c in CONDITIONS if c in out.columns]]


#HOLM STEP-DOWN ADJUSTED P-VALUES
def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj = np.minimum(1.0, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    out = np.empty_like(adj)
    out[order] = adj

    return out


#SENSING CONTRASTS WITHIN ONE SET OF RECORDS (PER SEED x SCENARIO, HOLM OVER ALL ROWS)
def sensing_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (scenario, seed), g in df.groupby(["scenario", "train_seed"]):
        for label, base, other in CONTRASTS:
            a = g[g.condition == base].set_index("seed").reached
            b = g[g.condition == other].set_index("seed").reached
            shared = a.index.intersection(b.index)

            if len(shared) != len(a) or len(shared) != len(b):
                raise ValueError(f"unpaired episodes: {scenario} s{seed} {base}/{other}")

            a, b = a.loc[shared].to_numpy(int), b.loc[shared].to_numpy(int)
            bw, ow = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
            rows.append({"scenario": scenario, "train_seed": int(seed), "contrast": label,
                         "blind": round(100 * a.mean(), 1), "sensing": round(100 * b.mean(), 1),
                         "gap": round(100 * (b.mean() - a.mean()), 1), "blind_only": bw,
                         "sensing_only_wins": ow, "p": mcnemar_exact(bw, ow)})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p.to_numpy())
    out["sig"] = out.p_holm < 0.05

    return out


#POOLED SENSING GAP PER CONTRAST x SCALE (DESCRIPTIVE + POOLED McNEMAR AS IN E8)
def pooled_gaps(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    df = df.assign(scale=df.scenario.str.extract(r"_s(\d)_")[0])

    for scale, g in df.groupby("scale"):
        for label, base, other in CONTRASTS:
            a = g[g.condition == base].set_index(["scenario", "train_seed", "seed"]).reached
            b = g[g.condition == other].set_index(["scenario", "train_seed", "seed"]).reached
            idx = a.index.intersection(b.index)
            a, b = a.loc[idx].to_numpy(int), b.loc[idx].to_numpy(int)
            bw, ow = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
            rows.append({"scale": int(scale), "contrast": label, "n": len(idx),
                         "blind": round(100 * a.mean(), 1), "sensing": round(100 * b.mean(), 1),
                         "gap": round(100 * (b.mean() - a.mean()), 1), "p": mcnemar_exact(bw, ow)})

    return pd.DataFrame(rows)


#SAME MODEL (CONDITION, SEED) BEFORE vs AFTER PHASE 2 ON IDENTICAL WORLDS
def before_after(before: pd.DataFrame, after: pd.DataFrame, by: str) -> pd.DataFrame:
    m = before.merge(after, on=KEY, suffixes=("_before", "_after"))

    if len(m) != len(after):
        raise ValueError(f"before/after pairing incomplete: {len(m)} of {len(after)}")

    for ax in ("x", "y", "z"):
        if not np.allclose(m[f"start_{ax}_before"], m[f"start_{ax}_after"], atol=1e-5):
            raise ValueError("before/after worlds differ")
    rows = []

    for (grp, cond), g in m.groupby([by, "condition"]):
        a, b = g.reached_before.to_numpy(int), g.reached_after.to_numpy(int)
        bw, aw = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
        rows.append({by: grp, "condition": cond, "n": len(g), "before": round(100 * a.mean(), 1),
                     "after": round(100 * b.mean(), 1), "change": round(100 * (b.mean() - a.mean()), 1),
                     "lost": bw, "gained": aw, "p": mcnemar_exact(bw, aw)})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p.to_numpy())

    return out


#WALL-GAP SUCCESS SPLIT BY WHETHER FLYING STRAIGHT REACHES THE GOAL IN THAT WORLD
def straight_split(df: pd.DataFrame) -> pd.DataFrame:
    ref = load(ROOT / "results/reference" / "wall_gap_straight" / "records.csv")
    ref = ref[["scenario", "seed", "reached"]].rename(columns={"reached": "straight_ok"})

    if ref.duplicated(["scenario", "seed"]).any():
        raise ValueError("straight reference has duplicate worlds")

    m = df.merge(ref, on=["scenario", "seed"], how="left")

    if m.straight_ok.isna().any():
        raise ValueError("worlds missing from the straight-flight reference")

    m["line"] = np.where(m.straight_ok == 1, "clear", "blocked")
    m["scale"] = m.scenario.str.extract(r"_s(\d)_")[0]
    t = m.groupby(["scale", "line", "condition"]).reached.mean().unstack() * 100

    return t[[c for c in CONDITIONS if c in t.columns]].round(1)


#WRITE ONE TABLE AS CSV + MARKDOWN
def write(df: pd.DataFrame, out: Path, name: str, index: bool = True) -> None:
    df.to_csv(out / f"{name}.csv", index=index)
    (out / f"{name}.md").write_text(df.to_markdown(index=index), encoding="utf-8")


#BUILD ALL PHASE-2 TABLES (stage2 & latest CHECKPOINTS) + PHASE-1 REFERENCE TABLES
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(ROOT / "results/phase2_tables"))
    ap.add_argument("--blind-only", action="store_true", help="only the blind tables (sensing records not ready)")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}

    e8 = load(ROOT / "results/compact_wall_gap")
    e7 = load(ROOT / "results/compact_eval_200ep")

    if args.blind_only:
        wg = load(ROOT / "results/phase2_wall_gap_stage2_blind")
        tl = load(ROOT / "results/phase2_training_like_stage2_blind")
        write(before_after(e8[e8.condition.isin(BLIND)], wg, "scenario"), out, "blind_wall_gap_before_after", False)
        write(before_after(e7[e7.condition.isin(BLIND)], tl, "scenario"), out, "blind_training_like_before_after", False)
        print("blind-only tables written to", out)
        return

    for ck in ("stage2", "latest"):
        wg = phase2("wall_gap", ck)
        tl = phase2("training_like", ck)

        for name, d, n in (("wall_gap", wg, 12000 * 2), ("training_like", tl, 8400 * 2)):
            if len(d) != n:
                raise ValueError(f"{name} {ck}: {len(d)} records, expected {n}")

        write(success_table(wg), out, f"wall_gap_success_{ck}")
        write(seed_range_table(wg), out, f"wall_gap_seed_range_{ck}")
        tests = sensing_tests(wg)
        write(tests, out, f"wall_gap_sensing_tests_{ck}", False)
        write(pooled_gaps(wg), out, f"wall_gap_pooled_gaps_{ck}", False)
        write(straight_split(wg), out, f"wall_gap_straight_split_{ck}")
        write(before_after(e8, wg, "scenario"), out, f"wall_gap_before_after_{ck}", False)
        write(success_table(tl), out, f"training_like_success_{ck}")
        write(before_after(e7, tl, "scenario"), out, f"training_like_before_after_{ck}", False)
        tl_overall = tl.groupby("condition").reached.mean().mul(100).round(1)
        e7_overall = e7.groupby("condition").reached.mean().mul(100).round(1)
        summary[ck] = {
            "sensing_tests_significant": int(tests.sig.sum()),
            "sensing_tests_significant_favouring_sensing": int((tests.sig & (tests.gap > 0)).sum()),
            "sensing_tests_significant_favouring_blind": int((tests.sig & (tests.gap < 0)).sum()),
            "sensing_tests_total": int(len(tests)),
            "pooled_gaps": pooled_gaps(wg).to_dict(orient="records"),
            "training_like_overall_before": e7_overall.to_dict(),
            "training_like_overall_after": tl_overall.to_dict(),
        }

    #PHASE-1 REFERENCE TABLES ON THE SAME WORLDS
    write(success_table(e8), out, "wall_gap_success_phase1")
    write(straight_split(e8), out, "wall_gap_straight_split_phase1")
    e8_tests = sensing_tests(e8)
    summary["phase1"] = {"sensing_tests_significant": int(e8_tests.sig.sum()), "sensing_tests_total": int(len(e8_tests)),
                         "pooled_gaps": pooled_gaps(e8).to_dict(orient="records")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
