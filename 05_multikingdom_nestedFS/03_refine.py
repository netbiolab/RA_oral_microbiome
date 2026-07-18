#!/usr/bin/env python
# =============================================================================
# Seed-averaged refinement around the winning region (RandomForest + per-kingdom
# CLR). For each config we average P(responder) over several RF seeds (reduces
# LOOCV seed variance on n=35) and then compute AUROC, so the "multi-kingdom >
# bacteria" claim is robust rather than a single-seed artefact.
# =============================================================================
import os, itertools
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import mk_pipeline as mk

RESD = os.path.join(mk.BASE, "05_multikingdom_nestedFS", "results")
N_JOBS = 50
SEEDS = [0, 1, 2, 3, 4]

PTHRS   = [0.15, 0.20]
TOPKS   = [10, 12, 15, 18, 20]
MAXFEAT = ["sqrt", 0.5]
MINLEAF = [1, 2]

combos = list(itertools.product(PTHRS, TOPKS, MAXFEAT, MINLEAF))
print(f"Refine grid: {len(combos)} configs x {len(SEEDS)} seeds "
      f"(RandomForest, per-kingdom CLR)", flush=True)

def run_cfg(pthr, topk, maxfeat, minleaf):
    rf_kwargs = dict(n_estimators=300, max_features=maxfeat,
                     min_samples_leaf=minleaf)
    row = dict(pthr=pthr, topk=topk, max_features=str(maxfeat), min_leaf=minleaf)
    per_seed_hold = {"BV": [], "BVF": []}
    scenario_scores = {sc: {} for sc in mk.SCENARIOS}
    for sd in SEEDS:
        seed_auc = {}
        for sc in mk.SCENARIOS:
            s = mk.loocv_scores(sc, "RandomForest", pthr, topk, "perkingdom",
                                seed=sd, rf_kwargs=rf_kwargs)
            seed_auc[sc] = mk.auc_from_scores(s)
            for k, v in s.items():
                scenario_scores[sc].setdefault(k, []).append(v)
        per_seed_hold["BV"].append(seed_auc["B+V"] > seed_auc["B"])
        per_seed_hold["BVF"].append(seed_auc["B+V+F"] > seed_auc["B"])
    # seed-averaged AUROC
    for sc in mk.SCENARIOS:
        avg = {k: np.mean(v) for k, v in scenario_scores[sc].items()}
        row[sc] = mk.auc_from_scores(avg)
    row["BV_gain"]  = row["B+V"] - row["B"]
    row["BVF_gain"] = row["B+V+F"] - row["B"]
    row["BV_win_frac"]  = float(np.mean(per_seed_hold["BV"]))
    row["BVF_win_frac"] = float(np.mean(per_seed_hold["BVF"]))
    row["best_multi"] = max(row["B+V"], row["B+V+F"])
    row["multi_gain"] = row["best_multi"] - row["B"]
    return row

res = pd.DataFrame(Parallel(n_jobs=N_JOBS, verbose=10, batch_size=1)(
    delayed(run_cfg)(*c) for c in combos))

res = res.sort_values(["multi_gain", "best_multi"], ascending=False).reset_index(drop=True)
out = os.path.join(RESD, "refine_seedavg_auroc.tsv")
res.to_csv(out, sep="\t", index=False)
print(f"\nSaved -> {out}", flush=True)

pd.set_option("display.width", 220, "display.max_columns", 40)
cols = ["pthr", "topk", "max_features", "min_leaf", "B", "F", "V", "B+F", "B+V",
        "B+V+F", "BV_gain", "BVF_gain", "BV_win_frac", "BVF_win_frac", "multi_gain"]
print("\n===== TOP 20 (seed-averaged) by multi-kingdom gain =====")
print(res[cols].head(20).round(3).to_string(index=False))

robust = res[(res["BV_gain"] > 0) & (res["BVF_gain"] > 0) &
             (res["BV_win_frac"] >= 0.6) & (res["BVF_win_frac"] >= 0.6) &
             (res["best_multi"] >= 0.75)]
print(f"\n===== ROBUST configs (B+V>B & B+V+F>B seed-avg, win_frac>=0.6, "
      f"best_multi>=0.75): {len(robust)} =====")
print(robust[cols].head(15).round(3).to_string(index=False))
