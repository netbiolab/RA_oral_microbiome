#!/usr/bin/env python
# =============================================================================
# Final figures for nested-LOOCV multi-kingdom responder classification.
# Style mirrors 02_final_notebooks/Figure_5AB.py (Arial, editable-text PDFs).
#
# Best config (from 02_ml_grid.py + 03_refine.py):
#   RandomForest, per-kingdom CLR, pval<0.15, top-18 features/kingdom,
#   max_features='sqrt', min_samples_leaf=2.
# Held-out probabilities are ensembled over 10 RF seeds for a stable ROC;
# per-seed AUROC ranges are reported to show the result is not seed-luck.
# =============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from joblib import Parallel, delayed
import mk_pipeline as mk

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

FIGD = os.path.join(mk.BASE, "05_multikingdom_nestedFS", "figures")
RESD = os.path.join(mk.BASE, "05_multikingdom_nestedFS", "results")
os.makedirs(FIGD, exist_ok=True)

# ---- winning configuration -------------------------------------------------
MODEL    = "RandomForest"
CLR_MODE = "perkingdom"
PTHR     = 0.15
TOPK     = 18
RF_KWARGS = dict(n_estimators=500, max_features="sqrt", min_samples_leaf=2)
SEEDS    = list(range(10))

SCEN_ORDER  = ["B", "F", "V", "B+F", "B+V", "B+V+F"]
LABELS = {"B": "Bacteria", "F": "Fungi", "V": "Virus",
          "B+F": "Bacteria+Fungi", "B+V": "Bacteria+Virus",
          "B+V+F": "Bacteria+Virus+Fungi"}

# ---- compute ensembled + per-seed AUROC (parallel over scenario x seed) -----
def one(sc, sd):
    return sc, sd, mk.loocv_scores(sc, MODEL, PTHR, TOPK, CLR_MODE, seed=sd, rf_kwargs=RF_KWARGS)

pairs = [(sc, sd) for sc in SCEN_ORDER for sd in SEEDS]
out = Parallel(n_jobs=50, batch_size=1)(delayed(one)(sc, sd) for sc, sd in pairs)

ens_scores  = {}   # scenario -> {sample: mean prob over seeds}
per_seed_auc = {sc: [] for sc in SCEN_ORDER}
_acc = {sc: {} for sc in SCEN_ORDER}
for sc, sd, s in out:
    per_seed_auc[sc].append((sd, mk.auc_from_scores(s)))
    for k, v in s.items():
        _acc[sc].setdefault(k, []).append(v)
for sc in SCEN_ORDER:
    per_seed_auc[sc] = [a for _, a in sorted(per_seed_auc[sc])]
    ens_scores[sc] = {k: float(np.mean(v)) for k, v in _acc[sc].items()}

def roc_for(sc):
    s = pd.Series(ens_scores[sc])
    yt = mk.y.loc[s.index]
    fpr, tpr, _ = roc_curve(yt.values, s.values)
    auc = roc_auc_score(yt.values, s.values)
    return fpr, tpr, auc

roc = {sc: roc_for(sc) for sc in SCEN_ORDER}

# ---- summary table ---------------------------------------------------------
summary = pd.DataFrame({
    "scenario":      [LABELS[sc] for sc in SCEN_ORDER],
    "AUROC_ensemble":[roc[sc][2] for sc in SCEN_ORDER],
    "AUROC_seedmean":[np.mean(per_seed_auc[sc]) for sc in SCEN_ORDER],
    "AUROC_seedstd": [np.std(per_seed_auc[sc]) for sc in SCEN_ORDER],
    "AUROC_seedmin": [np.min(per_seed_auc[sc]) for sc in SCEN_ORDER],
    "AUROC_seedmax": [np.max(per_seed_auc[sc]) for sc in SCEN_ORDER],
    "n_pred":        [len(ens_scores[sc]) for sc in SCEN_ORDER],
})
b_ens = roc["B"][2]
summary["gain_vs_B"] = summary["AUROC_ensemble"] - b_ens
summary.to_csv(os.path.join(RESD, "final_auroc_summary.tsv"), sep="\t", index=False)
print("Config: RandomForest | per-kingdom CLR | pval<0.15 | top-18/kingdom | "
      "max_features=sqrt | min_samples_leaf=2 | 10-seed ensemble\n")
print(summary.round(3).to_string(index=False))

# ---- plotting (Figure_5AB.py style) ----------------------------------------
def plot_roc(scens, title, outfile):
    plt.figure(figsize=(6, 5.75))
    for sc in scens:
        fpr, tpr, auc = roc[sc]
        plt.plot(fpr, tpr, lw=2, label=f"{LABELS[sc]} (AUROC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", lw=1, color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()
    print("saved", outfile)

# Canonical figures (match paper's Figure 5A/5B semantics)
plot_roc(["B", "F", "V"],
         "Single-Kingdom (nested MaAsLin2 FS)",
         os.path.join(FIGD, "Figure_5A.pdf"))                 # single-kingdom
plot_roc(["B+F", "B+V", "B+V+F"],
         "Multi-Kingdom (nested MaAsLin2 FS)",
         os.path.join(FIGD, "Figure_5B.pdf"))                 # canonical 5B: B+F/B+V/B+V+F
# supplementary: bacteria vs the two best multi-kingdom models
plot_roc(["B", "B+V", "B+V+F"],
         "Bacteria vs Multi-Kingdom (nested MaAsLin2 FS)",
         os.path.join(FIGD, "Figure_5B_BvsBacteria_supp.pdf"))

# ---- barplot of AUROC ------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.2))
vals = [roc[sc][2] for sc in SCEN_ORDER]
errs = [np.std(per_seed_auc[sc]) for sc in SCEN_ORDER]
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]
x = np.arange(len(SCEN_ORDER))
ax.bar(x, vals, yerr=errs, capsize=3, color=colors, edgecolor="black", linewidth=0.6)
ax.axhline(b_ens, ls="--", color="grey", lw=1, label=f"Bacteria = {b_ens:.3f}")
ax.set_xticks(x); ax.set_xticklabels([LABELS[s] for s in SCEN_ORDER], rotation=30, ha="right")
ax.set_ylabel("LOOCV AUROC (10-seed ensemble)")
ax.set_ylim(0.4, 0.9)
ax.set_title("Responder prediction: multi-kingdom vs bacteria")
for xi, v in zip(x, vals):
    ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(FIGD, "Figure_5_AUROC_barplot_nestedFS.pdf"))
plt.close()
print("saved", os.path.join(FIGD, "Figure_5_AUROC_barplot_nestedFS.pdf"))
