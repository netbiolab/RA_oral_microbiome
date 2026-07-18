#!/usr/bin/env python
# =============================================================================
# SHAP beeswarm for the multi-kingdom (B+V+F) Responder-vs-NonResponder model.
#
# SHAP explains a fixed model, so we train ONE final RandomForest on all 35
# samples using the winning multi-kingdom feature set (MaAsLin2 full-cohort
# selection: p<0.15, top-18/kingdom) with per-kingdom CLR and the same
# hyperparameters as the reported model. TreeExplainer (exact for tree
# ensembles) SHAP values are averaged over the same 10 seeds used for the
# ensemble ROC, giving a stable explanation. Class index 1 = Responder.
# =============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
import mk_pipeline as mk

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

FULLD = os.path.join(mk.BASE, "05_multikingdom_nestedFS", "maaslin_full")
FIGD  = os.path.join(mk.BASE, "05_multikingdom_nestedFS", "figures")
RESD  = os.path.join(mk.BASE, "05_multikingdom_nestedFS", "results")

PTHR, TOPK = 0.15, 18
KINGS = ["B", "V", "F"]                     # B+V+F
RF_KWARGS = dict(n_estimators=500, class_weight="balanced", n_jobs=1,
                 max_features="sqrt", min_samples_leaf=2)
SEEDS = list(range(10))

# ---- 1. select final feature set from full-cohort MaAsLin2 -----------------
orig_map = {}
for k in mk.MAT_FILES:
    nm = pd.read_csv(os.path.join(mk.FOLDS, f"namemap_{k}.tsv"), sep="\t")
    orig_map[k] = dict(zip(nm["maaslin_name"], nm["original"]))

sel = {}
for k in KINGS:
    fr = pd.read_csv(os.path.join(FULLD, f"{k}_full.tsv"), sep="\t")
    fr = fr[fr["metadata"] == "Responder_Status"].dropna(subset=["pval"]).sort_values("pval")
    sel[k] = fr[fr["pval"] < PTHR].head(TOPK)["feature"].tolist()
    print(f"{k}: {len(sel[k])} features selected")

# ---- 2. build per-kingdom CLR feature matrix (all 35 samples) --------------
blocks, disp_names = [], []
for k in KINGS:
    fk = [f for f in sel[k] if f in mk.mats[k].columns]
    Xk = mk.clr_transform(mk.mats[k].loc[mk.samples, fk])
    blocks.append(Xk)
    disp_names += [orig_map[k].get(f, f) for f in fk]
X = pd.concat(blocks, axis=1)

# make display names unique (bacteria table has duplicate species names)
seen, uniq = {}, []
for nm_ in disp_names:
    if nm_ in seen:
        seen[nm_] += 1; uniq.append(f"{nm_} ({seen[nm_]})")
    else:
        seen[nm_] = 0; uniq.append(nm_)
X.columns = uniq
y = mk.y.loc[mk.samples]
print(f"Final model matrix: {X.shape[0]} samples x {X.shape[1]} features "
      f"(B={len(sel['B'])}, V={len(sel['V'])}, F={len(sel['F'])})")

# ---- 3. seed-averaged TreeExplainer SHAP -----------------------------------
vals, bases = [], []
for sd in SEEDS:
    rf = RandomForestClassifier(random_state=sd, **RF_KWARGS).fit(X.values, y.values)
    expl = shap.TreeExplainer(rf)(X)          # (n, p, n_classes)
    v = np.asarray(expl.values)
    b = np.asarray(expl.base_values)
    if v.ndim == 2:                            # safety: (n, p) -> add class axis
        v = np.stack([-v, v], axis=-1)
        b = np.stack([-b, b], axis=-1) if b.ndim == 1 else b
    vals.append(v); bases.append(b)
avg_values = np.mean(vals, axis=0)             # (n, p, C)
avg_base   = np.mean(bases, axis=0)            # (n, C)
print("Averaged SHAP values shape:", avg_values.shape)

explanation = shap.Explanation(
    values=avg_values,
    base_values=avg_base,
    data=X.values,
    feature_names=list(X.columns),
)

# ---- 4. beeswarm (user's plotting schema) ----------------------------------
os.makedirs(FIGD, exist_ok=True)
plt.figure()
shap.plots.beeswarm(
    explanation[:, :, 1],
    max_display=20,
    show=False,
    group_remaining_features=False,
    s=7.5, plot_size=(4.5, 4.5)
)
plt.xlabel("SHAP value (impact on model output)", fontsize=7)
plt.ylabel("", fontsize=7)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.title("SHAP beeswarm: multikingdom", fontsize=7)
plt.tight_layout()
out = os.path.join(FIGD, "Figure_5C_260701.pdf")
plt.savefig(out)
plt.close()
print("saved", out)

# ---- 5. export ranked mean|SHAP| table -------------------------------------
mean_abs = np.abs(avg_values[:, :, 1]).mean(axis=0)
mean_signed = avg_values[:, :, 1].mean(axis=0)
tab = pd.DataFrame({"feature": X.columns,
                    "mean_abs_shap": mean_abs,
                    "mean_signed_shap": mean_signed})
tab = tab.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
tab.to_csv(os.path.join(RESD, "shap_feature_importance.tsv"), sep="\t", index=False)
print("\nTop 15 features by mean|SHAP| (class=Responder):")
print(tab.head(15).round(4).to_string(index=False))
