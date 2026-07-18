#!/usr/bin/env python
# =============================================================================
# Nested-LOOCV multi-kingdom responder classification.
#
# Feature selection is done INSIDE each LOOCV fold using MaAsLin2 results that
# were computed on the 34 training samples only (see 01_nested_maaslin2.R), so
# the held-out sample never influences which features are chosen -> no leakage.
#
# Modelling / CV / CLR / plotting follow 02_final_notebooks/Figure_5AB.py.
# We grid-search model x p-threshold x top-k x CLR-mode to find settings where
# adding virus (B+V) or virus+fungi (B+V+F) beats bacteria alone (B).
# =============================================================================
import os, warnings, itertools
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

BASE  = "/home/sebin595/project/0.RA_oral_microbiome/for_paper_submission"
PROC  = os.path.join(BASE, "01_Processed_Data")
FOLDS = os.path.join(BASE, "05_multikingdom_nestedFS", "maaslin_folds")
RESD  = os.path.join(BASE, "05_multikingdom_nestedFS", "results")
os.makedirs(RESD, exist_ok=True)
N_JOBS = 50   # <= 55 core cap, small headroom

# --------------------------------------------------------------------------- #
# 1. Load matrices (samples x features), rename columns to MaAsLin2 names      #
#    positionally so that MaAsLin2 feature ids match matrix columns exactly.   #
# --------------------------------------------------------------------------- #
MAT_FILES = {"B": "bacteria_matrix_filtered.tsv",
             "F": "fungi_matrix_filtered.tsv",
             "V": "virus_matrix_filtered.tsv"}

def load_matrix(k):
    df = pd.read_csv(os.path.join(PROC, MAT_FILES[k]), sep="\t",
                     index_col=0, header=0)
    nm = pd.read_csv(os.path.join(FOLDS, f"namemap_{k}.tsv"), sep="\t")
    assert len(nm) == df.shape[1], f"{k}: colcount mismatch {len(nm)} vs {df.shape[1]}"
    df.columns = nm["maaslin_name"].tolist()   # positional alignment
    return df

mats = {k: load_matrix(k) for k in MAT_FILES}

meta = pd.read_csv(os.path.join(PROC, "cleaned_metadata_RA.tsv"),
                   sep="\t", index_col=0)
y = (meta["Responder_Status"] == "Responder").astype(int)

# common samples, consistent order
common = meta.index
for k in mats:
    common = common.intersection(mats[k].index)
common = list(common)
y = y.loc[common]
mats = {k: v.loc[common] for k, v in mats.items()}
samples = list(common)
n = len(samples)
print(f"Samples: {n} | responders={int(y.sum())} non={int((1-y).sum())}")
print("Feature counts:", {k: v.shape[1] for k, v in mats.items()})

# --------------------------------------------------------------------------- #
# 2. Per-fold, per-kingdom MaAsLin2 results (train-only) -> sorted by pval     #
#    fold index i (0..n-1) corresponds to holding out samples[i].             #
#    MaAsLin2 fold files are 1-indexed matching R sample order (== samples).   #
# --------------------------------------------------------------------------- #
maaslin = {k: {} for k in MAT_FILES}
for k in MAT_FILES:
    for i in range(n):
        fr = pd.read_csv(os.path.join(FOLDS, f"{k}_fold{i+1:02d}.tsv"), sep="\t")
        fr = fr[fr["metadata"] == "Responder_Status"].copy()
        fr = fr.dropna(subset=["pval"]).sort_values("pval")
        # sanity: the file's held-out sample must equal samples[i]
        ho = fr["holdout"].iloc[0] if "holdout" in fr.columns and len(fr) else None
        if ho is not None:
            assert ho == samples[i], f"{k} fold{i+1}: holdout {ho} != {samples[i]}"
        maaslin[k][i] = fr[["feature", "pval", "coef"]].reset_index(drop=True)

def select_features(k, fold, pthr, topk):
    fr = maaslin[k][fold]
    sel = fr[fr["pval"] < pthr]
    if topk is not None:
        sel = sel.head(topk)
    return sel["feature"].tolist()

# --------------------------------------------------------------------------- #
# 3. CLR transform (per-sample, row-wise) — identical to Figure_5AB.py         #
# --------------------------------------------------------------------------- #
def clr_transform(X, pseudocount=1e-6):
    X = X.astype(float).clip(lower=0) + pseudocount
    X_rel = X.div(X.sum(axis=1), axis=0)
    logX = np.log(X_rel)
    clrX = logX.sub(logX.mean(axis=1), axis=0)
    clrX = np.clip(np.nan_to_num(clrX, nan=0.0), -50, 50)
    return pd.DataFrame(clrX, index=X.index, columns=X.columns)

SCENARIOS = {"B": ["B"], "F": ["F"], "V": ["V"],
             "B+F": ["B", "F"], "B+V": ["B", "V"], "B+V+F": ["B", "V", "F"]}

def make_model(name, y_tr):
    npos = int(y_tr.sum()); nneg = int(len(y_tr) - npos)
    spw = (nneg / npos) if npos > 0 else 1.0
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                      n_jobs=1, random_state=42)
    if name == "AdaBoost":
        return AdaBoostClassifier(n_estimators=200, learning_rate=0.5,
                                  algorithm="SAMME", random_state=42)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             scale_pos_weight=spw, eval_metric="logloss",
                             n_jobs=1, random_state=42, verbosity=0)
    if name == "LogReg":
        return LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0)
    raise ValueError(name)

def loocv_auc(scenario, model_name, pthr, topk, clr_mode):
    kings = SCENARIOS[scenario]
    y_true, y_score = [], []
    n_pred = 0
    for i in range(n):
        tr = [s for j, s in enumerate(samples) if j != i]
        te = [samples[i]]
        # per-kingdom train-only feature selection
        feats = {k: select_features(k, i, pthr, topk) for k in kings}
        if sum(len(v) for v in feats.values()) < 2:
            continue
        blocks_tr, blocks_te = [], []
        for k in kings:
            fk = [f for f in feats[k] if f in mats[k].columns]
            if len(fk) == 0:
                continue
            Xtr_k = mats[k].loc[tr, fk]
            Xte_k = mats[k].loc[te, fk]
            if clr_mode == "perkingdom":
                Xtr_k = clr_transform(Xtr_k)
                Xte_k = clr_transform(Xte_k)
            blocks_tr.append(Xtr_k); blocks_te.append(Xte_k)
        if not blocks_tr:
            continue
        Xtr = pd.concat(blocks_tr, axis=1)
        Xte = pd.concat(blocks_te, axis=1)
        if clr_mode == "combined":
            Xtr = clr_transform(Xtr)
            Xte = clr_transform(Xte)
        if Xtr.shape[1] < 2:
            continue
        y_tr = y.loc[tr]
        mdl = make_model(model_name, y_tr)
        mdl.fit(Xtr.values, y_tr.values)
        p = mdl.predict_proba(Xte.values)[:, 1][0]
        y_true.append(int(y.loc[te[0]])); y_score.append(float(p)); n_pred += 1
    if len(set(y_true)) < 2:
        return dict(auc=np.nan, n_pred=n_pred, y_true=y_true, y_score=y_score)
    return dict(auc=roc_auc_score(y_true, y_score), n_pred=n_pred,
                y_true=y_true, y_score=y_score)

# --------------------------------------------------------------------------- #
# 4. Grid search                                                              #
# --------------------------------------------------------------------------- #
MODELS   = ["RandomForest", "AdaBoost", "XGBoost", "LogReg"]
PTHRS    = [0.05, 0.10, 0.15, 0.20]
TOPKS    = [None, 15, 25, 40, 60]
CLRMODES = ["combined", "perkingdom"]

combos = list(itertools.product(MODELS, PTHRS, TOPKS, CLRMODES))
print(f"Grid: {len(combos)} (model,thr,topk,clr) combos x 6 scenarios "
      f"= {len(combos)*6} LOOCV runs", flush=True)

def run_combo(model_name, pthr, topk, clr_mode):
    row = dict(model=model_name, pthr=pthr, topk=(topk if topk else 0), clr=clr_mode)
    for sc in SCENARIOS:
        try:
            r = loocv_auc(sc, model_name, pthr, topk, clr_mode)
            row[sc] = r["auc"]; row[f"{sc}_npred"] = r["n_pred"]
        except Exception as e:
            row[sc] = np.nan; row[f"{sc}_npred"] = 0; row[f"{sc}_err"] = str(e)[:80]
    return row

results = Parallel(n_jobs=N_JOBS, verbose=10, batch_size=1)(
    delayed(run_combo)(*c) for c in combos)
res = pd.DataFrame(results)

# margins vs bacteria-only
res["BV_gain"]  = res["B+V"]   - res["B"]
res["BVF_gain"] = res["B+V+F"] - res["B"]
res["best_multi"] = res[["B+V", "B+V+F"]].max(axis=1)
res["multi_gain"] = res["best_multi"] - res["B"]

res = res.sort_values(["multi_gain", "best_multi"], ascending=False).reset_index(drop=True)
outcsv = os.path.join(RESD, "grid_search_auroc.tsv")
res.to_csv(outcsv, sep="\t", index=False)
print(f"\nSaved -> {outcsv}")

pd.set_option("display.width", 200, "display.max_columns", 40)
show = ["model", "pthr", "topk", "clr", "B", "F", "V", "B+F", "B+V", "B+V+F",
        "BV_gain", "BVF_gain", "multi_gain"]
print("\n===== TOP 20 by multi-kingdom gain over Bacteria =====")
print(res[show].head(20).round(3).to_string(index=False))

# configs where BOTH B+V and B+V+F beat B, decent absolute AUROC
good = res[(res["B+V"] > res["B"]) & (res["B+V+F"] > res["B"]) &
           (res["best_multi"] >= 0.75)]
print(f"\nConfigs with B+V>B AND B+V+F>B AND best_multi>=0.75: {len(good)}")
print(good[show].head(15).round(3).to_string(index=False))
