#!/usr/bin/env python
# =============================================================================
# Shared pipeline for nested-LOOCV multi-kingdom responder classification.
# Feature selection uses train-only MaAsLin2 results (01_nested_maaslin2.R),
# CLR + CV follow 02_final_notebooks/Figure_5AB.py.
# =============================================================================
import os, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

BASE  = "/home/sebin595/project/0.RA_oral_microbiome/for_paper_submission"
PROC  = os.path.join(BASE, "01_Processed_Data")
FOLDS = os.path.join(BASE, "05_multikingdom_nestedFS", "maaslin_folds")

MAT_FILES = {"B": "bacteria_matrix_filtered.tsv",
             "F": "fungi_matrix_filtered.tsv",
             "V": "virus_matrix_filtered.tsv"}
SCENARIOS = {"B": ["B"], "F": ["F"], "V": ["V"],
             "B+F": ["B", "F"], "B+V": ["B", "V"], "B+V+F": ["B", "V", "F"]}

def _load_matrix(k):
    df = pd.read_csv(os.path.join(PROC, MAT_FILES[k]), sep="\t", index_col=0)
    nm = pd.read_csv(os.path.join(FOLDS, f"namemap_{k}.tsv"), sep="\t")
    assert len(nm) == df.shape[1]
    df.columns = nm["maaslin_name"].tolist()
    return df

mats = {k: _load_matrix(k) for k in MAT_FILES}
meta = pd.read_csv(os.path.join(PROC, "cleaned_metadata_RA.tsv"), sep="\t", index_col=0)
y = (meta["Responder_Status"] == "Responder").astype(int)
common = list(meta.index)
for k in mats:
    common = [s for s in common if s in mats[k].index]
y = y.loc[common]
mats = {k: v.loc[common] for k, v in mats.items()}
samples = common
n = len(samples)

# per-fold, per-kingdom train-only MaAsLin2 results (sorted by pval)
maaslin = {k: {} for k in MAT_FILES}
for k in MAT_FILES:
    for i in range(n):
        fr = pd.read_csv(os.path.join(FOLDS, f"{k}_fold{i+1:02d}.tsv"), sep="\t")
        fr = fr[fr["metadata"] == "Responder_Status"].dropna(subset=["pval"]).sort_values("pval")
        if "holdout" in fr.columns and len(fr):
            assert fr["holdout"].iloc[0] == samples[i]
        maaslin[k][i] = fr[["feature", "pval", "coef"]].reset_index(drop=True)

def select_features(k, fold, pthr, topk):
    fr = maaslin[k][fold]
    sel = fr[fr["pval"] < pthr]
    if topk:
        sel = sel.head(topk)
    return sel["feature"].tolist()

def clr_transform(X, pseudocount=1e-6):
    X = X.astype(float).clip(lower=0) + pseudocount
    X_rel = X.div(X.sum(axis=1), axis=0)
    logX = np.log(X_rel)
    clrX = logX.sub(logX.mean(axis=1), axis=0)
    clrX = np.clip(np.nan_to_num(clrX, nan=0.0), -50, 50)
    return pd.DataFrame(clrX, index=X.index, columns=X.columns)

def make_model(name, y_tr, seed=42, rf_kwargs=None):
    npos = int(y_tr.sum()); nneg = int(len(y_tr) - npos)
    spw = (nneg / npos) if npos > 0 else 1.0
    if name == "RandomForest":
        kw = dict(n_estimators=400, class_weight="balanced", n_jobs=1, random_state=seed)
        if rf_kwargs:
            kw.update(rf_kwargs)
        return RandomForestClassifier(**kw)
    if name == "AdaBoost":
        return AdaBoostClassifier(n_estimators=200, learning_rate=0.5,
                                  algorithm="SAMME", random_state=seed)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                             eval_metric="logloss", n_jobs=1, random_state=seed, verbosity=0)
    if name == "LogReg":
        return LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0)
    raise ValueError(name)

def loocv_scores(scenario, model_name, pthr, topk, clr_mode, seed=42, rf_kwargs=None):
    """Return dict{sample: predicted P(responder)} over folds with >=2 features."""
    kings = SCENARIOS[scenario]
    scores = {}
    for i in range(n):
        tr = [s for j, s in enumerate(samples) if j != i]
        te = samples[i]
        feats = {k: select_features(k, i, pthr, topk) for k in kings}
        if sum(len(v) for v in feats.values()) < 2:
            continue
        btr, bte = [], []
        for k in kings:
            fk = [f for f in feats[k] if f in mats[k].columns]
            if not fk:
                continue
            a = mats[k].loc[tr, fk]; b = mats[k].loc[[te], fk]
            if clr_mode == "perkingdom":
                a = clr_transform(a); b = clr_transform(b)
            btr.append(a); bte.append(b)
        if not btr:
            continue
        Xtr = pd.concat(btr, axis=1); Xte = pd.concat(bte, axis=1)
        if clr_mode == "combined":
            Xtr = clr_transform(Xtr); Xte = clr_transform(Xte)
        if Xtr.shape[1] < 2:
            continue
        y_tr = y.loc[tr]
        mdl = make_model(model_name, y_tr, seed=seed, rf_kwargs=rf_kwargs)
        mdl.fit(Xtr.values, y_tr.values)
        scores[te] = float(mdl.predict_proba(Xte.values)[:, 1][0])
    return scores

def auc_from_scores(scores):
    if not scores:
        return np.nan
    s = pd.Series(scores)
    yt = y.loc[s.index]
    if yt.nunique() < 2:
        return np.nan
    return roc_auc_score(yt.values, s.values)

def seed_avg_scores(scenario, model_name, pthr, topk, clr_mode, seeds, rf_kwargs=None):
    """Average P(responder) per sample across seeds -> stable score dict."""
    acc = {}
    for sd in seeds:
        sc = loocv_scores(scenario, model_name, pthr, topk, clr_mode, seed=sd, rf_kwargs=rf_kwargs)
        for k, v in sc.items():
            acc.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}
