#!/usr/bin/env python
# =============================================================================
# Consensus feature panel: how often each feature is selected across the 35
# leave-one-out folds at the winning selection setting (pval<0.15, top-18 per
# kingdom). Features selected in most folds are the stable multi-kingdom
# responder biomarkers used by the models.
# =============================================================================
import os
from collections import Counter
import pandas as pd
import mk_pipeline as mk

RESD = os.path.join(mk.BASE, "05_multikingdom_nestedFS", "results")
PTHR, TOPK = 0.15, 18

# original (human-readable) names
name_orig = {}
for k in mk.MAT_FILES:
    nm = pd.read_csv(os.path.join(mk.FOLDS, f"namemap_{k}.tsv"), sep="\t")
    name_orig[k] = dict(zip(nm["maaslin_name"], nm["original"]))

rows = []
for k in mk.MAT_FILES:
    cnt = Counter()
    coefs = {}
    for i in range(mk.n):
        feats = mk.select_features(k, i, PTHR, TOPK)
        fr = mk.maaslin[k][i].set_index("feature")
        for f in feats:
            cnt[f] += 1
            coefs.setdefault(f, []).append(fr.loc[f, "coef"])
    for f, c in cnt.most_common():
        mean_coef = sum(coefs[f]) / len(coefs[f])
        rows.append(dict(kingdom=k, feature_original=name_orig[k].get(f, f),
                         maaslin_name=f, folds_selected=c,
                         frac_folds=round(c / mk.n, 3),
                         mean_coef=round(mean_coef, 3),
                         direction=("up_in_NonResponder" if mean_coef > 0
                                    else "up_in_Responder")))
df = pd.DataFrame(rows)
out = os.path.join(RESD, "consensus_feature_panel.tsv")
df.to_csv(out, sep="\t", index=False)
print(f"Saved -> {out}\n")

for k, kn in [("B", "Bacteria"), ("V", "Virus"), ("F", "Fungi")]:
    sub = df[df.kingdom == k]
    stable = sub[sub.folds_selected >= 30]
    print(f"=== {kn}: {sub.shape[0]} features ever selected | "
          f"{stable.shape[0]} selected in >=30/35 folds ===")
    print(sub.head(12)[["feature_original", "folds_selected",
                        "mean_coef", "direction"]].to_string(index=False))
    print()
