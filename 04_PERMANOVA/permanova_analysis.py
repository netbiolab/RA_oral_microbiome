#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PERMANOVA analysis for the RA oral-microbiome study.

For each feature table (bacteria, fungi, virus, pathway) the community
dissimilarity between samples is quantified with the Bray-Curtis metric on
*relative abundances* (each sample scaled to sum to 1).  Every clinical /
anthropometric variable is then tested one at a time with PERMANOVA
(the McArdle & Anderson 2001 method that vegan::adonis2 implements),
reporting pseudo-F, R2, the permutation p-value and a Benjamini-Hochberg
FDR q-value.

Two scenarios are produced:
  * "Follow-up"  : only the samples that have follow-up responsiveness data
                   (Good_Rx is recorded, i.e. treatment responders / non-
                   responders).  Baseline + follow-up disease-activity
                   variables are tested here.
  * "Full"       : all samples; cohort-wide anthropometric / clinical vars.

Outputs (written next to this script):
  * permanova_results_long.tsv          - one row per (scenario,table,variable)
  * permanova_<scenario>_wide.tsv       - variable x table matrix (R2 / P / Q)
  * permanova_results.xlsx              - all of the above, ready to paste
                                          into Word (Excel -> copy -> paste)
The wide tables are also echoed to the terminal (tab-separated) so they can
be copied straight from the console.

Implemented with numpy/scipy because R/vegan is not loadable in this
environment.  The PERMANOVA F and R2 were cross-checked against
skbio.stats.distance.permanova for categorical predictors (see --selftest).
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
BASE   = "/home/sebin595/project/0.RA_oral_microbiome/for_paper_submission"
ORIG   = os.path.join(BASE, "99_Original_Data")
META   = os.path.join(BASE, "01_Processed_Data", "cleaned_metadata.tsv")
OUTDIR = os.path.join(BASE, "04_PERMANOVA")

N_PERM = 999          # permutations (adonis2 default); min p-value = 1/(N_PERM+1)
SEED   = 42
MIN_N  = 8            # skip a test if fewer than this many samples have the variable

# Feature tables: (label, path, samples_axis, strip_suffix_from_sample_names)
TABLES = [
    ("Bacteria", os.path.join(ORIG, "bacteria_final_normalized_count.tsv"), "rows", None),
    ("Fungi",    os.path.join(ORIG, "fungi_final_normalized_count.tsv"),    "rows", "_bracken"),
    ("Virus",    os.path.join(ORIG, "virus_final_normalized_count.tsv"),    "cols", None),
    ("Pathway",  os.path.join(ORIG, "pathabundance_merged_filtered_final.tsv"), "cols", "_Abundance"),
]

# Metadata variables to test.  (column, display name, type)  type in {"num","cat"}
# Anthropometric + lifestyle + cohort-wide clinical variables (used in BOTH scenarios)
VARS_COMMON = [
    ("Age",            "Age",              "num"),
    ("Sex",            "Sex",              "cat"),
    ("Height",         "Height",           "num"),
    ("Weight",         "Weight",           "num"),
    ("BMI",            "BMI",              "num"),
    ("Alcohol_Use",    "Alcohol use",      "cat"),
    ("Smoking_Status", "Smoking status",   "cat"),
    ("Probiotics_Use", "Probiotics use",   "cat"),
    ("NSAIDs",         "NSAID use",        "cat"),
    ("ESR",            "ESR",              "num"),
    ("CRP",            "CRP",              "num"),
    ("WBC",            "WBC",              "num"),
    ("Plt",            "Platelet",         "num"),
]
# Only meaningful across the whole cohort (RA vs non-RA)
VARS_FULL_ONLY = [
    ("diagnosis",      "Diagnosis (RA vs non-RA)", "cat"),
]
# RA-only baseline + follow-up disease activity & responsiveness (Follow-up scenario)
VARS_FOLLOWUP_ONLY = [
    ("RF",             "Rheumatoid factor",        "num"),
    ("Anti_ccp",       "Anti-CCP",                 "num"),
    ("DAS_28_ESR",     "DAS28-ESR (baseline)",     "num"),
    ("DAS_28_CRP",     "DAS28-CRP (baseline)",     "num"),
    ("CDAI",           "CDAI (baseline)",          "num"),
    ("SDAI",           "SDAI (baseline)",          "num"),
    ("TJC.0.28",       "Tender joint count",       "num"),
    ("SJC.0.28",       "Swollen joint count",      "num"),
    ("PtVAS",          "Patient VAS",              "num"),
    ("PhVAS",          "Physician VAS",            "num"),
    ("DAS_28_ESR.1",   "DAS28-ESR (follow-up)",    "num"),
    ("DAS_28_CRP.1",   "DAS28-CRP (follow-up)",    "num"),
    ("CDAI.1",         "CDAI (follow-up)",         "num"),
    ("SDAI.1",         "SDAI (follow-up)",         "num"),
    ("Delta DAS_28_ESR", "Delta DAS28-ESR",        "num"),
    ("Good_Rx",        "Treatment response (Good_Rx)", "cat"),
]

SCENARIOS = {
    "Full":     VARS_COMMON + VARS_FULL_ONLY,                 # all samples
    "Follow-up": VARS_COMMON + VARS_FOLLOWUP_ONLY,            # Good_Rx recorded
}
SCENARIO_ORDER = ["Follow-up", "Full"]
TABLE_ORDER    = [t[0] for t in TABLES]


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_feature_table(path, samples_axis, strip):
    df = pd.read_csv(path, sep="\t", index_col=0)
    if samples_axis == "cols":          # features in rows -> transpose so samples are rows
        df = df.T
    if strip:
        df.index = df.index.str.replace(strip, "", regex=False)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df.index = df.index.astype(str)
    return df


def load_metadata(path):
    m = pd.read_csv(path, sep="\t", index_col=0,
                    na_values=["nan", "NaN", "NA", "", " ", "None"])
    m = m.set_index("saliva swab")
    m.index = m.index.astype(str)
    return m


def relative_abundance(df):
    """Row-normalise each sample to proportions (sum to 1)."""
    s = df.sum(axis=1)
    s = s.replace(0, np.nan)
    return df.div(s, axis=0).fillna(0.0)


# ----------------------------------------------------------------------------
# PERMANOVA (McArdle & Anderson 2001 / vegan::adonis2 for a single term)
# ----------------------------------------------------------------------------
def gower_center(dist_square):
    """Return the Gower double-centred matrix G from a full distance matrix."""
    n = dist_square.shape[0]
    d2 = dist_square ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    return -0.5 * J @ d2 @ J


def build_design(values, vtype):
    """Design matrix with intercept for one predictor (numeric or categorical)."""
    n = len(values)
    intercept = np.ones((n, 1))
    if vtype == "num":
        x = np.asarray(values, dtype=float).reshape(-1, 1)
        X = np.hstack([intercept, x])
    else:
        dummies = pd.get_dummies(pd.Series(values).astype(str),
                                 drop_first=True).to_numpy(dtype=float)
        X = np.hstack([intercept, dummies]) if dummies.size else intercept
    return X


def permanova(dist_full, values, vtype, n_perm=N_PERM, seed=SEED):
    """
    One-term PERMANOVA on a (sub)set distance matrix.

    dist_full : n x n Bray-Curtis distance matrix (already subset to samples
                that have the variable).
    Returns dict with F, R2, p, df_model, df_res, n.
    """
    n = dist_full.shape[0]
    G = gower_center(dist_full)
    X = build_design(values, vtype)

    rank = np.linalg.matrix_rank(X)
    df_model = rank - 1
    df_res = n - rank
    if df_model < 1 or df_res < 1:
        return None

    H = X @ np.linalg.pinv(X.T @ X) @ X.T
    ss_total = np.trace(G)
    ss_model = float(np.sum(H * G))          # == trace(H @ G), G & H symmetric
    ss_res = ss_total - ss_model
    if ss_total <= 0 or ss_res <= 0:
        return None

    F = (ss_model / df_model) / (ss_res / df_res)
    R2 = ss_model / ss_total

    rng = np.random.default_rng(seed)
    count = 1                                # observed statistic counts as one
    for _ in range(n_perm):
        p = rng.permutation(n)
        Gp = G[np.ix_(p, p)]
        ssm = float(np.sum(H * Gp))          # trace preserved under joint row/col perm
        ssr = ss_total - ssm
        Fp = (ssm / df_model) / (ssr / df_res)
        if Fp >= F - 1e-12:
            count += 1
    pval = count / (n_perm + 1)

    return {"F": F, "R2": R2, "p": pval,
            "df_model": df_model, "df_res": df_res, "n": n}


def bh_fdr(pvals):
    """Benjamini-Hochberg q-values."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return q
    pp = p[idx]
    order = np.argsort(pp)
    ranked = pp[order]
    m = len(pp)
    adj = ranked * m / (np.arange(1, m + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(m)
    out[order] = adj
    q[idx] = out
    return q


def stars(p):
    if p is None or np.isnan(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.1 else "ns"


def group_desc(values, vtype):
    if vtype == "num":
        return "continuous"
    vc = pd.Series(values).astype(str).value_counts()
    return ", ".join(f"{k} (n={v})" for k, v in vc.items())


# ----------------------------------------------------------------------------
# Optional self-test against skbio for a categorical predictor
# ----------------------------------------------------------------------------
def selftest():
    try:
        from skbio import DistanceMatrix
        from skbio.stats.distance import permanova as sk_permanova
    except Exception as e:
        print("skbio not available for self-test:", e); return
    meta = load_metadata(META)
    feat = load_feature_table(*TABLES[0][1:])
    common = [s for s in feat.index if s in meta.index]
    ra = relative_abundance(feat.loc[common])
    D = squareform(pdist(ra.to_numpy(), metric="braycurtis"))
    grp = meta.loc[common, "diagnosis"].astype(str)
    keep = grp.notna().to_numpy()
    Dk = D[np.ix_(keep, keep)]
    mine = permanova(Dk, grp[keep].to_numpy(), "cat")
    dm = DistanceMatrix(Dk, ids=[str(i) for i in range(keep.sum())])
    sk = sk_permanova(dm, grouping=list(grp[keep]), permutations=N_PERM)
    print("SELF-TEST (Bacteria ~ diagnosis)")
    print(f"  my   pseudo-F={mine['F']:.4f}  R2={mine['R2']:.4f}  p={mine['p']:.3f}")
    print(f"  skbio pseudo-F={sk['test statistic']:.4f}            p={sk['p-value']:.3f}")


# ----------------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------------
def run():
    meta = load_metadata(META)
    feats = {label: load_feature_table(path, axis, strip)
             for (label, path, axis, strip) in TABLES}

    # Bray-Curtis distance matrix per table (computed once on all matched samples;
    # subsetting a Bray-Curtis matrix == recomputing it on the subset, so this is
    # exact for every per-variable test).
    dist = {}
    order = {}
    for label, df in feats.items():
        common = [s for s in df.index if s in meta.index]
        ra = relative_abundance(df.loc[common])
        dist[label] = squareform(pdist(ra.to_numpy(), metric="braycurtis"))
        order[label] = list(ra.index)

    long_rows = []
    for scen in SCENARIO_ORDER:
        varlist = SCENARIOS[scen]
        # scenario sample mask
        if scen == "Follow-up":
            scen_samples = set(meta.index[meta["Good_Rx"].notna()])
        else:
            scen_samples = set(meta.index)

        for label in TABLE_ORDER:
            samples = order[label]
            D = dist[label]
            pos = {s: i for i, s in enumerate(samples)}

            recs = []
            for col, disp, vtype in varlist:
                if col not in meta.columns:
                    continue
                series = meta[col].copy()
                if vtype == "num":
                    series = pd.to_numeric(series, errors="coerce")

                # samples in this table AND this scenario AND with the value present
                sel = [s for s in samples if s in scen_samples and pd.notna(series.get(s))]
                if len(sel) < MIN_N:
                    continue
                vals = series.loc[sel]
                # need variation / >=2 groups
                if vtype == "num":
                    if vals.nunique() < 3:
                        continue
                else:
                    vc = vals.astype(str).value_counts()
                    if len(vc) < 2 or vc.min() < 2:
                        continue

                idx = [pos[s] for s in sel]
                Dsub = D[np.ix_(idx, idx)]
                res = permanova(Dsub, vals.to_numpy(), vtype)
                if res is None:
                    continue
                recs.append({
                    "Scenario": scen, "Table": label,
                    "Variable": disp, "Type": "continuous" if vtype == "num" else "categorical",
                    "Groups": group_desc(vals.to_numpy(), vtype),
                    "N": res["n"], "df": res["df_model"],
                    "PseudoF": res["F"], "R2": res["R2"], "P": res["p"],
                })

            # BH-FDR within this scenario x table
            qs = bh_fdr([r["P"] for r in recs])
            for r, q in zip(recs, qs):
                r["Q"] = q
                r["Sig"] = stars(r["P"])
                long_rows.append(r)

    long = pd.DataFrame(long_rows)
    return long


def fmt_p(x):
    if pd.isna(x):
        return ""
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def make_wide(long, scen):
    sub = long[long["Scenario"] == scen]
    # preserve variable order as first encountered
    var_order = list(dict.fromkeys(sub["Variable"]))
    cols = []
    data = {}
    for tbl in TABLE_ORDER:
        for metric in ["R2", "P", "Q"]:
            cols.append(f"{tbl} {metric}")
    for v in var_order:
        row = {}
        for tbl in TABLE_ORDER:
            cell = sub[(sub["Variable"] == v) & (sub["Table"] == tbl)]
            if len(cell):
                c = cell.iloc[0]
                row[f"{tbl} R2"] = f"{c['R2']:.3f}"
                row[f"{tbl} P"]  = fmt_p(c["P"]) + (f" {c['Sig']}" if c["Sig"] not in ("", "ns") else "")
                row[f"{tbl} Q"]  = fmt_p(c["Q"])
            else:
                row[f"{tbl} R2"] = row[f"{tbl} P"] = row[f"{tbl} Q"] = "-"
        data[v] = row
    wide = pd.DataFrame.from_dict(data, orient="index")[cols]
    wide.index.name = "Variable"
    return wide.reset_index()


def main():
    if "--selftest" in sys.argv:
        selftest()
        return

    os.makedirs(OUTDIR, exist_ok=True)
    long = run()

    # ---- long table -------------------------------------------------------
    long_out = long.copy()
    long_out["PseudoF"] = long_out["PseudoF"].map(lambda x: f"{x:.3f}")
    long_out["R2"]      = long_out["R2"].map(lambda x: f"{x:.4f}")
    long_out["P"]       = long_out["P"].map(fmt_p)
    long_out["Q"]       = long_out["Q"].map(fmt_p)
    long_out = long_out[["Scenario", "Table", "Variable", "Type", "Groups",
                         "N", "df", "PseudoF", "R2", "P", "Q", "Sig"]]
    long_path = os.path.join(OUTDIR, "permanova_results_long.tsv")
    long_out.to_csv(long_path, sep="\t", index=False)

    # ---- wide tables + console + xlsx ------------------------------------
    wides = {}
    for scen in SCENARIO_ORDER:
        wide = make_wide(long, scen)
        wides[scen] = wide
        wpath = os.path.join(OUTDIR, f"permanova_{scen.replace('-', '').lower()}_wide.tsv")
        wide.to_csv(wpath, sep="\t", index=False)

    with pd.ExcelWriter(os.path.join(OUTDIR, "permanova_results.xlsx"),
                        engine="openpyxl") as xw:
        for scen in SCENARIO_ORDER:
            wides[scen].to_excel(xw, sheet_name=f"{scen} (wide)"[:31], index=False)
        long_out.to_excel(xw, sheet_name="Long format", index=False)

    # ---- console output (tab separated, ready to copy) -------------------
    n_full = long[long["Scenario"] == "Full"]["N"].max()
    n_fu   = long[long["Scenario"] == "Follow-up"]["N"].max()
    print("=" * 100)
    print("PERMANOVA (Bray-Curtis on relative abundance, {} permutations, seed {})".format(N_PERM, SEED))
    print("R2 = variance explained; P = permutation p-value; Q = Benjamini-Hochberg FDR")
    print("Significance:  *** P<0.001   ** P<0.01   * P<0.05   . P<0.1")
    print("=" * 100)
    for scen in SCENARIO_ORDER:
        nmax = n_fu if scen == "Follow-up" else n_full
        title = ("SCENARIO 1 - Follow-up responsiveness cohort (Good_Rx recorded)"
                 if scen == "Follow-up"
                 else "SCENARIO 2 - Full cohort (all samples)")
        print("\n" + title + f"   [up to n={nmax} samples]")
        print("-" * 100)
        print(wides[scen].to_csv(sep="\t", index=False).rstrip())

    print("\n" + "=" * 100)
    print("Saved:")
    print("  " + long_path)
    for scen in SCENARIO_ORDER:
        print("  " + os.path.join(OUTDIR, f"permanova_{scen.replace('-', '').lower()}_wide.tsv"))
    print("  " + os.path.join(OUTDIR, "permanova_results.xlsx"))
    print("Tip: open the .xlsx in Excel, copy the sheet, and paste into Word as a table.")


if __name__ == "__main__":
    main()
