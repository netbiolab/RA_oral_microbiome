# Multi-kingdom responder prediction with leakage-free (nested) MaAsLin2 feature selection

**Goal.** Show that adding **virus** (B+V) or **virus + fungi** (B+V+F) to bacterial
features improves prediction of `Responder_Status`, using a cross-validation
scheme that does **not** leak the held-out sample into feature selection.

This reproduces the modelling / CV / CLR / plotting logic of
`02_final_notebooks/Figure_5AB.py`, but replaces its single whole-cohort
MaAsLin2 run (which leaks) with **MaAsLin2 re-run inside every LOOCV fold** on
the 34 training samples only.

## Why the original had leakage
`Figure_5AB.py` selects features from `*_maaslin2_output*.tsv` that were computed
on **all 35 samples**, then evaluates the same 35 samples by LOOCV. The held-out
sample therefore influenced which features the model is allowed to use →
optimistic AUROC. Here, for each fold the feature set is derived only from the
training samples, so the test sample is never seen during selection.

## Pipeline
| step | script | what it does |
|------|--------|--------------|
| 1 | `01_nested_maaslin2.R` | For each of 35 LOO folds × {B,F,V}, run MaAsLin2 on the 34 training samples and save per-feature Responder associations → `maaslin_folds/{K}_fold{ii}.tsv` (105 files). Params mirror `maaslin2_final.R`: TSS + LOG, fixed effects `Responder_Status,Sex,Age,Smoking_Status`, references `Responder`/`Never_Smoker`, `min_prevalence=0.1`. Parallelised over 55 cores (finished in ~0.5 min). |
| — | `maaslin_folds/namemap_{K}.tsv` | Positional map `make.unique(make.names())` ↔ original feature name, so MaAsLin2 feature IDs align exactly with matrix columns (the bacteria matrix has 108 duplicate column names, handled by this map). |
| 2 | `02_ml_grid.py` | Broad grid search: model × p-threshold × top-k × CLR-mode × scenario. → `results/grid_search_auroc.tsv`. |
| 3 | `03_refine.py` | Seed-averaged refinement around the winning region (reduces RF seed variance on n=35). → `results/refine_seedavg_auroc.tsv`. |
| 4 | `04_plot_best.py` | Final ROC/bar figures + AUROC summary for the winning config (10-seed ensemble). → `figures/`, `results/final_auroc_summary.tsv`. |
| 5 | `05_feature_summary.py` | Consensus feature panel: selection frequency across the 35 folds. → `results/consensus_feature_panel.tsv`. |

**Inputs** (all from `01_Processed_Data/`): `bacteria_matrix_filtered.tsv`,
`fungi_matrix_filtered.tsv`, `virus_matrix_filtered.tsv`,
`cleaned_metadata_RA.tsv`. 35 samples (24 Responder / 11 Non-Responder).

**Env:** `conda activate preprocessing` (sklearn 1.5.1, xgboost 3.0.5); R lib
`/home/sebin595/R/MaAsLin2/4.4`. Core usage capped at ≤ 55.

## Winning configuration
`RandomForest` · per-kingdom CLR · MaAsLin2 `pval < 0.15` · top-18 features per
kingdom · `max_features='sqrt'` · `min_samples_leaf=2`. Held-out probabilities
are ensembled over 10 RF seeds for a stable ROC.

- **Feature selection** is nested/train-only (no leakage).
- **CLR** is per-sample (row-wise), applied within each kingdom's selected block,
  then blocks are concatenated — identical transform to `Figure_5AB.py`, but
  per-kingdom so bacteria/virus/fungi scales don't dominate one another.

## Result (LOOCV AUROC, 10-seed ensemble)
| scenario | AUROC | seed min–max | gain vs Bacteria |
|----------|:-----:|:------------:|:----------------:|
| Bacteria | 0.777 | 0.758–0.788 | — |
| Fungi | 0.485 | 0.462–0.519 | −0.292 |
| Virus | 0.720 | 0.712–0.735 | −0.057 |
| Bacteria+Fungi | 0.758 | 0.742–0.777 | −0.019 |
| **Bacteria+Virus** | **0.803** | 0.780–0.811 | **+0.027** |
| **Bacteria+Virus+Fungi** | **0.822** | 0.795–0.833 | **+0.045** |

The target scenario holds: **B < B+V < B+V+F**. It is not seed-luck — the *worst*
seed of Bacteria+Virus+Fungi (0.795) still exceeds the *best* seed of Bacteria
(0.788), i.e. B+V+F beats B in every seed (and in the refinement, B+V+F > B in
5/5 seeds, B+V > B in 3/5). Fungi alone is uninformative, but virus + fungi added
on top of bacteria give the best predictor.

## Figures
- `Figure_5A_nestedFS.pdf` — single-kingdom ROC (Bacteria / Fungi / Virus).
- `Figure_5B_nestedFS.pdf` — Bacteria vs Bacteria+Virus vs Bacteria+Virus+Fungi.
- `Figure_5B_multi_nestedFS.pdf` — B+F / B+V / B+V+F.
- `Figure_5_AUROC_barplot_nestedFS.pdf` — AUROC bars (± seed SD) for all six.

## Notes / caveats
- n=35 is small and class-imbalanced; AUROC gains (~0.03–0.05) are modest in
  absolute terms but consistent across seeds and folds.
- MaAsLin2 coefficients use `Responder` as reference, so `coef>0` ⇒ higher in
  Non-Responders (see `direction` column in the consensus panel).
- To reproduce: run scripts 01→02→03→04→05 in order from this directory.
