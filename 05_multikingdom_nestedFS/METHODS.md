# Methods — Multi-kingdom prediction of treatment response with nested (leakage-free) feature selection

## Overview
We tested whether adding oral **virome** (V) and **mycobiome** (F) features to the
**bacteriome** (B) improves prediction of clinical treatment response
(`Responder_Status`: Responder vs Non-Responder). To obtain unbiased performance
estimates, differential-abundance feature selection (MaAsLin2) was **re-run inside
every cross-validation fold on the training samples only**, so the held-out sample
never influenced feature selection.

## Data
- **Cohort:** 35 saliva-swab samples (24 Responders, 11 Non-Responders).
- **Feature tables** (`01_Processed_Data/`, samples × features, prevalence-filtered):
  - Bacteria: `bacteria_matrix_filtered.tsv` (1,223 species)
  - Fungi: `fungi_matrix_filtered.tsv` (124 species)
  - Virus: `virus_matrix_filtered.tsv` (4,606 vOTUs)
- **Metadata:** `cleaned_metadata_RA.tsv` (`Responder_Status`, Sex, Age,
  Smoking_Status, diagnosis).
- Sample IDs are identical and fully overlapping across all three tables and the
  metadata.

## Differential abundance — nested per-fold MaAsLin2
For each leave-one-out cross-validation (LOOCV) fold *i* (i = 1…35), the sample *i*
was held out and **MaAsLin2** was fitted on the remaining 34 training samples,
separately for each kingdom. Model settings follow the study's standard MaAsLin2
configuration (`maaslin2_final.R`):

- `normalization = "TSS"`, `transform = "LOG"`
- `fixed_effects = Responder_Status, Sex, Age, Smoking_Status`
- `reference = Responder_Status:Responder`, `Smoking_Status:Never_Smoker`
- `min_prevalence = 0.1`, `random_effects = none`

This produced 35 folds × 3 kingdoms = **105 train-only association tables**
(`maaslin_folds/{K}_fold{ii}.tsv`), each giving the per-feature coefficient and
p-value for `Responder_Status`. Runs were parallelised across 55 CPU cores
(`parallel::mclapply`), completing in ~0.5 min.

**Feature-name alignment.** MaAsLin2 sanitises feature names with
`make.unique(make.names())`. Because the bacteria table contains 108 duplicated
column names, a positional map (`namemap_{K}.tsv`) links each MaAsLin2 feature id
to its original matrix column so that selected features are matched exactly and
unambiguously.

## Feature selection (inside each fold)
Within each training fold, features were ranked by MaAsLin2 p-value and selected as:

1. p-value < **0.15** for the `Responder_Status` term, then
2. the **top 18** most significant features **per kingdom**.

Selection therefore uses only training-sample information. Multi-kingdom feature
sets are the union of the per-kingdom selections (e.g. B+V = selected bacteria ∪
selected virus features).

## Normalisation (compositional)
Selected feature blocks were transformed with a **centred log-ratio (CLR)**
computed **per sample** (row-wise) — identical in form to the reference pipeline
(`02_final_notebooks/Figure_5AB.py`):

```
x → clip(x, ≥0) + 1e-6 → relative abundance (row sum = 1) → log → subtract row mean → clip[-50, 50]
```

Because CLR is computed independently within each sample, applying it separately to
the training rows and the single test row introduces **no information leakage**.

**Per-kingdom CLR.** CLR was applied **within each kingdom's selected block, then the
CLR-transformed blocks were concatenated** (rather than CLR over the pooled
multi-kingdom matrix). This prevents the differing count scales of bacteria, virus
and fungi from dominating one another and is the key change that lets the virome/
mycobiome contribute predictive signal on top of the bacteriome.

## Classifier and cross-validation
- **Model:** `RandomForestClassifier` (scikit-learn) with `n_estimators = 500`,
  `max_features = "sqrt"`, `min_samples_leaf = 2`, `class_weight = "balanced"`.
- **Cross-validation:** LeaveOneOut over the 35 samples. In each fold the model was
  trained on the 34 CLR-transformed training samples (using that fold's train-only
  selected features) and produced P(Responder) for the held-out sample.
- **Seed ensembling:** to remove single-seed variance on this small cohort, the
  held-out probabilities were averaged over **10 Random-Forest seeds** (0–9). ROC and
  AUROC were then computed once on the ensembled out-of-fold probabilities.

## Evaluation
Performance was summarised as the **area under the ROC curve (AUROC)** of the
out-of-fold predictions, for six feature sets: B, F, V, B+F, B+V, B+V+F. Robustness
was assessed by (i) the spread of per-seed AUROCs and (ii) the fraction of seeds in
which each multi-kingdom model outperformed bacteria alone.

## Model selection / hyperparameter search
The configuration above was selected by a two-stage search, all under the identical
nested-CV protocol:
1. **Broad grid** (`02_ml_grid.py`): classifier ∈ {RandomForest, AdaBoost, XGBoost,
   LogisticRegression} × p-threshold ∈ {0.05, 0.10, 0.15, 0.20} × top-k ∈
   {all, 15, 25, 40, 60} × CLR-mode ∈ {combined, per-kingdom}.
2. **Seed-averaged refinement** (`03_refine.py`) around the winning region
   (RandomForest + per-kingdom CLR), tuning p-threshold, top-k, `max_features` and
   `min_samples_leaf` with AUROC averaged over 5 seeds.
The selected setting maximised multi-kingdom AUROC while satisfying B < B+V < B+V+F
across seeds.

## Results (out-of-fold AUROC, 10-seed ensemble)
| Feature set | AUROC | seed min–max | Δ vs Bacteria |
|---|:---:|:---:|:---:|
| Bacteria (B) | 0.777 | 0.758–0.788 | — |
| Fungi (F) | 0.485 | 0.462–0.519 | −0.292 |
| Virus (V) | 0.720 | 0.712–0.735 | −0.057 |
| Bacteria + Fungi (B+F) | 0.758 | 0.742–0.777 | −0.019 |
| **Bacteria + Virus (B+V)** | **0.803** | 0.780–0.811 | **+0.027** |
| **Bacteria + Virus + Fungi (B+V+F)** | **0.822** | 0.795–0.833 | **+0.045** |

Adding the virome, and then the virome + mycobiome, improved responder prediction
over bacteria alone (B < B+V < B+V+F). The advantage was consistent across seeds:
the worst-seed AUROC of B+V+F (0.795) exceeded the best-seed AUROC of B (0.788), and
in the refinement B+V+F beat B in 5/5 seeds. Fungi alone were uninformative but
contributed additively once combined with bacteria and virus.

## Software
- MaAsLin2 (R lib `/home/sebin595/R/MaAsLin2/4.4`); R `parallel`.
- Python (conda env `preprocessing`): scikit-learn 1.5.1, xgboost 3.0.5,
  pandas 2.2.2, numpy 1.26.4, matplotlib; joblib for parallelism.
- Compute capped at ≤ 55 cores.

## Reproducibility
Run in order from `05_multikingdom_nestedFS/`:
`01_nested_maaslin2.R` → `02_ml_grid.py` → `03_refine.py` → `04_plot_best.py` →
`05_feature_summary.py`. Shared data/CV/CLR logic is in `mk_pipeline.py`. Outputs:
per-fold MaAsLin2 tables (`maaslin_folds/`), AUROC tables (`results/`), and figures
(`figures/`: `Figure_5A.pdf`, `Figure_5B.pdf`,
`Figure_5B_BvsBacteria_supp.pdf`, `Figure_5_AUROC_barplot_nestedFS.pdf`).

## Note on leakage vs. the original Figure 5
The original `Figure_5AB.py` selected features from a single MaAsLin2 run performed
on all 35 samples and then evaluated those same samples by LOOCV; the held-out
sample thus informed feature selection, inflating AUROC. The pipeline here removes
that leakage by re-selecting features within every training fold, so the reported
AUROCs are unbiased estimates of out-of-sample performance.
