# The Oral Microbiome in Rheumatoid Arthritis

This repository contains the data and analysis code accompanying our study of the
**salivary (oral) microbiome in rheumatoid arthritis (RA)**. Using shotgun
metagenomic profiling, we characterize multi-kingdom microbial communities
(bacteria, fungi, and viruses) together with their functional potential, and we
ask two questions:

1. **How does the oral microbiome differ between RA patients and non-RA controls?**
2. **Does the baseline oral microbiome distinguish treatment responders (R) from non-responders (NR)?**

The repository is organized so that every published figure can be reproduced from
the processed data tables and the analysis notebooks.

---

## Study overview

- **Sample type:** saliva swabs
- **Cohort:** 77 participants — 43 RA and 34 non-RA controls
- **RA treatment response:** within the RA group, patients are classified as
  responders (`Y`) or non-responders (`N`) based on the change in disease
  activity (Δ DAS-28-ESR)
- **Microbial kingdoms profiled:** bacteria, fungi, viruses
- **Functional profiling:** MetaCyc pathways and reactions (HUMAnN-style output)

---

## Repository structure

```
.
├── 00_Data_Preprocessing/      # Notebooks: raw → clean metadata, sample filtering
├── 01_Processed_Data/          # Cleaned metadata + filtered/relative-abundance matrices
├── 02_final_notebooks/         # Figure notebooks (Figs. 1-5)
├── 03_Output_Figures/          # Final figure panels (PDF), written out by the notebooks
├── 04_PERMANOVA/               # Beta-diversity PERMANOVA (Bray-Curtis) across all kingdoms
├── 05_multikingdom_nestedFS/   # Leakage-free nested-CV feature selection for the Fig. 5 responder-prediction model
└── 99_Original_Data/           # Raw normalized count tables, taxonomy, MaAsLin2 output
```

### `00_Data_Preprocessing/`
| File | Purpose |
|------|---------|
| `00_raw_to_clean_metadata.ipynb` | Clean and harmonize the clinical metadata |
| `01_filter_out_samples_from_matrices.ipynb` | Restrict count matrices to QC-passing samples |

### `01_Processed_Data/`
Cleaned metadata (`cleaned_metadata.tsv`, and the RA-only, responder-annotated
subset `cleaned_metadata_RA.tsv`) plus, for each kingdom and for the MetaCyc
pathways, the filtered count matrices and their relative-abundance
(`_rel_abundance`) versions.

### `02_final_notebooks/`
| File | Figure | Analysis |
|------|--------|----------|
| `updated_Figure_1.ipynb` | Fig. 1 | Bacteriome overview: alpha/beta diversity, ordination, and MaAsLin2-based associations |
| `updated_Figure_2_OAvsRA.ipynb` | Fig. 2 | Diversity comparison across non-RA (`OA`), Responder, and Non-Responder groups + MaAsLin2 (`maaslin2_OAvsRA`, `maaslin2_OA_vs_Responder`, `maaslin2_OA_vs_NonResponder`) |
| `updated_Figure_3.ipynb` | Fig. 3 | MetaCyc pathway differential abundance (Non-Responder vs OA, Responder vs Non-Responder) and pathway correlation analysis |
| `updated_Figure_4.ipynb` | Fig. 4 | Mycobiome & virome diversity/differential abundance, plus oral host–virus (HROV) profiling |

`OA` here labels the non-RA comparison group (`diagnosis == "Non_RA"` in
`cleaned_metadata.tsv`) as used in this repo's file/notebook naming.

### `03_Output_Figures/`
Destination for the PDF panels produced by the notebooks in
`02_final_notebooks/`. Empty until the notebooks are (re-)run.

### `04_PERMANOVA/`
Bray–Curtis PERMANOVA (`permanova_analysis.py`) testing every clinical /
anthropometric variable, one at a time, against each feature table (bacteria,
fungi, virus, MetaCyc pathways). Implemented in numpy/scipy (equivalent to
`vegan::adonis2`) rather than R, since R/vegan was not available in this
environment; F, R², and categorical p-values were cross-checked against
`skbio.stats.distance.permanova`. Two scenarios are run — **Follow-up**
(samples with treatment-response data) and **Full** (whole cohort) — with
results written to `permanova_results_long.tsv`, the per-scenario
`permanova_*_wide.tsv` matrices, and a combined `permanova_results.xlsx`.

### `05_multikingdom_nestedFS/`
Follow-up to Fig. 5 that removes cross-validation leakage from the responder
(R vs NR) prediction model. MaAsLin2 differential-abundance feature selection
is re-run **inside every LOOCV fold** (35 folds × bacteria/fungi/virus) so the
held-out sample never informs feature selection, then a Random Forest
(per-kingdom CLR features) is evaluated out-of-fold to test whether adding
virome (B+V) or virome+mycobiome (B+V+F) features improves AUROC over
bacteriome alone. See `05_multikingdom_nestedFS/README.md` and `METHODS.md`
for full details; in brief:

| step | script | output |
|------|--------|--------|
| 1 | `01_nested_maaslin2.R` | Per-fold MaAsLin2 associations → `maaslin_folds/` |
| 2 | `02_ml_grid.py` | Broad model/hyperparameter grid search → `results/grid_search_auroc.tsv` |
| 3 | `03_refine.py` | Seed-averaged refinement around the best region → `results/refine_seedavg_auroc.tsv` |
| 4 | `04_plot_best.py` | Final ROC/AUROC figures for the winning config → `figures/`, `results/final_auroc_summary.tsv` |
| 5 | `05_feature_summary.py` | Consensus feature panel across folds → `results/consensus_feature_panel.tsv` |
| 6 | `06_maaslin2_full.R` | Full-cohort (all 35 samples) MaAsLin2, used only to fix the feature set for SHAP → `maaslin_full/` |
| 7 | `07_shap_beeswarm.py` | SHAP beeswarm for the final multi-kingdom model → `results/shap_feature_importance.tsv` |

Best result: LOOCV AUROC 0.777 (Bacteria) → 0.803 (+Virus) → 0.822
(+Virus+Fungi), 10-seed ensembled.

### `99_Original_Data/`
Source tables produced upstream of this repository: normalized count matrices per
kingdom (`*_final_normalized_count.tsv`), bacterial taxonomy, host–virus profiles
(`HROV_HOST_PROFILE_vOTU_LEVEL.tsv`), pathway/reaction matrices, and the
per-feature MaAsLin2 differential-abundance results (including the
`maaslin2_OAvsRA/`, `maaslin2_OA_vs_Responder/`, and
`maaslin2_OA_vs_NonResponder/` run folders used by Fig. 2).

---

## Methods at a glance

- **Diversity** — alpha diversity (Chao1, Shannon, Simpson) and beta diversity
  with PERMANOVA on Bray–Curtis distances (`04_PERMANOVA/`).
- **Differential abundance** — MaAsLin2 (outputs provided in `99_Original_Data/`;
  re-run per-fold in `05_multikingdom_nestedFS/` for leakage-free feature
  selection).
- **Functional analysis** — MetaCyc pathway/reaction abundance with
  feature–feature correlation analysis (Fig. 3).
- **Machine learning** — Random Forest / XGBoost classifiers with SHAP-based
  feature attribution for responder-vs-non-responder prediction, including a
  leakage-free nested-CV variant (`05_multikingdom_nestedFS/`).

---

## Software environment

**Python** (figure notebooks + `04_PERMANOVA/`, `05_multikingdom_nestedFS/`):
`pandas`, `numpy`, `scipy`, `statsmodels`, `scikit-learn`, `xgboost`, `shap`,
`scikit-bio`, `matplotlib`, `seaborn`, `adjustText`

**R** (differential abundance in `05_multikingdom_nestedFS/`):
`Maaslin2`, `parallel`

**External tools:** MaAsLin2 (differential abundance)

A minimal Python setup:

```bash
pip install pandas numpy scipy statsmodels scikit-learn xgboost shap scikit-bio matplotlib seaborn adjustText
```

---

## Reproducing the figures

1. (Optional) Re-run preprocessing in `00_Data_Preprocessing/` to regenerate the
   tables in `01_Processed_Data/` from `99_Original_Data/`.
2. Unzip the HROV metadata and host information file in `99_Original_Data/`.
3. Open the relevant notebook in `02_final_notebooks/` and run all cells; PDFs
   are written to `03_Output_Figures/`.
4. Run `python 04_PERMANOVA/permanova_analysis.py` for the beta-diversity /
   PERMANOVA panels.
5. For the leakage-free Fig. 5 follow-up, run the scripts in
   `05_multikingdom_nestedFS/` in order (`01` → `07`) from inside that
   directory.

Scripts assume they are run from the repository root unless noted otherwise
(paths are relative, e.g. `./99_Original_Data/...`).

---

## Contact

For questions about the analysis or data, please open an issue or contact the
corresponding author listed in the publication.
