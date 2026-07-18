#!/usr/bin/env Rscript
# =============================================================================
# Full-cohort (all 35 samples) MaAsLin2 per kingdom, used ONLY to define the
# feature set of the final interpretation model (SHAP). Same parameters as the
# nested per-fold runs (01_nested_maaslin2.R / maaslin2_final.R).
# =============================================================================
suppressMessages({
  .libPaths("/home/sebin595/R/MaAsLin2/4.4")
  library(Maaslin2)
})

base   <- "/home/sebin595/project/0.RA_oral_microbiome/for_paper_submission"
proc   <- file.path(base, "01_Processed_Data")
outdir <- file.path(base, "05_multikingdom_nestedFS", "maaslin_full")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

metadata <- read.csv(file.path(proc, "cleaned_metadata_RA.tsv"),
                     sep = "\t", row.names = 1, check.names = FALSE)
metadata$Sex            <- factor(metadata$Sex)
metadata$Smoking_Status <- factor(metadata$Smoking_Status)
imd <- metadata[, c("Sex", "Age", "Smoking_Status", "Responder_Status")]

read_mat <- function(f) read.csv(file.path(proc, f), sep = "\t",
                                 row.names = 1, check.names = FALSE)
mats <- list(B = read_mat("bacteria_matrix_filtered.tsv"),
             F = read_mat("fungi_matrix_filtered.tsv"),
             V = read_mat("virus_matrix_filtered.tsv"))
samples <- rownames(imd)

for (k in names(mats)) {
  tmp <- file.path(tempdir(), paste0("full_", k))
  invisible(capture.output(suppressMessages(
    fit <- Maaslin2(
      input_data     = mats[[k]][samples, , drop = FALSE],
      input_metadata = imd,
      output         = tmp,
      fixed_effects  = c("Responder_Status", "Sex", "Age", "Smoking_Status"),
      normalization  = "TSS", transform = "LOG",
      reference      = c("Responder_Status:Responder", "Smoking_Status:Never_Smoker"),
      min_prevalence = 0.1, plot_heatmap = FALSE, plot_scatter = FALSE, cores = 1)
  )))
  ar <- fit$results
  ar <- ar[ar$metadata == "Responder_Status",
           c("feature", "metadata", "value", "coef", "stderr", "pval", "qval")]
  write.table(ar, file.path(outdir, paste0(k, "_full.tsv")),
              sep = "\t", quote = FALSE, row.names = FALSE)
  cat(sprintf("%s: %d associations | p<0.15: %d\n",
              k, nrow(ar), sum(ar$pval < 0.15, na.rm = TRUE)))
}
cat("Full-cohort MaAsLin2 done.\n")
