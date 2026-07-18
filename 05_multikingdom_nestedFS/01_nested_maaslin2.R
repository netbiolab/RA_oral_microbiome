#!/usr/bin/env Rscript
# =============================================================================
# Nested (leave-one-out) MaAsLin2 feature selection.
#
# For every LOOCV fold (hold out 1 of 35 samples) and every kingdom (B/F/V),
# run MaAsLin2 on ONLY the 34 training samples and save the per-feature
# association results for Responder_Status. Selecting features from these
# train-only results inside the ML loop removes the data-leakage that occurs
# when MaAsLin2 is run once on the full cohort (as in Figure_5AB.py).
#
# MaAsLin2 parameters mirror maaslin2_final.R (TSS + LOG, same covariates,
# same references, min_prevalence = 0.1).
# =============================================================================

suppressMessages({
  .libPaths("/home/sebin595/R/MaAsLin2/4.4")
  library(Maaslin2)
  library(parallel)
})

base   <- "/home/sebin595/project/0.RA_oral_microbiome/for_paper_submission"
proc   <- file.path(base, "01_Processed_Data")
outdir <- file.path(base, "05_multikingdom_nestedFS", "maaslin_folds")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

NCORES <- 55

# ---- metadata --------------------------------------------------------------
metadata <- read.csv(file.path(proc, "cleaned_metadata_RA.tsv"),
                     sep = "\t", row.names = 1, check.names = FALSE)
metadata$Sex            <- factor(metadata$Sex)
metadata$Smoking_Status <- factor(metadata$Smoking_Status)
input_metadata_full <- metadata[, c("Sex", "Age", "Smoking_Status", "Responder_Status")]

# ---- count matrices (samples x features), all from 01_Processed_Data -------
read_mat <- function(f) read.csv(file.path(proc, f), sep = "\t",
                                 row.names = 1, check.names = FALSE)
mats <- list(
  B = read_mat("bacteria_matrix_filtered.tsv"),
  F = read_mat("fungi_matrix_filtered.tsv"),
  V = read_mat("virus_matrix_filtered.tsv")
)

samples <- rownames(input_metadata_full)
n <- length(samples)
cat(sprintf("Samples: %d | B=%d F=%d V=%d features\n",
            n, ncol(mats$B), ncol(mats$F), ncol(mats$V)))

# ---- job list: fold (held-out sample) x kingdom ----------------------------
jobs <- list()
for (i in seq_len(n)) {
  for (k in names(mats)) {
    jobs[[length(jobs) + 1]] <- list(fold = i, holdout = samples[i], kingdom = k)
  }
}
cat(sprintf("Total MaAsLin2 jobs: %d (%d folds x 3 kingdoms)\n", length(jobs), n))

run_job <- function(job) {
  i  <- job$fold
  k  <- job$kingdom
  ho <- job$holdout
  outfile <- file.path(outdir, sprintf("%s_fold%02d.tsv", k, i))
  if (file.exists(outfile)) return(sprintf("skip  %s", basename(outfile)))

  train_samples <- setdiff(samples, ho)
  md  <- input_metadata_full[train_samples, , drop = FALSE]
  dat <- mats[[k]][train_samples, , drop = FALSE]

  tmpout <- file.path(tempdir(), sprintf("ml_%s_f%02d_%s", k, i,
                                         gsub("[^0-9A-Za-z]", "", ho)))
  res <- tryCatch({
    invisible(capture.output(suppressMessages(
      fit <- Maaslin2(
        input_data     = dat,
        input_metadata = md,
        output         = tmpout,
        fixed_effects  = c("Responder_Status", "Sex", "Age", "Smoking_Status"),
        random_effects = c(),
        normalization  = "TSS",
        transform      = "LOG",
        reference      = c("Responder_Status:Responder",
                           "Smoking_Status:Never_Smoker"),
        min_prevalence = 0.1,
        plot_heatmap   = FALSE,
        plot_scatter   = FALSE,
        cores          = 1
      )
    )))
    ar <- fit$results
    ar <- ar[ar$metadata == "Responder_Status",
             c("feature", "metadata", "value", "coef", "stderr", "pval", "qval")]
    ar$holdout <- ho
    write.table(ar, outfile, sep = "\t", quote = FALSE, row.names = FALSE)
    unlink(tmpout, recursive = TRUE, force = TRUE)
    sprintf("done  %s (%d feats)", basename(outfile), nrow(ar))
  }, error = function(e) {
    unlink(tmpout, recursive = TRUE, force = TRUE)
    sprintf("ERROR %s fold%02d: %s", k, i, conditionMessage(e))
  })
  res
}

t0 <- Sys.time()
res <- mclapply(jobs, run_job, mc.cores = NCORES, mc.preschedule = FALSE)
cat(unlist(res), sep = "\n")
cat(sprintf("\nElapsed: %.1f min\n",
            as.numeric(difftime(Sys.time(), t0, units = "mins"))))

nerr <- sum(grepl("^ERROR", unlist(res)))
cat(sprintf("Errors: %d / %d\n", nerr, length(jobs)))
