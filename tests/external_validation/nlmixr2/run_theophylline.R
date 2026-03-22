#!/usr/bin/env Rscript
# ============================================================================
# External validation: Theophylline 1-compartment oral model in nlmixr2
#
# Dataset : Boeckmann et al. (1992) theophylline (nlmixr2data::theo_sd)
# Model   : 1-cmt oral, proportional error, IIV on KA / CL / V (diagonal)
# Methods : FO, FOCE-INTERACTION
#
# Outputs
#   ../data/theophylline_boeckmann.csv  — NONMEM-format data for openpkpd
#   reference/theophylline_fo.json      — nlmixr2 FO reference
#   reference/theophylline_foce.json    — nlmixr2 FOCE-I reference
#
# Run from the nlmixr2/ directory:
#   Rscript run_theophylline.R
# ============================================================================

suppressPackageStartupMessages({
  library(nlmixr2)
  library(nlmixr2data)
  library(jsonlite)
})

# ---------------------------------------------------------------------------
# 1. Prepare data for nlmixr2 (keep native EVID=101)
# ---------------------------------------------------------------------------
dat_nlmixr <- theo_sd
cat(sprintf("Dataset: %d rows, %d subjects\n",
            nrow(dat_nlmixr), length(unique(dat_nlmixr$ID))))

# Rows used in likelihood: EVID=0 observations with positive time
n_obs_used <- sum(dat_nlmixr$EVID == 0 & dat_nlmixr$TIME > 0)
cat(sprintf("Observations used in likelihood: %d\n", n_obs_used))

# ---------------------------------------------------------------------------
# 2. Export NONMEM-standard CSV for openpkpd (EVID: 101→1, add MDV)
# ---------------------------------------------------------------------------
dat_export <- dat_nlmixr[, c("ID", "TIME", "AMT", "DV", "EVID", "WT")]
dat_export$EVID[dat_export$EVID == 101] <- 1      # dose rows
dat_export$MDV  <- 0L
dat_export$MDV[dat_export$EVID == 1]   <- 1L      # dose rows: ignore DV
dat_export$MDV[dat_export$EVID == 0 & dat_export$TIME == 0] <- 1L  # pre-dose obs

write.csv(dat_export, file = "../data/theophylline_boeckmann.csv",
          row.names = FALSE, quote = FALSE)
cat("Wrote ../data/theophylline_boeckmann.csv\n")

# ---------------------------------------------------------------------------
# 3. Model definition  (parameterisation mirrors openpkpd ADVAN2/TRANS2)
#    KA = exp(lka + eta.ka),  CL = exp(lcl + eta.cl),  V = exp(lv + eta.v)
#    DV ~ proportional error: sd(DV) = prop.err * IPRED
# ---------------------------------------------------------------------------
one_cmt_oral <- function() {
  ini({
    lka <- log(1.5);  label("log(KA) h-1")
    lcl <- log(2.8);  label("log(CL) L/h")
    lv  <- log(32.0); label("log(V) L")
    eta.ka ~ 0.09
    eta.cl ~ 0.09
    eta.v  ~ 0.09
    prop.err <- 0.1
  })
  model({
    ka <- exp(lka + eta.ka)
    cl <- exp(lcl + eta.cl)
    v  <- exp(lv  + eta.v)
    d/dt(depot)   <- -ka * depot
    d/dt(central) <- ka * depot - (cl / v) * central
    cp <- central / v
    cp ~ prop(prop.err)
  })
}

# ---------------------------------------------------------------------------
# Helper: extract results → list for JSON serialisation
# ---------------------------------------------------------------------------
extract_results <- function(fit, method_name) {
  fe <- fixef(fit)
  ka_est  <- exp(fe["lka"])
  cl_est  <- exp(fe["lcl"])
  v_est   <- exp(fe["lv"])

  vc         <- fit$omega
  omega_diag <- diag(vc)

  pe <- fit$parFixedDf
  prop_err_est <- pe["prop.err", "Estimate"]

  ofv_val <- fit$objDf$OBJF[1]

  list(
    method           = method_name,
    software         = "nlmixr2",
    software_version = as.character(packageVersion("nlmixr2")),
    dataset          = "boeckmann_theo_sd",
    n_subjects       = length(unique(dat_nlmixr$ID)),
    n_obs_in_likelihood = n_obs_used,
    ofv              = ofv_val,
    theta = list(KA = ka_est, CL = cl_est, V = v_est),
    omega_diag = list(
      KA = omega_diag[1],
      CL = omega_diag[2],
      V  = omega_diag[3]
    ),
    sigma_prop_err_variance = prop_err_est^2,
    raw_message = fit$message,
    meta = list(
      description          = paste("nlmixr2", method_name,
                                   "reference for openpkpd external validation"),
      externally_validated = TRUE,
      reference_software   = "nlmixr2",
      model                = "1-cmt oral, IIV on KA/CL/V, prop error",
      parameterisation     = "log-normal: KA=exp(lka+eta.ka), etc.",
      tolerance_notes      = paste(
        "OFV compared within 10 units; THETA within 25%.",
        "Differences expected from numerical integration vs analytical solution",
        "and different gradient computation methods."
      )
    )
  )
}

save_json <- function(results, filename) {
  path <- file.path("reference", filename)
  write_json(results, path, pretty = TRUE, auto_unbox = TRUE)
  cat(sprintf("Wrote %s\n", path))
}

# ---------------------------------------------------------------------------
# 4. FO estimation
# ---------------------------------------------------------------------------
cat("\n── Running FO ──────────────────────────────────────────────────────\n")
fit_fo <- tryCatch(
  nlmixr2(one_cmt_oral, dat_nlmixr, est = "fo",
          control = foControl(print = 0)),
  error = function(e) { cat("FO failed:", conditionMessage(e), "\n"); NULL }
)

if (!is.null(fit_fo)) {
  cat(sprintf("FO   OFV=%.4f  KA=%.4f  CL=%.4f  V=%.4f\n",
              fit_fo$objDf$OBJF[1],
              exp(fixef(fit_fo)["lka"]),
              exp(fixef(fit_fo)["lcl"]),
              exp(fixef(fit_fo)["lv"])))
  save_json(extract_results(fit_fo, "FO"), "theophylline_fo.json")
} else {
  cat("Skipping FO JSON\n")
}

# ---------------------------------------------------------------------------
# 5. FOCE-INTERACTION estimation
# ---------------------------------------------------------------------------
cat("\n── Running FOCE-INTERACTION ─────────────────────────────────────────\n")
fit_foce <- tryCatch(
  nlmixr2(one_cmt_oral, dat_nlmixr, est = "focei",
          control = foceiControl(maxOuterIterations = 9999, print = 0)),
  error = function(e) { cat("FOCEI failed:", conditionMessage(e), "\n"); NULL }
)

if (!is.null(fit_foce)) {
  cat(sprintf("FOCE OFV=%.4f  KA=%.4f  CL=%.4f  V=%.4f\n",
              fit_foce$objDf$OBJF[1],
              exp(fixef(fit_foce)["lka"]),
              exp(fixef(fit_foce)["lcl"]),
              exp(fixef(fit_foce)["lv"])))
  save_json(extract_results(fit_foce, "FOCEI"), "theophylline_foce.json")
} else {
  cat("Skipping FOCEI JSON\n")
}

cat("\nDone.\n")
