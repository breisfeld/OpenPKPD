#!/usr/bin/env Rscript
# ============================================================================
# External validation: Phenobarbital neonatal PK benchmark in nlmixr2
#
# Dataset : tests/external_validation/data/phenobarbital_simulated.csv
# Model   : 1-cmt IV bolus, WT-scaled CL/V, proportional error, IIV on CL / V
# Methods : FO
#
# Outputs
#   reference/phenobarbital_fo.json      — nlmixr2 FO reference
#
# Run from the nlmixr2/ directory:
#   Rscript run_phenobarbital.R
# ============================================================================

repo_lib <- normalizePath("../../../.r-lib", mustWork = FALSE)
.libPaths(c(repo_lib, .libPaths()))

suppressPackageStartupMessages({
  library(nlmixr2)
  library(jsonlite)
})

dat_nlmixr <- read.csv("../data/phenobarbital_simulated.csv")
cat(sprintf("Dataset: %d rows, %d subjects\n",
            nrow(dat_nlmixr), length(unique(dat_nlmixr$ID))))

n_obs_used <- sum(dat_nlmixr$EVID == 0 & dat_nlmixr$MDV == 0)
cat(sprintf("Observations used in likelihood: %d\n", n_obs_used))

phenobarbital_model <- function() {
  ini({
    tcl <- 0.0047
    tv  <- 0.96
    eta.cl ~ 0.0361
    eta.v  ~ 0.0256
    prop.err <- 0.1
  })
  model({
    cl <- tcl * WT * exp(eta.cl)
    v  <- tv  * WT * exp(eta.v)
    d/dt(centr) <- -(cl / v) * centr
    cp <- centr / v
    cp ~ prop(prop.err)
  })
}

extract_results <- function(fit, method_name) {
  fe <- fixef(fit)
  omega_diag <- diag(fit$omega)
  prop_err_est <- fit$parFixedDf["prop.err", "Estimate"]

  list(
    method = method_name,
    software = "nlmixr2",
    software_version = as.character(packageVersion("nlmixr2")),
    dataset = "phenobarbital_simulated",
    n_subjects = length(unique(dat_nlmixr$ID)),
    n_obs_in_likelihood = n_obs_used,
    ofv = unname(fit$objDf$OBJF[1]),
    theta = list(
      CL_per_kg = unname(fe["tcl"]),
      V_per_kg = unname(fe["tv"])
    ),
    omega_diag = list(
      CL = unname(omega_diag[1]),
      V = unname(omega_diag[2])
    ),
    sigma_prop_err_variance = unname(prop_err_est^2),
    raw_message = as.character(fit$message),
    meta = list(
      description = paste("nlmixr2", method_name,
                          "reference for openpkpd external validation"),
      externally_validated = TRUE,
      reference_software = "nlmixr2",
      model = "phenobarbital 1-cmt IV bolus, WT-scaled CL/V, prop error",
      parameterisation = "CL=tcl*WT*exp(eta.cl), V=tv*WT*exp(eta.v)",
      tolerance_notes = paste(
        "This benchmark compares FO estimates on the same simulated dataset.",
        "FOCEI is not yet stable enough on this benchmark to freeze as a release gate."
      )
    )
  )
}

save_json <- function(results, filename) {
  path <- file.path("reference", filename)
  write_json(results, path, pretty = TRUE, auto_unbox = TRUE, na = "null")
  cat(sprintf("Wrote %s\n", path))
}

cat("\n── Running FO ──────────────────────────────────────────────────────\n")
fit_fo <- nlmixr2(
  phenobarbital_model,
  dat_nlmixr,
  est = "fo",
  control = foControl(print = 0)
)
save_json(extract_results(fit_fo, "FO"), "phenobarbital_fo.json")

cat("\nDone.\n")
