#!/usr/bin/env Rscript
# ============================================================================
# External validation: joint PK/PD warfarin benchmark in nlmixr2
#
# Dataset : nlmixr2data::warfarin
# Model   : joint 4-state PK/PD model with endpoint routing by DVID
# Methods : FO, FOCE-INTERACTION
#
# Outputs
#   ../data/warfarin_pkpd.csv             — NONMEM-format mixed-endpoint data
#   ../data/warfarin_pkpd_4.csv           — 4-subject reduced mixed-endpoint data
#   ../data/warfarin_pkpd_6.csv           — 6-subject reduced mixed-endpoint data
#   reference/warfarin_pkpd_fo.json       — nlmixr2 FO reference
#   reference/warfarin_pkpd_foce.json     — nlmixr2 FOCE-I reference
#   reference/warfarin_pkpd_4_fo.json     — nlmixr2 FO reference (4-subject reduced)
#   reference/warfarin_pkpd_6_fo.json     — nlmixr2 FO reference (6-subject reduced)
#
# Run from the nlmixr2/ directory:
#   Rscript run_warfarin_pkpd.R
# ============================================================================

repo_lib <- normalizePath("../../../.r-lib", mustWork = FALSE)
.libPaths(c(repo_lib, .libPaths()))

suppressPackageStartupMessages({
  library(nlmixr2)
  library(nlmixr2data)
  library(jsonlite)
})

dat_nlmixr <- warfarin
dat_export <- dat_nlmixr
dat_export$DVID_NUM <- ifelse(as.character(dat_export$dvid) == "cp", 1L, 2L)
dat_export$CMT_NUM <- ifelse(dat_export$evid != 0L, 1L,
                             ifelse(dat_export$DVID_NUM == 1L, 3L, 4L))
dat_export$MDV <- ifelse(dat_export$evid == 0L, 0L, 1L)
dat_export <- dat_export[, c("id", "time", "amt", "dv", "evid", "MDV", "CMT_NUM", "DVID_NUM", "wt")]
names(dat_export) <- c("ID", "TIME", "AMT", "DV", "EVID", "MDV", "CMT", "DVID", "WT")

write.csv(dat_export, file = "../data/warfarin_pkpd.csv",
          row.names = FALSE, quote = FALSE)
cat("Wrote ../data/warfarin_pkpd.csv\n")

dat_reduced_nlmixr <- subset(dat_nlmixr, id <= 4)
dat_reduced_export <- subset(dat_export, ID <= 4)
write.csv(dat_reduced_export, file = "../data/warfarin_pkpd_4.csv",
          row.names = FALSE, quote = FALSE)
cat("Wrote ../data/warfarin_pkpd_4.csv\n")

dat_reduced6_nlmixr <- subset(dat_nlmixr, id <= 6)
dat_reduced6_export <- subset(dat_export, ID <= 6)
write.csv(dat_reduced6_export, file = "../data/warfarin_pkpd_6.csv",
          row.names = FALSE, quote = FALSE)
cat("Wrote ../data/warfarin_pkpd_6.csv\n")

joint_warfarin <- function() {
  ini({
    tktr <- log(1.0)
    tka <- log(1.0)
    tcl <- log(0.13)
    tv <- log(8.0)
    temax <- c(0.01, 0.8, 0.999)
    tec50 <- log(1.0)
    tkout <- log(0.05)
    te0 <- log(100.0)

    eta.cl ~ 0.08
    eta.v ~ 0.05
    eta.ec50 ~ 0.10
    eta.kout ~ 0.05
    eta.e0 ~ 0.05

    pk.prop.err <- 0.12
    pk.add.err <- 0.25
    pd.add.err <- 4.0
  })
  model({
    ktr <- exp(tktr)
    ka <- exp(tka)
    cl <- exp(tcl + eta.cl)
    v <- exp(tv + eta.v)
    emax <- temax
    ec50 <- exp(tec50 + eta.ec50)
    kout <- exp(tkout + eta.kout)
    e0 <- exp(te0 + eta.e0)

    cp <- center / v
    pd <- 1 - emax * cp / (ec50 + cp)

    d/dt(depot) = -ktr * depot
    d/dt(gut) = ktr * depot - ka * gut
    d/dt(center) = ka * gut - cl / v * center
    d/dt(resp) = kout * e0 * (pd - 1) - kout * resp

    cp ~ prop(pk.prop.err) + add(pk.add.err)
    effect <- e0 + resp
    effect ~ add(pd.add.err) | pca
  })
}

joint_warfarin_reduced <- function() {
  ini({
    tktr <- log(1.0)
    tka <- log(1.0)
    tcl <- log(0.14)
    tv <- log(7.5)
    temax <- c(0.01, 0.9, 0.999)
    tec50 <- log(1.3)
    tkout <- log(0.05)
    te0 <- log(96.0)

    eta.e0 ~ 1e-8

    pk.prop.err <- 0.15
    pk.add.err <- 0.5
    pd.add.err <- 3.8
  })
  model({
    ktr <- exp(tktr)
    ka <- exp(tka)
    cl <- exp(tcl)
    v <- exp(tv)
    emax <- temax
    ec50 <- exp(tec50)
    kout <- exp(tkout)
    e0 <- exp(te0 + eta.e0)

    cp <- center / v
    pd <- 1 - emax * cp / (ec50 + cp)

    d/dt(depot) = -ktr * depot
    d/dt(gut) = ktr * depot - ka * gut
    d/dt(center) = ka * gut - cl / v * center
    d/dt(resp) = kout * e0 * (pd - 1) - kout * resp

    cp ~ prop(pk.prop.err) + add(pk.add.err)
    effect <- e0 + resp
    effect ~ add(pd.add.err) | pca
  })
}

extract_results <- function(fit, method_name, dataset_name, data_for_counts,
                            omega_names, description, notes) {
  pe <- fit$parFixedDf
  om <- diag(fit$omega)
  omega_diag <- as.list(stats::setNames(as.list(unname(om[seq_along(omega_names)])), omega_names))

  list(
    method = method_name,
    software = "nlmixr2",
    software_version = as.character(packageVersion("nlmixr2")),
    dataset = dataset_name,
    n_subjects = length(unique(data_for_counts$id)),
    n_obs_in_likelihood = sum(data_for_counts$evid == 0),
    ofv = unname(fit$objDf$OBJF[1]),
    theta = list(
      KTR = unname(exp(pe["tktr", "Estimate"])),
      KA = unname(exp(pe["tka", "Estimate"])),
      CL = unname(exp(pe["tcl", "Estimate"])),
      V = unname(exp(pe["tv", "Estimate"])),
      EMAX = unname(pe["temax", "Estimate"]),
      EC50 = unname(exp(pe["tec50", "Estimate"])),
      KOUT = unname(exp(pe["tkout", "Estimate"])),
      E0 = unname(exp(pe["te0", "Estimate"])),
      PK_PROP_ERR = unname(pe["pk.prop.err", "Estimate"]),
      PK_ADD_ERR = unname(pe["pk.add.err", "Estimate"]),
      PD_ADD_ERR = unname(pe["pd.add.err", "Estimate"])
    ),
    omega_diag = omega_diag,
    raw_message = as.character(fit$message),
    meta = list(
      description = description,
      externally_validated = TRUE,
      reference_software = "nlmixr2",
      model = "Joint 4-state warfarin PK/PD model with PK and PD endpoints routed by DVID",
      parameterisation = "KTR/KA/CL/V/EMAX/EC50/KOUT/E0 with PK add+prop error and PD additive error",
      endpoint_coding = list(dose = 0, cp = 1, pca = 2),
      notes = notes
    )
  )
}

save_json <- function(results, filename) {
  path <- file.path("reference", filename)
  write_json(results, path, pretty = TRUE, auto_unbox = TRUE)
  cat(sprintf("Wrote %s\n", path))
}

cat("\n── Running FO ──────────────────────────────────────────────────────\n")
fit_fo <- nlmixr2(joint_warfarin, dat_nlmixr, est = "fo",
                  control = foControl(print = 0))
save_json(
  extract_results(
    fit_fo, "FO", "warfarin_joint_pkpd", dat_nlmixr,
    c("CL", "V", "EC50", "KOUT", "E0"),
    "nlmixr2 FO joint PK/PD warfarin reference for openpkpd external validation",
    paste("This mixed-endpoint benchmark is intended for the first empirical",
          "DVID-routed external validation path.")
  ),
  "warfarin_pkpd_fo.json"
)

cat("\n── Running FOCE-INTERACTION ─────────────────────────────────────────\n")
fit_foce <- nlmixr2(joint_warfarin, dat_nlmixr, est = "focei",
                    control = foceiControl(print = 0, maxOuterIterations = 200))
save_json(
  extract_results(
    fit_foce, "FOCEI", "warfarin_joint_pkpd", dat_nlmixr,
    c("CL", "V", "EC50", "KOUT", "E0"),
    "nlmixr2 FOCEI joint PK/PD warfarin reference for openpkpd external validation",
    paste("This mixed-endpoint benchmark is intended for the first empirical",
          "DVID-routed external validation path.")
  ),
  "warfarin_pkpd_foce.json"
)

cat("\n── Running FO (4-subject reduced benchmark) ─────────────────────────\n")
fit_reduced_fo <- nlmixr2(joint_warfarin_reduced, dat_reduced_nlmixr, est = "fo",
                          control = foControl(print = 0))
save_json(
  extract_results(
    fit_reduced_fo, "FO", "warfarin_joint_pkpd_4", dat_reduced_nlmixr,
    c("E0"),
    "nlmixr2 FO joint PK/PD warfarin 4-subject reduced benchmark",
    "Reduced empirical mixed-endpoint benchmark for runtime-practical external validation."
  ),
  "warfarin_pkpd_4_fo.json"
)

cat("\n── Running FO (6-subject reduced benchmark) ─────────────────────────\n")
fit_reduced6_fo <- nlmixr2(joint_warfarin_reduced, dat_reduced6_nlmixr, est = "fo",
                           control = foControl(print = 0))
save_json(
  extract_results(
    fit_reduced6_fo, "FO", "warfarin_joint_pkpd_6", dat_reduced6_nlmixr,
    c("E0"),
    "nlmixr2 FO joint PK/PD warfarin 6-subject reduced benchmark",
    paste("Second-tier reduced empirical mixed-endpoint benchmark with practical runtime",
          "and broader coverage than the 4-subject release-gated path.")
  ),
  "warfarin_pkpd_6_fo.json"
)

cat("\nDone.\n")