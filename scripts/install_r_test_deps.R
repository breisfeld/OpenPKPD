#!/usr/bin/env Rscript

# Install or verify the local R packages used by external-validation tests.
#
# Default behavior:
#   Rscript --vanilla scripts/install_r_test_deps.R
#
# Verification only:
#   Rscript --vanilla scripts/install_r_test_deps.R --check
#
# Force reinstall:
#   Rscript --vanilla scripts/install_r_test_deps.R --force

args <- commandArgs(trailingOnly = TRUE)
check_only <- "--check" %in% args
force_install <- "--force" %in% args

required_packages <- c(
  "PKNCA",
  "nlmixr2",
  "nlmixr2data",
  "jsonlite"
)

`%||%` <- function(x, y) {
  if (!is.null(x) && length(x) > 0) x else y
}

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_arg[1] %||% ""), mustWork = FALSE)
repo_root <- normalizePath(file.path(dirname(script_path), ".."), mustWork = TRUE)
local_lib <- file.path(repo_root, ".r-lib")

dir.create(local_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(unique(c(local_lib, .libPaths())))

package_is_available <- function(pkg) {
  suppressWarnings(requireNamespace(pkg, quietly = TRUE))
}

print_status <- function() {
  cat("R test dependency library:", local_lib, "\n")
  for (pkg in required_packages) {
    if (package_is_available(pkg)) {
      cat(sprintf("  - %-12s %s\n", pkg, as.character(utils::packageVersion(pkg))))
    } else {
      cat(sprintf("  - %-12s MISSING\n", pkg))
    }
  }
}

missing_packages <- function() {
  required_packages[!vapply(required_packages, package_is_available, logical(1))]
}

to_install <- if (force_install) required_packages else missing_packages()

if (check_only) {
  print_status()
  missing <- missing_packages()
  if (length(missing) > 0) {
    cat("Missing R test dependencies:", paste(missing, collapse = ", "), "\n")
    quit(status = 1)
  }
  quit(status = 0)
}

if (length(to_install) == 0) {
  cat("All required R test dependencies are already installed.\n")
  print_status()
  quit(status = 0)
}

options(repos = c(CRAN = "https://cloud.r-project.org"))
cat("Installing R test dependencies into", local_lib, "\n")
cat("Packages:", paste(to_install, collapse = ", "), "\n")

utils::install.packages(
  to_install,
  lib = local_lib,
  dependencies = TRUE,
  Ncpus = max(1L, parallel::detectCores(logical = TRUE) - 1L)
)

missing <- missing_packages()
print_status()

if (length(missing) > 0) {
  cat("Installation incomplete; still missing:", paste(missing, collapse = ", "), "\n")
  quit(status = 1)
}

cat("R test dependencies installed successfully.\n")
