## ----setup, echo=FALSE--------------------------------------------------------
knitr::knit_engines$set(cpp = function(options) {
  options$engine <- "Rcpp"
  knitr::engine_output(options, options$code, out = "")
})
IS_GITHUB <- Sys.getenv("IS_GITHUB") != ""

## ----results='asis', echo=FALSE, eval=IS_GITHUB-------------------------------
# cat('
# [![R-CMD-check](https://github.com/qsbase/qs2/workflows/R-CMD-check/badge.svg)](https://github.com/qsbase/qs2/actions)
# [![CRAN-Status-Badge](https://www.r-pkg.org/badges/version/qs2)](https://cran.r-project.org/package=qs2)
# [![CRAN-Downloads-Badge](https://cranlogs.r-pkg.org/badges/qs2)](https://cran.r-project.org/package=qs2)
# [![CRAN-Downloads-Total-Badge](https://cranlogs.r-pkg.org/badges/grand-total/qs2)](https://cran.r-project.org/package=qs2)
# ')

## ----eval=FALSE---------------------------------------------------------------
# qs_save(data, "myfile.qs2")
# data <- qs_read("myfile.qs2")

## ----eval=FALSE---------------------------------------------------------------
# install.packages("qs2")

## ----eval=FALSE---------------------------------------------------------------
# remotes::install_cran("qs2", type = "source", configure.args = "--with-simd=AVX2")

## ----eval=FALSE---------------------------------------------------------------
# file_qs2 <- tempfile(fileext = ".qs2")
# file_rds <- tempfile(fileext = ".RDS")
# x <- runif(1e6)
# 
# # save `x` with qs_save
# qs_save(x, file_qs2)
# 
# # convert the file to RDS
# qs_to_rds(input_file = file_qs2, output_file = file_rds)
# 
# # read `x` back in with `readRDS`
# xrds <- readRDS(file_rds)
# stopifnot(identical(x, xrds))

## ----eval=FALSE---------------------------------------------------------------
# qs_save(data, "myfile.qs2")
# data <- qs_read("myfile.qs2", validate_checksum = TRUE)

## ----eval=FALSE---------------------------------------------------------------
# x <- serialize(mtcars, connection = NULL)
# xz <- zstd_compress_raw(x, compress_level = 3)
# x2 <- zstd_decompress_raw(xz)
# stopifnot(identical(x, x2))

## ----eval=FALSE---------------------------------------------------------------
# infile <- tempfile()
# writeBin(as.raw(1:5), infile)
# zfile <- tempfile(fileext = ".zst")
# zstd_compress_file(infile, zfile, compress_level = 1)
# outfile <- tempfile()
# zstd_decompress_file(zfile, outfile)
# stopifnot(identical(readBin(infile, "raw", 5), readBin(outfile, "raw", 5)))

## ----eval=FALSE---------------------------------------------------------------
# # library(data.table)
# save_file <- tempfile(fileext = ".csv.zst")
# 
# # write out zstd compressed table
# zstd_out(data.table::fwrite, mtcars, file = save_file)
# 
# # read in zstd compressed table
# dt <- zstd_in(data.table::fread, file = save_file)

## ----eval=FALSE---------------------------------------------------------------
# qd_save(data, "myfile.qs2")
# data <- qd_read("myfile.qs2")

