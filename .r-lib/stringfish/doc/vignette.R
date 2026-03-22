## ----setup, echo=FALSE--------------------------------------------------------
IS_GITHUB <- Sys.getenv("IS_GITHUB") != ""

## ----results='asis', echo=FALSE, eval=IS_GITHUB-------------------------------
# cat('
# [![R-CMD-check](https://github.com/traversc/stringfish/workflows/R-CMD-check/badge.svg)](https://github.com/traversc/stringfish/actions)
# [![CRAN-Status-Badge](https://www.r-pkg.org/badges/version/stringfish)](https://cran.r-project.org/package=stringfish)
# [![CRAN-Downloads-Badge](https://cranlogs.r-pkg.org/badges/stringfish)](https://cran.r-project.org/package=stringfish)
# [![CRAN-Downloads-Total-Badge](https://cranlogs.r-pkg.org/badges/grand-total/stringfish)](https://cran.r-project.org/package=stringfish)
# ')

## ----eval=FALSE---------------------------------------------------------------
# install.packages("stringfish", type="source", configure.args="--with-simd=AVX2")

## ----echo=FALSE, results='asis'-----------------------------------------------
if(IS_GITHUB) {
  cat('![](vignettes/bench_v2.png "bench_v2"){width=576px}')
} else {
  cat('![](bench_v2.png "bench_v2"){width=576px}')
}

## ----eval=FALSE---------------------------------------------------------------
# sf_alternate_case("hello world")
# [1] "hElLo wOrLd"

