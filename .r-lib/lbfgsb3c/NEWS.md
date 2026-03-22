# Version: 2024-3.5 changes

* Added function pointer interface (instead of only low level abi interface)

# lbfgsb3c 2024-3.4 changes

* LTO fixes and remove unused code for Fortran fixes

# lbfgsb3c 2024-3.3 changes

* Fixed `lmm`.  In prior version with the R interface `lmm` was not
  being passed through correctly (though it was passed through in C
  correctly)

* Fixed some bugs in printout

* Reverted the code to fix some of the issues in the FORTRAN code

# lbfgsb3c 2020-3.3 changes

* Now allow `rho=NULL` to work the same as if `rho` was not supplied

* Be More careful about `$convergence` by adding a default value of `NA_INTEGER`

* `$convergence` is now an integer instead of a real number

* Fix too many arguments for format as requested by CRAN

* Added a `NEWS.md` file to track changes to the package.

# lbfgsb3c 2020-3.2 prior changes and information

To do

Add test using a plain C function for optimization. lbfgsb3c is
supposed to handle this.

--------------------------------------------------------------
2019-03-19
    o Packages lbfgsb3 and lbfgsb3c merged into latter. Vignette added.
    o Suppressed printout when trace>2 and starting (f not defined)

2015-01-20
    o Fixup line longer than 72 chars in lbfgsb.f. Undeclared
      integer itask in errclb subroutine. Thanks to Berend Hasselman.

New package lbfgsb3 2014.7.31
