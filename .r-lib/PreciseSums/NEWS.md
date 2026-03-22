# PreciseSums 0.7

* Created a new non-binary linkage method with 'PreciseSumsPtr.h' in
  addition to 'PreciseSums.h'.  To break binary package dependencies,
  the `rxode2` internals use the function pointers, while the models
  built by `rxode2` use the binary linkage.

* As requested by `CRAN` use strict headers.

# PreciseSums 0.6

* Updated linkage for C.

* Added a `NEWS.md` file to track changes to the package.
