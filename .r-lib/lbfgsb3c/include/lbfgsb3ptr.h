#ifndef __LBFGS3PTR_H__
#define __LBFGS3PTR_H__
#include <stdlib.h> // for NULL
#include <Rmath.h>
#include <R_ext/Rdynload.h>

#if defined(__cplusplus)
extern "C" {
#endif

  typedef double optimfn(int n, double *par, void *ex);

  typedef void optimgr(int n, double *par, double *gr, void *ex);

  typedef void (*lbfgsb3_fn)(int n, int lmm, double *x, double *lower,
                             double *upper, int *nbd, double *Fmin, optimfn fn,
                             optimgr gr, int *fail, void *ex, double factr,
                             double pgtol, int *fncount, int *grcount,
                             int maxit, char *msg, int trace, int nREPORT, double atol,
                             double rtol, double *g);
  extern lbfgsb3_fn lbfgsb3C;

  static inline SEXP iniLbfgsb3ptr0(SEXP p) {
    if (lbfgsb3C == NULL) {
      lbfgsb3C = (lbfgsb3_fn) R_ExternalPtrAddrFn(VECTOR_ELT(p, 0));
    }
    return R_NilValue;
  }

#define iniLbfgsb3                             \
  lbfgsb3_fn lbfgsb3C = NULL;                   \
  SEXP iniLbfgsb3ptr(SEXP ptr) {                \
    return iniLbfgsb3ptr0(ptr);                 \
  }

#if defined(__cplusplus)
}
#endif

#endif
