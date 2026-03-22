#ifndef __N1QN1C_H__
#define __N1QN1C_H__

#if defined(__cplusplus)
extern "C" {
#endif


typedef int (*S_fp) (int *, int *, double *, double *, double *, int *, float *, double *, int *);
typedef void (*S2_fp) (int *, int *, double *, double *, double *, int *, float *, double *, int *);
typedef  int (*U_fp)(int *, int *, double *, double *, double *, int *, float *, double *, int *);


typedef void (*n1qn1F_t)(S2_fp simul, int n[], double x[], double f[], double g[], double var[], double eps[],
                         int mode[], int niter[], int nsim[], int imp[], int lp[], double zm[], int izs[],
                         float rzs[], double dzs[]);
extern n1qn1F_t n1qn1F;

typedef void (*n1qn1F2_t)(S2_fp simul, int n[], double x[], double f[], double g[], double var[], double eps[],
int mode[], int niter[], int nsim[], int imp[], double zm[], int izs[],
                     float rzs[], double dzs[]);
extern n1qn1F2_t n1qn1F2;

typedef void (*n1qn1__t)(S2_fp simul, int n[], double x[], double f[], double g[], double var[], double eps[],
                           int mode[], int niter[], int nsim[], int imp[], double zm[], int izs[],
                           float rzs[], double dzs[], int id[]);
extern n1qn1__t n1qn1_;

static inline SEXP iniN1qn1cPtrs0(SEXP p) {
  if (n1qn1F == NULL) {
    n1qn1F = (n1qn1F_t) R_ExternalPtrAddrFn(VECTOR_ELT(p, 0));
    n1qn1F2 = (n1qn1F2_t) R_ExternalPtrAddrFn(VECTOR_ELT(p, 1));
    n1qn1_ = (n1qn1__t) R_ExternalPtrAddrFn(VECTOR_ELT(p, 2));
  }
  return R_NilValue;
}

#define iniN1qn1c              \
  n1qn1F_t n1qn1F = NULL;      \
  n1qn1F2_t n1qn1F2 = NULL;    \
  n1qn1__t n1qn1_ = NULL;      \
  SEXP iniN1qn1cPtrs(SEXP p) { \
    iniN1qn1cPtrs0(p);          \
    return R_NilValue;          \
  }

#if defined(__cplusplus)
}
#endif

#endif // __N1QN1C_H__
