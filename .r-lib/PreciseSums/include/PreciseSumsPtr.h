#ifndef __PreciseSumsPtr_H__
#define __PreciseSumsPtr_H__

#define PS_DoubleSum 5
#define PS_PairwiseSum 1
#define PS_FSum 2
#define PS_KahanSum 3
#define PS_NeumaierSum 4

#define PS_ProdLD 1
#define PS_Prod 2
#define PS_LogifyProd 3

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>
#include <Rdefines.h>
#include <Rconfig.h>
#include <R_ext/Rdynload.h>
#if defined(__cplusplus)
extern "C" {
#endif

  typedef double (*PreciseSums_sum_type)(double*, int);
  extern PreciseSums_sum_type PreciseSums_sum;

  typedef double (*PreciseSums_prod_type)(double*, int);
  extern PreciseSums_prod_type PreciseSums_prod;

  typedef double (*PreciseSums_prod_r_type)(double*, double*, int, int);
  extern PreciseSums_prod_r_type PreciseSums_prod_r;

  typedef double (*PreciseSums_sum_r_type)(double*, int, double*, int, int);
  extern PreciseSums_sum_r_type PreciseSums_sum_r;

  typedef int (*PreciseSums_sum_get_type)(void);
  extern PreciseSums_sum_get_type PreciseSums_sum_get;

  typedef int (*PreciseSums_prod_get_type)(void);
  extern PreciseSums_prod_get_type PreciseSums_prod_get;


  static inline void iniPreciseSumsPtr0(SEXP ptrLst) {
    if (PreciseSums_sum == NULL) {
      PreciseSums_sum = (PreciseSums_sum_type) R_ExternalPtrAddrFn(VECTOR_ELT(ptrLst, 0));
      PreciseSums_prod = (PreciseSums_prod_type) R_ExternalPtrAddrFn(VECTOR_ELT(ptrLst, 1));
      PreciseSums_sum_r = (PreciseSums_sum_r_type) R_ExternalPtrAddrFn(VECTOR_ELT(ptrLst, 2));
      PreciseSums_prod_r = (PreciseSums_prod_r_type) R_ExternalPtrAddrFn(VECTOR_ELT(ptrLst, 3));
      PreciseSums_sum_get = (PreciseSums_sum_get_type) R_ExternalPtrAddrFn(VECTOR_ELT(ptrLst, 4));
      PreciseSums_prod_get = (PreciseSums_prod_get_type) R_ExternalPtrAddrFn(VECTOR_ELT(ptrLst, 5));
    }
  }
  #define iniPreciseSums \
    PreciseSums_sum_type PreciseSums_sum; \
    PreciseSums_prod_type PreciseSums_prod; \
    PreciseSums_sum_r_type PreciseSums_sum_r; \
    PreciseSums_prod_r_type PreciseSums_prod_r; \
    PreciseSums_sum_get_type PreciseSums_sum_get; \
    PreciseSums_prod_get_type PreciseSums_prod_get; \
    SEXP iniPreciseSumsPtr(SEXP ptr) { \
      iniPreciseSumsPtr0(ptr); \
      return R_NilValue; \
    }

#if defined(__cplusplus)
}
#endif
#endif
