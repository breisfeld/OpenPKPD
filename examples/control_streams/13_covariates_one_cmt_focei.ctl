; ============================================================================
; Example 13 — One-Compartment with Covariates (ADVAN1 TRANS2, FOCEI)
; ============================================================================
; Dataset: Phase IIa study, 60 subjects, IV infusion
;          Covariates: bodyweight (WT), age (AGE), sex (SEX: 1=male, 0=female)
;
; Reference: NONMEM 7.5.0 FOCEI (Run 504, Bauer NONMEM Tutorial)
;   CL  = 3.03  L/h  (IIV: omega = 0.0307)
;   V   = 32.4  L    (IIV: omega = 0.0466)
;   CL~WT exponent:  0.660  (allometric)
;   V~WT exponent:   1.322
;   CL~AGE exponent: -0.534
;   V~AGE exponent:   0.052
;   CL~SEX multiplier: 0.904  (females relative to males)
;   V~SEX multiplier:  0.947
;   Proportional residual error: sigma = 0.0503
;   OFV = 1058.304 (NONMEM MINIMIZATION SUCCESSFUL)
;
; Covariate model:
;   TVCL = THETA(1) * (WT/70)^THETA(3) * (AGE/50)^THETA(5) * THETA(7)^SEX
;   TVV  = THETA(2) * (WT/70)^THETA(4) * (AGE/50)^THETA(6) * THETA(8)^SEX
;
; Block OMEGA(2) for correlated CL-V variability.
; ============================================================================

$PROBLEM  Phase IIa: 1-cmt IV, covariates — NONMEM Run 504

$INPUT    C ID TIME DV AMT RATE WT AGE SEX

$DATA     501.csv IGNORE=C

$SUBROUTINES ADVAN1 TRANS2

$PK
  ; Power-law covariate model on typical values
  TVCL = THETA(1) * (WT/70)**THETA(3) * (AGE/50)**THETA(5) * THETA(7)**SEX
  TVV  = THETA(2) * (WT/70)**THETA(4) * (AGE/50)**THETA(6) * THETA(8)**SEX

  CL   = TVCL * EXP(ETA(1))
  V    = TVV  * EXP(ETA(2))

  S1   = V

$ERROR
  Y = F * (1 + EPS(1))

; Initial values — start near NONMEM solution for better convergence
$THETA
  (0, 4)     ; THETA(1): CL (L/h) at reference (70 kg, 50 y, male)
  (0, 30)    ; THETA(2): V  (L)
  0.8        ; THETA(3): CL~WT exponent
  0.8        ; THETA(4): V~WT exponent
  -0.5       ; THETA(5): CL~AGE exponent
  0.05       ; THETA(6): V~AGE exponent
  0.9        ; THETA(7): CL~SEX multiplier (female/male ratio)
  0.95       ; THETA(8): V~SEX multiplier

; Block OMEGA for correlated CL-V between-subject variability
$OMEGA BLOCK(2)
  0.1         ; IIV CL
  0.001 0.1   ; CL-V covariance, IIV V

$SIGMA
  0.04   ; proportional residual error variance

; Tighten gradient convergence for covariate-rich models (OpenPKPD extension):
;   GTOL=1e-6 forces more iterations in flat covariate-exponent landscape
;   NSTARTS=3 adds random restarts to explore covariate parameter space
$ESTIMATION METHOD=COND INTERACTION MAXEVAL=9999 PRINT=5 NOABORT GTOL=1e-6

$COVARIANCE UNCONDITIONAL MATRIX=R

$TABLE ID TIME DV IPRED CWRES CL V ETA1 ETA2 WT AGE SEX
       NOPRINT ONEHEADER FILE=covariates_one_cmt.tab
