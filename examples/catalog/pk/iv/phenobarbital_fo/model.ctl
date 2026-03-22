; ============================================================================
; Example 12 — Phenobarbital Neonatal Population PK (FO)
; ============================================================================
; Dataset: Simulated neonatal phenobarbital dataset
;          Based on Grasela & Donn (1985), 59 neonates, sparse sampling
;          Multiple IV bolus doses, birthweight covariate
;
; Reference: Grasela TH Jr, Donn SM (1985)
;   "Neonatal population pharmacokinetics of phenobarbital derived from
;    routine clinical data."
;   Dev Pharmacol Ther, 8(6):374-83. PMID: 4075936
;
; Published parameter estimates (NONMEM FO):
;   CL = 0.0047 L/h/kg  (BSV: CV ~19%)
;   V  = 0.96   L/kg    (BSV: CV ~16%)
;   t½ ≈ 141 h (for 1 kg neonate at typical parameters)
;
; Parameterisation:
;   CL_i = THETA(1) * WT * exp(ETA(1))   [allometric weight scaling]
;   V_i  = THETA(2) * WT * exp(ETA(2))
;   Proportional residual error: Y = F * (1 + THETA(3)*EPS(1))
;
; Data file: phenobarbital_simulated.csv (generated with seed=42,
;   identical population parameters to Grasela & Donn 1985)
; ============================================================================

$PROBLEM  Phenobarbital neonatal population PK — Grasela & Donn 1985

$INPUT    ID TIME DV AMT RATE EVID MDV WT APGAR

$DATA     ../../../../shared_data/phenobarbital/phenobarbital_simulated.csv IGNORE=@

$SUBROUTINES ADVAN1 TRANS1

$PK
  ; Weight-scaled typical values (WT in kg)
  TVCL = THETA(1) * WT
  TVV  = THETA(2) * WT

  CL   = TVCL * EXP(ETA(1))
  V    = TVV  * EXP(ETA(2))

  S1   = V

$ERROR
  IPRED = F
  W     = IPRED * THETA(3)
  IRES  = DV - IPRED
  IWRES = IRES / W
  Y     = IPRED + W * EPS(1)

; Initial values close to published Grasela & Donn estimates
$THETA
  (0.0001, 0.005, 0.05)   ; THETA(1): CL per kg (L/h/kg)  → total CL = 0.005 * WT
  (0.10,   1.0,  5.0)    ; THETA(2): V per kg  (L/kg)     → total V  = 1.0 * WT
  (0.001,  0.20, 1.0)    ; THETA(3): proportional error SD

; Between-subject variability on CL and V
$OMEGA
  0.04   ; IIV CL (CV ~20%)
  0.025  ; IIV V  (CV ~16%)

; Residual error (proportional SD encoded in THETA(3))
$SIGMA
  1.0 FIX

$ESTIMATION METHOD=0 MAXEVAL=500 PRINT=10

$TABLE ID TIME DV IPRED IRES IWRES ETA1 ETA2 WT
       NOPRINT ONEHEADER FILE=phenobarbital.tab
