; ============================================================================
; Example 10 — Warfarin Population PK (1-compartment oral, FOCEI)
; ============================================================================
; Dataset: nlmixr2data::warfarin, PK observations only (DVID="cp")
;          32 subjects, 251 concentration observations
;          Oral dosing, single 100 mg dose
;
; Reference: nlmixr2 FOCEI (v5.0.0)
;   KA  = 0.648  h⁻¹   (IIV: omega = 0.434)
;   CL  = 0.136  L/h   (IIV: omega = 0.073)
;   V   = 8.168  L     (IIV: omega = 0.038)
;   Proportional residual error variance = 0.0505
;   OFV = 474.61
;
; Model parameterisation:
;   KA = exp(THETA(1) + ETA(1))   [log-normal IIV]
;   CL = exp(THETA(2) + ETA(2))
;   V  = exp(THETA(3) + ETA(3))
;   IPRED = F
;   Y = F * (1 + EPS(1))          [proportional error]
; ============================================================================

$PROBLEM  Warfarin PK — 1-cmt oral FOCEI (nlmixr2 reference)

$INPUT    ID TIME AMT DV EVID WT MDV

$DATA     warfarin.csv IGNORE=@

$SUBROUTINES ADVAN2 TRANS2

$PK
  KA = EXP(THETA(1) + ETA(1))
  CL = EXP(THETA(2) + ETA(2))
  V  = EXP(THETA(3) + ETA(3))
  S2 = V

$ERROR
  IPRED = F
  W     = IPRED * THETA(4)
  IRES  = DV - IPRED
  IWRES = IRES / W
  Y     = IPRED + W * EPS(1)

; Initial values — log scale for KA, CL, V
$THETA
  (,  -0.4)    ; THETA(1): log(KA) h-1  → KA ~ 0.67
  (, -2.0)     ; THETA(2): log(CL) L/h  → CL ~ 0.14
  (,  2.1)     ; THETA(3): log(V) L     → V  ~ 8.2
  (0.001, 0.1, 1.0)  ; THETA(4): proportional error SD

; Diagonal OMEGA for KA, CL, V (IIV on log scale)
$OMEGA
  0.4    ; IIV KA
  0.07   ; IIV CL
  0.04   ; IIV V

; Residual error — fixed to 1 (proportional SD in THETA(4))
$SIGMA
  1.0 FIX

$ESTIMATION METHOD=COND INTERACTION MAXEVAL=9999 PRINT=5

$TABLE ID TIME DV IPRED IRES IWRES CWRES ETA1 ETA2 ETA3
       NOPRINT ONEHEADER FILE=warfarin_pk.tab
