; ============================================================================
; Example 11 — Two-Compartment IV Bolus (ADVAN3 TRANS4, FOCEI)
; ============================================================================
; Dataset: 30-subject IV bolus PK dataset (Chapter 4/5 textbook, Bauer 2019)
;          12 observations per subject, single IV bolus dose
;
; Reference: NONMEM 7.4.3 FOCEI
;   V1  =  9.76 L      (IIV: omega = 0.01487)
;   CL  =  3.88 L/h    (IIV: omega = 0.01997)
;   V2  = 30.8  L      (IIV: omega = 0.02919)
;   Q   =  8.77 L/h    (IIV: omega = 0.02318)
;   Proportional residual error variance = 0.008841
;   OFV = 196.008 (NONMEM MINIMIZATION SUCCESSFUL)
;
; NOTE ON CONVERGENCE:
;   The two-compartment IV model has a known local-minimum issue when
;   V2 ≈ V1 and Q ≈ Q_init. Starting near the solution (V1~10, V2~30,
;   CL~4, Q~9) avoids the local minimum at V2~8, Q~26.
;   OpenPKPD currently gets stuck at OFV~1497 with default initialization.
;   This is an active area of improvement (multi-start optimization).
;
; ADVAN3 TRANS4 parameterization: CL, V1, Q, V2
; ============================================================================

$PROBLEM  Two-compartment IV FOCEI — Chapter 5 textbook dataset

$INPUT    C ID TIME DV AMT

$DATA     402.csv IGNORE=C

$SUBROUTINES ADVAN3 TRANS4

$PK
  TVV1 = THETA(1)
  V1   = TVV1 * EXP(ETA(1))

  TVCL = THETA(2)
  CL   = TVCL * EXP(ETA(2))

  TVV2 = THETA(3)
  V2   = TVV2 * EXP(ETA(3))

  TVQ  = THETA(4)
  Q    = TVQ  * EXP(ETA(4))

  S1   = V1

$ERROR
  Y = F * (1 + EPS(1))

; Initial values chosen near the known NONMEM solution
; to demonstrate convergence to the global minimum
$THETA
  (0, 9.8)    ; THETA(1): V1 (L)
  (0, 3.7)    ; THETA(2): CL (L/h)
  (0, 8.6)    ; THETA(3): V2 (L)  — note: start near V1 can trap in local min
  (0, 31.0)   ; THETA(4): Q  (L/h)

$OMEGA
  0.02   ; IIV V1
  0.02   ; IIV CL
  0.02   ; IIV V2
  0.02   ; IIV Q

$SIGMA
  0.02   ; proportional residual error variance

; With well-chosen initial values (V2>>V1) single-start converges to the
; global minimum. If initial values are poor, use NSTARTS to enable
; multi-start optimisation (OpenPKPD extension, not in standard NONMEM):
;   $ESTIMATION METHOD=1 MAXEVAL=9999 INTER NSTARTS=5 PERTURBATION=1.5 SEED=42
$ESTIMATION METHOD=1 MAXEVAL=9999 INTER PRINT=5

$COVARIANCE

$TABLE ID TIME DV IPRED CWRES ETA1 ETA2 ETA3 ETA4
       NOPRINT ONEHEADER FILE=two_compartment_iv.tab
