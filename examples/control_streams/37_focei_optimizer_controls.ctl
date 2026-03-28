; Internal curated example
; Purpose: demonstrate OpenPKPD-native FOCEI optimizer controls in a stable PK benchmark
; Dataset: examples/control_streams/warfarin.csv
; Reference family: nlmixr2 warfarin PK FOCEI

$PROBLEM  Warfarin PK — FOCEI optimizer controls demo

$INPUT    ID TIME AMT DV EVID WT MDV

$DATA     examples/control_streams/warfarin.csv IGNORE=@

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

$THETA
  (,  -0.4)
  (, -2.0)
  (,  2.1)
  (0.001, 0.1, 1.0)

$OMEGA
  0.4
  0.07
  0.04

$SIGMA
  1.0 FIX

$ESTIMATION METHOD=COND INTERACTION MAXEVAL=200 NSTARTS=3 GTOL=1E-6 OUTEROPT=L-BFGS-B FALLBACKOPT=POWELL FALLBACKMAXEVAL=40 RETAINBEST RETRYONABNORMAL RETRYOMEGASCALE=0.5,0.25,0.1 PRINT=5
