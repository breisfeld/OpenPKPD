; Source: tests/fixtures/control_streams/theophylline.ctl
; Fuller theophylline example with residual calculations and a $TABLE block.
$PROBLEM Theophylline 1-compartment oral model
$DATA theophylline.csv IGNORE=@
$INPUT ID TIME AMT DV EVID MDV WT
$SUBROUTINES ADVAN2 TRANS2

$PK
  KA = THETA(1) * EXP(ETA(1))
  CL = THETA(2) * EXP(ETA(2))
  V  = THETA(3) * EXP(ETA(3))

$ERROR
  IPRED = F
  W     = THETA(4) * IPRED
  Y     = IPRED + W * EPS(1)
  IRES  = DV - IPRED
  IWRES = IRES / W

$THETA
  (0,   1.5,  20)    ; 1 KA (hr-1)
  (0,  0.04,   2)    ; 2 CL/WT (L/hr/kg)
  (0,  0.50,   5)    ; 3 V/WT (L/kg)
  (0.01, 0.10, 0.50) ; 4 proportional residual error

$OMEGA
  0.48   ; 1 KA
  0.07   ; 2 CL
  0.02   ; 3 V

$SIGMA
  1 FIXED ; proportional (W carries all error)

$ESTIMATION METHOD=COND INTER MAXEVAL=9999 SIGDIG=3 PRINT=5
$COVARIANCE

$TABLE ID TIME DV IPRED PRED CWRES IWRES IRES RES ETA1 ETA2 ETA3
       NOPRINT FILE=sdtab.theophylline