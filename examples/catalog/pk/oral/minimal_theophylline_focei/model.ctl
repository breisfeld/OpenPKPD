; Source: docs/examples/06_from_control_stream.md
; Minimal documented control stream suitable for parser and GUI demos.
$PROBLEM Theophylline 1-compartment oral FO
$DATA ../../../../shared_data/theophylline/theo.csv IGNORE=#
$INPUT ID TIME AMT DV EVID
$SUBROUTINES ADVAN2 TRANS2

$PK
  KA = THETA(1) * EXP(ETA(1))
  CL = THETA(2) * EXP(ETA(2))
  V  = THETA(3) * EXP(ETA(3))

$ERROR
  IPRED = F
  W = THETA(4) * IPRED
  Y = IPRED + W * EPS(1)

$THETA (0.01, 1.5, 20) (0, 0.04, 2) (0, 0.50, 5) (0.01, 0.10, 0.50)
$OMEGA 0.48 0.07 0.02
$SIGMA 1 FIXED
$ESTIMATION METHOD=ZERO MAXEVAL=500
$COVARIANCE
$TABLE ID TIME DV PRED IPRED CWRES NOAPPEND NOPRINT FILE=sdtab
