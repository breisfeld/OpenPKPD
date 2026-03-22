; Source: examples/20_saem_estimation.py
$PROBLEM Theophylline oral SAEM example
$DATA ../../../../shared_data/theophylline/theophylline_saem.csv
$INPUT ID TIME AMT DV EVID MDV CMT RATE ADDL II SS
$SUBROUTINES ADVAN2 TRANS2

$PK
  KA = THETA(1) * EXP(ETA(1))
  CL = THETA(2) * EXP(ETA(2))
  V  = THETA(3) * EXP(ETA(3))

$ERROR
  Y = F * (1 + EPS(1))

$THETA
  (0.3, 1.5, 8.0)    ; KA
  (0.5, 3.0, 15.0)   ; CL
  (10.0, 35.0, 80.0) ; V

$OMEGA
  0.09
  0.06
  0.04

$SIGMA
  0.02

$ESTIMATION METHOD=SAEM