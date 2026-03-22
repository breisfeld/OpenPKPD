; Source: examples/02_warfarin_foce.py
$PROBLEM Warfarin 1-cmt oral FOCE
$DATA warfarin.csv
$INPUT ID TIME AMT DV EVID MDV WT
$SUBROUTINES ADVAN2 TRANS2

$PK
  KA = THETA(1) * EXP(ETA(1))
  CL = THETA(2) * EXP(ETA(2))
  V  = THETA(3) * EXP(ETA(3))

$ERROR
  Y = F * (1 + EPS(1))

$THETA
  (0.01, 0.9, 20)   ; KA
  (0.001, 0.13, 5)  ; CL
  (0.1, 8.7, 200)   ; V

$OMEGA
  0.4
  0.3
  0.3

$SIGMA
  0.05

$ESTIMATION METHOD=COND INTER MAXEVAL=800