; Source: examples/01_theophylline_fo.py
$PROBLEM Theophylline 1-cmt oral FO
$DATA theophylline.csv
$INPUT ID TIME AMT DV EVID MDV WT
$SUBROUTINES ADVAN2 TRANS2

$PK
  KA = THETA(1) * EXP(ETA(1))
  CL = THETA(2) * EXP(ETA(2))
  V  = THETA(3) * EXP(ETA(3))

$ERROR
  Y = F * (1 + EPS(1))

$THETA
  (0.01, 1.5, 20)   ; KA
  (0.001, 0.08, 5)  ; CL
  (0.1, 30, 500)    ; V

$OMEGA
  0.5
  0.3
  0.3

$SIGMA
  0.1

$ESTIMATION METHOD=ZERO MAXEVAL=500