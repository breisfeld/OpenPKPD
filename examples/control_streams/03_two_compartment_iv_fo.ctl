; Source: examples/03_two_compartment_iv.py
$PROBLEM 2-cmt IV ADVAN3 FO
$DATA two_compartment_iv.csv
$INPUT ID TIME AMT DV EVID MDV
$SUBROUTINES ADVAN3 TRANS1

$PK
  CL  = THETA(1) * EXP(ETA(1))
  V1  = THETA(2) * EXP(ETA(2))
  Q   = THETA(3)
  V2  = THETA(4)
  K   = CL / V1
  K12 = Q / V1
  K21 = Q / V2

$ERROR
  Y = F * (1 + EPS(1))

$THETA
  (0.01, 1.6, 30)   ; CL
  (1.0, 8.0, 100)   ; V1
  (0.1, 0.64, 10)   ; Q
  (1.0, 8.0, 100)   ; V2

$OMEGA
  0.4
  0.4

$SIGMA
  0.05

$ESTIMATION METHOD=ZERO MAXEVAL=600