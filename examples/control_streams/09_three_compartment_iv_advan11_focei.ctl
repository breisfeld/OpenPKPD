; Source: examples/09_three_compartment.py
$PROBLEM 3-compartment IV ADVAN11 FOCE
$DATA three_compartment_iv.csv
$INPUT ID TIME AMT DV EVID MDV
$SUBROUTINES ADVAN11 TRANS1

$PK
  CL  = THETA(1) * EXP(ETA(1))
  V1  = THETA(2) * EXP(ETA(2))
  Q2  = THETA(3)
  V2  = THETA(4)
  Q3  = THETA(5)
  V3  = THETA(6)
  K   = CL / V1
  K12 = Q2 / V1
  K21 = Q2 / V2
  K13 = Q3 / V1
  K31 = Q3 / V3

$ERROR
  Y = F * (1 + EPS(1))

$THETA
  (0.01, 2.0, 50.0)   ; CL
  (0.5, 10.0, 200.0)  ; V1
  (0.01, 1.5, 20.0)   ; Q2
  (1.0, 30.0, 500.0)  ; V2
  (0.01, 0.5, 10.0)   ; Q3
  (1.0, 50.0, 500.0)  ; V3

$OMEGA
  0.1
  0.1

$SIGMA
  0.05

$ESTIMATION METHOD=COND INTER MAXEVAL=500