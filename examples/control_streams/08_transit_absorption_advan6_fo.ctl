; Source: examples/08_ode_transit_absorption.py
; Advanced ODE showcase using a transit-absorption $DES block.
$PROBLEM Transit absorption ADVAN6
$DATA transit_absorption.csv
$INPUT ID TIME AMT DV EVID MDV
$SUBROUTINES ADVAN6 TRANS1 TOL=9

$PK
  MTT = THETA(1) * EXP(ETA(1))
  CL  = THETA(2) * EXP(ETA(2))
  V   = THETA(3) * EXP(ETA(3))
  KTR = 4.0 / MTT
  K   = CL / V

$DES
  DADT(1) = -KTR * A(1)
  DADT(2) =  KTR * A(1) - KTR * A(2)
  DADT(3) =  KTR * A(2) - KTR * A(3)
  DADT(4) =  KTR * A(3) - K   * A(4)

$ERROR
  Y = F * (1 + EPS(1))

$THETA
  (0.1, 2.0, 20.0)   ; MTT
  (0.1, 5.0, 50.0)   ; CL
  (5.0, 50.0, 500.0) ; V

$OMEGA
  0.1
  0.1
  0.1

$SIGMA
  0.05

$ESTIMATION METHOD=ZERO MAXEVAL=300