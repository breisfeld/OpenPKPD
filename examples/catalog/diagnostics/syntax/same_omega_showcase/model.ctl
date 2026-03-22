; Source: docs/user_guide/control_stream.md and examples/23_iov_model.py
; Compact syntax showcase for repeated $OMEGA BLOCK(...) SAME usage.
$PROBLEM SAME OMEGA syntax showcase
$DATA ../../../../shared_data/internal/same_omega.csv
$INPUT ID TIME AMT DV EVID MDV OCC
$SUBROUTINES ADVAN2 TRANS2

$PK
  KA = THETA(1)
  CL = THETA(2) * EXP(ETA(1) + ETA(2))
  V  = THETA(3)

$ERROR
  Y = F * (1 + EPS(1))

$THETA
  (0.2, 1.2, 6.0)   ; KA
  (0.5, 3.0, 15.0)  ; CL
  (8.0, 30.0, 80.0) ; V

$OMEGA BLOCK(1)
  0.04
$OMEGA BLOCK(1) SAME

$SIGMA
  0.01

$ESTIMATION METHOD=ZERO MAXEVAL=400
