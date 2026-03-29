; OpenPKPD curated example
; Purpose: demonstrate the currently supported `$SIMULATION` runtime subset.
; This runs the native simulation path with a fixed seed and two subproblems.

$PROBLEM Theophylline simulation-only subset
$INPUT ID TIME AMT DV EVID MDV
$DATA theo.csv IGNORE=@
$SUBROUTINES ADVAN2 TRANS2
$PK
KA = THETA(1)
CL = THETA(2)
V  = THETA(3)
$ERROR
Y = F*(1 + EPS(1))
$THETA 1.5 0.08 30
$OMEGA 0.0
$SIGMA 0.05
$SIMULATION (12345) ONLYSIMULATION SUBPROBLEMS=2 TRUE=FINAL
