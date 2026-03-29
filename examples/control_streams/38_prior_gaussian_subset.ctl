; OpenPKPD curated example
; Purpose: demonstrate the currently supported `$PRIOR` / `$THETAP` /
; `$THETAPV` / `$OMEGAP` / `$OMEGAPD` runtime subset.
; Intended for control-stream authoring, migration guidance, and parser/runtime
; inspection rather than full NONMEM prior parity.

$PROBLEM Theophylline FOCE with Gaussian prior subset
$INPUT ID TIME AMT DV EVID MDV
$DATA theo.csv IGNORE=@
$SUBROUTINES ADVAN2 TRANS2
$PK
KA = THETA(1)*EXP(ETA(1))
CL = THETA(2)*EXP(ETA(2))
V  = THETA(3)*EXP(ETA(3))
$ERROR
Y = F*(1 + EPS(1))
$THETA (0.01,1.5,20) (0.001,0.08,5) (0.1,30,500)
$OMEGA 0.5 0.3 0.3
$SIGMA 0.1
$PRIOR NWPRI NTHETA=3 NETA=3
$THETAP 1.4 0.09 28.0
$THETAPV 0.25 0.04 9.0
$OMEGAP 0.25 0.10 0.10
$OMEGAPD 4 4 4
$ESTIMATION METHOD=COND INTER MAXEVAL=99
$COVARIANCE
