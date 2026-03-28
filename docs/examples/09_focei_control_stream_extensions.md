# Example 9 — FOCEI Optimizer Controls from a Control Stream

**Script:** `examples/26_control_stream_optimizer_extensions.py`

This example shows the OpenPKPD-native `$ESTIMATION` extensions for FOCEI by
parsing and inspecting a control stream.

## Control stream

```text
$PROBLEM Warfarin PK — FOCEI optimizer controls demo
$INPUT ID TIME AMT DV EVID WT MDV
$DATA examples/control_streams/warfarin.csv IGNORE=@
$SUBROUTINES ADVAN2 TRANS2
$PK
  KA = EXP(THETA(1) + ETA(1))
  CL = EXP(THETA(2) + ETA(2))
  V  = EXP(THETA(3) + ETA(3))
  S2 = V
$ERROR
  IPRED = F
  W     = IPRED * THETA(4)
  Y     = IPRED + W * EPS(1)
$THETA (, -0.4) (, -2.0) (, 2.1) (0.001, 0.1, 1.0)
$OMEGA 0.4 0.07 0.04
$SIGMA 1 FIX
$ESTIMATION METHOD=COND INTER MAXEVAL=200 NSTARTS=3 GTOL=1E-6 \
  OUTEROPT=L-BFGS-B FALLBACKOPT=POWELL FALLBACKMAXEVAL=40 \
  RETAINBEST RETRYONABNORMAL RETRYOMEGASCALE=0.5,0.25,0.1
```

## Inspecting it

```bash
python examples/26_control_stream_optimizer_extensions.py
openpkpd parse examples/control_streams/37_focei_optimizer_controls.ctl --json
```

## Notes

- `OUTEROPT=` chooses the primary outer optimizer
- `FALLBACKOPT=` adds a short polish phase after the main optimizer
- `RETAINBEST` keeps the lowest-OFV point seen during the search
- `RETRYONABNORMAL` and `RETRYOMEGASCALE=` enable structured FOCEI retry behavior

This control stream is primarily a syntax and parser showcase. It is suitable
for `openpkpd parse ...` and GUI/control-stream inspection. It is not currently
documented as a recommended fit benchmark.

These keywords are OpenPKPD extensions. They are parsed and executable by the
native runner, but they are not standard NONMEM syntax.
