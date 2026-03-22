## NMsim two-compartment BLOCK OMEGA FO

- Source repo: `NMautoverse/NMsim`
- Source URL: <https://github.com/NMautoverse/NMsim>
- Upstream path: `inst/examples/nonmem/xgxr022.mod`
- Bundled shared dataset: `examples/shared_data/nmautoverse/xgxr2.csv`
- License: `MIT + file LICENSE`

This imported bundle captures the original FO two-compartment example with correlated ETA structure while making it runnable from the curated catalog.

OpenPKPD adaptations are intentionally minimal:
- renamed the upstream `.mod` file to `model.ctl`
- rewired `$DATA` to the curated `examples/shared_data/` location
- retained the original NONMEM-style structure and illustrative output-file references

The shared dataset is a documented local subset; see `examples/data/README.md` for source and subset-policy details.

