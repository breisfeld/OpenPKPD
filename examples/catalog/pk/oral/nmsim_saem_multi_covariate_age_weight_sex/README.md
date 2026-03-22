## NMsim SAEM multi-covariate age/weight/sex

- Source repo: `NMautoverse/NMsim`
- Source URL: <https://github.com/NMautoverse/NMsim>
- Upstream path: `inst/examples/nonmem/xgxr134.mod`
- Bundled shared dataset: `examples/shared_data/nmautoverse/xgxr2covs.csv`
- License: `MIT + file LICENSE`

This imported bundle captures the original SAEM multi-covariate workflow with age, weight, and sex effects while keeping the catalog self-contained.

OpenPKPD adaptations are intentionally minimal:
- renamed the upstream `.mod` file to `model.ctl`
- rewired `$DATA` to the curated `examples/shared_data/` location
- retained the original NONMEM-style structure, including illustrative `MSFO` and output-file paths

The shared dataset is a documented local subset; see `examples/data/README.md` for source and retained-subject details.

