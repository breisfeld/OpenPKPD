## NMdata SAEM age covariate

- Source repo: `NMautoverse/NMdata`
- Source URL: <https://github.com/NMautoverse/NMdata>
- Upstream path: `inst/examples/nonmem/xgxr132.mod`
- Bundled shared dataset: `examples/shared_data/nmautoverse/xgxr2covs.csv`
- License: `MIT + file LICENSE`

This imported bundle demonstrates the original SAEM-plus-age-covariate workflow while resolving to an in-repo shared dataset for GUI and testing use.

OpenPKPD adaptations are intentionally minimal:
- renamed the upstream `.mod` file to `model.ctl`
- rewired `$DATA` to the curated `examples/shared_data/` location
- retained the original NONMEM-style structure, including illustrative `MSFO` and output-file paths

The shared dataset is a documented local subset; see `examples/data/README.md` for source and retained-subject details.

