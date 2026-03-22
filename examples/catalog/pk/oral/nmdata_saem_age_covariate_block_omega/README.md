## NMdata SAEM age covariate BLOCK OMEGA

- Source repo: `NMautoverse/NMdata`
- Source URL: <https://github.com/NMautoverse/NMdata>
- Upstream path: `inst/examples/nonmem/xgxr133.mod`
- Bundled shared dataset: `examples/shared_data/nmautoverse/xgxr2covs.csv`
- License: `MIT + file LICENSE`

This imported bundle preserves the original SAEM example variant with a `BLOCK(2)` OMEGA structure while making it discoverable from the curated catalog.

OpenPKPD adaptations are intentionally minimal:
- renamed the upstream `.mod` file to `model.ctl`
- rewired `$DATA` to the curated `examples/shared_data/` location
- retained the original NONMEM-style structure, including illustrative `MSFO` and output-file paths

The shared dataset is a documented local subset; see `examples/data/README.md` for source and retained-subject details.

