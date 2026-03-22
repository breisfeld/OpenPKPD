## NMsim ADVAN13 one-compartment DES

- Source repo: `NMautoverse/NMsim`
- Source URL: <https://github.com/NMautoverse/NMsim>
- Upstream path: `inst/examples/nonmem/xgxr046.mod`
- Bundled shared dataset: `examples/shared_data/nmautoverse/xgxr12.csv`
- License: `MIT + file LICENSE`

This imported bundle preserves the original ADVAN13 + `$MODEL` + `$DES` example while making it usable from the manifest-backed catalog.

OpenPKPD adaptations are intentionally minimal:
- renamed the upstream `.mod` file to `model.ctl`
- rewired `$DATA` to the curated `examples/shared_data/` location
- retained the original NONMEM-style structure, including illustrative `MSFO` and output-file paths

The shared dataset is a documented local subset; see `examples/data/README.md` for source and retained-subject details.

