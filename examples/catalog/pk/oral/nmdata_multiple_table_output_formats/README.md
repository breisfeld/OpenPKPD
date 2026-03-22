## NMdata multiple table output formats

- Source repo: `NMautoverse/NMdata`
- Source URL: <https://github.com/NMautoverse/NMdata>
- Upstream path: `inst/examples/nonmem/xgxr018.mod`
- Bundled shared dataset: `examples/shared_data/nmautoverse/xgxr4.csv`
- License: `MIT + file LICENSE`

This imported bundle preserves the original example's focus on multiple `$TABLE` output styles while making it runnable from the curated catalog.

OpenPKPD adaptations are intentionally minimal:
- renamed the upstream `.mod` file to `model.ctl`
- rewired `$DATA` to the curated `examples/shared_data/` location
- retained the original NONMEM-style control-stream structure and illustrative output-file references

The shared dataset is a lightweight documented subset; see `examples/data/README.md` for source and subset-policy details.

