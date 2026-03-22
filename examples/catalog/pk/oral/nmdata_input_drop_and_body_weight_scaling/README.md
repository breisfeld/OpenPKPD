## NMdata input drop and body-weight scaling

- Source repo: `NMautoverse/NMdata`
- Source URL: <https://github.com/NMautoverse/NMdata>
- Upstream path: `inst/examples/nonmem/xgxr002.mod`
- Bundled shared dataset: `examples/shared_data/nmautoverse/xgxr1.csv`
- License: `MIT + file LICENSE`

This catalog bundle vendors a lightweight in-repo copy of the imported control stream for GUI, parser, and demo workflows.

OpenPKPD adaptations are intentionally minimal:
- renamed the upstream `.mod` file to `model.ctl`
- rewired `$DATA` to the curated `examples/shared_data/` location
- kept the original NONMEM-style structure and illustrative output paths as closely as practical

The bundled dataset is a documented local subset; see `examples/data/README.md` for the retained subject window and source-summary details.

