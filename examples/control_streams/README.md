Curated NONMEM-style control-stream exemplars for OpenPKPD GUI/demo use.

This library includes both internal curated examples and imported upstream
examples gathered conservatively from documented sources.

Internal examples were gathered from documented repository sources:

- `examples/*.py` example scripts
- `docs/examples/06_from_control_stream.md`
- `docs/user_guide/control_stream.md`
- existing parser/fixture control streams in `tests/`

Imported upstream examples were adapted from:

- `NMautoverse/NMdata` — `License: MIT + file LICENSE`
- `NMautoverse/NMsim` — `License: MIT + file LICENSE`

Notes:

- The `.ctl` files are intended to be opened, inspected, and adapted.
- Imported upstream examples now resolve to curated local CSV subsets under `examples/data/`.
- Those local CSVs are intentionally lightweight and are meant for parser/GUI/demo workflows rather than full upstream reproduction.
- Advanced syntax showcase files are included for authoring/reference even when they are primarily parse-oriented examples.
- Imported upstream examples retain their original NONMEM structure as closely as practical.
- Imported upstream examples were conservatively adapted by renaming `.mod` to `.ctl`, adding provenance headers, and retaining relative `$DATA`/`FILE=` paths.

Internal curated exemplars:

- `00_theophylline_fixture_diagnostics.ctl` — fuller theophylline example from the existing test fixture
- `01_theophylline_oral_fo.ctl` — 1-compartment oral FO example
- `02_warfarin_oral_focei.ctl` — warfarin oral FOCEI example
- `03_two_compartment_iv_fo.ctl` — 2-compartment IV example
- `06_minimal_theophylline_focei.ctl` — documented minimal control stream from the user docs
- `08_transit_absorption_advan6_fo.ctl` — ODE/transit absorption example with `$DES`
- `09_three_compartment_iv_advan11_focei.ctl` — 3-compartment IV ADVAN11 example
- `20_theophylline_oral_saem.ctl` — SAEM estimation example derived from the SAEM script
- `23_same_omega_showcase.ctl` — compact syntax showcase for repeated `$OMEGA ... SAME`

Imported upstream exemplars:

- `30_nmdata_input_drop_and_body_weight_scaling.ctl` — NMdata `xgxr002.mod`; `$INPUT` rename/drop and body-weight scaling
- `31_nmdata_multiple_table_output_formats.ctl` — NMdata `xgxr018.mod`; multiple `$TABLE` output styles and formatting
- `32_nmdata_saem_age_covariate.ctl` — NMdata `xgxr132.mod`; SAEM + age covariate + IMP follow-up
- `33_nmdata_saem_age_covariate_block_omega.ctl` — NMdata `xgxr133.mod`; SAEM + age covariate + `BLOCK(2)` OMEGA
- `34_nmsim_two_compartment_block_omega_fo.ctl` — NMsim `xgxr022.mod`; FO example with correlated ETA block
- `35_nmsim_advan13_one_compartment_des.ctl` — NMsim `xgxr046.mod`; ADVAN13 + `$MODEL` + `$DES`
- `36_nmsim_saem_multi_covariate_age_weight_sex.ctl` — NMsim `xgxr134.mod`; SAEM with age, weight, and sex covariates

Imported dataset subsets used by those examples:

- `30_nmdata_input_drop_and_body_weight_scaling.ctl` → `examples/data/xgxr1.csv`
- `31_nmdata_multiple_table_output_formats.ctl` → `examples/data/xgxr4.csv`
- `32_nmdata_saem_age_covariate.ctl` → `examples/data/xgxr2covs.csv`
- `33_nmdata_saem_age_covariate_block_omega.ctl` → `examples/data/xgxr2covs.csv`
- `34_nmsim_two_compartment_block_omega_fo.ctl` → `examples/data/xgxr2.csv`
- `35_nmsim_advan13_one_compartment_des.ctl` → `examples/data/xgxr12.csv`
- `36_nmsim_saem_multi_covariate_age_weight_sex.ctl` → `examples/data/xgxr2covs.csv`

Provenance summary for imported examples:

- Source repo and upstream path are recorded in each imported file header.
- Upstream package license for both source repositories is `MIT + file LICENSE`.
- The local CSV subsets are documented in `examples/data/README.md`, including upstream source paths, license notes, and retained subject-ID windows.
- OpenPKPD adaptations are intentionally minimal and are documented in each file header.