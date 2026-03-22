Curated local CSV subsets for imported NONMEM control-stream examples.

These files make the imported control streams under `examples/control_streams/`
usable in-repo for parsing, GUI import, and lightweight demo workflows without
vendoring the full upstream example datasets.

Sources and licensing:

- `xgxr1.csv` and `xgxr4.csv` were subset from `NMautoverse/NMdata`
  (`inst/examples/data/`; license: `MIT + file LICENSE`).
- `xgxr2.csv`, `xgxr12.csv`, and `xgxr2covs.csv` were subset from
  `NMautoverse/NMsim` (`inst/examples/data/`; license: `MIT + file LICENSE`).

Subset policy:

- Keep only a small contiguous subject window so the example library stays
  lightweight while preserving realistic NONMEM-style row structure.
- `xgxr1.csv`, `xgxr4.csv`, and `xgxr2.csv` keep subject IDs `31..42`.
- `xgxr12.csv` and `xgxr2covs.csv` keep subject IDs `151..162`.
- Column order and raw values are retained from upstream for the kept rows.
- The vendored files are stored without a header row so they load cleanly via
  the imported examples' `$INPUT` records.

Control-stream mapping:

- `30_nmdata_input_drop_and_body_weight_scaling.ctl` -> `xgxr1.csv`
- `31_nmdata_multiple_table_output_formats.ctl` -> `xgxr4.csv`
- `32_nmdata_saem_age_covariate.ctl` -> `xgxr2covs.csv`
- `33_nmdata_saem_age_covariate_block_omega.ctl` -> `xgxr2covs.csv`
- `34_nmsim_two_compartment_block_omega_fo.ctl` -> `xgxr2.csv`
- `35_nmsim_advan13_one_compartment_des.ctl` -> `xgxr12.csv`
- `36_nmsim_saem_multi_covariate_age_weight_sex.ctl` -> `xgxr2covs.csv`

