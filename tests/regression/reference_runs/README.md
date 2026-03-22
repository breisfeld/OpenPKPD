# Regression reference status

Files in this directory are **internal regression baselines**.

- They are generated from fixed OpenPKPD runs.
- They are used to detect unintended behavioral drift.
- They are **not automatically external scientific references**.

Each JSON file may include a `_meta` block that states:

- `reference_kind`: currently `internal-baseline`
- `externally_validated`: currently `false` unless explicitly upgraded later
- `validation_level`: optional, e.g. `literature-aligned synthetic benchmark`
- `external_method_references`: optional list of canonical papers behind the expectations
- `validation_note`: optional docs page describing the current validation scope
- dataset / method / seed provenance

External or literature-backed validation should be tracked separately and only
promoted into these baselines once the expected tolerances and provenance are
documented.