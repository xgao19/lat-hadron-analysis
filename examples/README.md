# Examples

Canonical DA example set.

## Layout

- `l64c64a076_m140/data/`: tracked example CSV and HDF5 inputs
- `l64c64a076_m140/analysis/`: tracked notebook outputs
- `l64c64a076_m140/input_files/`: tracked plain-text workflow inputs

## Tracked Data

- `examples/l64c64a076_m140/data/c2pt_csv/`
- `examples/l64c64a076_m140/data/qda/`

## Tracked Analysis

- `examples/l64c64a076_m140/analysis/0-effective-mass/`
- `examples/l64c64a076_m140/analysis/1-c2pt-fit/`
- `examples/l64c64a076_m140/analysis/2-bm/`
- `examples/l64c64a076_m140/analysis/3-normalize/`
- `examples/l64c64a076_m140/analysis/4-FT/`

Only the `mode3` downstream path is kept in the example tree.

## Data Contract

- keep only `b_X`
- keep only `OT5`
- keep only `bT = 0`
- remove `b_Y`
- remove `OZ5`

The reusable notebook and input-file templates are documented in the repository root README.
