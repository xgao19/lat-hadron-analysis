# DA Analysis Input Files

These plain-text templates mirror the notebook-based DA example under
`examples/l64c64a076_m140/analysis/`.

Run them from the repository root in this order:

1. `lqcd-analysis da-nstate-fit examples/l64c64a076_m140/input_files/2-bm/da_nstate_k0.txt`
2. `lqcd-analysis da-nstate-fit examples/l64c64a076_m140/input_files/2-bm/da_nstate_k6.txt`
3. `lqcd-analysis da-normalize examples/l64c64a076_m140/input_files/3-normalize/da_normalize_mode3.txt`
4. `lqcd-analysis da-fourier examples/l64c64a076_m140/input_files/4-FT/da_fourier_mode3.txt`

The output directories are the same ones used by the notebook example tree, so
rerunning these files reproduces the tracked notebook results in place.
