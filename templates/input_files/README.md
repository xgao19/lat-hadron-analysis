# Input-File Templates

These plain-text templates mirror the original CLI workflow and use repository-relative example data.
Both compact runnable examples and comment-rich annotated examples are provided.

Run them from the repository root.

## TGEVP Example

```bash
python scripts/ss_2pt_tgevp_extract.py ss-2pt-tgevp \
  templates/input_files/tgevp_example_realdata.txt
```

Annotated version:

```text
templates/input_files/tgevp_example_realdata_annotated.txt
```

## N-State Fit Example

```bash
python scripts/fit_2pt_nstate.py 2pt-nstate-fit \
  templates/input_files/nstate_fit_example_realdata.txt
```

Annotated version:

```text
templates/input_files/nstate_fit_example_realdata_annotated.txt
```

## Plot Example

Run the N-state fit example first, then:

```bash
python scripts/plot_2pt_results.py \
  examples/outputs/plot_2pt_realdata \
  examples/outputs/nstate_fit_realdata/l64c64a076_m140_fit_k0_pz0/tables/l64c64a076_m140_fit_k0_pz0_normal_correlator_mean.txt \
  examples/outputs/nstate_fit_realdata/l64c64a076_m140_fit_k0_pz0/tables/l64c64a076_m140_fit_k0_pz0_normal_effective_mass_tmax12.txt \
  examples/outputs/nstate_fit_realdata/l64c64a076_m140_fit_k0_pz0/tables/l64c64a076_m140_fit_k0_pz0_normal_2state_tmax12_fits.txt \
  --nstates 2 \
  --model normal \
  --title l64c64a076_m140_fit_k0_pz0 \
  --nt 64
```
