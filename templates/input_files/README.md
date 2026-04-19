# Input-File Templates

These plain-text templates mirror the original CLI workflow and use repository-relative example data.
Both compact runnable examples and comment-rich annotated examples are provided.

Run them from the repository root.

## Two-Point Templates

### TGEVP Example

```bash
python scripts/two_point/ss_2pt_tgevp_extract.py ss-2pt-tgevp \
  templates/input_files/two_point/tgevp_example_realdata.txt
```

Annotated version:

```text
templates/input_files/two_point/tgevp_example_realdata_annotated.txt
```

### N-State Fit Example

```bash
python scripts/two_point/fit_2pt_nstate.py 2pt-nstate-fit \
  templates/input_files/two_point/nstate_fit_example_realdata.txt
```

Annotated version:

```text
templates/input_files/two_point/nstate_fit_example_realdata_annotated.txt
```

### Effective Mass Example

```bash
lqcd-analysis 2pt-effective-mass \
  templates/input_files/two_point/effective_mass_example_realdata.txt
```

Annotated version:

```text
templates/input_files/two_point/effective_mass_example_realdata_annotated.txt
```

## TMDWF Templates

### TMDWF Ratio-Fit Example

```bash
lqcd-analysis tmdwf-nstate-fit \
  templates/input_files/tmdwf/tmdwf_nstate_example.txt
```

or:

```bash
python scripts/tmdwf/fit_tmdwf_nstate.py tmdwf-nstate-fit \
  templates/input_files/tmdwf/tmdwf_nstate_example.txt
```

Annotated version:

```text
templates/input_files/tmdwf/tmdwf_nstate_example_annotated.txt
```

This TMDWF template is a workflow template rather than a fully runnable tracked
example dataset. You should replace the HDF5 path, dataset template details, and
the two-point fit root plus `two_point_fit_window_by_pz` mapping with values
that match your local data layout. Each mapping entry should give the explicit
`[tmin, tmax]` window to reuse for a given momentum.

Current grouped-output behavior:

- outputs are grouped by `bT`, not split into a separate file per `bz`
- grouped summaries contain one parseable block per `bz`
- grouped fit / sample / curve files include an explicit leading `bz` column

## Plot Example

Run the N-state fit example first, then:

```bash
python scripts/two_point/plot_2pt_results.py \
  examples/outputs/plot_2pt_realdata \
  examples/outputs/nstate_fit_realdata/l64c64a076_m140_fit_k0_pz0/tables/l64c64a076_m140_fit_k0_pz0_normal_correlator_mean.txt \
  examples/outputs/nstate_fit_realdata/l64c64a076_m140_fit_k0_pz0/tables/l64c64a076_m140_fit_k0_pz0_normal_effective_mass_tmax12.txt \
  examples/outputs/nstate_fit_realdata/l64c64a076_m140_fit_k0_pz0/tables/l64c64a076_m140_fit_k0_pz0_normal_2state_tmax12_fits.txt \
  --nstates 2 \
  --model normal \
  --title l64c64a076_m140_fit_k0_pz0 \
  --nt 64
```
