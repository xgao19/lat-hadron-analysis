# Lattice QCD Analysis

Repository scaffold for lattice QCD data analysis, with a Python package layout,
basic correlator utilities, tests, and a clean directory structure for raw and
processed data.

## Planned Scope

- Correlator IO helpers
- Effective mass extraction
- Jackknife and bootstrap analysis
- Fit pipelines for two-point functions, TMDWF observables, and future TMDPDF workflows
- Reproducible analysis scripts and notebooks

## Project Layout

```text
.
├── configs/              Example configuration files
├── docs/                 Notes and methodology docs
├── examples/
│   ├── data/             Tracked example correlator CSV files
│   └── outputs/          Ignored example run products
├── scripts/              Domain-organized command-line entry points
├── src/lqcd_analysis/    Python source package
│   ├── common/           Shared reusable helpers and infrastructure
│   ├── two_point/        Two-point analysis workflows
│   ├── tmdpdf_pion/      Pion TMDPDF scaffolding
│   ├── tmdpdf_proton/    Proton TMDPDF scaffolding
│   ├── gpd_pion/         Pion GPD scaffolding
│   ├── gpd_proton/       Proton GPD scaffolding
│   └── tmdwf/            TMDWF analysis scaffolding
├── templates/            Domain-organized notebook and input-file templates
└── tests/                Unit tests
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m unittest discover
```

## Next Steps

1. Add project-specific analysis routines under the appropriate domain package in `src/lqcd_analysis/`.
2. Add or extend templates under `templates/`.
3. Add more tracked example datasets under `examples/data/` if useful.
4. Keep run products under ignored output directories such as `examples/outputs/`.

## SS 2pt TGEVP Driver

You can run the current SS two-point TGEVP extractor with:

```bash
lqcd-analysis ss-2pt-tgevp input_k0_SS.txt --seed 123
```

or:

```bash
python scripts/two_point/ss_2pt_tgevp_extract.py ss-2pt-tgevp input_k0_SS.txt --seed 123
```

If `--results-dir` is not provided, outputs are written to `results/` next to the input file.

Expected input file format:

```text
l64c64a076_m140_SS_k0_pz* 64 64 0.076
c2pt /path/to/c2pt_5_5_k0_pz*_real.csv
pzlist 0 1 2
fold_t periodic
tsrange 0 20
```

The analysis writes:

- `results/<title>_tgevp_summary.txt`
- `results/<title>_tgevp_correlation.txt`
- `results/samples/<title>_tgevp_samples.txt`

## Bootstrap N-State 2pt Fit

Traditional multi-exponential fits are available as a separate workflow:

```bash
python scripts/two_point/fit_2pt_nstate.py 2pt-nstate-fit configs/example_nstate_fit.txt
```

Expected extra input keys:

```text
model symmetric
nstates 1 2 3
fold_t periodic
tmax auto
binsize 1
bootstrap_samples auto
bootstrap_size auto
seed 2026
plot true
```

Optional input:

- `pz0_ground_energy <value>`:
  provide the pz=0 ground-state energy in lattice units. When present, the 1-state plateau search uses the lattice dispersion target `sqrt(E0_pz0^2 + (2*pi*pz/Ns)^2)` to pre-filter plateau candidates before the usual plateau ranking. This affects only plateau selection, not the nonlinear fit model.

`fold_t` options:

- `fold_t false`
- `fold_t none`
- `fold_t true`
- `fold_t periodic`
- `fold_t antiperiodic`

Meaning:

- `false` or `none`: do not fold
- `true` or `periodic`: use symmetric folding with `t` and `Nt-t`
- `antiperiodic`: use antisymmetric folding with `t` and `Nt-t`

Notes on the workflow:

- `tmax auto` is chosen as the first time slice where the effective-mass relative error reaches 50%, or `Nt/2 - 7`, whichever is smaller.
- The 2-state fit is initialized from the suggested 1-state plateau.
- The 3-state fit is initialized from the suggested 2-state plateau.
- Plateau suggestion looks for the longest contiguous `tmin` window where adjacent ground-state energies are statistically consistent and the fits remain reasonably well behaved.
- If `matplotlib` is unavailable, the code writes a small text note instead of a plot file.
- Plotting is now handled by a reusable module in `src/lqcd_analysis/two_point/plotting.py`.
- Plot outputs convert fitted energies and effective masses from lattice units `E*a` into physical units in MeV using the provided lattice spacing.
- After each 2pt fit run, an editable notebook is written under `results_dir/notebook_plots/`. The notebook calls the reusable plotting module so you can tweak paths, styles, and which state to draw.

## TMDWF N-State Ratio Fit

The repository also includes a focused TMDWF ratio-fitting workflow for the
`gamma_t gamma_5` / `gamma_z gamma_5` insertion cases:

```bash
lqcd-analysis tmdwf-nstate-fit input_tmdwf.txt
```

or:

```bash
python scripts/tmdwf/fit_tmdwf_nstate.py tmdwf-nstate-fit input_tmdwf.txt
```

This first implementation:

- fits only the TMDWF ratio
- supports `T5` (`gamma_t gamma_5`) and `Z5` (`gamma_z gamma_5`)
- supports only `1-state` and `2-state` fits
- fits real and imaginary parts separately
- uses fixed two-point amplitudes and energies loaded from a two-point plateau table
- does not yet couple to two-point bootstrap-sample amplitudes and energies

Expected key input fields include:

```text
fit_target ratio
fit_component both
nstates 1 2
gmlist T5
tmax auto
qtmdwf_h5 /path/to/file_or_pattern.h5
dataset_path_template {gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}
two_point_plateau_table /path/to/2pt_plateau_pz*_tmax#_plateau.txt
c2pt /path/to/c2pt_pz*_real.csv
fold_t periodic
tsrange 0 20
```

The TMDWF workflow:

- expands HDF5 dataset paths using `{gm}`, `{eta}`, `{pz}`, `{Tdir}`, `{bT}`, and `{bz}`
- resolves `tmax` from the two-point plateau filename token `_tmax<digits>_plateau.txt` when `tmax` is omitted or set to `auto`
- lets an explicit integer `tmax` override any inferred filename value
- averages over the requested `Tdir` entries
- combines `+bz` and `-bz` when `bz != 0`
- applies the phase factor `exp(-i * phase * bz / 2)` with `phase = 2*pi*pz/Ns`
- applies operator-dependent preprocessing before folding:
  - `T5`: first-half `+1`, second-half `-1`
  - `Z5`: multiply the whole correlator by `-i`
- applies operator-dependent numerator models:
  - `T5`: the original `gamma_t gamma_5` form
  - `Z5`: the `gamma_z gamma_5` form with the lattice-momentum factor `Pz / E_i`
- folds the time dependence after preprocessing
- writes outputs grouped by `bT` rather than one file per `bz`
- stores one parseable `bz` block per grouped summary file
- records two-point provenance in each summary block:
  - `two_point_plateau_table_resolved`
  - `two_point_tmax_source`
  - `two_point_tmax_inferred`
- records downstream-friendly metadata in grouped fit tables:
  - `shared_window_flag`
  - `reference_eta`
  - `reference_bT`
  - `reference_bz`
  - `plateau_tmax_used`

For a fixed `(title, gm, eta, bT, component, nstates)` combination, the grouped
TMDWF outputs now look like:

- `..._summary.txt`: one clearly separated block per `bz`
- `..._fit.txt`: one row per `bz`
- `..._samples.txt`: one row per `(bz, sample_id)`
- `..._curve.txt`: one row per `(bz, t)`

Template inputs are provided in:

- `templates/input_files/tmdwf/tmdwf_nstate_example.txt`
- `templates/input_files/tmdwf/tmdwf_nstate_example_annotated.txt`

A matching notebook template is also provided in:

- `templates/tmdwf/tmdwf_nstate_template.ipynb`

These TMDWF templates are intentionally example workflow templates rather than
fully runnable tracked examples, because they depend on user-local HDF5 data.

## TMDWF Normalize

The extracted TMDWF matrix elements can also be normalized in a separate
downstream step, using only the grouped fit/sample outputs produced by the
TMDWF N-state fit:

```bash
lqcd-analysis tmdwf-normalize input_tmdwf_normalize.txt
```

This normalization workflow:

- reads grouped `..._fit.txt` and `..._samples.txt` outputs from the TMDWF fit workflow
- normalizes the matrix element sample-by-sample before summarizing
- writes grouped normalized outputs that keep the familiar `m0_mean`, `m0_err`, and `m0` columns
- can be used as an intermediate step before the downstream Fourier workflow

Supported normalization modes are:

- `mode1`: `m0(pz, bT, bz) / m0(pz, bT=0, bz=0)`
- `mode2`: `m0(pz, bT, bz) / m0(pz=0, bT, bz=0)`
- `mode3`: `[m0(pz, bT, bz) / m0(pz, bT=0, bz=0)] / [m0(pz=0, bT, bz=0) / m0(pz=0, bT=0, bz=0)]`

Template inputs are provided in:

- `templates/input_files/tmdwf/tmdwf_normalize_example.txt`
- `templates/input_files/tmdwf/tmdwf_normalize_example_annotated.txt`

A matching notebook template is also provided in:

- `templates/tmdwf/tmdwf_normalize_template.ipynb`

## TMDWF Fourier

The repository also provides a separate downstream Fourier workflow that reads
existing grouped TMDWF matrix-element outputs and performs the post-fit cosine
transform:

```bash
lqcd-analysis tmdwf-fourier input_tmdwf_fourier.txt
```

This Fourier workflow:

- reads grouped TMDWF fit/sample outputs, including normalized outputs when desired
- treats the extracted `m0(bT, bz)` as the matrix element to transform in `bz`
- performs the cosine transform on a refined `z` grid
- writes downstream table, sample, and plot outputs without rerunning the nonlinear fit

Template inputs are provided in:

- `templates/input_files/tmdwf/tmdwf_fourier_example.txt`
- `templates/input_files/tmdwf/tmdwf_fourier_example_annotated.txt`

A matching notebook template is also provided in:

- `templates/tmdwf/tmdwf_fourier_template.ipynb`

## Workflows

Both workflows are supported:

- Input-file workflow:
  use plain-text input files plus the existing CLI/scripts
- Notebook-template workflow:
  copy a notebook from `templates/`, edit the user-input cell, validate it, and run the same backend interactively

Current notebook templates:

- `templates/two_point/tgevp_template.ipynb`
- `templates/two_point/nstate_fit_template.ipynb`
- `templates/tmdwf/tmdwf_nstate_template.ipynb`
- `templates/tmdwf/tmdwf_normalize_template.ipynb`
- `templates/tmdwf/tmdwf_fourier_template.ipynb`
- `templates/two_point/plot_2pt_template.ipynb`

Notebook templates are thin wrappers around the existing analysis code. They are intended for clarity and interactive use, while the plain-text input-file workflow remains the stable batch-style interface.
Each template now uses the same notebook-facing pattern: edit a single `workflow_config` object, validate it, and then call one `run_*_from_notebook(...)` function.
Each notebook template also includes an `Option Guide` markdown cell right after the user-input cell, describing the main options, their expected choices, and their practical effect.

The helper bridge for notebooks lives in `src/lqcd_analysis/notebook_workflows.py`. It renders notebook configs into the same or nearly the same text fields used by the existing input-file parsers, and then calls the same backend runners in the `two_point` and `tmdwf` packages.
The two workflows use different default output locations by design: notebook workflows default to the current working directory, while plain-text input-file workflows default to a results directory next to the input file.
For `nstatefit`, the user-facing `fit_mode` option supports both `uncorrelated` and `correlated` fits. In correlated mode the code builds one shared covariance matrix from the full bootstrap ensemble and reuses it across the mean fit and bootstrap fits; if a correlated sample/window fit fails, it falls back to a diagonal fit using only the covariance diagonal.
The N-state fit outputs also track this fallback usage: fit tables include a `fallback_uncorrelated_successes` column for each `tmin` window, and summaries report the representative window's fallback count.

## Runnable Examples

The repository now includes tracked example correlator data under:

- `examples/data/l64c64a076_m140/comb_c2pt_csv/`

and plain-text example inputs under:

- `templates/input_files/two_point/tgevp_example_realdata.txt`
- `templates/input_files/two_point/nstate_fit_example_realdata.txt`
- `templates/input_files/tmdwf/tmdwf_nstate_example.txt`
- `templates/input_files/tmdwf/tmdwf_normalize_example.txt`
- `templates/input_files/tmdwf/tmdwf_fourier_example.txt`
- `templates/input_files/two_point/plot_2pt_example_command.txt`
- `templates/input_files/two_point/tgevp_example_realdata_annotated.txt`
- `templates/input_files/two_point/nstate_fit_example_realdata_annotated.txt`
- `templates/input_files/tmdwf/tmdwf_nstate_example_annotated.txt`
- `templates/tmdwf/tmdwf_nstate_template.ipynb`

These examples use repository-relative paths, so they can be run directly from the repository root.
The `_annotated.txt` variants include inline comments describing each option and are meant to mirror the notebook `Option Guide` cells in plain-text form.
The TMDWF templates are the exception: they are structurally complete templates, but you should point them at your own local HDF5 datasets and two-point plateau tables before running.

Example outputs are intentionally ignored by git and should go under:

- `examples/outputs/`

This lets the repository keep realistic data and templates tracked, while avoiding noisy result folders, plots, logs, and notebooks in version control.

## Package Organization

- `lqcd_analysis.common` contains genuinely shared infrastructure such as bootstrap helpers, folding utilities, generic correlator helpers, and fit-table schema parsing.
- `lqcd_analysis.two_point` contains the current two-point analysis implementations, including N-state fitting, plotting, TGEVP, and two-point-specific CSV loading.
- `lqcd_analysis.tmdwf` contains the current TMDWF ratio-fitting workflow and TMDWF-specific HDF5/data-model helpers.
- `lqcd_analysis.tmdpdf_pion` is a reserved subpackage for future pion TMDPDF workflows.
- `lqcd_analysis.tmdpdf_proton` is a reserved subpackage for future proton TMDPDF workflows.
- `lqcd_analysis.gpd_pion` is a reserved subpackage for future pion GPD workflows.
- `lqcd_analysis.gpd_proton` is a reserved subpackage for future proton GPD workflows.
