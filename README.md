# Lattice QCD Analysis

Repository scaffold for lattice QCD data analysis, with a Python package layout,
basic correlator utilities, tests, and a clean directory structure for raw and
processed data.

## Planned Scope

- Correlator IO helpers
- Effective mass extraction
- Jackknife and bootstrap analysis
- Fit pipelines for two-point functions and DA observables
- Reproducible analysis scripts and notebooks

## Project Layout

```text
.
├── docs/                 Notes and methodology docs
├── examples/
│   └── l64c64a076_m140/   Canonical DA example data, notebooks, and inputs
├── scripts/              Domain-organized command-line entry points
├── src/lqcd_analysis/    Python source package
│   ├── common/           Shared reusable helpers and infrastructure
│   ├── two_point/        Two-point analysis workflows
│   └── da/               DA analysis workflows
├── templates/            Notebook and input-file templates
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
3. Add more tracked example datasets under `examples/l64c64a076_m140/data/` if useful.
4. Keep run products under ignored output directories next to the example tree.

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

## Standalone Effective Mass

If you want the effective-mass tables without running the full N-state fit,
use the standalone two-point effective-mass workflow:

```bash
lqcd-analysis 2pt-effective-mass templates/input_files/two_point/meff.txt
```

or:

```bash
python -m lqcd_analysis.cli 2pt-effective-mass \
  templates/input_files/two_point/meff.txt
```

This workflow:

- reads correlator CSV files
- applies the requested folding and retained `tsrange`
- bootstraps the correlator data
- writes one effective-mass table per momentum

Template inputs are provided in:

- `templates/input_files/two_point/meff.txt`
- `templates/input_files/two_point/meff_annotated.txt`

A matching notebook template is also provided in:

- `templates/two_point/meff.ipynb`

## Bootstrap N-State 2pt Fit

Traditional multi-exponential fits are available as a separate workflow:

```bash
python scripts/two_point/fit_2pt_nstate.py 2pt-nstate-fit \
  templates/input_files/two_point/nstate_2.txt
```

Expected extra input keys:

```text
model symmetric
nstates 1 2 3
fold_t periodic
pz0_ground_energy 0.42
fix_ground_energy_from_dispersion true
# notebook workflow_config: tmin_window = {0: [0, 4], 5: [0, 6]}
# notebook workflow_config: tmax = {0: 12, 5: 12}
# plain-text input: tmin_window /path/to/2pt_tmin_windows.txt
# plain-text input: tmax /path/to/2pt_tmax.txt
binsize 1
bootstrap_samples auto
bootstrap_size auto
seed 2026
plot true
```

Optional input:

- `pz0_ground_energy <value>`:
  provide the pz=0 ground-state energy in lattice units. The code uses it to build the dispersion target for the initial `E0` guess, and also as the fixed `E0` value when you enable dispersion anchoring.
- `fix_ground_energy_from_dispersion true|false`:
  when `true`, the nonlinear 2pt fit fixes the ground-state energy to the lattice-dispersion target derived from `pz0_ground_energy`. When `false`, the fit still uses that target as the initial `E0` guess if `pz0_ground_energy` is provided.
- `tmin_window <path>`:
  preferred plain-text scan-window table with rows of the form `pz tmin_start tmin_end`.
  The fit scans `tmin` from the start up to the end while keeping `tmax` fixed.
- `tmax <value or path>`:
  fixed upper bound for each momentum. For multi-momentum notebook configs,
  this can be provided as a per-momentum mapping that the notebook helper
  materializes into a temporary table.
Recommended default usage:

- fix trusted scan windows explicitly per momentum with notebook `tmin_window`
  and `tmax`, or the plain-text equivalents
- provide `pz0_ground_energy`
- set `fix_ground_energy_from_dispersion true`

This is the repository’s current legacy-aligned default path for production
analysis.

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

- The 2-state fit reads the previous 1-state fit table from the matching state-specific output directory and uses that row to build its initial guess and priors.
- The 3-state fit reads the previous 2-state fit table from the matching state-specific output directory and uses that row to build its initial guess and priors.
- Prior quick reference:
  - `1-state`: no low-state prior.
  - `2-state`: `low_state_prior_tmin[pz]` selects one row from the previous `1-state` table, and that row provides the `E0` prior.
  - `3-state`: the same selector reads one row from the previous `2-state` table, and that row provides the `E0` and `E1` priors.
  - `pz0_ground_energy` supplies the dispersion target for the initial `E0` guess, and becomes the fixed `E0` only when `fix_ground_energy_from_dispersion` is enabled.
- For multistate runs, keep the lower-state output directory available before launching the higher-state fit.
- `lambda_prior` only scales the prior residuals after the row has been selected.
- If `matplotlib` is unavailable, the code writes a small text note instead of a plot file.
- Plotting is now handled by a reusable module in `src/lqcd_analysis/two_point/plotting.py`.
- A more detailed combined workflow guide lives in `docs/analysis_DA_agent/lqcd_agent_month1_detailed_guide_CN.pdf`.

- Plot outputs convert fitted energies and effective masses from lattice units `E*a` into physical units in GeV using the provided lattice spacing.
- Effective-mass plots no longer draw a `tmax` guide line; both effective-mass and energy plots start their displayed data from `tmin = 2` by default.
- After each 2pt fit run, an editable notebook is written under `results_dir/notebook_plots/`. The notebook calls the reusable plotting module so you can tweak paths, styles, and which state to draw.

## DA N-State Ratio Fit

The repository also includes a focused DA ratio-fitting workflow for the
`gamma_t gamma_5` / `gamma_z gamma_5` insertion cases:

```bash
lqcd-analysis da-nstate-fit input_da.txt
```

or:

```bash
python scripts/da/fit_da_nstate.py da-nstate-fit input_da.txt
```

This first implementation:

- fits only the DA ratio
- supports `T5` (`gamma_t gamma_5`) and `Z5` (`gamma_z gamma_5`)
- supports only `1-state` and `2-state` fits
- fits real and imaginary parts separately
- uses fixed two-point amplitudes and energies loaded from a two-point fit-summary table
- can optionally couple to two-point bootstrap-sample amplitudes and energies
- can optionally run a `decay_constant_check` mode that forces `bT = bz = 0`, fits only the real part, and scans nearby fit windows around the requested `fit_window`

Expected key input fields include:

```text
fit_target ratio
fit_component both
nstates 1 2
gmlist T5
decay_constant_check false
two_point_fit_sample_coupled false
# notebook workflow_config: fit_window = {5: [6, 12], 6: [6, 12]}
# plain-text input: fit_window /path/to/da_fit_windows.txt
qda_h5 /path/to/file_or_pattern.h5
dataset_path_template {gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}
two_point_fit_window_table /path/to/2pt_fit_pz*_tmax#_fit_window.txt
c2pt /path/to/c2pt_pz*_real.csv
fold_t periodic
tsrange 0 20
```

The DA workflow:

- expands HDF5 dataset paths using `{gm}`, `{eta}`, `{pz}`, `{Tdir}`, `{bT}`, and `{bz}`
- resolves `tmax` from the two-point fit-summary filename token `_tmax<digits>_fit_window.txt`
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
  - `two_point_fit_table_resolved`
  - `two_point_tmax_source`
  - `two_point_tmax_inferred`
- records downstream-friendly metadata in grouped fit tables:
  - `fit_window_tmax_used`
- when `decay_constant_check` is true, restricts the fit to `bT = bz = 0`, scans `tmin - 2` to `tmin + 2` and `tmax - 2` to `tmax`, also scans the two-point reference `tmin - 1` to `tmin + 1`, and writes a dedicated `*_decay_constant_check_summary.txt`
- the decay-constant check summary and console output report the decay constant in GeV together with its relative error and `chi2_dof`; the summary also records the fixed `bT = bz = 0`, the base `tfit`, a base-fit block selected from the sweep results using the same row format as the sweep table, and the associated two-point `tmin`

For a fixed `(title, gm, eta, bT, component, nstates)` combination, the grouped
DA outputs now look like:

- `..._summary.txt`: one clearly separated block per `bz`
- `..._fit.txt`: one row per `bz`
- `..._samples.txt`: one row per `(bz, sample_id)`
- `..._curve.txt`: one row per `(bz, t)`

Template inputs are provided in:

- `templates/input_files/da/da_nstate_example.txt`
- `templates/input_files/da/da_nstate_example_annotated.txt`

A matching notebook template is also provided in:

- `templates/da/da_nstate_template.ipynb`

The combined Month 1 guide also covers the DA ratio flow in detail, so there is no separate DA guide file anymore.

Recommended default usage:

- use explicit per-momentum fit windows as the default path
- in notebooks, prefer `fit_window`
- in plain-text inputs, use `fit_window`
- treat this as the repository’s current production-style recommendation when
  you already have trusted `trange` choices from prior analysis or inspection

Useful extra DA fit controls:

- `fit_window <path>`:
  preferred plain-text fit-window table. Supported row formats are
  `pz tmin tmax` and `gm pz tmin tmax`.
- `fit_window` in notebook `workflow_config`:
  preferred default notebook-facing fit-window form. Use a dictionary like
  `{5: [6, 12], 6: [6, 12]}` to fix the fit window per momentum, or a nested
  form like `{"T5": {5: [6, 12]}}` when you want `gm`-specific windows. The
  notebook helper materializes this into the backend fit-window table format
  automatically.
- `decay_constant_check` in notebook `workflow_config`:
  set this to `True` for the decay-constant check mode. In that mode the
  workflow ignores `bTlist` and `bzlist`, uses only the real part at
  `bT = bz = 0`, and scans nearby fit windows around the requested
  `fit_window`. It also scans the two-point reference `tmin` around the value
  selected by `two_point_fit_window_by_pz`. For the initial `fit_window`,
  a practical starting point is a `tmin` equal to the chosen two-point `tmin`
  or slightly larger, with `tmax` chosen to keep the DA/2pt ratio smooth
  and non-odd. The scan only keeps windows with `tmin + 1 < tmax`, so each
  candidate window has at least 3 points.
These DA templates are intentionally example workflow templates rather than
fully runnable tracked examples, because they depend on user-local HDF5 data.

## DA Normalize

The extracted DA matrix elements can also be normalized in a separate
downstream step, using only the grouped fit/sample outputs produced by the
DA N-state fit:

```bash
lqcd-analysis da-normalize input_da_normalize.txt
```

This normalization workflow:

- reads grouped `..._fit.txt` and `..._samples.txt` outputs from the DA fit workflow
- normalizes the matrix element sample-by-sample before summarizing
- when both grouped `real` and `imag` sample tables are available, it first forms complex bootstrap samples, performs the normalization in the complex plane, and only then writes the requested component
- writes grouped normalized outputs that keep the familiar `m0_mean`, `m0_err`, and `m0` columns
- can be used as an intermediate step before the downstream Fourier workflow

Supported normalization modes are:

- `mode1`: `m0(pz, bT, bz) / m0(pz, bT=0, bz=0)`
- `mode2`: `m0(pz, bT, bz) / m0(pz=0, bT, bz=0)`
- `mode3`: `[m0(pz, bT, bz) / m0(pz, bT=0, bz=0)] / [m0(pz=0, bT, bz=0) / m0(pz=0, bT=0, bz=0)]`

Template inputs are provided in:

- `templates/input_files/da/da_normalize_example.txt`
- `templates/input_files/da/da_normalize_example_annotated.txt`

A matching notebook template is also provided in:

- `templates/da/da_normalize_template.ipynb`

## DA Fourier

The repository also provides a separate downstream Fourier workflow that reads
existing grouped DA matrix-element outputs and performs the post-fit cosine
transform:

```bash
lqcd-analysis da-fourier input_da_fourier.txt
```

This Fourier workflow:

- reads grouped DA fit/sample outputs, including normalized outputs when desired
- treats the extracted `m0(bT, bz)` as the matrix element to transform in `bz`
- performs the cosine transform on a refined `z` grid
- writes downstream table, sample, and plot outputs without rerunning the nonlinear fit

Template inputs are provided in:

- `templates/input_files/da/da_fourier_example.txt`
- `templates/input_files/da/da_fourier_example_annotated.txt`

A matching notebook template is also provided in:

- `templates/da/da_fourier_template.ipynb`

## DA Ratio Fourier-per-t

An alternate x-space chain starts by Fourier transforming the raw
DA/C2pt ratio at each time slice:

```bash
lqcd-analysis da-ratio-fourier-t input_da_ratio_fourier_t.txt
```

This workflow:

- reads raw DA numerator HDF5 data and the matching C2pt denominator
- constructs bootstrap samples of the ratio for each `bz` and `t`
- performs the cosine transform in `bz` separately at every `t`
- writes `q(x,t)` tables and bootstrap samples for the x-space fit step

Template inputs are provided in:

- `templates/input_files/da/da_ratio_fourier_t_example.txt`
- `templates/input_files/da/da_ratio_fourier_t_example_annotated.txt`

A matching notebook template is also provided in:

- `templates/da/da_ratio_fourier_t_template.ipynb`

## DA x-space N-State Fit

The x-space fit workflow reads `q(x,t)` samples and fits the time dependence
independently at each x:

```bash
lqcd-analysis da-x-nstate-fit input_da_x_nstate_fit.txt
```

This workflow:

- reads `..._ratio_fourier_t_samples.txt` outputs from `da-ratio-fourier-t`
- uses the same two-point fit amplitudes and energies as the ordinary DA fit
- fits the selected `fit_window` independently for every x value
- writes `q0(x)` x-fit tables and bootstrap sample tables

Template inputs are provided in:

- `templates/input_files/da/da_x_nstate_fit_example.txt`
- `templates/input_files/da/da_x_nstate_fit_example_annotated.txt`

A matching notebook template is also provided in:

- `templates/da/da_x_nstate_fit_template.ipynb`

## DA x-fit Normalize

The x-fit normalization workflow normalizes the new `q0(x)` outputs using old
bare matrix-element outputs from `da-nstate-fit`:

```bash
lqcd-analysis da-xfit-normalize input_da_xfit_normalize.txt
```

This workflow:

- reads new `..._xfit_samples.txt` outputs from `da-x-nstate-fit`
- reads old bare `m0(bz)` fit/sample outputs from `da-nstate-fit`
- applies `mode1`, `mode2`, or `mode3` in the old bare matrix-element space
- Fourier transforms the old bare and normalized `m0(bz)` samples on the new x grid
- applies the resulting x-space normalization factor sample-by-sample to `q0(x)`

It intentionally does not read old Fourier outputs.

Template inputs are provided in:

- `templates/input_files/da/da_xfit_normalize_example.txt`
- `templates/input_files/da/da_xfit_normalize_example_annotated.txt`

A matching notebook template is also provided in:

- `templates/da/da_xfit_normalize_template.ipynb`

## Recommended DA Downstream Chain

The intended downstream chain is: 1. `da-nstate-fit` 2. optional `da-normalize` 3. `da-fourier` (cosine Fourier transform in bz).

For the x-space alternative: 1. `da-ratio-fourier-t` 2. `da-x-nstate-fit` 3. `da-xfit-normalize`.

The data products passed between these steps are:

- `da-nstate-fit`:
  writes grouped matrix-element outputs in `m0(bT, bz)`, including grouped fit tables and grouped bootstrap sample tables.
- `da-normalize`:
  reads those grouped `m0` outputs and writes normalized grouped `m0` outputs with the same general structure.
- `da-fourier`:
  reads either raw grouped `m0` outputs or normalized grouped `m0` outputs, then writes Fourier-space `q(x; pz, bT)` tables and bootstrap samples.

The key convention is:

- `normalization_mode` must stay consistent across the downstream chain.
- `raw` means: read original DA fit outputs.
- `mode1` / `mode2` / `mode3` mean: read the corresponding normalized outputs.

Supported chains:
- raw chain: `fit -> fourier(raw)`
- normalized chain: `fit -> normalize(modeX) -> fourier(modeX)`
- x-space fit chain: `ratio-fourier-t -> x-nstate-fit -> xfit-normalize`

## Workflows

Both workflows are supported:

- Input-file workflow:
  use plain-text input files plus the existing CLI/scripts
- Notebook-template workflow:
  copy a notebook from `templates/`, edit the user-input cell, validate it, and run the same backend interactively

Current notebook templates:

- `templates/two_point/tgevp.ipynb`
- `templates/two_point/nstate_1.ipynb`
- `templates/two_point/nstate_2.ipynb`
- `templates/da/da_nstate_template.ipynb`
- `templates/da/da_normalize_template.ipynb`
- `templates/da/da_fourier_template.ipynb`
- `templates/da/da_ratio_fourier_t_template.ipynb`
- `templates/da/da_x_nstate_fit_template.ipynb`
- `templates/da/da_xfit_normalize_template.ipynb`
- `templates/two_point/plot_2pt.ipynb`

Notebook templates are thin wrappers around the existing analysis code. They are intended for clarity and interactive use, while the plain-text input-file workflow remains the stable batch-style interface.
Each template now uses the same notebook-facing pattern: edit a single `workflow_config` object, validate it, and then call one `run_*_from_notebook(...)` function.
Each notebook template also includes an `Option Guide` markdown cell right after the user-input cell, describing the main options, their expected choices, and their practical effect.

The helper bridge for notebooks lives in `src/lqcd_analysis/notebook_workflows.py`. Notebook workflows take their configuration directly from the notebook `workflow_config` object and call the same backend runners in the `two_point` and `da` packages; they do not require a separate input file.
The two workflows use different default output locations by design: notebook workflows default to the current working directory, while plain-text input-file workflows default to a results directory next to the input file.
For `nstatefit`, the user-facing `fit_mode` option supports both `uncorrelated` and `correlated` fits. In correlated mode the code builds one shared covariance matrix from the full bootstrap ensemble and reuses it across the mean fit and bootstrap fits; if a correlated sample/window fit fails, it falls back to a diagonal fit using only the covariance diagonal.
The N-state fit outputs also track this fallback usage: fit tables include a `fallback_uncorrelated_successes` column for each `tmin` window, and summaries report the representative window's fallback count.

## Runnable Examples

The repository now includes tracked example correlator data under:

- `examples/l64c64a076_m140/data/c2pt_csv/`
- `examples/l64c64a076_m140/data/qda/`

and plain-text example inputs under:

- `templates/input_files/two_point/tgevp.txt`
- `templates/input_files/two_point/nstate_1.txt`
- `templates/input_files/two_point/nstate_2.txt`
- `templates/input_files/two_point/meff.txt`
- `templates/input_files/two_point/plot_2pt.txt`
- `templates/input_files/da/da_nstate_example.txt`
- `templates/input_files/da/da_normalize_example.txt`
- `templates/input_files/da/da_fourier_example.txt`
- `templates/input_files/da/da_ratio_fourier_t_example.txt`
- `templates/input_files/da/da_x_nstate_fit_example.txt`
- `templates/input_files/da/da_xfit_normalize_example.txt`
- `templates/input_files/two_point/plot_2pt.txt`
- `templates/input_files/two_point/tgevp_annotated.txt`
- `templates/input_files/two_point/nstate_1_annotated.txt`
- `templates/input_files/two_point/nstate_2_annotated.txt`
- `templates/input_files/da/da_nstate_example_annotated.txt`
- `templates/input_files/da/da_ratio_fourier_t_example_annotated.txt`
- `templates/input_files/da/da_x_nstate_fit_example_annotated.txt`
- `templates/input_files/da/da_xfit_normalize_example_annotated.txt`
- `templates/da/da_nstate_template.ipynb`

These examples use repository-relative paths, so they can be run directly from the repository root.
The `_annotated.txt` variants include inline comments describing each option and are meant to mirror the notebook `Option Guide` cells in plain-text form.
The DA templates are the exception: they are structurally complete templates, but you should point them at your own local HDF5 datasets and two-point fit-summary tables before running.

Example notebook outputs live under the tracked per-example analysis tree, for example:

- `examples/l64c64a076_m140/analysis/`

This keeps the canonical data, templates, and generated notebooks together without introducing a separate outputs tree.

## Package Organization

- `lqcd_analysis.common` contains genuinely shared infrastructure such as bootstrap helpers, folding utilities, generic correlator helpers, and fit-table schema parsing.
- `lqcd_analysis.two_point` submodules contain the current two-point analysis implementations, including N-state fitting, effective-mass extraction, plotting, TGEVP, and two-point-specific CSV loading. Import from the submodules directly.
- `lqcd_analysis.da` submodules contain the current DA ratio-fitting workflow and DA-specific HDF5/data-model helpers. Import from the submodules directly.
