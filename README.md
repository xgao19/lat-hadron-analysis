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
pz0_ground_energy 0.42
fix_ground_energy_from_dispersion true
# notebook workflow_config: fit_window = {0: [4, 12], 5: [6, 12]}
# plain-text input: fit_window /path/to/2pt_fit_windows.txt
binsize 1
bootstrap_samples auto
bootstrap_size auto
seed 2026
plot true
```

Optional input:

- `pz0_ground_energy <value>`:
  provide the pz=0 ground-state energy in lattice units. This serves as the dispersion-reference input when you want to fix the ground-state energy.
- `fix_ground_energy_from_dispersion true|false`:
  when `true`, the nonlinear 2pt fit fixes the ground-state energy to the lattice-dispersion target derived from `pz0_ground_energy`. This is the repository-native analogue to the legacy fixed-`E0` setup.
- `fit_window <path>`:
  preferred plain-text fit-window table with rows of the form `pz tmin tmax`.
  After folding, the correlator is clipped directly to that window for the
  requested momentum.
Recommended default usage:

- fix trusted fit windows explicitly per momentum with notebook `fit_window`
  or plain-text `fit_window`
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

- The 2-state fit is initialized from the selected 1-state fit window summary.
- The 3-state fit is initialized from the selected 2-state fit window summary.
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
- uses fixed two-point amplitudes and energies loaded from a two-point fit-summary table
- does not yet couple to two-point bootstrap-sample amplitudes and energies

Expected key input fields include:

```text
fit_target ratio
fit_component both
nstates 1 2
gmlist T5
# notebook workflow_config: fit_window = {5: [6, 12], 6: [6, 12]}
# plain-text input: fit_window /path/to/tmdwf_fit_windows.txt
qtmdwf_h5 /path/to/file_or_pattern.h5
dataset_path_template {gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}
two_point_plateau_table /path/to/2pt_fit_pz*_tmax#_plateau.txt
c2pt /path/to/c2pt_pz*_real.csv
fold_t periodic
tsrange 0 20
```

The TMDWF workflow:

- expands HDF5 dataset paths using `{gm}`, `{eta}`, `{pz}`, `{Tdir}`, `{bT}`, and `{bz}`
- resolves `tmax` from the two-point fit-summary filename token `_tmax<digits>_plateau.txt`
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

Recommended default usage:

- use explicit per-momentum fit windows as the default path
- in notebooks, prefer `fit_window`
- in plain-text inputs, use `fit_window`
- treat this as the repository’s current production-style recommendation when
  you already have trusted `trange` choices from prior analysis or inspection

Useful extra TMDWF fit controls:

- `fit_window <path>`:
  preferred plain-text fit-window table. Supported row formats are
  `pz tmin tmax` and `gm pz tmin tmax`.
- `fit_window` in notebook `workflow_config`:
  preferred default notebook-facing fit-window form. Use a dictionary like
  `{5: [6, 12], 6: [6, 12]}` to fix the fit window per momentum, or a nested
  form like `{"T5": {5: [6, 12]}}` when you want `gm`-specific windows. The
  notebook helper materializes this into the backend fit-window table format
  automatically.
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
- when both grouped `real` and `imag` sample tables are available, it first forms complex bootstrap samples, performs the normalization in the complex plane, and only then writes the requested component
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

## TMDWF CS-Kernel Extraction

The repository also provides a downstream Collins-Soper kernel extraction
workflow that reads already-generated TMDWF Fourier outputs:

```bash
lqcd-analysis tmdwf-cs-kernel input_tmdwf_cs_kernel.txt
```

This CS-kernel workflow:

- reads one Fourier bootstrap sample table for each requested `(gm, eta, bT, pz, component, nstates, normalization_mode)` combination
- converts lattice momentum integers into physical momenta using the ensemble metadata
- preserves the legacy type-2 estimator logic, including the absolute value inside the logarithm
- by default uses the full requested `pzlist` in one extraction group: `P1 = pzmin`, `P2 = all larger requested momenta`
- also supports `pair_mode adjacent` for neighboring pairs like `5-6`, `6-7`, `7-8`
- writes bootstrap samples, 16/50/84 summary bands, and chi2/dof diagnostics
- currently supports only the legacy `CG` matching scheme for type-2 extraction, with clean validation

Expected key input fields include:

```text
title_pattern demo_tmdwf_cs_kernel
input_root /path/to/tmdwf_fourier_outputs
ns 64
lattice_spacing_fm 0.076
gmlist T5
etalist eta0
component real
nstates 1
normalization_mode raw
mu 2.0
scheme CG
extraction_type type2
pair_mode all
kernel_labels LO NLO NLL
bTrange 0 4
pzrange 2 6
x_window 0.2 0.8
plot true
```

Option guide:

- `title_pattern`:
  same per-`pz` title pattern used upstream by the TMDWF fit/Fourier workflows, for example `l64c64a076_m140_tmdwf_pz*`.
- `input_root`:
  root directory containing existing TMDWF Fourier outputs. The CS workflow resolves the usual Fourier table/sample filenames automatically from the repository naming convention.
- `ns`, `lattice_spacing_fm`:
  ensemble metadata. `ns` and `lattice_spacing_fm` determine the physical momentum unit `2*pi/(Ns*a*fmGeV)`.
- `gmlist`, `etalist`:
  select which operator/insertion channels to read from the existing Fourier outputs.
- `component`, `nstates`:
  select which Fourier output family to consume.
- `normalization_mode`:
  one of `raw`, `mode1`, `mode2`, or `mode3`. This matches the upstream Fourier-output mode and is recorded explicitly in the CS-kernel outputs.
- `mu`:
  perturbative matching scale in GeV passed into the legacy `CS_Dgamma` correction object.
- `scheme`:
  matching-scheme selector. The current type-2 implementation supports only `CG`, and invalid choices raise a clear error.
- `extraction_type`:
  keep this as `type2` for the legacy qTMDWF CS-kernel method.
- `pair_mode`:
  controls which momentum combinations are fit.
  `all` means a single extraction group per `bT` using `pzmin` as `P1` and all larger requested momenta as `P2`.
  `adjacent` means one extraction group per neighboring pair, e.g. `5-6`, `6-7`, `7-8`.
- `kernel_labels`:
  one or more perturbative labels to run in batch. The legacy order mapping is preserved exactly:
  `LO -> 0`, `NLO/NLL -> 1`, `NNLO/NNLL -> 2`.
- `bTlist` / `bTrange`:
  which transverse separations to process.
- `pzlist` / `pzrange`:
  which lattice momentum integers to process. How they are grouped into `P1/P2` combinations is controlled explicitly by `pair_mode`.
- `x_window`:
  `x` range used in the extraction fit. The default `[0.2, 0.8]` reproduces the legacy script.
- `plot`:
  whether to also write a quick summary PDF band plot for each output group.
  When `pair_mode adjacent`, the workflow also writes an automatic breakdown plot showing
  the data log-ratio term, the matching correction, and the total estimator for all adjacent pairs.
- `results_dir`:
  output root for summaries, band tables, bootstrap samples, diagnostics, and optional plots.

Expected input table format:

- The workflow reads the repository-native Fourier sample table format:
  one row per `(sample_id, x)` with a `q_sample` column.
- Internally those long-form rows are regrouped into a bootstrap matrix with shape `(n_samples, n_x)` before the CS estimator and constant fit are applied.

For each `(kernel_label, bT, pair-group)` combination, the workflow writes:

- `*_summary.txt`
- `tables/*_band.txt`
- `samples/*_samples.txt`
- `diagnostics/*_diagnostics.txt`
- `plots/*_band.pdf` when `plot true`
- `plots/*_adjacent_breakdown.pdf` when `plot true` and `pair_mode adjacent`

Template inputs are provided in:

- `templates/input_files/tmdwf/tmdwf_cs_kernel_example.txt`
- `templates/input_files/tmdwf/tmdwf_cs_kernel_example_annotated.txt`

A matching notebook template is also provided in:

- `templates/tmdwf/tmdwf_cs_kernel_template.ipynb`

## Recommended TMDWF Downstream Chain

The intended downstream chain is now:

1. `tmdwf-nstate-fit`
2. optional `tmdwf-normalize`
3. `tmdwf-fourier`
4. `tmdwf-cs-kernel`

The data products passed between these steps are:

- `tmdwf-nstate-fit`:
  writes grouped matrix-element outputs in `m0(bT, bz)`, including grouped fit tables and grouped bootstrap sample tables.
- `tmdwf-normalize`:
  reads those grouped `m0` outputs and writes normalized grouped `m0` outputs with the same general structure.
- `tmdwf-fourier`:
  reads either raw grouped `m0` outputs or normalized grouped `m0` outputs, then writes Fourier-space `q(x; pz, bT)` tables and bootstrap samples.
- `tmdwf-cs-kernel`:
  reads those Fourier-space bootstrap outputs and performs the legacy type-2 multi-`Pz` CS-kernel extraction.

Practical choices:

- If you want the CS kernel from the unnormalized matrix element chain:
  run `tmdwf-fourier` with `normalization_mode raw`, then run `tmdwf-cs-kernel` with `normalization_mode raw`.
- If you want the CS kernel from one of the normalized matrix-element chains:
  first run `tmdwf-normalize` with `mode1`, `mode2`, or `mode3`, then run `tmdwf-fourier` with the same mode, then run `tmdwf-cs-kernel` with that same mode.

The key convention is:

- `normalization_mode` must stay consistent across the downstream chain.
- `raw` means:
  read original TMDWF fit outputs.
- `mode1` / `mode2` / `mode3` mean:
  read the corresponding normalized outputs derived from the normalization workflow.

This means the repository now supports these two common chains cleanly:

- raw chain:
  `fit -> fourier(raw) -> cs-kernel(raw)`
- normalized chain:
  `fit -> normalize(modeX) -> fourier(modeX) -> cs-kernel(modeX)`

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
- `templates/tmdwf/tmdwf_cs_kernel_template.ipynb`
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
- `templates/input_files/tmdwf/tmdwf_cs_kernel_example.txt`
- `templates/input_files/two_point/plot_2pt_example_command.txt`
- `templates/input_files/two_point/tgevp_example_realdata_annotated.txt`
- `templates/input_files/two_point/nstate_fit_example_realdata_annotated.txt`
- `templates/input_files/tmdwf/tmdwf_nstate_example_annotated.txt`
- `templates/input_files/tmdwf/tmdwf_cs_kernel_example_annotated.txt`
- `templates/tmdwf/tmdwf_nstate_template.ipynb`

These examples use repository-relative paths, so they can be run directly from the repository root.
The `_annotated.txt` variants include inline comments describing each option and are meant to mirror the notebook `Option Guide` cells in plain-text form.
The TMDWF templates are the exception: they are structurally complete templates, but you should point them at your own local HDF5 datasets and two-point fit-summary tables before running.

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
