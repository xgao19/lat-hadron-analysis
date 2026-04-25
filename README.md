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
├── docs/                 Notes and methodology docs
├── examples/
│   ├── data/             Tracked example correlator CSV files
│   └── outputs/          Ignored example run products
├── scripts/              Domain-organized command-line entry points
├── src/lqcd_analysis/    Python source package
│   ├── common/           Shared reusable helpers and infrastructure
│   ├── two_point/        Two-point analysis workflows
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

## Standalone Effective Mass

If you want the effective-mass tables without running the full N-state fit,
use the standalone two-point effective-mass workflow:

```bash
lqcd-analysis 2pt-effective-mass templates/input_files/two_point/effective_mass_example_realdata.txt
```

or:

```bash
python -m lqcd_analysis.cli 2pt-effective-mass \
  templates/input_files/two_point/effective_mass_example_realdata.txt
```

This workflow:

- reads correlator CSV files
- applies the requested folding and retained `tsrange`
- bootstraps the correlator data
- writes one effective-mass table per momentum

Template inputs are provided in:

- `templates/input_files/two_point/effective_mass_example_realdata.txt`
- `templates/input_files/two_point/effective_mass_example_realdata_annotated.txt`

A matching notebook template is also provided in:

- `templates/two_point/effective_mass_template.ipynb`

## Bootstrap N-State 2pt Fit

Traditional multi-exponential fits are available as a separate workflow:

```bash
python scripts/two_point/fit_2pt_nstate.py 2pt-nstate-fit \
  templates/input_files/two_point/nstate_fit_2state_example_realdata.txt
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
- A more detailed working guide lives in `docs/two_point_nstate_fit_guide.md`.

- Plot outputs convert fitted energies and effective masses from lattice units `E*a` into physical units in GeV using the provided lattice spacing.
- Effective-mass plots no longer draw a `tmax` guide line; both effective-mass and energy plots start their displayed data from `tmin = 2` by default.
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
- can optionally run a `decay_constant_check` mode that forces `bT = bz = 0`, fits only the real part, and scans nearby fit windows around the requested `fit_window`

Expected key input fields include:

```text
fit_target ratio
fit_component both
nstates 1 2
gmlist T5
decay_constant_check false
# notebook workflow_config: fit_window = {5: [6, 12], 6: [6, 12]}
# plain-text input: fit_window /path/to/tmdwf_fit_windows.txt
qtmdwf_h5 /path/to/file_or_pattern.h5
dataset_path_template {gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}
two_point_fit_window_table /path/to/2pt_fit_pz*_tmax#_fit_window.txt
c2pt /path/to/c2pt_pz*_real.csv
fold_t periodic
tsrange 0 20
```

The TMDWF workflow:

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

A more detailed workflow guide lives in:

- `docs/tmdwf_nstate_fit_guide.md`

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
- `decay_constant_check` in notebook `workflow_config`:
  set this to `True` for the decay-constant check mode. In that mode the
  workflow ignores `bTlist` and `bzlist`, uses only the real part at
  `bT = bz = 0`, and scans nearby fit windows around the requested
  `fit_window`. It also scans the two-point reference `tmin` around the value
  selected by `two_point_fit_window_by_pz`. For the initial `fit_window`,
  a practical starting point is a `tmin` equal to the chosen two-point `tmin`
  or slightly larger, with `tmax` chosen to keep the TMDWF/2pt ratio smooth
  and non-odd. The scan only keeps windows with `tmin + 1 < tmax`, so each
  candidate window has at least 3 points.
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
- fits the legacy type-2 CS-kernel directly in the scalar `real` or `imag` component channel, using the second-formula relation between `P1`, `P2`, and `gamma^{\overline{MS}}`
- by default uses the full requested `pzlist` in one extraction group, with the shared `P1` taken from the first requested momentum unless you set `reference_p1`
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
reference_p1 auto
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
  `all` means a single extraction group per `bT` using a shared `P1` and all other requested momenta as `P2`.
  `adjacent` means one extraction group per neighboring pair, e.g. `5-6`, `6-7`, `7-8`.
  `fixed_p1` means one extraction group per `P2` using a shared `P1`; it only writes the pairwise breakdown plot, not the band plot.
  `reference_p1` is optional and fixes that shared `P1` explicitly; when omitted, the workflow uses `pzlist[0]`.
  `fixed_p1` output filenames include a `fixedp1_...` tag so they do not collide with the `all`-mode files.
  The fit is done in the chosen scalar `component` channel; `real` and `imag` are both supported.
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
  Internally those long-form rows are regrouped into a bootstrap matrix with shape `(n_samples, n_x)` before the direct gamma fit is applied.

For each `(kernel_label, bT, pair-group)` combination, the workflow writes:

- `*_summary.txt`
- `tables/*_band.txt`
- `samples/*_samples.txt`
- `diagnostics/*_diagnostics.txt`
- `plots/*_band.pdf` when `plot true` and `pair_mode all`
- `plots/*_pairwise_breakdown.pdf` when `plot true` and `pair_mode adjacent` or `pair_mode fixed_p1`
- `fixed_p1` breakdown plots include a `fixedp1_refpz...` tag in the filename

Template inputs are provided in:

- `templates/input_files/tmdwf/tmdwf_cs_kernel_example.txt`
- `templates/input_files/tmdwf/tmdwf_cs_kernel_example_annotated.txt`

A matching notebook template is also provided in:

- `templates/tmdwf/tmdwf_cs_kernel_template.ipynb`

## TMDWF Joint CS-Kernel Effective Surface

For multi-ensemble studies, the repository also provides a joint downstream
fit that reads Fourier bootstrap samples directly and fits the effective
anomalous dimension

```text
gamma_eff(x, bT)
```

at each specified x independently, parameterizing the bT dependence with a 1D
piecewise-linear or cubic spline. The fit uses the direct type-2 evolution
relation simultaneously across all ensembles, analytically eliminating one
nuisance amplitude per `(ensemble, bT)` group. This path does not consume
`tmdwf-cs-kernel` or `tmdwf-cs-kernel-average` outputs. It is intended as the
clean first-stage joint fit before adding discretization, finite-volume, or
large-momentum systematic terms.

```bash
lqcd-analysis tmdwf-cs-kernel-joint input_tmdwf_cs_kernel_joint.txt
```

This joint workflow:

- reads repository-native Fourier bootstrap sample tables from each requested
  ensemble
- keeps each ensemble's own physical `Pz` and physical `bT = nT * a`
- fits `gamma_eff(bT)` independently at each `x_knots` point using a 1D bT
  spline, then writes the combined `gamma_eff(x,bT)` surface
- uses the direct type-2 evolution relation in the scalar `real` or `imag`
  component channel, with the matching correction evaluated at each x's actual
  data-grid value (the nearest grid point to the requested x_knot)
- analytically eliminates one nuisance amplitude for each
  `(ensemble, bT)` group during each per-x nonlinear spline fit
- writes the bootstrap surface, summary quantiles, chi2/dof diagnostics, and
  spline coefficients for reconstructing `gamma_eff(bT)` at arbitrary bT values

Expected key input fields include:

```text
gm T5
eta eta0
component real
nstates 2
normalization_mode mode3
mu 2.0
scheme CG
kernel_label LO
reference_p1_gev 1.0
x_window 0.2 0.8
x_knots 0.2 0.3 0.4 0.5 0.6 0.7 0.8
bT_knots_fm 0.05 0.10 0.15 0.20 0.25 0.30
spline_kind linear
plot true
progress true
progress_every 10
ensemble l48a060 /path/l48 l48_pz* 48 0.060 pz=1:5 bT=0:20
ensemble l64a050 /path/l64 l64_pz* 64 0.050 pz=3:8 bT=0:30
results_dir results_tmdwf_cs_kernel_joint
```

Option guide:

- `gm`, `eta`, `component`, `nstates`, `normalization_mode`:
  select the Fourier output family to consume. These settings are shared by
  all ensembles in the joint fit.
- `mu`, `scheme`, `kernel_label`:
  matching settings used in the same type-2 correction machinery as
  `tmdwf-cs-kernel`. The first version supports one `kernel_label` per run.
- `reference_p1_gev`:
  physical reference momentum scale in GeV used in the direct evolution
  formula. This is not a lattice momentum integer.
- `x_window`:
  x range. Observations whose actual data-grid x value falls outside this
  window are excluded from the fit.
- `x_knots`:
  the x values at which `gamma_eff(x,bT)` is fitted independently. Each
  `x_knot` is mapped to the nearest point on the data x-grid; the actual
  x value is recorded in the output. If omitted, the workflow picks up to 6
  evenly spaced points within `x_window`.
- `bT_knots_fm`:
  spline knots (in fm) for the bT-direction spline parameterizing
  `gamma_eff(bT)` at each x. If omitted, the workflow uses the unique
  physical bT values across all ensembles, capped at 8 knots.
- `spline_kind`:
  interpolation kind for the bT spline. Use `linear` for the piecewise-linear
  hat basis or `cubic` for a natural cubic spline basis.
- `plot`:
  whether to write one `gamma_eff` vs x band plot for each `bT_knots_fm`
  value, per-x pz-diagnostics plots showing data vs fit for each
  `(ensemble, bT)` group, and a diagnostics notebook for redrawing
  those plots from saved outputs.
- `progress`, `progress_every`:
  whether to print bootstrap-fit progress (per x and per sample) and how many
  completed bootstrap samples to wait between progress reports. If
  `progress_every` is omitted, the workflow reports roughly every 5% of the
  bootstrap samples.
- `ensemble`:
  one line per ensemble, with
  `label input_root title_pattern Ns lattice_spacing_fm pz=<list/range> bT=<list/range>`.
  Lists are comma-separated; inclusive ranges use `start:stop`. The
  `title_pattern` follows the existing convention where `*` is replaced by the
  lattice momentum integer.
- `results_dir`:
  output root for the joint summary, surface table, bootstrap surface samples,
  spline coefficients, and diagnostics.

The workflow writes:

- `joint_gamma_eff/*_summary.txt`
- `joint_gamma_eff/tables/*_surface.txt`
- `joint_gamma_eff/samples/*_samples.txt`
- `joint_gamma_eff/samples/*_coefficients.txt`
- `joint_gamma_eff/diagnostics/*_diagnostics.txt`
- `joint_gamma_eff/plots/*_x_band.pdf`
- `joint_gamma_eff/plots/diagnostics/*_x{X}_*_pz_diagnostics.pdf`
- `joint_gamma_eff/plots/diagnostics/*_diagnostics_notebook.ipynb`

The `*_coefficients.txt` file records the spline coefficients for every
bootstrap sample at each x, so that `gamma_eff(bT)` can be reconstructed at
any bT value using `_spline_basis(bT_values, bT_knots_fm, kind=spline_kind) @ coeffs`.
The `bT_knots_fm` and `spline_kind` are recorded in `*_summary.txt`.

The `*_pz_diagnostics.pdf` files show, for each fitted x, data points and the
reconstructed fit band in `O vs pz` space organized per ensemble. The
`*_diagnostics_notebook.ipynb` notebook redraws these plots from saved outputs
and supports custom x/bT selection plus cross-ensemble comparison at matching
physical bT values.

Template inputs are provided in:

- `templates/input_files/tmdwf/tmdwf_cs_kernel_joint_example.txt`
- `templates/input_files/tmdwf/tmdwf_cs_kernel_joint_example_annotated.txt`

A matching notebook template is also provided in:

- `templates/tmdwf/tmdwf_cs_kernel_joint_template.ipynb`

## TMDWF CS-Kernel Averaging

The repository also provides a downstream averaging step that consumes existing
CS-kernel outputs and averages the x-dependent results over the selected x
region for each `bT`.

Two interfaces are supported:

- Notebook template: self-contained `workflow_config`, no separate input file
- Plain-text batch input: `input_tmdwf_cs_kernel_average.txt`

Notebook template:

- `templates/tmdwf/tmdwf_cs_kernel_average_template.ipynb`

Plain-text batch example:

```bash
lqcd-analysis tmdwf-cs-kernel-average input_tmdwf_cs_kernel_average.txt
```

This averaging workflow:

- reads the repository-native CS-kernel band/sample tables produced by the
  downstream CS-kernel extraction workflow
- selects x values using either the explicit `x_range` or the physical cuts
  `2*x*pz>1 GeV`, `2*(1-x)*pz>1 GeV`, `bT*pz*x>0.5`, and `bT*pz*(1-x)>0.5`
- averages the selected x-dependent results for each bootstrap sample
- reports the sample-mean statistical error and the within-sample systematic error
- writes one averaged value per `bT` together with the selection summary and full bootstrap summary table
- automatically writes a `bT` summary plot with statistical and total error bars

Expected key input fields include:

```text
title_pattern demo_tmdwf_cs_kernel_average
input_root /path/to/tmdwf_cs_kernel_outputs
lattice_spacing_fm 0.076
gm T5
eta eta0
component real
nstates 1
normalization_mode raw
scheme CG
extraction_type type2
kernel_label LO
pair_mode fixed_p1
reference_p1 5
bTrange 0 4
x_range 0.25 0.75
reference_pz_labels 5-6 6-7 7-8
results_dir /path/to/tmdwf_cs_kernel_average
```

Option guide:

- `title_pattern`:
  same per-`pz` title pattern used upstream by the TMDWF fit/Fourier/CS-kernel workflows.
- `input_root`:
  root directory containing existing TMDWF CS-kernel outputs. The averaging workflow resolves the usual CS-kernel band/sample filenames automatically from the repository naming convention.
- `lattice_spacing_fm`:
  lattice spacing in fm. The averaging workflow uses it to add `bT_fm` to the values table and to plot the summary on a fm horizontal axis.
- `gm`, `eta`:
  operator and insertion-channel selectors used to identify the correct CS-kernel outputs.
- `component`, `nstates`:
  select which Fourier/CS-kernel family to consume.
- `normalization_mode`:
  one of `raw`, `mode1`, `mode2`, or `mode3`. This must match the upstream Fourier/CS-kernel output mode.
- `scheme`:
  matching-scheme selector. The current implementation supports only `CG` for the type-2 TMDWF workflow.
- `extraction_type`:
  keep this as `type2` for the legacy qTMDWF CS-kernel method.
- `kernel_label`:
  which perturbative CS-kernel label to average. The workflow currently expects one label at a time.
- `pair_mode`:
  optional source filter. Use `all`, `adjacent`, or `fixed_p1` to select which CS-kernel extraction runs to average. If omitted, all matching modes are used.
- `reference_p1`:
  optional fixed shared `P1` filter. When provided, or when set to `auto`, only CS-kernel outputs with that reference momentum are averaged. If omitted, no `P1` filter is applied.
- `bTlist` / `bTrange`:
  which transverse separations to process.
- `x_range`:
  optional explicit `x_min x_max` interval. When provided, the workflow uses that x window directly and skips the automatic physical x-selection cuts.
- `reference_pz_labels`:
  which CS-kernel pair-group labels to include in the averaging. These are matched against the CS-kernel output metadata, so `5-6` selects the source group whose reference momentum label is `5-6` and whose numeric reference momentum is `5`. If omitted, all matching labels are used.
- `results_dir`:
  output root for the averaged summary file, grouped tables, bootstrap samples, and selection metadata.

Expected input data shape:

- The workflow reads the repository-native CS-kernel summary bands and bootstrap sample tables.
- It groups the CS-kernel x-dependent results by `bT`. If `x_range` is provided, it uses that interval directly; otherwise it applies the physical x-selection cuts. It also applies any requested source filters such as `pair_mode` or `reference_p1`, then averages across the selected x values for each bootstrap sample.
- The output summary uses the mean of the sample means as the central value, a percentile-based statistical error from the sample means, and the mean within-sample standard deviation as the systematic error.

For each requested `bT`, the workflow writes:

- `*_summary.txt`
- `tables/*_values.txt`
- `tables/*_selection.txt`
- `samples/*_samples.txt`
- `plots/*_bT_average.pdf`

## Recommended TMDWF Downstream Chain

The intended downstream chain is now:

1. `tmdwf-nstate-fit`
2. optional `tmdwf-normalize`
3. `tmdwf-fourier`
4. `tmdwf-cs-kernel`
5. optional `tmdwf-cs-kernel-average`

For the joint effective-surface path, replace steps 4 and 5 with:

4. `tmdwf-cs-kernel-joint`

The data products passed between these steps are:

- `tmdwf-nstate-fit`:
  writes grouped matrix-element outputs in `m0(bT, bz)`, including grouped fit tables and grouped bootstrap sample tables.
- `tmdwf-normalize`:
  reads those grouped `m0` outputs and writes normalized grouped `m0` outputs with the same general structure.
- `tmdwf-fourier`:
  reads either raw grouped `m0` outputs or normalized grouped `m0` outputs, then writes Fourier-space `q(x; pz, bT)` tables and bootstrap samples.
- `tmdwf-cs-kernel`:
  reads those Fourier-space bootstrap outputs and performs the legacy type-2 multi-`Pz` CS-kernel extraction.
- `tmdwf-cs-kernel-average`:
  reads those CS-kernel outputs and averages the x-dependent results over the selected x window for each `bT`.
  It can also filter the source outputs by `pair_mode` and `reference_p1`
  when you need to separate `all`, `adjacent`, or `fixed_p1` extraction runs.
- `tmdwf-cs-kernel-joint`:
  reads Fourier-space bootstrap outputs directly from one or more ensembles
  and fits a shared continuous `gamma_eff(x,bT)` surface.

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
- averaged CS-kernel chain:
  `fit -> normalize(optional) -> fourier -> cs-kernel -> cs-kernel-average`
- joint effective-surface chain:
  `fit -> normalize(optional) -> fourier -> cs-kernel-joint`

## Workflows

Both workflows are supported:

- Input-file workflow:
  use plain-text input files plus the existing CLI/scripts
- Notebook-template workflow:
  copy a notebook from `templates/`, edit the user-input cell, validate it, and run the same backend interactively

Current notebook templates:

- `templates/two_point/tgevp_template.ipynb`
- `templates/two_point/nstate_fit_1state_template.ipynb`
- `templates/two_point/nstate_fit_2state_template.ipynb`
- `templates/tmdwf/tmdwf_nstate_template.ipynb`
- `templates/tmdwf/tmdwf_normalize_template.ipynb`
- `templates/tmdwf/tmdwf_fourier_template.ipynb`
- `templates/tmdwf/tmdwf_cs_kernel_template.ipynb`
- `templates/tmdwf/tmdwf_cs_kernel_joint_template.ipynb`
- `templates/two_point/plot_2pt_template.ipynb`

Notebook templates are thin wrappers around the existing analysis code. They are intended for clarity and interactive use, while the plain-text input-file workflow remains the stable batch-style interface.
Each template now uses the same notebook-facing pattern: edit a single `workflow_config` object, validate it, and then call one `run_*_from_notebook(...)` function.
Each notebook template also includes an `Option Guide` markdown cell right after the user-input cell, describing the main options, their expected choices, and their practical effect.

The helper bridge for notebooks lives in `src/lqcd_analysis/notebook_workflows.py`. Notebook workflows take their configuration directly from the notebook `workflow_config` object and call the same backend runners in the `two_point` and `tmdwf` packages; they do not require a separate input file.
The two workflows use different default output locations by design: notebook workflows default to the current working directory, while plain-text input-file workflows default to a results directory next to the input file.
For `nstatefit`, the user-facing `fit_mode` option supports both `uncorrelated` and `correlated` fits. In correlated mode the code builds one shared covariance matrix from the full bootstrap ensemble and reuses it across the mean fit and bootstrap fits; if a correlated sample/window fit fails, it falls back to a diagonal fit using only the covariance diagonal.
The N-state fit outputs also track this fallback usage: fit tables include a `fallback_uncorrelated_successes` column for each `tmin` window, and summaries report the representative window's fallback count.

## Runnable Examples

The repository now includes tracked example correlator data under:

- `examples/data/l64c64a076_m140/comb_c2pt_csv/`

and plain-text example inputs under:

- `templates/input_files/two_point/tgevp_example_realdata.txt`
- `templates/input_files/two_point/nstate_fit_1state_example_realdata.txt`
- `templates/input_files/two_point/nstate_fit_2state_example_realdata.txt`
- `templates/input_files/tmdwf/tmdwf_nstate_example.txt`
- `templates/input_files/tmdwf/tmdwf_normalize_example.txt`
- `templates/input_files/tmdwf/tmdwf_fourier_example.txt`
- `templates/input_files/tmdwf/tmdwf_cs_kernel_example.txt`
- `templates/input_files/tmdwf/tmdwf_cs_kernel_joint_example.txt`
- `templates/input_files/two_point/plot_2pt_example_command.txt`
- `templates/input_files/two_point/tgevp_example_realdata_annotated.txt`
- `templates/input_files/two_point/nstate_fit_1state_example_realdata_annotated.txt`
- `templates/input_files/two_point/nstate_fit_2state_example_realdata_annotated.txt`
- `templates/input_files/tmdwf/tmdwf_nstate_example_annotated.txt`
- `templates/input_files/tmdwf/tmdwf_cs_kernel_example_annotated.txt`
- `templates/input_files/tmdwf/tmdwf_cs_kernel_joint_example_annotated.txt`
- `templates/tmdwf/tmdwf_nstate_template.ipynb`

These examples use repository-relative paths, so they can be run directly from the repository root.
The `_annotated.txt` variants include inline comments describing each option and are meant to mirror the notebook `Option Guide` cells in plain-text form.
The TMDWF templates are the exception: they are structurally complete templates, but you should point them at your own local HDF5 datasets and two-point fit-summary tables before running.

Example outputs are intentionally ignored by git and should go under:

- `examples/outputs/`

This lets the repository keep realistic data and templates tracked, while avoiding noisy result folders, plots, logs, and notebooks in version control.

## Package Organization

- `lqcd_analysis.common` contains genuinely shared infrastructure such as bootstrap helpers, folding utilities, generic correlator helpers, and fit-table schema parsing.
- `lqcd_analysis.two_point` submodules contain the current two-point analysis implementations, including N-state fitting, effective-mass extraction, plotting, TGEVP, and two-point-specific CSV loading. Import from the submodules directly.
- `lqcd_analysis.tmdwf` submodules contain the current TMDWF ratio-fitting workflow and TMDWF-specific HDF5/data-model helpers. Import from the submodules directly.
