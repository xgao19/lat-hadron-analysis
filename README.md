# Lattice QCD Analysis

Repository scaffold for lattice QCD data analysis, with a Python package layout,
basic correlator utilities, tests, and a clean directory structure for raw and
processed data.

## Planned Scope

- Correlator IO helpers
- Effective mass extraction
- Jackknife and bootstrap analysis
- Fit pipelines for two-point and three-point functions
- Reproducible analysis scripts and notebooks

## Project Layout

```text
.
├── configs/              Example configuration files
├── docs/                 Notes and methodology docs
├── examples/
│   ├── data/             Tracked example correlator CSV files
│   └── outputs/          Ignored example run products
├── scripts/              Reproducible command-line entry points
├── src/lqcd_analysis/    Python source package
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

1. Add project-specific analysis routines under `src/lqcd_analysis/`.
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
python scripts/ss_2pt_tgevp_extract.py ss-2pt-tgevp input_k0_SS.txt --seed 123
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
python scripts/fit_2pt_nstate.py 2pt-nstate-fit configs/example_nstate_fit.txt
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

- `tmax auto` is chosen as the first time slice where the effective-mass relative error reaches 50%, or `Nt/2 - 4`, whichever is smaller.
- The 2-state fit is initialized from the suggested 1-state plateau.
- The 3-state fit is initialized from the suggested 2-state plateau.
- Plateau suggestion looks for the longest contiguous `tmin` window where adjacent ground-state energies are statistically consistent and the fits remain reasonably well behaved.
- If `matplotlib` is unavailable, the code writes a small text note instead of a plot file.
- Plotting is now handled by a reusable module in `src/lqcd_analysis/plotting_2pt.py`.
- Plot outputs convert fitted energies and effective masses from lattice units `E*a` into physical units in MeV using the provided lattice spacing.
- After each 2pt fit run, an editable notebook is written under `results_dir/notebook_plots/`. The notebook calls the reusable plotting module so you can tweak paths, styles, and which state to draw.

## Workflows

Both workflows are supported:

- Input-file workflow:
  use plain-text input files plus the existing CLI/scripts
- Notebook-template workflow:
  copy a notebook from `templates/`, edit the user-input cell, validate it, and run the same backend interactively

Current notebook templates:

- `templates/tgevp_template.ipynb`
- `templates/nstate_fit_template.ipynb`
- `templates/plot_2pt_template.ipynb`

Notebook templates are thin wrappers around the existing analysis code. They are intended for clarity and interactive use, while the plain-text input-file workflow remains the stable batch-style interface.
Each template now uses the same notebook-facing pattern: edit a single `workflow_config` object, validate it, and then call one `run_*_from_notebook(...)` function.
Each notebook template also includes an `Option Guide` markdown cell right after the user-input cell, describing the main options, their expected choices, and their practical effect.

The helper bridge for notebooks lives in `src/lqcd_analysis/notebook_workflows.py`. It renders notebook configs into the same or nearly the same text fields used by the existing input-file parsers, and then calls the same backend runners.
The two workflows use different default output locations by design: notebook workflows default to the current working directory, while plain-text input-file workflows default to a results directory next to the input file.
For `nstatefit`, the user-facing `fit_mode` option supports both `uncorrelated` and `correlated` fits. In correlated mode the code builds one shared covariance matrix from the full bootstrap ensemble and reuses it across the mean fit and bootstrap fits; if a correlated sample/window fit fails, it falls back to a diagonal fit using only the covariance diagonal.
The N-state fit outputs also track this fallback usage: fit tables include a `fallback_uncorrelated_successes` column for each `tmin` window, and summaries report the representative window's fallback count.

## Runnable Examples

The repository now includes tracked example correlator data under:

- `examples/data/l64c64a076_m140/comb_c2pt_csv/`

and plain-text example inputs under:

- `templates/input_files/tgevp_example_realdata.txt`
- `templates/input_files/nstate_fit_example_realdata.txt`
- `templates/input_files/plot_2pt_example_command.txt`
- `templates/input_files/tgevp_example_realdata_annotated.txt`
- `templates/input_files/nstate_fit_example_realdata_annotated.txt`

These examples use repository-relative paths, so they can be run directly from the repository root.
The `_annotated.txt` variants include inline comments describing each option and are meant to mirror the notebook `Option Guide` cells in plain-text form.

Example outputs are intentionally ignored by git and should go under:

- `examples/outputs/`

This lets the repository keep realistic data and templates tracked, while avoiding noisy result folders, plots, logs, and notebooks in version control.
