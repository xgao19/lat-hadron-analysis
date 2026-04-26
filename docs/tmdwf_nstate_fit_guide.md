# TMDWF N-State Fit Guide

This note collects the current best-practice workflow for the TMDWF `n-state` ratio-fit pipeline in this repository.

The goal is practical use:

- run the fit in a stable order
- choose the two-point reference windows with clear intent
- understand how the TMDWF ratio is built from the correlators and the two-point fit
- use `decay_constant_check` as a diagnostic tool for fit-window selection
- avoid the common failure modes that come from poor two-point windows or excited-state contamination

Relevant templates:

- `templates/tmdwf/tmdwf_nstate_template.ipynb`
- `templates/input_files/tmdwf/tmdwf_nstate_example.txt`
- `templates/input_files/tmdwf/tmdwf_nstate_example_annotated.txt`

## What The Workflow Fits

The TMDWF `n-state` workflow fits a ratio:

```text
ratio(t) = TMDWF_numerator(t) / two_point_correlator(t)
```

The numerator is built from the extracted TMDWF matrix element and the selected operator channel:

- `T5` uses the `gamma_t gamma_5` form
- `Z5` uses the `gamma_z gamma_5` form and includes the lattice-momentum factor `Pz / E_i`

The denominator is the two-point correlator, folded according to the requested folding mode.

The fit parameters are the matrix elements `m_i` for the chosen `nstates`.
In practice:

- `m0` is the ground-state matrix element
- `m1` is the first excited-state matrix element
- the fit is done on bootstrap samples
- the reported central values come from the bootstrap-centered fit results

## Recommended Workflow

1. Pick a stable two-point reference window for each momentum.
2. Run the TMDWF `n-state` fit.
3. Inspect the grouped fit summary and the `m0` vs `bz` plots.
4. If you want a decay-constant-style diagnostic, turn on `decay_constant_check`.
5. Use the scan in `decay_constant_check` mode to pick a better `fit_window`.
6. If the scan still does not stabilize the decay constant, revisit the two-point reference windows.

## Main Inputs

The TMDWF workflow uses three groups of inputs.

### 1. Data Settings

These settings define what raw data is read.

- `title_pattern`
- `pzlist`
- `gmlist`
- `etalist`
- `Tdirlist`
- `bTlist` / `bTrange`
- `bzlist` / `bzrange`
- `qtmdwf_h5`
- `dataset_path_template`
- `tsrange`
- `decay_constant_check`

Important notes:

- `qtmdwf_h5` may use `{gm}` and `{pz}` placeholders through the notebook helper.
- `dataset_path_template` may use `{gm}`, `{eta}`, `{pz}`, `{Tdir}`, `{bT}`, and `{bz}`.
- `decay_constant_check = true` forces `bT = 0` and `bz = 0`.
- In `decay_constant_check` mode, the workflow uses only the real part.

### 2. Two-Point Correlator Input

This block provides the denominator correlator and the reference two-point fit windows.

- `c2pt`
- `fold_t`
- `two_point_fit_root`
- `two_point_fit_window_by_pz`

This is the most important human-input step.

You usually want to:

- inspect the two-point plateau as `tmin` changes
- find a stable plateau region
- choose a good `tmin` inside that plateau
- keep `tmax` reasonable, but `tmax` is usually less critical than `tmin`

The workflow uses this two-point fit table to supply fixed amplitudes and energies for the TMDWF ratio model by default.
If you enable `two_point_fit_sample_coupled`, it instead reads the matching bootstrap sample row from the two-point sample table for each TMDWF bootstrap sample.

### 3. TMDWF Fit Settings

These settings control the nonlinear fit itself.

- `fit_target` should stay `ratio`
- `fit_component` is normally `both`
- `nstates` is usually `1`, `2`, or both
- `fit_window`
- `binsize`
- `bootstrap_samples`
- `bootstrap_size`
- `seed`
- `plot`
- `results_dir`

In normal ratio-fit mode, `fit_component = both` means the workflow fits real and imaginary parts separately.

In `decay_constant_check` mode:

- the workflow always uses the real part
- it ignores `bTlist` and `bzlist`
- it only uses `bT = bz = 0`
- it scans nearby fit windows around the requested `fit_window`

## Fit-Window Logic

The `fit_window` input is the main tuning knob for the TMDWF ratio fit.

In normal mode:

- each momentum has a base `[tmin, tmax]` window
- the fit uses that window directly
- `tmin` is usually the primary choice
- `tmax` is secondary, but it should still stay in a region where the TMDWF/2pt ratio does not show obvious loss of signal or strange behavior
- the best `tmax` for the TMDWF ratio fit can differ from the two-point fit `tmax`

In `decay_constant_check` mode:

- the workflow takes the base `[tmin, tmax]`
- it scans `tmin - 2` to `tmin + 2`
- it scans `tmax - 2` to `tmax`
- only valid windows inside the retained time range are kept, with `tmin + 1 < tmax` so each candidate window has at least 3 points
- the two-point reference `tmin` also scans `tmin - 1` to `tmin + 1` around the row selected by `two_point_fit_window_by_pz`
- conceptually, the scan is organized as: pick a two-point reference `tmin`, then scan the TMDWF `fit_window` around it

This scan is the reason the new `decay_constant_check` mode exists.
In the ideal continuum-like limit, the extracted decay constant should not depend strongly on `pz` or the ensemble.
If it changes, the most common practical culprit is excited-state contamination.
Scanning the fit window helps you see which choice moves the result closest to the expected value.

## Physical Interpretation

The workflow is extracting the matrix element from a ratio.

For the diagnostic mode:

- the ground-state matrix element is reported in physical units, `GeV`
- the conversion uses the lattice spacing
- the printed quantity is the decay constant estimate in the current implementation

The code converts the fitted lattice-unit value with:

```text
value_GeV = value_lattice * (hbar*c in GeV fm) / a_fm
```

So the output is not just a raw lattice number.
It is the physical value used to compare different momenta and ensembles.

## Output Products

The normal TMDWF `n-state` workflow writes grouped outputs by `bT`.

For each `(title, gm, eta, bT, component, nstates)` combination, it writes:

- `*_summary.txt`
- `tables/*_fit.txt`
- `samples/*_samples.txt`
- `tables/*_curve.txt`
- optional `plots/*_ratio_fit.pdf`

For `decay_constant_check` mode, it additionally writes:

- `*_decay_constant_check_summary.txt`

The diagnostic summary records:

- fixed `bT = 0` and `bz = 0`
- the base `tfit`
- a separate base-fit block selected from the sweep results, using the same row format as the sweep table
- the scanned `tmin` and `tmax`
- the associated two-point fit `tmin`
- the decay constant in GeV
- its relative bootstrap uncertainty
- `chi2_dof`
- `pvalue`
- the `m0` vs `bz` plot only uses even `bT` values
- the associated two-point fit provenance

The console output also prints the same diagnostic information in a compact line per scanned window.

## What Needs Human Judgment

There are two places where human judgment matters most.

### 1. Choosing `two_point_fit_window_by_pz`

This is the first and usually most important decision.

What to look for:

- a stable plateau in the two-point fit as `tmin` changes
- a `tmin` inside that plateau where the signal is still good
- a `tmax` that does not run too far into the noise-dominated region

Recommended heuristic:

- `tmin` matters more than `tmax`
- pick a plateau row with decent central values and reasonable errors
- if the selected TMDWF result behaves poorly, try a nearby two-point window first

### 2. Choosing `fit_window` for `decay_constant_check`

This is the main reason the new diagnostic mode exists.

What to look for:

- the decay constant should be as flat as possible across nearby windows
- the best window is usually the one closest to the `pz = 0` result
- if several windows behave well, prefer the one closest to the `pz = 0` result
- `chi2` should not be in the worst-looking group among the nearby candidates
- the relative error should also not be in the worst-looking group among the nearby candidates
- for large `z`, the real part should not become overly negative, especially at large `tmax`
- if the trend looks wrong at large `tmax`, shrink `tmax` first even when the decay constant is already close to the ideal value
- `tmin` is still the first thing to tune
- `tmax` is a secondary tuning knob, but it should still avoid the region where the ratio becomes noisy or visibly distorted

Recommended heuristic:

- first compare windows around the same two-point reference
- if every scan point still looks bad, change the two-point reference window and try again
- if the result is still unstable after that, the data may simply be too noisy or too contaminated to support a clean extraction
- for the initial `fit_window`, usually start with a `tmin` equal to the chosen two-point `tmin` or a little larger, and choose `tmax` by requiring the TMDWF/2pt ratio to stay smooth and non-odd
- when the decay constant is already close to the ideal value, still prefer smaller `tmax` if the large-`z` real part starts to drift negative or loses the expected trend

In other words:

- fix the two-point reference first
- then tune the TMDWF `fit_window`
- only after that consider changing the underlying data selection strategy

## Practical Troubleshooting

- If the fit fails for many windows, check the two-point reference first.
- If the decay constant changes strongly with `tmin`, the early-time region is probably still contaminated by excited states.
- If the decay constant changes strongly with `tmax`, the late-time region may be too noisy or the ratio may have entered a region with odd behavior.
- If the decay constant improves noticeably after changing the two-point reference `tmin`, the original two-point plateau row was probably not the best choice.
- If the result is unstable across all scanned windows, try a different two-point plateau row.
- If that still does not help, the ensemble or momentum channel may simply be too noisy for a clean check.

## Summary

The TMDWF `n-state` workflow is a ratio fit with fixed two-point inputs.
The two most important human decisions are:

1. choosing a stable two-point reference window
2. choosing a TMDWF fit window that suppresses excited-state contamination

`decay_constant_check` is the diagnostic mode that makes the second step much easier.
It isolates `bT = bz = 0`, uses only the real part, scans nearby windows, and reports the decay constant in GeV so you can compare against the expected ideal behavior.
