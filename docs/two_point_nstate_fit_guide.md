# Two-Point N-State Fit Guide

This note collects the current best-practice workflow for the two-point `n-state` fit pipeline in this repository.

The goal is practical use:

- run the fit in a stable order
- choose the most important parameters with clear intent
- understand how the low-state prior is selected
- avoid the common failure modes that come from overfitting or unstable windows

Relevant templates:

- `templates/two_point/nstate_1.ipynb`
- `templates/two_point/nstate_2.ipynb`
- `templates/input_files/two_point/nstate_1.txt`
- `templates/input_files/two_point/nstate_2.txt`

## Recommended Workflow

1. Run `1-state` first.
2. Inspect the `1-state` fit table and summary.
3. Pick a stable `tmin` row for the next stage by setting `low_state_prior_tmin[pz]`.
4. Run `2-state`.
5. If you want `3-state`, repeat the same pattern from the `2-state` table.

This order matters because the higher-state fit reads the lower-state fit table from the matching previous-state output directory as its prior source.
It does not rerun the lower-state fit for you, so make sure the lower-state output directory is already present before launching the higher-state run.

## Typical Ensemble Workflow

When I start from a new ensemble, I usually follow this sequence:

1. Run a `1-state` fit for `pz = 0` first.
2. Use the `pz = 0` plateau to identify a stable ground-state mass.
3. Choose the plateau energy either by:
   - averaging the plateau region, or
   - taking the point whose fitted energy is closest to the plateau center.
4. Run `1-state` fits for all momenta with `pz0_ground_energy` provided, but keep `fix_ground_energy_from_dispersion = false`.
5. Plot the dispersion-based ground-state energy as a reference line in the energy and effective-mass plots.
6. Check that the dispersion-based energy lies within the plateau uncertainty band for the other momenta.
7. Turn `fix_ground_energy_from_dispersion = true` and run the `2-state` fit.
8. If a `3-state` fit is really needed, keep `E0` fixed in the same way and build the `E1` prior from the `2-state` plateau.

This is the workflow I use because it gives a clean separation between:

- identifying the ground-state mass
- checking the dispersion relation
- locking `E0`
- constraining only the excited-state energies with priors

In practice:

- `2-state` windows are usually narrower than the `1-state` windows
- `3-state` windows are usually narrower than the `2-state` windows
- the right edge of the higher-state window is often only one or two time slices beyond the point where the lower-state plateau begins
- amplitudes are usually left unconstrained
- the energies are the quantities that most often need explicit control
- the mean-fit warm start is anchored at the first third of the `tmin` scan window instead of chaining from every previous window
- the previous-state output root should already exist before you launch the higher-state fit
- the plotted energy and effective-mass axes are shown in GeV in the current workflow

## Quick Decision Map

| Parameter | What it controls | Typical choice |
| --- | --- | --- |
| `tmin_window` | Which `tmin` values are scanned | Start conservative, then narrow around the stable plateau |
| `tmax` | Upper edge of the fit window | Fix it once the late-time noise starts to dominate |
| `pz0_ground_energy` | Dispersion anchor for the initial `E0` guess, and for fixed `E0` when enabled | Set from the trusted `pz = 0` ground-state estimate |
| `fix_ground_energy_from_dispersion` | Whether `E0` is fixed during the fit | `false` for initial checking, `true` for the production `2-state` and `3-state` fits |
| `low_state_prior_tmin` | Which lower-state row becomes the prior source | Pick a stable plateau row with a sensible energy center and moderate uncertainty |
| `lambda_prior` | Prior strength | Keep at `1.0` unless you need a stronger or weaker prior |
| `nstates` | Model complexity | Use the smallest value that still captures the data |
| `fit_mode` | Diagonal vs correlated fit | Prefer `correlated` when the covariance is stable |
| `shrinkage_lambda` | Covariance stabilization | Usually leave it to the automatic candidate order |
| `A_n` | Amplitude parameters | Usually leave unconstrained |
| `E_n` | Energy parameters | The main quantities to constrain with fixed anchors or priors |

## Parameter Hierarchy

The current workflow has three layers of decision-making:

1. Data and scan setup
2. Fit model and fit window
3. Prior and anchoring settings

Keep those layers separate when you tune a run. Most confusion comes from mixing them together.

## The Most Important Parameters

### `tmin_window`

This is the main scan control.

- It defines the allowed `tmin` range for each momentum.
- The fit scans within that range rather than fitting a single hand-picked point.
- Choose it to exclude obviously noisy early-time data while keeping enough points for a stable fit.

Practical rule:

- start with a conservative window
- prefer a window that gives stable parameters across nearby `tmin` values
- avoid choosing a window that is too wide just because it runs

### `tmax`

This fixes the upper edge of the fit window.

- Too small: you may throw away useful signal.
- Too large: you may include noise-dominated late-time data.

Practical rule:

- keep `tmax` fixed per momentum once you have a reasonable choice
- do not let `tmax` float just to improve the fit numerically

### `nstates`

This should be treated as the complexity budget of the model.

- `1-state`: baseline extraction of the ground state
- `2-state`: first excited correction
- `3-state`: more flexible, but easier to overfit

Practical rule:

- use the smallest `nstates` that captures the data
- do not jump to a larger state count unless the lower-state fit is clearly inadequate

### `fix_ground_energy_from_dispersion`

This decides whether the ground-state energy is anchored by the dispersion relation.

- `true`: the fit fixes `E0` to the dispersion target derived from `pz0_ground_energy`
- `false`: `E0` is left free

Even when this flag is `false`, the current workflow still uses the dispersion target as the initial `E0` guess whenever `pz0_ground_energy` is provided.

Practical rule:

- keep it `true` when the dispersion anchor is trusted
- turn it off only when you explicitly want the fit to determine `E0` from the correlator

### `pz0_ground_energy`

This is the pz=0 anchor used to build the dispersion target.

- It sets the initial `E0` guess whenever it is provided.
- It becomes the fixed `E0` value only when `fix_ground_energy_from_dispersion=true`.
- It is not a prior row selector.
- It is a physics anchor for the ground state.

Practical rule:

- set it only when you trust the reference value
- do not confuse it with `low_state_prior_tmin`

### `low_state_prior_tmin`

This is the key higher-state prior selector.

- It is a momentum-indexed dictionary.
- For each momentum `pz`, it points to one exact `tmin` row in the previous lower-state fit table.
- That row becomes the prior source for the next state count.

Current behavior:

- `2-state` reads one row from the previous `1-state` table
- `3-state` reads one row from the previous `2-state` table

Practical rule:

- choose a row that is stable, not borderline
- prefer a row inside the low-state plateau region
- prefer a row whose energy center is close to the dispersion expectation
- avoid using a row with obviously huge uncertainty
- avoid using a row with visible fit instability or tiny bootstrap success

### `lambda_prior`

This controls the strength of the prior residual.

- `1.0` is the default
- larger values make the prior stronger
- `0.0` disables the prior contribution

Practical rule:

- keep the default unless you have a reason to strengthen or weaken the prior
- tune the prior row first before changing `lambda_prior`

### `fit_mode`

This decides whether the fit uses the full covariance matrix.

- `uncorrelated`: diagonal errors only
- `correlated`: full covariance fit

Practical rule:

- prefer `correlated` when the covariance matrix is well behaved
- if the fit becomes numerically fragile, let the shrinkage machinery stabilize it

### `A_n` versus `E_n`

In this workflow I usually constrain the energies much more than the amplitudes.

- `E_n` are the main quantities that benefit from priors or fixed anchoring.
- `A_n` are often stable enough to leave free.

Practical rule:

- if you need to simplify the fit, start by controlling the energies
- only add amplitude constraints if you have a specific instability that requires them

### `shrinkage_lambda`

This is used internally to stabilize the covariance matrix.

The current candidate list is tried in order:

- `0.0`
- `0.1`
- `0.2`
- `0.3`
- `0.5`
- `1.0`

Interpretation:

- `0.0` means no shrinkage
- `1.0` means keep only the diagonal
- intermediate values trade off stability and correlation information

Practical rule:

- do not tune this first
- rely on the automatic order unless you have a specific covariance issue

## How Prior Selection Works Now

The current logic is simple and explicit:

1. You choose `low_state_prior_tmin[pz]`.
2. The code loads that exact row from the previous lower-state fit table.
3. The row is converted into one or more energy priors.
4. `lambda_prior` scales the prior residuals.

The important consequence is that the prior no longer comes from an automatically aggregated fit window. It comes from one row that you selected on purpose.

### What each mode gets

- `1-state`: no low-state prior
- `2-state`: prior on `E0`
- `3-state`: priors on `E0` and `E1`

If `fix_ground_energy_from_dispersion=true`, then `E0` is fixed by the dispersion anchor. In that case, the `E0` prior still contributes to the residual bookkeeping, but it no longer changes the fitted value of `E0`.

## Recommended Selection Strategy

When choosing the prior row from a lower-state fit table:

1. Start from the lower-state scan output.
2. Look for a row with stable parameters and a reasonable `chi2/dof`.
3. Prefer a row that sits on the plateau rather than an early unstable point.
4. Use the same momentum’s row, not a row borrowed from another momentum.

If the fit table contains multiple acceptable rows, prefer the row that is easiest to justify physically, not just the row with the smallest numerical `chi2/dof`.

## Typical Failure Modes

### Overly aggressive `tmin_window`

Symptoms:

- unstable energies
- wildly varying amplitudes
- poor bootstrap success

Fix:

- move the window later
- narrow the scan
- reduce the state count if needed

### Prior row chosen too early

Symptoms:

- prior pulls the fit toward an unstable low-state result
- higher-state fit becomes sensitive to the prior

Fix:

- move `low_state_prior_tmin` to a more stable row

### Too many states

Symptoms:

- parameters blow up
- excited-state energies become unphysical
- bootstrap samples become unstable

Fix:

- drop back to a lower `nstates`
- only add a state when the data clearly support it

### Covariance instability

Symptoms:

- correlated fit fails
- shrinkage has to fall back to diagonal behavior

Fix:

- keep the correlated fit, but let the shrinkage sequence stabilize it
- if the problem persists, simplify the fit window

## What To Trust

Trust these outputs in roughly this order:

1. Stability across nearby `tmin`
2. Bootstrap success rate
3. `chi2/dof` and `p-value`
4. Physical plausibility of the extracted energies
5. Consistency across momenta

Do not trust a single number in isolation.

## Short Version

If you only remember one thing:

- use `1-state` to find a stable low-state row
- set `low_state_prior_tmin[pz]` explicitly from that row
- keep `tmin_window` conservative
- keep `nstates` as small as possible
- use `fix_ground_energy_from_dispersion` only when the dispersion anchor is trusted
