# Examples

This directory contains tracked example input data and ignored run outputs.

## Layout

- `data/`: tracked example correlator CSV files copied into the repository
- `outputs/`: ignored run products produced by example workflows

## Directly Runnable Example Workflows

Notebook-based templates:

- `templates/tgevp_template.ipynb`
- `templates/nstate_fit_1state_template.ipynb`
- `templates/nstate_fit_2state_template.ipynb`
- `templates/plot_2pt_template.ipynb`

Input-file-based templates:

- `templates/input_files/tgevp_example_realdata.txt`
- `templates/input_files/nstate_fit_1state_example_realdata.txt`
- `templates/input_files/nstate_fit_2state_example_realdata.txt`

The templates are configured to use repository-relative paths so they can be run directly from the repository root.
The n-state templates use `low_state_prior_tmin` to select the lower-order fit-table row used as the prior source for higher-state fits, and they expect the previous-state output directory to already exist when you launch a higher-state run.

Prior behavior in the current n-state workflow:

- `1-state`: no low-state prior is used.
- `2-state`: the code reads the exact `tmin` row selected by `low_state_prior_tmin[pz]` from the previous `1-state` fit table and uses that row to build the `E0` prior.
- `3-state`: the code reads the exact `tmin` row selected by `low_state_prior_tmin[pz]` from the previous `2-state` fit table and uses that row to build the `E0` and `E1` priors.
- `pz0_ground_energy` supplies the dispersion target for the initial `E0` guess, and becomes the fixed `E0` only when `fix_ground_energy_from_dispersion` is enabled.
- `fix_ground_energy_from_dispersion` only controls whether that dispersion target is used as a fixed ground-state energy during the fit.
- `lambda_prior` only scales the prior residuals once the prior row has been selected.
- The higher-state notebook does not rerun the lower-state fit; it reads the lower-state table from the matching state-specific output directory.
