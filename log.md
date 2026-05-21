# Change Log

Update this file before every commit. Keep each entry short: one or two concise paragraphs at most.

## Uncommitted - EMFF HDF5 Ratio Workflow

Migrated the EMFF ratio workflow to read 2pt correlators directly from HDF5, use the Eq. (6) ratio from arXiv:2102.06047, compute energies with `hadron_mass_gev` through the dispersion relation, average transverse momentum orbits at the raw-correlator level before constructing ratios, and expose the workflow through the CLI.

Updated the active analysis notebook, repo templates, README, `AGENTS.md`, `SESSION_MEMORY.md`, and this log to match the backend workflow and commit documentation practice. Verified targeted EMFF tests, module compilation, template JSON validity, and successful execution of the active analysis notebook.
