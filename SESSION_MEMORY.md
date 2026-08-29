# Session Memory

Durable project memory for future sessions. Keep this file concise and focused on reusable practices, pitfalls, fixes, and key project conventions.

## Practices
- Keep analysis workflows slim: one canonical path per task, no compatibility aliases, no fallback branches unless explicitly requested.
- Before every commit, update `SESSION_MEMORY.md` and `log.md`.
- Keep `log.md` entries short: one or two concise paragraphs at most.
- Code comments must be English only.
- Do not revert unrelated user changes; inspect dirty files before editing nearby code.

## EMFF Project Memory
- EMFF 2pt correlators are read directly from HDF5, not CSV. Canonical reader: `src/lqcd_analysis/emff/io.py::load_emff_c2pt_correlator`.
- Expected 2pt HDF5 dataset path pattern: `SS/{sink_gamma}/PX{px}PY{py}PZ{pz}`.
- Source gamma is encoded in the 2pt filename, for example `src5`; sink gamma and momentum are encoded in the HDF5 path.
- Current EMFF ratio uses Eq. (6) of arXiv:2102.06047 with `Pi = Pf - q`.
- Energies use `E(P)=sqrt(hadron_mass_gev^2 + |P|^2)`. Keep the mass input named `hadron_mass_gev`, not pion-specific.
- Transverse orbit averaging should average raw `C3pt` and corresponding initial-state `C2pt` before ratio construction.
- Do not merge `qz` with `-qz` by default; longitudinal momentum transfer can change energy when `Pf` is nonzero.

## Pitfalls And Fixes
- If old CSV 2pt logic appears after the HDF5 migration, remove it instead of keeping a parallel path.
- Avoid vague names like "HDF5 patterns" for EMFF data after both 2pt and 3pt became HDF5; prefer specific names such as 2pt HDF5 path or 3pt HDF5 path.
- Do not average already-built ratios for momentum orbits; average correlators first to preserve the intended estimator.
- Notebook files under `/Users/xiang/Desktop/docs/...` are outside the repo and usually need escalated write permission.
- TMDWF mode3 normalization uses all four factors sample-by-sample: `(target / same-pz bT0 bz0) / (pz0 same-bT bz0 / pz0 bT0 bz0)`, in the complex plane when both real and imag samples exist.
- TMDWF adjacent-b n-state difference fitting is experimental only; keep `tmdwf-nstate-fit` as the default production matrix-element extraction path. The diff workflow writes downstream-compatible grouped fit/sample tables plus edge, graph, and plaquette diagnostics.
- CS kernel averaged central values should use the bootstrap median with percentile half-widths, not the arithmetic mean.
- TMDWF joint CS-kernel corrections enter as additive shifts to `gamma_MSbar` inside the evolution exponent. Use fixed reference scales `a0=0.1 fm`, `b0=1.0 fm`, and `p0=1.0 GeV` so a2, inverse-bT, pz1, and pz2 nuisance coefficients stay dimensionless; correction priors may be disabled explicitly.
- TMDWF joint CS-kernel x reflection fits one representative from each x and 1-x pair, then mirrors outputs back to the requested x grid.

## Useful Checks
- Targeted tests: `python3 -m pytest tests/test_emff_fit.py`
- Targeted TMDWF CS kernel tests: `python3 -m pytest tests/test_tmdwf_cs_kernel_average.py`
- Compile EMFF code: `python3 -m compileall -q src/lqcd_analysis/emff src/lqcd_analysis/notebook_workflows.py`
- Validate EMFF template notebooks with `python3 -m json.tool`.
