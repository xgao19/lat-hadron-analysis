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
- TMDWF mode3 sample normalization uses each bootstrap target divided by the matching pz=0 same-bT sample, then multiplies by one central complex factor from pz=0,bT=0 over same-pz,bT=0 references.
- CS kernel averaged central values should use the bootstrap median with percentile half-widths, not the arithmetic mean.

## Useful Checks
- Targeted tests: `python3 -m pytest tests/test_emff_fit.py`
- Targeted TMDWF CS kernel tests: `python3 -m pytest tests/test_tmdwf_cs_kernel_average.py`
- Compile EMFF code: `python3 -m compileall -q src/lqcd_analysis/emff src/lqcd_analysis/notebook_workflows.py`
- Validate EMFF template notebooks with `python3 -m json.tool`.
