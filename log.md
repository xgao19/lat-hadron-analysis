# Change Log

Update this file before every commit. Keep each entry short: one or two concise paragraphs at most.

## Uncommitted - TMDWF Mode3 Normalization

Updated TMDWF mode3 normalization so bootstrap samples use the pz=0 same-bT denominator with a central complex bT=0 correction factor, and reused the same rule in x-fit normalization. CS kernel average summaries now report bootstrap medians with percentile half-widths.

Added focused tests for the median-centered CS kernel average and the revised mode3 sample outputs. Verified TMDWF module compilation and the CS kernel average test file; the broader `tests/test_tmdwf_fit.py` collection is currently blocked by its stale `fit_tmdwf_mean_component` import.
