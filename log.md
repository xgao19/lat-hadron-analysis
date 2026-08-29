# Change Log

Update this file before every commit. Keep each entry short: one or two concise paragraphs at most.

## Uncommitted - TMDWF Mode3 Sample Normalization

Changed `tmdwf-normalize` and `tmdwf-xfit-normalize` mode3 back to the full sample-by-sample ratio of all four matrix-element factors, removing the central-factor shortcut. Updated README guidance, durable memory, and the normalization regression test; restored the mean-only TMDWF helper expected by the focused fit tests.

## Uncommitted - Experimental TMDWF Adjacent-b Difference Fit

Added an experimental `tmdwf-nstate-diff-fit` workflow with paired-bootstrap adjacent-b edge fits, graph reconstruction, downstream-compatible grouped fit/sample outputs, diagnostics, CLI dispatch, notebook helpers, templates, and focused tests.

Updated README guidance to keep direct `tmdwf-nstate-fit` as the recommended production path while documenting the new diff workflow as a diagnostic and method-comparison option.

## Uncommitted - TMDWF Joint CS Kernel Corrections

Updated the TMDWF joint CS-kernel correction model so nuisance terms shift `gamma_MSbar` additively inside the evolution exponent. Added an independent symmetric `1/Pz` channel, optional omission of all correction priors, ordered correction-parameter/output blocks, and unambiguous labels for fixed-reference multi-momentum extraction.

Retained x-reflection planning so x and 1-x share one independent fit while the workflow still writes mirrored outputs for all requested x values. Updated summaries, diagnostics notebooks, README guidance, and focused tests for pz1-only, pz2-only, combined correction blocks, no-prior fits, output files, and legacy pz2 behavior.
