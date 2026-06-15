# Change Log

Update this file before every commit. Keep each entry short: one or two concise paragraphs at most.

## Uncommitted - TMDWF Joint CS Kernel Corrections

Updated the TMDWF joint CS-kernel correction model so nuisance factors multiply `gamma_MSbar` inside the evolution exponent with fixed dimensionless reference scales for a2, inverse-bT, and pz2 shapes. Summary outputs now record those scales and the independent/output x-fit counts.

Added x-reflection planning so x and 1-x share one independent fit while the workflow still writes mirrored outputs for all requested x values. Added focused tests for the correction formula, summary metadata, and reflected x outputs.
