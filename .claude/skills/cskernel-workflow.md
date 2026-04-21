---
name: cskernel-workflow
description: Complete guide for TMDWF and CS-kernel analysis workflow across ensembles
author: Claude Code
version: 1.0
tags: [lqcd, tmdwf, cskernel, workflow, ensemble]
---

# CS-kernel Analysis Workflow Skill

## Command
`/cskernel-workflow` or `/tmdwf-analysis`

## Description
This skill provides a complete, experience-based guide for executing the TMDWF to CS-kernel extraction workflow across LQCD ensembles. Based on successful completion of `analysis_l64c64a076_m140_src5`, it includes all learned patterns, pitfalls, and verification steps.

## When to Use
- When starting a new ensemble analysis
- When encountering errors in the workflow
- When verifying step-by-step progress
- When checking parameter configurations

## Quick Start
1. Navigate to the ensemble directory: `cd /Users/xiang/Desktop/docs/0-2026/projects/1-CSkernel/analysis_l*_src5`
2. Run the skill: `/cskernel-workflow`
3. Follow the step-by-step guide below

## Complete Workflow Guide

### 1. Ensemble Identification
First, extract parameters from the directory name:
- Format: `l{ns}c64a{spacing}_m140_src5`
- Example: `l48c64a060` → `ns=48`, `lattice_spacing_fm=0.060`

### 2. Pre-flight Checklist
Before starting, verify:
```bash
# Check data files exist
ls /Users/xiang/Desktop/docs/0-2025/projects/3-CSkernel/data/{ensemble_name}/

# Expected structure:
# comb_c2pt_csv/    # Two-point correlator CSVs
# comb_qTMDWF/      # TMDWF HDF5 files
```

### 3. Strict Execution Order
**DO NOT DEVIATE FROM THIS ORDER:**

1. **Two-point 1-state fit** (`1-c2pt-fit/nstate_fit_k*.ipynb`)
2. **Two-point 2-state fit** (`1-c2pt-fit/nstate_fit_k*_2state.ipynb`)
3. **TMDWF 2-state fit** (`2-bm/tmdwf_nstate_k*.ipynb`)
4. **Mode2 normalization** (`3-normalize/tmdwf_normalize_template.ipynb`)
5. **Fourier transform** (`4-FT/tmdwf_fourier_template.ipynb`)
6. **CS-kernel extraction** (`5-CSkernel/tmdwf_cs_kernel_run.ipynb`)

### 4. Critical Configuration Changes

#### Universal Changes (all notebooks)
- `title_pattern`: `l{ns}c64a{spacing}_m140_fit_pz*` (remove `_k0`/`_k6`)
- `results_dir`: Remove `_k0`/`_k6` suffixes

#### Path Dependencies
```
TMDWF fit: two_point_fit_root → ../1-c2pt-fit/results_nstate_fit_2state
Normalize: input_root → ../2-bm/results_fit_nst2
Fourier: input_root → ../3-normalize
CS-kernel: input_root → ../4-FT
```

#### Ensemble-Specific Parameters
**Must adjust for each ensemble:**
- `ns`, `nt`: Spatial/temporal lattice size
- `lattice_spacing_fm`: From directory name (0.040, 0.050, 0.060, 0.076)
- `pzlist`: Check actual momentum data available
- `two_point_fit_window_by_pz`: Adjust based on data quality
- `fit_window`: Adjust for TMDWF fits

#### Data File Paths
Replace template paths in notebooks:
- `c2pt`: `/Users/xiang/Desktop/docs/0-2025/projects/3-CSkernel/data/{ensemble_name}/comb_c2pt_csv/...`
- `qtmdwf_h5`: `/Users/xiang/Desktop/docs/0-2025/projects/3-CSkernel/data/{ensemble_name}/comb_qTMDWF/...`

### 5. Key Pitfalls & Solutions

#### Pitfall 1: k0/k6 naming confusion
**Symptom**: `FileNotFoundError` for `*_k0_*` or `*_k6_*` files
**Solution**: Use unified naming without k-distinction in `title_pattern` and `results_dir`

#### Pitfall 2: Mode2 normalization fails
**Symptom**: Missing pz=0 reference data
**Solution**: `pzlist` in normalize must include `[0] + high_pz_list`

#### Pitfall 3: Path linking errors
**Symptom**: Cannot find previous step outputs
**Solution**: Verify each `input_root`/`two_point_fit_root` points to correct relative path

#### Pitfall 4: Parameter mismatch
**Symptom**: Dimension errors or empty outputs
**Solution**: Double-check `ns`, `nt`, `lattice_spacing_fm` match ensemble

### 6. Step-by-Step Verification

#### After Two-point fits:
```bash
ls 1-c2pt-fit/results_nstate_fit_1state/
ls 1-c2pt-fit/results_nstate_fit_2state/
# Should see per-pz directories with tables/ subdirectories
```

#### After TMDWF fit:
```bash
ls 2-bm/results_fit_nst2/
# Should have summary.txt, tables/, samples/, plots/
```

#### After Normalize:
```bash
ls 3-normalize/
# All requested pz should have directories, including pz=0
```

#### After Fourier:
```bash
ls 4-FT/
# Each (pz,bT) should have .txt, _samples.txt, and .pdf
```

#### After CS-kernel:
```bash
ls 5-CSkernel/
# Should have l{ns}c64a{spacing}_m140_fit_pzmultiPz/ with diagnostics/, tables/, etc.
```

### 7. Momentum Configuration Guide

#### Typical configurations:
- **k0 dataset**: Usually has pz=0 only
- **k6 dataset**: Usually has pz=5,6,7,8 (check actual data)
- **Normalize**: `pzlist = [0, 5, 6, 7, 8]` (must include 0)
- **Fourier/CS-kernel**: `pzlist = [5, 6, 7, 8]` (only high momenta)

### 8. Error Diagnosis Flowchart

```
Error occurs → Check error message
    ↓
FileNotFoundError → Check: 1) Path exists? 2) Previous step output? 3) title_pattern match?
    ↓
ParameterError → Check: 1) ns/nt match? 2) pzlist in range? 3) spacing correct?
    ↓
RuntimeError → Check: 1) Data file integrity? 2) Memory limits? 3) Python dependencies?
    ↓
Success but no output → Check: 1) Notebook cell executed? 2) Logs show processing?
```

### 9. Optimization Tips

#### For faster testing:
- Reduce `bootstrap_samples` from 200 to 50
- Test with single pz first
- Test with `bTlist = [0, 1]` only

#### For production runs:
- Use full `bTlist = list(range(0, 21))`
- Use full `bzlist = list(range(0, 21))`
- `bootstrap_samples = 200` for good statistics

### 10. Final Deliverables Checklist

Each completed ensemble should have:
- [ ] `1-c2pt-fit/results_nstate_fit_1state/`
- [ ] `1-c2pt-fit/results_nstate_fit_2state/`
- [ ] `2-bm/results_fit_nst2/`
- [ ] `3-normalize/`
- [ ] `4-FT/`
- [ ] `5-CSkernel/l{ns}c64a{spacing}_m140_fit_pzmultiPz/`

### 11. Common Questions

#### Q: Should I run k0 and k6 in parallel?
**A**: Yes, but monitor resource usage. Better to run sequentially for easier debugging.

#### Q: What if an ensemble has different momentum range?
**A**: Check the actual CSV files in `comb_c2pt_csv/` to see available pz values.

#### Q: How to adjust fit windows?
**A**: Start with values from similar ensembles, adjust if fits look poor (check χ²).

#### Q: Can I skip steps if re-running?
**A**: Only if previous outputs are verified correct. Otherwise, run full chain.

### 12. Quick Reference Commands

```bash
# Check ensemble parameters from directory name
basename $(pwd) | sed 's/.*l\([0-9]*\)c64a\([0-9]*\).*/\1 \2/'

# Verify data files exist
find /Users/xiang/Desktop/docs/0-2025/projects/3-CSkernel/data/ -name "*$(basename $(pwd) | cut -d'_' -f1)*" -type d

# Quick test single pz
sed -i '' "s/pzlist = \[.*\]/pzlist = [5]/" 2-bm/tmdwf_nstate_k6.ipynb
```

## Examples

### Example 1: Starting new ensemble
```
$ cd /Users/xiang/Desktop/docs/0-2026/projects/1-CSkernel/analysis_l48c64a060_m140_src5
$ /cskernel-workflow
# Follow steps 1-6 in order
```

### Example 2: Debugging failed step
```
$ /cskernel-workflow
# Check "Error Diagnosis Flowchart" and "Key Pitfalls & Solutions"
```

### Example 3: Verification after completion
```
$ /cskernel-workflow
# Run "Final Deliverables Checklist" to verify all outputs
```

## Version History
- 1.0 (2026-04-19): Initial version based on analysis_l64c64a076_m140_src5 experience

---

*This skill encapsulates all lessons learned from successfully completing the analysis_l64c64a076_m140_src5 workflow. Use it as your primary reference for subsequent ensemble analyses.*