# Lattice QCD Analysis

Repository scaffold for lattice QCD data analysis, with a Python package layout,
basic correlator utilities, tests, and a clean directory structure for raw and
processed data.

## Planned Scope

- Correlator IO helpers
- Effective mass extraction
- Jackknife and bootstrap analysis
- Fit pipelines for two-point and three-point functions
- Reproducible analysis scripts and notebooks

## Project Layout

```text
.
├── configs/           Example configuration files
├── data/
│   ├── processed/     Derived data products
│   └── raw/           Raw input data (kept out of git)
├── docs/              Notes and methodology docs
├── notebooks/         Exploratory notebooks
├── scripts/           Reproducible batch scripts
├── src/lqcd_analysis/ Python source package
└── tests/             Unit tests
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Next Steps

1. Add your first correlator dataset loader in `src/lqcd_analysis/io.py`.
2. Extend `src/lqcd_analysis/correlators.py` with fit and resampling routines.
3. Add project-specific configs under `configs/`.
4. Create a GitHub repository and add it as `origin`.

