from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from .nstate_fit import parse_nstate_fit_input, run_nstate_fit
from .plotting_2pt import plot_nstate_outputs
from .tgevp import parse_tgevp_input, run_ss_2pt_tgevp

TGEVP_INPUT_KEYS = {
    "title_pattern",
    "ns",
    "nt",
    "lattice_spacing_fm",
    "c2pt",
    "pzlist",
    "fold_t",
    "tsrange",
    "binsize",
    "bootstrap_samples",
    "bootstrap_size",
    "seed",
    "results_dir",
}
TGEVP_RUN_KEYS = {
    "binsize",
    "bootstrap_samples",
    "bootstrap_size",
    "seed",
    "results_dir",
}
NSTATE_INPUT_KEYS = {
    "title_pattern",
    "ns",
    "nt",
    "lattice_spacing_fm",
    "c2pt",
    "pzlist",
    "fold_t",
    "tsrange",
    "model",
    "nstates",
    "tmax",
    "binsize",
    "bootstrap_samples",
    "bootstrap_size",
    "seed",
    "plot",
    "results_dir",
}
NSTATE_RUN_KEYS = {"results_dir"}
PLOT_REQUIRED_KEYS = {
    "output_dir",
    "correlator_table",
    "meff_table",
    "fit_table",
    "nstates",
    "model",
    "title",
    "nt",
}


def _as_scalar_string(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value)
    return str(value)


def _subset_config(config: dict[str, Any], allowed_keys: set[str]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key in allowed_keys and value is not None}


def render_tgevp_input_text(config: dict[str, Any]) -> str:
    config = _subset_config(config, TGEVP_INPUT_KEYS)
    required = [
        "title_pattern",
        "ns",
        "nt",
        "lattice_spacing_fm",
        "c2pt",
        "pzlist",
        "fold_t",
        "tsrange",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"missing TGEVP notebook config keys: {missing}")

    lines = [
        f"{config['title_pattern']} {config['ns']} {config['nt']} {config['lattice_spacing_fm']}",
        f"c2pt {_as_scalar_string(config['c2pt'])}",
        f"pzlist {_as_scalar_string(config['pzlist'])}",
        f"fold_t {_as_scalar_string(config['fold_t'])}",
        f"tsrange {_as_scalar_string(config['tsrange'])}",
    ]
    for optional_key in ("binsize", "bootstrap_samples", "bootstrap_size", "seed", "results_dir"):
        if optional_key in config and config[optional_key] is not None:
            lines.append(f"{optional_key} {_as_scalar_string(config[optional_key])}")
    return "\n".join(lines) + "\n"


def render_nstate_fit_input_text(config: dict[str, Any]) -> str:
    config = _subset_config(config, NSTATE_INPUT_KEYS)
    required = [
        "title_pattern",
        "ns",
        "nt",
        "lattice_spacing_fm",
        "c2pt",
        "pzlist",
        "fold_t",
        "tsrange",
        "model",
        "nstates",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"missing N-state notebook config keys: {missing}")

    lines = [
        f"{config['title_pattern']} {config['ns']} {config['nt']} {config['lattice_spacing_fm']}",
        f"c2pt {_as_scalar_string(config['c2pt'])}",
        f"pzlist {_as_scalar_string(config['pzlist'])}",
        f"fold_t {_as_scalar_string(config['fold_t'])}",
        f"tsrange {_as_scalar_string(config['tsrange'])}",
        f"model {_as_scalar_string(config['model'])}",
        f"nstates {_as_scalar_string(config['nstates'])}",
    ]

    for optional_key in (
        "tmax",
        "binsize",
        "bootstrap_samples",
        "bootstrap_size",
        "seed",
        "plot",
        "results_dir",
    ):
        if optional_key in config and config[optional_key] is not None:
            lines.append(f"{optional_key} {_as_scalar_string(config[optional_key])}")
    return "\n".join(lines) + "\n"


def _materialize_input_text(text: str, suffix: str) -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix="lqcd_notebook_"))
    path = tmpdir / suffix
    path.write_text(text, encoding="utf-8")
    return path


def validate_tgevp_notebook_config(config: dict[str, Any]):
    input_path = _materialize_input_text(render_tgevp_input_text(config), "input_tgevp.txt")
    return parse_tgevp_input(input_path)


def validate_nstate_notebook_config(config: dict[str, Any]):
    input_path = _materialize_input_text(render_nstate_fit_input_text(config), "input_nstate.txt")
    return parse_nstate_fit_input(input_path)


def run_tgevp_from_notebook(
    config: dict[str, Any],
    **overrides: Any,
) -> list[Path]:
    input_path = _materialize_input_text(render_tgevp_input_text(config), "input_tgevp.txt")
    run_config = {
        "binsize": 1,
        "bootstrap_samples": None,
        "bootstrap_size": None,
        "seed": 2026,
        "results_dir": None,
    }
    run_config.update(_subset_config(config, TGEVP_RUN_KEYS))
    run_config.update({key: value for key, value in overrides.items() if value is not None})
    if run_config["results_dir"] is None:
        # In notebooks, default to the notebook's working directory so outputs
        # appear alongside the notebook unless the user overrides the path.
        run_config["results_dir"] = Path.cwd()
    return run_ss_2pt_tgevp(
        input_path,
        binsize=run_config["binsize"],
        bootstrap_samples=run_config["bootstrap_samples"],
        bootstrap_size=run_config["bootstrap_size"],
        seed=run_config["seed"],
        results_dir=run_config["results_dir"],
    )


def run_nstate_fit_from_notebook(
    config: dict[str, Any],
    **overrides: Any,
) -> list[Path]:
    input_path = _materialize_input_text(render_nstate_fit_input_text(config), "input_nstate.txt")
    run_config = {"results_dir": None}
    run_config.update(_subset_config(config, NSTATE_RUN_KEYS))
    run_config.update({key: value for key, value in overrides.items() if value is not None})
    if run_config["results_dir"] is None:
        # Match the TGEVP notebook workflow: default to the notebook's working
        # directory when no explicit output directory is provided.
        run_config["results_dir"] = Path.cwd()
    return run_nstate_fit(input_path, results_dir=run_config["results_dir"])


def validate_plot_2pt_notebook_config(config: dict[str, Any]) -> dict[str, Any]:
    missing = PLOT_REQUIRED_KEYS - set(config)
    if missing:
        raise ValueError(f"missing plot notebook config keys: {sorted(missing)}")
    return {key: config[key] for key in sorted(PLOT_REQUIRED_KEYS)}


def run_plot_2pt_from_notebook(config: dict[str, Any]) -> list[Path]:
    validated = validate_plot_2pt_notebook_config(config)
    return plot_nstate_outputs(
        output_dir=validated["output_dir"],
        correlator_table=validated["correlator_table"],
        meff_table=validated["meff_table"],
        fit_table=validated["fit_table"],
        nstates=int(validated["nstates"]),
        model=str(validated["model"]),
        title=str(validated["title"]),
        nt=int(validated["nt"]),
    )


def pretty_print_config(config: dict[str, Any]) -> str:
    return json.dumps(config, indent=2, ensure_ascii=False, default=str)
