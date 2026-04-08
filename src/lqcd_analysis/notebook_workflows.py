from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from .tmdwf.fit_nstate import parse_tmdwf_fit_input, run_tmdwf_nstate_fit
from .two_point.fit_nstate import parse_nstate_fit_input, run_nstate_fit
from .two_point.plotting import plot_nstate_outputs
from .two_point.tgevp import parse_tgevp_input, run_ss_2pt_tgevp

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
    "fit_mode",
    "pz0_ground_energy",
    "nstates",
    "tmax",
    "binsize",
    "bootstrap_samples",
    "bootstrap_size",
    "seed",
    "lambda_prior",
    "plot",
    "results_dir",
}
NSTATE_RUN_KEYS = {"results_dir"}
TMDWF_INPUT_KEYS = {
    "title_pattern",
    "ns",
    "nt",
    "lattice_spacing_fm",
    "fit_target",
    "fit_component",
    "nstates",
    "pzlist",
    "gmlist",
    "etalist",
    "Tdirlist",
    "bTlist",
    "bTrange",
    "bzlist",
    "bzrange",
    "binsize",
    "bootstrap_samples",
    "bootstrap_size",
    "seed",
    "tmin",
    "tmax",
    "qtmdwf_h5",
    "dataset_path_template",
    "two_point_plateau_table",
    "c2pt",
    "fold_t",
    "tsrange",
    "plot",
    "results_dir",
}
TMDWF_RUN_KEYS = {"results_dir"}
PLOT_REQUIRED_KEYS = {
    "output_dir",
    "correlator_table",
    "meff_table",
    "fit_table",
    "nstates",
    "model",
    "title",
    "nt",
    "lattice_spacing_fm",
}
PLOT_OPTIONAL_KEYS = {"plateau_table"}


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
        "fit_mode",
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
        f"fit_mode {_as_scalar_string(config.get('fit_mode', 'uncorrelated'))}",
        f"nstates {_as_scalar_string(config['nstates'])}",
    ]

    for optional_key in (
        "pz0_ground_energy",
        "tmax",
        "binsize",
        "bootstrap_samples",
        "bootstrap_size",
        "seed",
        "lambda_prior",
        "plot",
        "results_dir",
    ):
        if optional_key in config and config[optional_key] is not None:
            lines.append(f"{optional_key} {_as_scalar_string(config[optional_key])}")
    return "\n".join(lines) + "\n"


def render_tmdwf_fit_input_text(config: dict[str, Any]) -> str:
    config = _subset_config(config, TMDWF_INPUT_KEYS)
    required = [
        "title_pattern",
        "ns",
        "nt",
        "lattice_spacing_fm",
        "fit_target",
        "fit_component",
        "nstates",
        "pzlist",
        "gmlist",
        "etalist",
        "Tdirlist",
        "tmin",
        "tmax",
        "qtmdwf_h5",
        "dataset_path_template",
        "two_point_plateau_table",
        "c2pt",
        "fold_t",
        "tsrange",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"missing TMDWF notebook config keys: {missing}")
    if "bTlist" not in config and "bTrange" not in config:
        raise ValueError("missing TMDWF notebook config key: bTlist or bTrange")
    if "bzlist" not in config and "bzrange" not in config:
        raise ValueError("missing TMDWF notebook config key: bzlist or bzrange")

    lines = [
        f"{config['title_pattern']} {config['ns']} {config['nt']} {config['lattice_spacing_fm']}",
        f"fit_target {_as_scalar_string(config['fit_target'])}",
        f"fit_component {_as_scalar_string(config['fit_component'])}",
        f"nstates {_as_scalar_string(config['nstates'])}",
        f"pzlist {_as_scalar_string(config['pzlist'])}",
        f"gmlist {_as_scalar_string(config['gmlist'])}",
        f"etalist {_as_scalar_string(config['etalist'])}",
        f"Tdirlist {_as_scalar_string(config['Tdirlist'])}",
    ]
    if "bTlist" in config and config["bTlist"] is not None:
        lines.append(f"bTlist {_as_scalar_string(config['bTlist'])}")
    elif "bTrange" in config and config["bTrange"] is not None:
        lines.append(f"bTrange {_as_scalar_string(config['bTrange'])}")
    if "bzlist" in config and config["bzlist"] is not None:
        lines.append(f"bzlist {_as_scalar_string(config['bzlist'])}")
    elif "bzrange" in config and config["bzrange"] is not None:
        lines.append(f"bzrange {_as_scalar_string(config['bzrange'])}")

    lines.extend(
        [
            f"tmin {_as_scalar_string(config['tmin'])}",
            f"tmax {_as_scalar_string(config['tmax'])}",
            f"qtmdwf_h5 {_as_scalar_string(config['qtmdwf_h5'])}",
            f"dataset_path_template {_as_scalar_string(config['dataset_path_template'])}",
            f"two_point_plateau_table {_as_scalar_string(config['two_point_plateau_table'])}",
            f"c2pt {_as_scalar_string(config['c2pt'])}",
            f"fold_t {_as_scalar_string(config['fold_t'])}",
            f"tsrange {_as_scalar_string(config['tsrange'])}",
        ]
    )
    for optional_key in (
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


def _guess_notebook_dir() -> Path:
    """Prefer the actual VS Code notebook location over Path.cwd().

    In VS Code Jupyter sessions, Path.cwd() may point to a temporary working
    directory rather than the .ipynb file's directory. When available, use the
    injected __vsc_ipynb_file__ variable from the live IPython user namespace.
    """
    try:
        from IPython import get_ipython
    except ImportError:
        return Path.cwd().resolve()

    shell = get_ipython()
    if shell is None:
        return Path.cwd().resolve()

    notebook_path = shell.user_ns.get("__vsc_ipynb_file__")
    if notebook_path:
        return Path(notebook_path).resolve().parent
    return Path.cwd().resolve()


def validate_tgevp_notebook_config(config: dict[str, Any]):
    input_path = _materialize_input_text(render_tgevp_input_text(config), "input_tgevp.txt")
    return parse_tgevp_input(input_path)


def validate_nstate_notebook_config(config: dict[str, Any]):
    input_path = _materialize_input_text(render_nstate_fit_input_text(config), "input_nstate.txt")
    return parse_nstate_fit_input(input_path)


def validate_tmdwf_notebook_config(config: dict[str, Any]):
    input_path = _materialize_input_text(render_tmdwf_fit_input_text(config), "input_tmdwf.txt")
    return parse_tmdwf_fit_input(input_path)


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
        run_config["results_dir"] = _guess_notebook_dir()
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
        run_config["results_dir"] = _guess_notebook_dir()
    return run_nstate_fit(input_path, results_dir=run_config["results_dir"])


def run_tmdwf_fit_from_notebook(
    config: dict[str, Any],
    **overrides: Any,
) -> list[Path]:
    input_path = _materialize_input_text(render_tmdwf_fit_input_text(config), "input_tmdwf.txt")
    run_config = {"results_dir": None}
    run_config.update(_subset_config(config, TMDWF_RUN_KEYS))
    run_config.update({key: value for key, value in overrides.items() if value is not None})
    if run_config["results_dir"] is None:
        run_config["results_dir"] = _guess_notebook_dir()
    return run_tmdwf_nstate_fit(input_path, results_dir=run_config["results_dir"])


def validate_plot_2pt_notebook_config(config: dict[str, Any]) -> dict[str, Any]:
    missing = PLOT_REQUIRED_KEYS - set(config)
    if missing:
        raise ValueError(f"missing plot notebook config keys: {sorted(missing)}")
    allowed = PLOT_REQUIRED_KEYS | PLOT_OPTIONAL_KEYS
    return {key: config[key] for key in sorted(allowed) if key in config}


def run_plot_2pt_from_notebook(config: dict[str, Any]) -> list[Path]:
    validated = validate_plot_2pt_notebook_config(config)
    return plot_nstate_outputs(
        output_dir=validated["output_dir"],
        correlator_table=validated["correlator_table"],
        meff_table=validated["meff_table"],
        fit_table=validated["fit_table"],
        plateau_table=validated.get("plateau_table"),
        nstates=int(validated["nstates"]),
        model=str(validated["model"]),
        title=str(validated["title"]),
        nt=int(validated["nt"]),
        lattice_spacing_fm=float(validated["lattice_spacing_fm"]),
    )


def pretty_print_config(config: dict[str, Any]) -> str:
    return json.dumps(config, indent=2, ensure_ascii=False, default=str)
