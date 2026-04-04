from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from .nstate_fit import parse_nstate_fit_input, run_nstate_fit
from .plotting_2pt import plot_nstate_outputs
from .tgevp import parse_tgevp_input, run_ss_2pt_tgevp


def _as_scalar_string(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value)
    return str(value)


def render_tgevp_input_text(config: dict[str, Any]) -> str:
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
    return "\n".join(lines) + "\n"


def render_nstate_fit_input_text(config: dict[str, Any]) -> str:
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
    *,
    binsize: int = 1,
    bootstrap_samples: int | None = None,
    bootstrap_size: int | None = None,
    seed: int = 2026,
    results_dir: str | Path | None = None,
) -> list[Path]:
    input_path = _materialize_input_text(render_tgevp_input_text(config), "input_tgevp.txt")
    return run_ss_2pt_tgevp(
        input_path,
        binsize=binsize,
        bootstrap_samples=bootstrap_samples,
        bootstrap_size=bootstrap_size,
        seed=seed,
        results_dir=results_dir,
    )


def run_nstate_fit_from_notebook(
    config: dict[str, Any],
    *,
    results_dir: str | Path | None = None,
) -> list[Path]:
    input_path = _materialize_input_text(render_nstate_fit_input_text(config), "input_nstate.txt")
    return run_nstate_fit(input_path, results_dir=results_dir)


def run_plot_2pt_from_notebook(config: dict[str, Any]) -> list[Path]:
    required = {
        "output_dir",
        "correlator_table",
        "meff_table",
        "fit_table",
        "nstates",
        "model",
        "title",
        "nt",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"missing plot notebook config keys: {sorted(missing)}")
    return plot_nstate_outputs(
        output_dir=config["output_dir"],
        correlator_table=config["correlator_table"],
        meff_table=config["meff_table"],
        fit_table=config["fit_table"],
        nstates=int(config["nstates"]),
        model=str(config["model"]),
        title=str(config["title"]),
        nt=int(config["nt"]),
    )


def pretty_print_config(config: dict[str, Any]) -> str:
    return json.dumps(config, indent=2, ensure_ascii=False, default=str)
