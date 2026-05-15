from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from .DA.fourier import (
    parse_da_fourier_input,
    run_da_fourier_workflow,
)
from .DA.fit_nstate import parse_da_fit_input, run_da_nstate_fit
from .DA.normalize import parse_da_normalize_input, run_da_normalization
from .DA.ratio_fourier_t import (
    parse_da_ratio_fourier_t_input,
    run_da_ratio_fourier_t_workflow,
)
from .DA.x_nstate_fit import (
    parse_da_x_nstate_fit_input,
    run_da_x_nstate_fit_workflow,
)
from .DA.xfit_normalize import (
    parse_da_xfit_normalize_input,
    run_da_xfit_normalization,
)
from .two_point.effective_mass import parse_effective_mass_input, run_effective_mass_workflow
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
    "tmax",
    "model",
    "fit_mode",
    "pz0_ground_energy",
    "fix_ground_energy_from_dispersion",
    "nstates",
    "tmin_window",
    "binsize",
    "bootstrap_samples",
    "bootstrap_size",
    "seed",
    "low_state_prior_tmin",
    "lambda_prior",
    "plot",
    "results_dir",
}
NSTATE_RUN_KEYS = {"results_dir"}
EFFECTIVE_MASS_INPUT_KEYS = {
    "title_pattern",
    "ns",
    "nt",
    "lattice_spacing_fm",
    "c2pt",
    "pzlist",
    "fold_t",
    "tsrange",
    "model",
    "binsize",
    "bootstrap_samples",
    "bootstrap_size",
    "seed",
    "results_dir",
}
EFFECTIVE_MASS_RUN_KEYS = {"results_dir"}
DA_INPUT_KEYS = {
    "title_pattern",
    "ns",
    "nt",
    "lattice_spacing_fm",
    "decay_constant_check",
    "two_point_fit_sample_coupled",
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
    "fit_window",
    "qda_h5",
    "dataset_path_template",
    "two_point_fit_root",
    "two_point_fit_window_by_pz",
    "c2pt",
    "fold_t",
    "tsrange",
    "plot",
    "results_dir",
}
DA_RUN_KEYS = {"results_dir"}
DA_FOURIER_KEYS = {
    "title_pattern",
    "input_root",
    "ns",
    "lattice_spacing_fm",
    "pzlist",
    "gmlist",
    "etalist",
    "bTlist",
    "bTrange",
    "component",
    "nstates",
    "normalization_mode",
    "x_values",
    "x_range",
    "x_count",
    "zstep_fm",
    "interpolation_kind",
    "plot",
    "results_dir",
}
DA_FOURIER_RUN_KEYS = {"results_dir"}
DA_RATIO_FOURIER_T_KEYS = {
    "title_pattern",
    "ns",
    "nt",
    "lattice_spacing_fm",
    "pzlist",
    "gmlist",
    "etalist",
    "Tdirlist",
    "bTlist",
    "bTrange",
    "bzlist",
    "bzrange",
    "component",
    "qda_h5",
    "dataset_path_template",
    "c2pt",
    "fold_t",
    "tsrange",
    "binsize",
    "bootstrap_samples",
    "bootstrap_size",
    "seed",
    "x_values",
    "x_range",
    "x_count",
    "zstep_fm",
    "interpolation_kind",
    "results_dir",
}
DA_RATIO_FOURIER_T_RUN_KEYS = {"results_dir"}
DA_X_NSTATE_FIT_KEYS = {
    "title_pattern",
    "input_root",
    "ns",
    "nt",
    "lattice_spacing_fm",
    "pzlist",
    "gmlist",
    "etalist",
    "bTlist",
    "bTrange",
    "component",
    "nstates",
    "fit_window",
    "two_point_fit_root",
    "two_point_fit_window_by_pz",
    "two_point_fit_sample_coupled",
    "results_dir",
}
DA_X_NSTATE_FIT_RUN_KEYS = {"results_dir"}
DA_XFIT_NORMALIZE_KEYS = {
    "title_pattern",
    "input_root",
    "bare_matrix_root",
    "ns",
    "lattice_spacing_fm",
    "pzlist",
    "gmlist",
    "etalist",
    "bTlist",
    "bTrange",
    "bzlist",
    "bzrange",
    "component",
    "nstates",
    "normalization_mode",
    "zstep_fm",
    "interpolation_kind",
    "results_dir",
}
DA_XFIT_NORMALIZE_RUN_KEYS = {"results_dir"}
DA_NORMALIZE_KEYS = {
    "title_pattern",
    "input_root",
    "ns",
    "lattice_spacing_fm",
    "pzlist",
    "gmlist",
    "etalist",
    "bTlist",
    "bTrange",
    "bzlist",
    "bzrange",
    "component",
    "nstates",
    "normalization_mode",
    "plot",
    "results_dir",
}
DA_NORMALIZE_RUN_KEYS = {"results_dir"}
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
PLOT_OPTIONAL_KEYS = set()


def _as_scalar_string(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value)
    return str(value)


def _subset_config(config: dict[str, Any], allowed_keys: set[str]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key in allowed_keys and value is not None}


def _materialize_da_two_point_fit_window_file(window_by_pz: dict[Any, Any]) -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix="lqcd_da_two_point_fit_window_"))
    path = tmpdir / "two_point_fit_window_by_pz.txt"
    lines: list[str] = []
    for pz, window in window_by_pz.items():
        if not isinstance(window, (list, tuple)) or len(window) != 2:
            raise ValueError(f"invalid two_point_fit_window entry for pz={pz}; expected [tmin, tmax]")
        tmin = int(window[0])
        tmax = int(window[1])
        if tmax < tmin:
            raise ValueError(f"invalid two_point_fit_window entry for pz={pz}; tmax must be >= tmin")
        lines.append(f"{int(pz)} {tmin} {tmax}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


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
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"missing TGEVP notebook config keys: {missing}")

    lines = [
        f"{config['title_pattern']} {config['ns']} {config['nt']} {config['lattice_spacing_fm']}",
        f"c2pt {_as_scalar_string(config['c2pt'])}",
        f"pzlist {_as_scalar_string(config['pzlist'])}",
        f"fold_t {_as_scalar_string(config['fold_t'])}",
    ]
    if "tsrange" in config and config["tsrange"] is not None:
        lines.append(f"tsrange {_as_scalar_string(config['tsrange'])}")
    for optional_key in ("binsize", "bootstrap_samples", "bootstrap_size", "seed", "results_dir"):
        if optional_key in config and config[optional_key] is not None:
            lines.append(f"{optional_key} {_as_scalar_string(config[optional_key])}")
    return "\n".join(lines) + "\n"


def render_nstate_fit_input_text(config: dict[str, Any]) -> str:
    if isinstance(config.get("tmax"), dict):
        config = dict(config)
        config["tmax"] = str(_materialize_nstate_tmax_file(config["tmax"]))
    if isinstance(config.get("tmin_window"), dict):
        config = dict(config)
        config["tmin_window"] = str(
            _materialize_nstate_tmin_window_file(config["tmin_window"])
        )
    if isinstance(config.get("low_state_prior_tmin"), dict):
        config = dict(config)
        config["low_state_prior_tmin"] = str(
            _materialize_nstate_low_state_prior_tmin_file(config["low_state_prior_tmin"])
        )
    config = _subset_config(config, NSTATE_INPUT_KEYS)
    required = [
        "title_pattern",
        "ns",
        "nt",
        "lattice_spacing_fm",
        "c2pt",
        "pzlist",
        "fold_t",
        "tmax",
        "model",
        "fit_mode",
        "nstates",
        "tmin_window",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"missing N-state notebook config keys: {missing}")

    lines = [
        f"{config['title_pattern']} {config['ns']} {config['nt']} {config['lattice_spacing_fm']}",
        f"c2pt {_as_scalar_string(config['c2pt'])}",
        f"pzlist {_as_scalar_string(config['pzlist'])}",
        f"fold_t {_as_scalar_string(config['fold_t'])}",
        f"tmax {_as_scalar_string(config['tmax'])}",
        f"model {_as_scalar_string(config['model'])}",
        f"fit_mode {_as_scalar_string(config.get('fit_mode', 'uncorrelated'))}",
        f"nstates {_as_scalar_string(config['nstates'])}",
        f"tmin_window {_as_scalar_string(config['tmin_window'])}",
    ]

    for optional_key in (
        "pz0_ground_energy",
        "fix_ground_energy_from_dispersion",
        "binsize",
        "bootstrap_samples",
        "bootstrap_size",
        "seed",
        "low_state_prior_tmin",
        "lambda_prior",
        "plot",
        "results_dir",
    ):
        if optional_key in config and config[optional_key] is not None:
            lines.append(f"{optional_key} {_as_scalar_string(config[optional_key])}")
    return "\n".join(lines) + "\n"


def render_effective_mass_input_text(config: dict[str, Any]) -> str:
    config = _subset_config(config, EFFECTIVE_MASS_INPUT_KEYS)
    required = [
        "title_pattern",
        "ns",
        "nt",
        "lattice_spacing_fm",
        "c2pt",
        "pzlist",
        "fold_t",
        "model",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"missing effective-mass notebook config keys: {missing}")

    lines = [
        f"{config['title_pattern']} {config['ns']} {config['nt']} {config['lattice_spacing_fm']}",
        f"c2pt {_as_scalar_string(config['c2pt'])}",
        f"pzlist {_as_scalar_string(config['pzlist'])}",
        f"fold_t {_as_scalar_string(config['fold_t'])}",
        f"model {_as_scalar_string(config['model'])}",
    ]
    if "tsrange" in config and config["tsrange"] is not None:
        lines.append(f"tsrange {_as_scalar_string(config['tsrange'])}")
    for optional_key in ("binsize", "bootstrap_samples", "bootstrap_size", "seed", "results_dir"):
        if optional_key in config and config[optional_key] is not None:
            lines.append(f"{optional_key} {_as_scalar_string(config[optional_key])}")
    return "\n".join(lines) + "\n"


def render_da_fit_input_text(config: dict[str, Any]) -> str:
    if isinstance(config.get("fit_window"), dict):
        config = dict(config)
        config["fit_window"] = str(
            _materialize_da_fit_window_file(config["fit_window"])
        )
    if isinstance(config.get("two_point_fit_window_by_pz"), dict):
        config = dict(config)
        config["two_point_fit_window_by_pz"] = str(
            _materialize_da_two_point_fit_window_file(config["two_point_fit_window_by_pz"])
        )
    config = _subset_config(config, DA_INPUT_KEYS)
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
        "fit_window",
        "qda_h5",
        "dataset_path_template",
        "two_point_fit_root",
        "two_point_fit_window_by_pz",
        "c2pt",
        "fold_t",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"missing DA notebook config keys: {missing}")
    if "bTlist" not in config and "bTrange" not in config:
        raise ValueError("missing DA notebook config key: bTlist or bTrange")
    if "bzlist" not in config and "bzrange" not in config:
        raise ValueError("missing DA notebook config key: bzlist or bzrange")

    lines = [
        f"{config['title_pattern']} {config['ns']} {config['nt']} {config['lattice_spacing_fm']}",
        f"decay_constant_check {_as_scalar_string(config.get('decay_constant_check', False))}",
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
            f"fit_window {_as_scalar_string(config['fit_window'])}",
            # qda_h5 is passed through unchanged so notebook configs may use
            # either {pz}/{gm} placeholders or the legacy * pz wildcard.
            f"qda_h5 {_as_scalar_string(config['qda_h5'])}",
            f"dataset_path_template {_as_scalar_string(config['dataset_path_template'])}",
            f"two_point_fit_root {_as_scalar_string(config['two_point_fit_root'])}",
            f"two_point_fit_window_by_pz {_as_scalar_string(config['two_point_fit_window_by_pz'])}",
            f"c2pt {_as_scalar_string(config['c2pt'])}",
            f"fold_t {_as_scalar_string(config['fold_t'])}",
        ]
    )
    if "tsrange" in config and config["tsrange"] is not None:
        lines.append(f"tsrange {_as_scalar_string(config['tsrange'])}")
    for optional_key in (
        "binsize",
        "bootstrap_samples",
        "bootstrap_size",
        "seed",
        "plot",
        "results_dir",
        "two_point_fit_sample_coupled",
    ):
        if optional_key in config and config[optional_key] is not None:
            lines.append(f"{optional_key} {_as_scalar_string(config[optional_key])}")
    return "\n".join(lines) + "\n"


def render_da_fourier_input_text(config: dict[str, Any]) -> str:
    config = _subset_config(config, DA_FOURIER_KEYS)
    required = [
        "title_pattern",
        "input_root",
        "ns",
        "lattice_spacing_fm",
        "pzlist",
        "gmlist",
        "etalist",
        "component",
        "nstates",
        "normalization_mode",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"missing DA Fourier notebook config keys: {missing}")
    if "bTlist" not in config and "bTrange" not in config:
        raise ValueError("missing DA Fourier notebook config key: bTlist or bTrange")

    lines = [
        f"title_pattern {_as_scalar_string(config['title_pattern'])}",
        f"input_root {_as_scalar_string(config['input_root'])}",
        f"ns {_as_scalar_string(config['ns'])}",
        f"lattice_spacing_fm {_as_scalar_string(config['lattice_spacing_fm'])}",
        f"pzlist {_as_scalar_string(config['pzlist'])}",
        f"gmlist {_as_scalar_string(config['gmlist'])}",
        f"etalist {_as_scalar_string(config['etalist'])}",
    ]
    if "bTlist" in config and config["bTlist"] is not None:
        lines.append(f"bTlist {_as_scalar_string(config['bTlist'])}")
    elif "bTrange" in config and config["bTrange"] is not None:
        lines.append(f"bTrange {_as_scalar_string(config['bTrange'])}")
    lines.extend(
        [
            f"component {_as_scalar_string(config['component'])}",
            f"nstates {_as_scalar_string(config['nstates'])}",
            f"normalization_mode {_as_scalar_string(config['normalization_mode'])}",
        ]
    )
    if "x_values" in config and config["x_values"] is not None:
        lines.append(f"x_values {_as_scalar_string(config['x_values'])}")
    elif "x_range" in config and config["x_range"] is not None:
        lines.append(f"x_range {_as_scalar_string(config['x_range'])}")
        if "x_count" in config and config["x_count"] is not None:
            lines.append(f"x_count {_as_scalar_string(config['x_count'])}")
    for optional_key in ("zstep_fm", "interpolation_kind", "plot", "results_dir"):
        if optional_key in config and config[optional_key] is not None:
            lines.append(f"{optional_key} {_as_scalar_string(config[optional_key])}")
    return "\n".join(lines) + "\n"


def render_da_ratio_fourier_t_input_text(config: dict[str, Any]) -> str:
    config = _subset_config(config, DA_RATIO_FOURIER_T_KEYS)
    required = [
        "title_pattern",
        "ns",
        "nt",
        "lattice_spacing_fm",
        "pzlist",
        "gmlist",
        "etalist",
        "Tdirlist",
        "component",
        "qda_h5",
        "dataset_path_template",
        "c2pt",
        "fold_t",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"missing DA ratio-Fourier-t notebook config keys: {missing}")
    if "bTlist" not in config and "bTrange" not in config:
        raise ValueError("missing DA ratio-Fourier-t notebook config key: bTlist or bTrange")
    if "bzlist" not in config and "bzrange" not in config:
        raise ValueError("missing DA ratio-Fourier-t notebook config key: bzlist or bzrange")

    lines = [
        f"{config['title_pattern']} {config['ns']} {config['nt']} {config['lattice_spacing_fm']}",
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
            f"component {_as_scalar_string(config['component'])}",
            f"qda_h5 {_as_scalar_string(config['qda_h5'])}",
            f"dataset_path_template {_as_scalar_string(config['dataset_path_template'])}",
            f"c2pt {_as_scalar_string(config['c2pt'])}",
            f"fold_t {_as_scalar_string(config['fold_t'])}",
        ]
    )
    if "tsrange" in config and config["tsrange"] is not None:
        lines.append(f"tsrange {_as_scalar_string(config['tsrange'])}")
    if "x_values" in config and config["x_values"] is not None:
        lines.append(f"x_values {_as_scalar_string(config['x_values'])}")
    elif "x_range" in config and config["x_range"] is not None:
        lines.append(f"x_range {_as_scalar_string(config['x_range'])}")
        if "x_count" in config and config["x_count"] is not None:
            lines.append(f"x_count {_as_scalar_string(config['x_count'])}")
    for optional_key in (
        "binsize",
        "bootstrap_samples",
        "bootstrap_size",
        "seed",
        "zstep_fm",
        "interpolation_kind",
        "results_dir",
    ):
        if optional_key in config and config[optional_key] is not None:
            lines.append(f"{optional_key} {_as_scalar_string(config[optional_key])}")
    return "\n".join(lines) + "\n"


def render_da_x_nstate_fit_input_text(config: dict[str, Any]) -> str:
    if isinstance(config.get("fit_window"), dict):
        config = dict(config)
        config["fit_window"] = str(_materialize_da_fit_window_file(config["fit_window"]))
    if isinstance(config.get("two_point_fit_window_by_pz"), dict):
        config = dict(config)
        config["two_point_fit_window_by_pz"] = str(
            _materialize_da_two_point_fit_window_file(config["two_point_fit_window_by_pz"])
        )
    config = _subset_config(config, DA_X_NSTATE_FIT_KEYS)
    required = [
        "title_pattern",
        "input_root",
        "ns",
        "nt",
        "lattice_spacing_fm",
        "pzlist",
        "gmlist",
        "etalist",
        "component",
        "nstates",
        "fit_window",
        "two_point_fit_root",
        "two_point_fit_window_by_pz",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"missing DA x-space N-state notebook config keys: {missing}")
    if "bTlist" not in config and "bTrange" not in config:
        raise ValueError("missing DA x-space N-state notebook config key: bTlist or bTrange")

    lines = [
        f"title_pattern {_as_scalar_string(config['title_pattern'])}",
        f"input_root {_as_scalar_string(config['input_root'])}",
        f"ns {_as_scalar_string(config['ns'])}",
        f"nt {_as_scalar_string(config['nt'])}",
        f"lattice_spacing_fm {_as_scalar_string(config['lattice_spacing_fm'])}",
        f"pzlist {_as_scalar_string(config['pzlist'])}",
        f"gmlist {_as_scalar_string(config['gmlist'])}",
        f"etalist {_as_scalar_string(config['etalist'])}",
    ]
    if "bTlist" in config and config["bTlist"] is not None:
        lines.append(f"bTlist {_as_scalar_string(config['bTlist'])}")
    elif "bTrange" in config and config["bTrange"] is not None:
        lines.append(f"bTrange {_as_scalar_string(config['bTrange'])}")
    lines.extend(
        [
            f"component {_as_scalar_string(config['component'])}",
            f"nstates {_as_scalar_string(config['nstates'])}",
            f"fit_window {_as_scalar_string(config['fit_window'])}",
            f"two_point_fit_root {_as_scalar_string(config['two_point_fit_root'])}",
            f"two_point_fit_window_by_pz {_as_scalar_string(config['two_point_fit_window_by_pz'])}",
        ]
    )
    for optional_key in ("two_point_fit_sample_coupled", "results_dir"):
        if optional_key in config and config[optional_key] is not None:
            lines.append(f"{optional_key} {_as_scalar_string(config[optional_key])}")
    return "\n".join(lines) + "\n"


def render_da_xfit_normalize_input_text(config: dict[str, Any]) -> str:
    config = _subset_config(config, DA_XFIT_NORMALIZE_KEYS)
    required = [
        "title_pattern",
        "input_root",
        "bare_matrix_root",
        "ns",
        "lattice_spacing_fm",
        "pzlist",
        "gmlist",
        "etalist",
        "component",
        "nstates",
        "normalization_mode",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"missing DA x-fit normalization notebook config keys: {missing}")
    if "bTlist" not in config and "bTrange" not in config:
        raise ValueError("missing DA x-fit normalization notebook config key: bTlist or bTrange")
    if "bzlist" not in config and "bzrange" not in config:
        raise ValueError("missing DA x-fit normalization notebook config key: bzlist or bzrange")

    lines = [
        f"title_pattern {_as_scalar_string(config['title_pattern'])}",
        f"input_root {_as_scalar_string(config['input_root'])}",
        f"bare_matrix_root {_as_scalar_string(config['bare_matrix_root'])}",
        f"ns {_as_scalar_string(config['ns'])}",
        f"lattice_spacing_fm {_as_scalar_string(config['lattice_spacing_fm'])}",
        f"pzlist {_as_scalar_string(config['pzlist'])}",
        f"gmlist {_as_scalar_string(config['gmlist'])}",
        f"etalist {_as_scalar_string(config['etalist'])}",
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
            f"component {_as_scalar_string(config['component'])}",
            f"nstates {_as_scalar_string(config['nstates'])}",
            f"normalization_mode {_as_scalar_string(config['normalization_mode'])}",
        ]
    )
    for optional_key in ("zstep_fm", "interpolation_kind", "results_dir"):
        if optional_key in config and config[optional_key] is not None:
            lines.append(f"{optional_key} {_as_scalar_string(config[optional_key])}")
    return "\n".join(lines) + "\n"


def render_da_normalize_input_text(config: dict[str, Any]) -> str:
    config = _subset_config(config, DA_NORMALIZE_KEYS)
    required = [
        "title_pattern",
        "input_root",
        "ns",
        "lattice_spacing_fm",
        "pzlist",
        "gmlist",
        "etalist",
        "component",
        "nstates",
        "normalization_mode",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"missing DA normalization notebook config keys: {missing}")
    if "bTlist" not in config and "bTrange" not in config:
        raise ValueError("missing DA normalization notebook config key: bTlist or bTrange")
    if "bzlist" not in config and "bzrange" not in config:
        raise ValueError("missing DA normalization notebook config key: bzlist or bzrange")

    lines = [
        f"title_pattern {_as_scalar_string(config['title_pattern'])}",
        f"input_root {_as_scalar_string(config['input_root'])}",
        f"ns {_as_scalar_string(config['ns'])}",
        f"lattice_spacing_fm {_as_scalar_string(config['lattice_spacing_fm'])}",
        f"pzlist {_as_scalar_string(config['pzlist'])}",
        f"gmlist {_as_scalar_string(config['gmlist'])}",
        f"etalist {_as_scalar_string(config['etalist'])}",
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
            f"component {_as_scalar_string(config['component'])}",
            f"nstates {_as_scalar_string(config['nstates'])}",
            f"normalization_mode {_as_scalar_string(config['normalization_mode'])}",
        ]
    )
    for optional_key in ("plot", "results_dir"):
        if optional_key in config and config[optional_key] is not None:
            lines.append(f"{optional_key} {_as_scalar_string(config[optional_key])}")
    return "\n".join(lines) + "\n"


def _materialize_input_text(text: str, suffix: str) -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix="lqcd_notebook_"))
    path = tmpdir / suffix
    path.write_text(text, encoding="utf-8")
    return path


def _materialize_da_fit_window_file(fit_window: dict[Any, Any]) -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix="lqcd_da_fit_windows_"))
    path = tmpdir / "da_fit_window.txt"
    lines: list[str] = []

    def _parse_window(window: Any, label: str) -> tuple[int, int]:
        if not isinstance(window, (list, tuple)) or len(window) != 2:
            raise ValueError(
                f"invalid fit_window entry for {label}; expected [tmin, tmax]"
            )
        tmin = int(window[0])
        tmax = int(window[1])
        if tmax < tmin:
            raise ValueError(
                f"invalid fit_window entry for {label}; tmax must be >= tmin"
            )
        return tmin, tmax

    for key, value in fit_window.items():
        if isinstance(value, dict):
            gm = str(key)
            for pz_key, window in value.items():
                tmin, tmax = _parse_window(window, f"{gm}/{pz_key}")
                lines.append(f"{gm} {int(pz_key)} {tmin} {tmax}")
        else:
            tmin, tmax = _parse_window(value, str(key))
            lines.append(f"{int(key)} {tmin} {tmax}")

    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def _materialize_nstate_tmin_window_file(tmin_window: dict[Any, Any]) -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix="lqcd_nstate_tmin_windows_"))
    path = tmpdir / "nstate_tmin_window.txt"
    lines: list[str] = []
    for key, value in tmin_window.items():
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(
                f"invalid tmin_window entry for {key}; expected [tmin, tmax]"
            )
        tmin = int(value[0])
        tmax = int(value[1])
        if tmax < tmin:
            raise ValueError(
                f"invalid tmin_window entry for {key}; tmax must be >= tmin"
            )
        lines.append(f"{int(key)} {tmin} {tmax}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def _materialize_nstate_low_state_prior_tmin_file(prior_tmin_by_pz: dict[Any, Any]) -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix="lqcd_nstate_low_state_prior_tmin_"))
    path = tmpdir / "nstate_low_state_prior_tmin.txt"
    lines: list[str] = []
    for key, value in prior_tmin_by_pz.items():
        lines.append(f"{int(key)} {int(value)}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def _materialize_nstate_tmax_file(tmax_by_pz: dict[Any, Any]) -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix="lqcd_nstate_tmax_"))
    path = tmpdir / "nstate_tmax.txt"
    lines: list[str] = []
    for key, value in tmax_by_pz.items():
        if isinstance(value, (list, tuple)) or isinstance(value, dict):
            raise ValueError(f"invalid tmax entry for {key}; expected an integer")
        lines.append(f"{int(key)} {int(value)}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
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


def validate_effective_mass_notebook_config(config: dict[str, Any]):
    input_path = _materialize_input_text(render_effective_mass_input_text(config), "input_effective_mass.txt")
    return parse_effective_mass_input(input_path)


def validate_da_notebook_config(config: dict[str, Any]):
    input_path = _materialize_input_text(render_da_fit_input_text(config), "input_da.txt")
    return parse_da_fit_input(input_path)


def validate_da_fourier_notebook_config(config: dict[str, Any]) -> dict[str, Any]:
    input_path = _materialize_input_text(render_da_fourier_input_text(config), "input_da_fourier.txt")
    parsed = parse_da_fourier_input(input_path)
    validated = {
        "title_pattern": parsed.title_pattern,
        "input_root": parsed.input_root,
        "ns": parsed.ns,
        "lattice_spacing_fm": parsed.lattice_spacing_fm,
        "pzlist": parsed.pzlist,
        "gmlist": parsed.gmlist,
        "etalist": parsed.etalist,
        "bTlist": parsed.bTlist,
        "component": parsed.component,
        "nstates": parsed.nstates,
        "normalization_mode": parsed.normalization_mode,
        "x_values": parsed.x_values,
        "zstep_fm": parsed.zstep_fm,
        "interpolation_kind": parsed.interpolation_kind,
        "plot": parsed.make_plots,
    }
    if "results_dir" in config and config["results_dir"] is not None:
        validated["results_dir"] = Path(config["results_dir"])
    return validated


def validate_da_ratio_fourier_t_notebook_config(config: dict[str, Any]) -> dict[str, Any]:
    input_path = _materialize_input_text(
        render_da_ratio_fourier_t_input_text(config),
        "input_da_ratio_fourier_t.txt",
    )
    parsed = parse_da_ratio_fourier_t_input(input_path)
    validated = {
        "title_pattern": parsed.title_pattern,
        "ns": parsed.ns,
        "nt": parsed.nt,
        "lattice_spacing_fm": parsed.lattice_spacing_fm,
        "pzlist": parsed.pzlist,
        "gmlist": parsed.gmlist,
        "etalist": parsed.etalist,
        "Tdirlist": parsed.tdirlist,
        "bTlist": parsed.bTlist,
        "bzlist": parsed.bzlist,
        "component": parsed.component,
        "x_values": parsed.x_values,
        "zstep_fm": parsed.zstep_fm,
        "interpolation_kind": parsed.interpolation_kind,
    }
    if "results_dir" in config and config["results_dir"] is not None:
        validated["results_dir"] = Path(config["results_dir"])
    return validated


def validate_da_x_nstate_fit_notebook_config(config: dict[str, Any]) -> dict[str, Any]:
    input_path = _materialize_input_text(
        render_da_x_nstate_fit_input_text(config),
        "input_da_x_nstate_fit.txt",
    )
    parsed = parse_da_x_nstate_fit_input(input_path)
    validated = {
        "title_pattern": parsed.title_pattern,
        "input_root": parsed.input_root,
        "ns": parsed.ns,
        "nt": parsed.nt,
        "lattice_spacing_fm": parsed.lattice_spacing_fm,
        "pzlist": parsed.pzlist,
        "gmlist": parsed.gmlist,
        "etalist": parsed.etalist,
        "bTlist": parsed.bTlist,
        "component": parsed.component,
        "nstates": parsed.nstates,
        "fit_window": parsed.fit_window,
        "two_point_fit_root": parsed.two_point_fit_root,
        "two_point_fit_sample_coupled": parsed.two_point_fit_sample_coupled,
    }
    if "results_dir" in config and config["results_dir"] is not None:
        validated["results_dir"] = Path(config["results_dir"])
    return validated


def validate_da_xfit_normalize_notebook_config(config: dict[str, Any]) -> dict[str, Any]:
    input_path = _materialize_input_text(
        render_da_xfit_normalize_input_text(config),
        "input_da_xfit_normalize.txt",
    )
    parsed = parse_da_xfit_normalize_input(input_path)
    validated = {
        "title_pattern": parsed.title_pattern,
        "input_root": parsed.input_root,
        "bare_matrix_root": parsed.bare_matrix_root,
        "ns": parsed.ns,
        "lattice_spacing_fm": parsed.lattice_spacing_fm,
        "pzlist": parsed.pzlist,
        "gmlist": parsed.gmlist,
        "etalist": parsed.etalist,
        "bTlist": parsed.bTlist,
        "bzlist": parsed.bzlist,
        "component": parsed.component,
        "nstates": parsed.nstates,
        "normalization_mode": parsed.normalization_mode,
        "zstep_fm": parsed.zstep_fm,
        "interpolation_kind": parsed.interpolation_kind,
    }
    if "results_dir" in config and config["results_dir"] is not None:
        validated["results_dir"] = Path(config["results_dir"])
    return validated


def validate_da_normalize_notebook_config(config: dict[str, Any]) -> dict[str, Any]:
    input_path = _materialize_input_text(render_da_normalize_input_text(config), "input_da_normalize.txt")
    parsed = parse_da_normalize_input(input_path)
    validated = {
        "title_pattern": parsed.title_pattern,
        "input_root": parsed.input_root,
        "ns": parsed.ns,
        "lattice_spacing_fm": parsed.lattice_spacing_fm,
        "pzlist": parsed.pzlist,
        "gmlist": parsed.gmlist,
        "etalist": parsed.etalist,
        "bTlist": parsed.bTlist,
        "bzlist": parsed.bzlist,
        "component": parsed.component,
        "nstates": parsed.nstates,
        "normalization_mode": parsed.normalization_mode,
        "plot": parsed.make_plots,
    }
    if "results_dir" in config and config["results_dir"] is not None:
        validated["results_dir"] = Path(config["results_dir"])
    return validated


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


def run_effective_mass_from_notebook(
    config: dict[str, Any],
    **overrides: Any,
) -> list[Path]:
    input_path = _materialize_input_text(render_effective_mass_input_text(config), "input_effective_mass.txt")
    run_config = {"results_dir": None}
    run_config.update(_subset_config(config, EFFECTIVE_MASS_RUN_KEYS))
    run_config.update({key: value for key, value in overrides.items() if value is not None})
    if run_config["results_dir"] is None:
        run_config["results_dir"] = _guess_notebook_dir()
    return run_effective_mass_workflow(input_path, results_dir=run_config["results_dir"])


def run_da_fit_from_notebook(
    config: dict[str, Any],
    **overrides: Any,
) -> list[Path]:
    input_path = _materialize_input_text(render_da_fit_input_text(config), "input_da.txt")
    run_config = {"results_dir": None}
    run_config.update(_subset_config(config, DA_RUN_KEYS))
    run_config.update({key: value for key, value in overrides.items() if value is not None})
    if run_config["results_dir"] is None:
        run_config["results_dir"] = _guess_notebook_dir()
    return run_da_nstate_fit(input_path, results_dir=run_config["results_dir"])


def run_da_fourier_from_notebook(
    config: dict[str, Any],
    **overrides: Any,
) -> list[Path]:
    run_config = {"results_dir": None}
    run_config.update(_subset_config(config, DA_FOURIER_RUN_KEYS))
    run_config.update({key: value for key, value in overrides.items() if value is not None})
    if run_config["results_dir"] is None:
        run_config["results_dir"] = _guess_notebook_dir()
    input_path = _materialize_input_text(render_da_fourier_input_text(config), "input_da_fourier.txt")
    return run_da_fourier_workflow(input_path, results_dir=run_config["results_dir"])


def run_da_ratio_fourier_t_from_notebook(
    config: dict[str, Any],
    **overrides: Any,
) -> list[Path]:
    run_config = {"results_dir": None}
    run_config.update(_subset_config(config, DA_RATIO_FOURIER_T_RUN_KEYS))
    run_config.update({key: value for key, value in overrides.items() if value is not None})
    if run_config["results_dir"] is None:
        run_config["results_dir"] = _guess_notebook_dir()
    input_path = _materialize_input_text(
        render_da_ratio_fourier_t_input_text(config),
        "input_da_ratio_fourier_t.txt",
    )
    return run_da_ratio_fourier_t_workflow(input_path, results_dir=run_config["results_dir"])


def run_da_x_nstate_fit_from_notebook(
    config: dict[str, Any],
    **overrides: Any,
) -> list[Path]:
    run_config = {"results_dir": None}
    run_config.update(_subset_config(config, DA_X_NSTATE_FIT_RUN_KEYS))
    run_config.update({key: value for key, value in overrides.items() if value is not None})
    if run_config["results_dir"] is None:
        run_config["results_dir"] = _guess_notebook_dir()
    input_path = _materialize_input_text(
        render_da_x_nstate_fit_input_text(config),
        "input_da_x_nstate_fit.txt",
    )
    return run_da_x_nstate_fit_workflow(input_path, results_dir=run_config["results_dir"])


def run_da_xfit_normalize_from_notebook(
    config: dict[str, Any],
    **overrides: Any,
) -> list[Path]:
    run_config = {"results_dir": None}
    run_config.update(_subset_config(config, DA_XFIT_NORMALIZE_RUN_KEYS))
    run_config.update({key: value for key, value in overrides.items() if value is not None})
    if run_config["results_dir"] is None:
        run_config["results_dir"] = _guess_notebook_dir()
    input_path = _materialize_input_text(
        render_da_xfit_normalize_input_text(config),
        "input_da_xfit_normalize.txt",
    )
    return run_da_xfit_normalization(input_path, results_dir=run_config["results_dir"])


def run_da_normalize_from_notebook(
    config: dict[str, Any],
    **overrides: Any,
) -> list[Path]:
    run_config = {"results_dir": None}
    run_config.update(_subset_config(config, DA_NORMALIZE_RUN_KEYS))
    run_config.update({key: value for key, value in overrides.items() if value is not None})
    if run_config["results_dir"] is None:
        run_config["results_dir"] = _guess_notebook_dir()
    input_path = _materialize_input_text(render_da_normalize_input_text(config), "input_da_normalize.txt")
    return run_da_normalization(input_path, results_dir=run_config["results_dir"])


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
        nstates=int(validated["nstates"]),
        model=str(validated["model"]),
        title=str(validated["title"]),
        nt=int(validated["nt"]),
        lattice_spacing_fm=float(validated["lattice_spacing_fm"]),
    )


def pretty_print_config(config: dict[str, Any]) -> str:
    return json.dumps(config, indent=2, ensure_ascii=False, default=str)
