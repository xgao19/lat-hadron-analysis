from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from ..common.bootstrap import bin_samples, bootstrap_indices as common_bootstrap_indices, bootstrap_means
from ..common.constants import MIN_POSITIVE
from ..common.parsing import (
    load_fit_window_table,
    parse_bool,
    parse_fold_t,
    parse_int_list_or_range,
    parse_optional_int,
    parse_tsrange,
)
from ..common.utils import apply_fold_t, robust_mean_and_error
from ..two_point.io import load_correlator_csv
from .fit_nstate import (
    TMDWFFitResult,
    TMDWFOutputRecord,
    TMDWFRatioRecord,
    _component_list,
    _load_two_point_sample_parameters,
    _two_point_sample_table_path,
    _write_component_outputs,
    _write_ratio_outputs,
    fit_tmdwf_component,
    sanitize_token,
)
from .io import (
    _load_tmdwf_correlator_from_handle,
    expand_template,
    resolve_qtmdwf_h5_path,
    resolve_two_point_fit_reference,
)
from .models import evaluate_tmdwf_ratio, normalize_tmdwf_operator

Node = tuple[int, int]


@dataclass(frozen=True)
class TMDWFNStateDiffInput:
    title_pattern: str
    ns: int
    nt: int
    lattice_spacing_fm: float
    two_point_fit_sample_coupled: bool
    fit_component: str
    nstates: tuple[int, ...]
    pzlist: tuple[int, ...]
    gmlist: tuple[str, ...]
    etalist: tuple[str, ...]
    tdirlist: tuple[str, ...]
    bTlist: tuple[int, ...]
    bzlist: tuple[int, ...]
    binsize: int
    bootstrap_samples: int | None
    bootstrap_size: int | None
    seed: int
    fit_window: str
    qtmdwf_h5: str
    dataset_path_template: str
    c2pt: str
    fold_t: str
    tsrange: tuple[int, int]
    two_point_fit_root: str
    two_point_fit_window_by_pz: dict[int, tuple[int, int]]
    results_dir: Path


@dataclass(frozen=True)
class DifferenceEdge:
    kind: str
    source: Node
    target: Node


@dataclass(frozen=True)
class EdgeFitResult:
    edge: DifferenceEdge
    delta_samples: np.ndarray
    sigma: np.ndarray
    chi2_dof_samples: np.ndarray
    pvalue_samples: np.ndarray


@dataclass(frozen=True)
class GraphReconstruction:
    samples_by_node: dict[Node, np.ndarray]
    residual_chi2_dof_samples: np.ndarray


def parse_tmdwf_nstate_diff_input(
    path: str | Path,
    results_dir: str | Path | None = None,
) -> TMDWFNStateDiffInput:
    file_path = Path(path)
    entries: dict[str, list[str]] = {}
    first_tokens: list[str] | None = None

    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = stripped.split()
            if first_tokens is None:
                first_tokens = tokens
            entries[tokens[0]] = tokens[1:]

    if first_tokens is None or len(first_tokens) < 4:
        raise ValueError("the first non-empty line must be: title Ns Nt a_fm")

    required = {
        "fit_component",
        "nstates",
        "pzlist",
        "gmlist",
        "etalist",
        "Tdirlist",
        "qtmdwf_h5",
        "dataset_path_template",
        "two_point_fit_root",
        "two_point_fit_window_by_pz",
        "c2pt",
        "fold_t",
        "fit_window",
    }
    missing = required - entries.keys()
    if missing:
        raise ValueError(f"missing required keys in {file_path}: {sorted(missing)}")
    if "bTlist" not in entries and "bTrange" not in entries:
        raise ValueError(f"missing required key in {file_path}: bTlist or bTrange")
    if "bzlist" not in entries and "bzrange" not in entries:
        raise ValueError(f"missing required key in {file_path}: bzlist or bzrange")

    fit_component = entries["fit_component"][0].lower()
    if fit_component not in {"real", "imag", "both"}:
        raise ValueError("fit_component must be one of: real, imag, both")
    nstates = tuple(sorted({int(item) for item in entries["nstates"]}))
    if not nstates or any(state not in {1, 2} for state in nstates):
        raise ValueError("TMDWF nstate diff nstates must contain only 1 and/or 2")
    for gm in entries["gmlist"]:
        normalize_tmdwf_operator(gm)

    bTlist = parse_int_list_or_range(entries, "bTlist", "bTrange")
    bzlist = parse_int_list_or_range(entries, "bzlist", "bzrange")
    if (0, 0) not in {(bT, bz) for bT in bTlist for bz in bzlist}:
        raise ValueError("TMDWF nstate diff fit requires anchor bT=0 and bz=0")

    pz_values = tuple(int(item) for item in entries["pzlist"])
    two_point_fit_window_path = Path(entries["two_point_fit_window_by_pz"][0])
    two_point_fit_window_by_pz = load_fit_window_table(two_point_fit_window_path)
    missing_windows = sorted(set(pz_values) - {pz for gm, pz in two_point_fit_window_by_pz if gm is None})
    if missing_windows:
        raise ValueError(f"two_point_fit_window_by_pz is missing entries for pz values: {missing_windows}")

    input_path = file_path.resolve()
    return TMDWFNStateDiffInput(
        title_pattern=first_tokens[0],
        ns=int(first_tokens[1]),
        nt=int(first_tokens[2]),
        lattice_spacing_fm=float(first_tokens[3]),
        two_point_fit_sample_coupled=parse_bool(entries.get("two_point_fit_sample_coupled", ["false"])[0]),
        fit_component=fit_component,
        nstates=nstates,
        pzlist=pz_values,
        gmlist=tuple(entries["gmlist"]),
        etalist=tuple(entries["etalist"]),
        tdirlist=tuple(entries["Tdirlist"]),
        bTlist=bTlist,
        bzlist=bzlist,
        binsize=int(entries.get("binsize", ["1"])[0]),
        bootstrap_samples=parse_optional_int(entries.get("bootstrap_samples", ["auto"])[0]),
        bootstrap_size=parse_optional_int(entries.get("bootstrap_size", ["auto"])[0]),
        seed=int(entries.get("seed", ["2026"])[0]),
        fit_window=entries["fit_window"][0],
        qtmdwf_h5=entries["qtmdwf_h5"][0],
        dataset_path_template=entries["dataset_path_template"][0],
        c2pt=entries["c2pt"][0],
        fold_t=parse_fold_t(entries),
        tsrange=parse_tsrange(entries, int(first_tokens[2])),
        two_point_fit_root=entries["two_point_fit_root"][0],
        two_point_fit_window_by_pz={
            pz: window for (gm, pz), window in two_point_fit_window_by_pz.items() if gm is None
        },
        results_dir=(
            Path(entries["results_dir"][0])
            if "results_dir" in entries and results_dir is None
            else ((input_path.parent / "results_tmdwf_nstate_diff_fit") if results_dir is None else Path(results_dir))
        ),
    )


def _build_edges(bT_values: tuple[int, ...], bz_values: tuple[int, ...]) -> tuple[DifferenceEdge, ...]:
    bT_sorted = tuple(sorted(bT_values))
    bz_sorted = tuple(sorted(bz_values))
    edges: list[DifferenceEdge] = []
    for bT in bT_sorted:
        for left, right in zip(bz_sorted[:-1], bz_sorted[1:], strict=True):
            edges.append(DifferenceEdge(kind="bz", source=(bT, left), target=(bT, right)))
    for bz in bz_sorted:
        for lower, upper in zip(bT_sorted[:-1], bT_sorted[1:], strict=True):
            edges.append(DifferenceEdge(kind="bT", source=(lower, bz), target=(upper, bz)))
    return tuple(edges)


def _design_matrix(
    times: np.ndarray,
    amplitudes: np.ndarray,
    energies: np.ndarray,
    nt: int,
    *,
    pz: int,
    ns: int,
    gm: str,
    nstates: int,
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for idx in range(nstates):
        unit = np.zeros(nstates, dtype=float)
        unit[idx] = 1.0
        columns.append(
            np.asarray(
                evaluate_tmdwf_ratio(
                    times,
                    amplitudes,
                    energies,
                    unit,
                    nt,
                    gm=gm,
                    pz=pz,
                    ns=ns,
                ),
                dtype=float,
            )
        )
    return np.column_stack(columns)


def _component_window(samples: np.ndarray, tmin: int, tmax: int, component: str) -> np.ndarray:
    window = samples[:, tmin : tmax + 1]
    if component == "real":
        return np.asarray(np.real(window), dtype=float)
    if component == "imag":
        return np.asarray(np.imag(window), dtype=float)
    raise ValueError("component must be real or imag")


def fit_tmdwf_edge_delta(
    delta_ratio_samples: np.ndarray,
    amplitudes: np.ndarray,
    energies: np.ndarray,
    nt: int,
    pz: int,
    ns: int,
    gm: str,
    tmin: int,
    tmax: int,
    component: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data_samples = _component_window(delta_ratio_samples, tmin, tmax, component)
    times = np.arange(tmin, tmax + 1, dtype=int)
    sigma = np.nanstd(data_samples, axis=0, ddof=1)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0.0), sigma, MIN_POSITIVE)
    amplitudes = np.asarray(amplitudes, dtype=float)
    energies = np.asarray(energies, dtype=float)
    amplitudes_by_sample = amplitudes if amplitudes.ndim == 2 else None
    energies_by_sample = energies if energies.ndim == 2 else None
    nstates = amplitudes.shape[-1]
    delta_params = np.full((data_samples.shape[0], nstates), np.nan, dtype=float)
    chi2_dof_samples = np.full(data_samples.shape[0], np.nan, dtype=float)
    pvalue_samples = np.full(data_samples.shape[0], np.nan, dtype=float)

    for sample_id, sample_data in enumerate(data_samples):
        fit_amplitudes = amplitudes_by_sample[sample_id] if amplitudes_by_sample is not None else amplitudes
        fit_energies = energies_by_sample[sample_id] if energies_by_sample is not None else energies
        design = _design_matrix(
            times,
            fit_amplitudes,
            fit_energies,
            nt,
            pz=pz,
            ns=ns,
            gm=gm,
            nstates=nstates,
        )
        if not np.all(np.isfinite(design)) or not np.all(np.isfinite(sample_data)):
            continue
        weighted_design = design / sigma[:, None]
        weighted_data = sample_data / sigma
        params, _, _, _ = np.linalg.lstsq(weighted_design, weighted_data, rcond=None)
        residual = weighted_design @ params - weighted_data
        chi2_value = float(np.dot(residual, residual))
        dof = max(len(sample_data) - nstates, 1)
        delta_params[sample_id] = params
        chi2_dof_samples[sample_id] = chi2_value / dof
        pvalue_samples[sample_id] = float(chi2.sf(chi2_value, dof))

    param_sigma = np.full(nstates, MIN_POSITIVE, dtype=float)
    for idx in range(nstates):
        valid = delta_params[:, idx][np.isfinite(delta_params[:, idx])]
        if valid.size > 1:
            current = float(np.std(valid, ddof=1))
            param_sigma[idx] = current if np.isfinite(current) and current > 0.0 else MIN_POSITIVE
    return delta_params, param_sigma, chi2_dof_samples, pvalue_samples


def _reachable_nodes(anchor: Node, edges: list[tuple[Node, Node]]) -> set[Node]:
    adjacency: dict[Node, set[Node]] = {}
    for source, target in edges:
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)
    reachable = {anchor}
    frontier = [anchor]
    while frontier:
        node = frontier.pop()
        for neighbor in adjacency.get(node, set()):
            if neighbor in reachable:
                continue
            reachable.add(neighbor)
            frontier.append(neighbor)
    return reachable


def reconstruct_tmdwf_graph_samples(
    nodes: tuple[Node, ...],
    edges: tuple[DifferenceEdge, ...],
    edge_results: dict[tuple[Node, Node], EdgeFitResult],
    anchor_samples: np.ndarray,
    *,
    anchor: Node = (0, 0),
) -> GraphReconstruction:
    n_boot, nstates = anchor_samples.shape
    samples_by_node = {node: np.full((n_boot, nstates), np.nan, dtype=float) for node in nodes}
    residual_chi2_dof_samples = np.full((n_boot, nstates), np.nan, dtype=float)

    for sample_id in range(n_boot):
        for state_idx in range(nstates):
            anchor_value = anchor_samples[sample_id, state_idx]
            if not np.isfinite(anchor_value):
                continue
            valid_edge_rows: list[tuple[DifferenceEdge, float, float]] = []
            connectivity_edges: list[tuple[Node, Node]] = []
            for edge in edges:
                result = edge_results[(edge.source, edge.target)]
                delta_value = result.delta_samples[sample_id, state_idx]
                sigma = result.sigma[state_idx]
                if not np.isfinite(delta_value) or not np.isfinite(sigma) or sigma <= 0.0:
                    continue
                valid_edge_rows.append((edge, float(delta_value), float(sigma)))
                connectivity_edges.append((edge.source, edge.target))
            reachable = _reachable_nodes(anchor, connectivity_edges)
            if anchor not in reachable:
                continue
            active_nodes = tuple(node for node in nodes if node in reachable and node != anchor)
            node_index = {node: idx for idx, node in enumerate(active_nodes)}
            samples_by_node[anchor][sample_id, state_idx] = anchor_value
            if not active_nodes:
                residual_chi2_dof_samples[sample_id, state_idx] = 0.0
                continue

            rows: list[np.ndarray] = []
            rhs_values: list[float] = []
            used_edges: list[tuple[DifferenceEdge, float, float]] = []
            for edge, delta_value, sigma in valid_edge_rows:
                if edge.source not in reachable or edge.target not in reachable:
                    continue
                row = np.zeros(len(active_nodes), dtype=float)
                rhs = delta_value
                if edge.target == anchor:
                    rhs -= anchor_value
                else:
                    row[node_index[edge.target]] += 1.0
                if edge.source == anchor:
                    rhs += anchor_value
                else:
                    row[node_index[edge.source]] -= 1.0
                weight = 1.0 / sigma
                rows.append(row * weight)
                rhs_values.append(rhs * weight)
                used_edges.append((edge, delta_value, sigma))
            if not rows:
                continue
            design = np.vstack(rows)
            rhs = np.asarray(rhs_values, dtype=float)
            solution, _, _, _ = np.linalg.lstsq(design, rhs, rcond=None)
            for node, idx in node_index.items():
                samples_by_node[node][sample_id, state_idx] = solution[idx]

            chi2_value = 0.0
            for edge, delta_value, sigma in used_edges:
                source_value = anchor_value if edge.source == anchor else solution[node_index[edge.source]]
                target_value = anchor_value if edge.target == anchor else solution[node_index[edge.target]]
                chi2_value += float(((target_value - source_value - delta_value) / sigma) ** 2)
            dof = max(len(used_edges) - len(active_nodes), 1)
            residual_chi2_dof_samples[sample_id, state_idx] = chi2_value / dof
    return GraphReconstruction(samples_by_node=samples_by_node, residual_chi2_dof_samples=residual_chi2_dof_samples)


def _fit_result_from_reconstructed_samples(
    node: Node,
    sample_params: np.ndarray,
    ratio_samples: np.ndarray,
    amplitudes: np.ndarray,
    energies: np.ndarray,
    nt: int,
    pz: int,
    ns: int,
    gm: str,
    tmin: int,
    tmax: int,
    component: str,
) -> TMDWFFitResult:
    data_samples = _component_window(ratio_samples, tmin, tmax, component)
    times = np.arange(tmin, tmax + 1, dtype=int)
    sigma = np.nanstd(data_samples, axis=0, ddof=1)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0.0), sigma, MIN_POSITIVE)
    amplitudes = np.asarray(amplitudes, dtype=float)
    energies = np.asarray(energies, dtype=float)
    amplitudes_by_sample = amplitudes if amplitudes.ndim == 2 else None
    energies_by_sample = energies if energies.ndim == 2 else None
    chi2_dof_samples = np.full(sample_params.shape[0], np.nan, dtype=float)
    pvalue_samples = np.full(sample_params.shape[0], np.nan, dtype=float)
    for sample_id, params in enumerate(sample_params):
        if not np.all(np.isfinite(params)):
            continue
        fit_amplitudes = amplitudes_by_sample[sample_id] if amplitudes_by_sample is not None else amplitudes
        fit_energies = energies_by_sample[sample_id] if energies_by_sample is not None else energies
        model_values = evaluate_tmdwf_ratio(
            times,
            fit_amplitudes,
            fit_energies,
            params,
            nt,
            gm=gm,
            pz=pz,
            ns=ns,
        )
        residual = (np.asarray(model_values, dtype=float) - data_samples[sample_id]) / sigma
        chi2_value = float(np.dot(residual, residual))
        dof = max(len(times) - sample_params.shape[1], 1)
        chi2_dof_samples[sample_id] = chi2_value / dof
        pvalue_samples[sample_id] = float(chi2.sf(chi2_value, dof))

    success_mask = np.all(np.isfinite(sample_params), axis=1)
    if not np.any(success_mask):
        return TMDWFFitResult(
            params=np.full(sample_params.shape[1], np.nan, dtype=float),
            chi2=float("nan"),
            chi2_dof=float("nan"),
            pvalue=float("nan"),
            success=False,
            message=f"graph reconstruction failed for bT={node[0]}, bz={node[1]}",
        )
    params_mean = []
    for idx in range(sample_params.shape[1]):
        params_mean.append(robust_mean_and_error(sample_params[success_mask, idx])[0])
    chi2_dof_mean = robust_mean_and_error(chi2_dof_samples[success_mask])[0]
    pvalue_mean = robust_mean_and_error(pvalue_samples[success_mask])[0]
    return TMDWFFitResult(
        params=np.asarray(params_mean, dtype=float),
        chi2=float("nan"),
        chi2_dof=chi2_dof_mean,
        pvalue=pvalue_mean,
        success=True,
        message=f"graph reconstruction from {int(np.count_nonzero(success_mask))} successful samples",
    )


def _write_edge_diagnostics(
    output_root: Path,
    stem: str,
    edge_results: dict[tuple[Node, Node], EdgeFitResult],
) -> list[Path]:
    diagnostics_dir = output_root / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    summary_path = diagnostics_dir / f"{stem}_edge_deltas.txt"
    sample_path = diagnostics_dir / f"{stem}_edge_delta_samples.txt"
    ordered = list(edge_results.values())
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "\t".join(
                [
                    "edge_id",
                    "kind",
                    "source_bT",
                    "source_bz",
                    "target_bT",
                    "target_bz",
                    "state",
                    "delta_mean",
                    "delta_err",
                    "chi2_dof",
                    "pvalue",
                ]
            )
            + "\n"
        )
        for edge_id, result in enumerate(ordered):
            valid = np.all(np.isfinite(result.delta_samples), axis=1)
            for state_idx in range(result.delta_samples.shape[1]):
                values = result.delta_samples[valid, state_idx]
                mean, err = robust_mean_and_error(values)
                chi2_dof = robust_mean_and_error(result.chi2_dof_samples[valid])[0]
                pvalue = robust_mean_and_error(result.pvalue_samples[valid])[0]
                handle.write(
                    "\t".join(
                        [
                            str(edge_id),
                            result.edge.kind,
                            str(result.edge.source[0]),
                            str(result.edge.source[1]),
                            str(result.edge.target[0]),
                            str(result.edge.target[1]),
                            f"m{state_idx}",
                            f"{mean:.10e}",
                            f"{err:.10e}",
                            f"{chi2_dof:.10e}",
                            f"{pvalue:.10e}",
                        ]
                    )
                    + "\n"
                )
    with sample_path.open("w", encoding="utf-8") as handle:
        header = ["edge_id", "kind", "source_bT", "source_bz", "target_bT", "target_bz", "sample_id", "success"]
        header += [f"dm{idx}" for idx in range(ordered[0].delta_samples.shape[1])] if ordered else []
        handle.write("\t".join(header) + "\n")
        for edge_id, result in enumerate(ordered):
            for sample_id, params in enumerate(result.delta_samples):
                success = int(np.all(np.isfinite(params)))
                row = [
                    str(edge_id),
                    result.edge.kind,
                    str(result.edge.source[0]),
                    str(result.edge.source[1]),
                    str(result.edge.target[0]),
                    str(result.edge.target[1]),
                    str(sample_id),
                    str(success),
                    *[f"{value:.10e}" for value in params],
                ]
                handle.write("\t".join(row) + "\n")
    return [summary_path, sample_path]


def _write_graph_diagnostics(output_root: Path, stem: str, reconstruction: GraphReconstruction) -> Path:
    diagnostics_dir = output_root / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    path = diagnostics_dir / f"{stem}_graph_reconstruction.txt"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("state\tresidual_chi2_dof_mean\tresidual_chi2_dof_err\tsuccessful_samples\n")
        for state_idx in range(reconstruction.residual_chi2_dof_samples.shape[1]):
            values = reconstruction.residual_chi2_dof_samples[:, state_idx]
            valid = values[np.isfinite(values)]
            mean, err = robust_mean_and_error(valid)
            handle.write(f"m{state_idx}\t{mean:.10e}\t{err:.10e}\t{valid.size}\n")
    return path


def _write_plaquette_diagnostics(
    output_root: Path,
    stem: str,
    bT_values: tuple[int, ...],
    bz_values: tuple[int, ...],
    edge_results: dict[tuple[Node, Node], EdgeFitResult],
) -> Path:
    diagnostics_dir = output_root / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    path = diagnostics_dir / f"{stem}_plaquette_closure.txt"
    bT_sorted = tuple(sorted(bT_values))
    bz_sorted = tuple(sorted(bz_values))
    with path.open("w", encoding="utf-8") as handle:
        handle.write("bT0\tbT1\tbz0\tbz1\tstate\tclosure_mean\tclosure_err\n")
        for bT0, bT1 in zip(bT_sorted[:-1], bT_sorted[1:], strict=True):
            for bz0, bz1 in zip(bz_sorted[:-1], bz_sorted[1:], strict=True):
                required = [
                    ((bT0, bz0), (bT0, bz1)),
                    ((bT0, bz1), (bT1, bz1)),
                    ((bT1, bz0), (bT1, bz1)),
                    ((bT0, bz0), (bT1, bz0)),
                ]
                if any(key not in edge_results for key in required):
                    continue
                bottom = edge_results[required[0]].delta_samples
                right = edge_results[required[1]].delta_samples
                top = edge_results[required[2]].delta_samples
                left = edge_results[required[3]].delta_samples
                closure = bottom + right - top - left
                for state_idx in range(closure.shape[1]):
                    values = closure[:, state_idx][np.isfinite(closure[:, state_idx])]
                    mean, err = robust_mean_and_error(values)
                    handle.write(f"{bT0}\t{bT1}\t{bz0}\t{bz1}\tm{state_idx}\t{mean:.10e}\t{err:.10e}\n")
    return path


def _write_bz_chain_diagnostics(
    output_root: Path,
    stem: str,
    bT_values: tuple[int, ...],
    bz_values: tuple[int, ...],
    edge_results: dict[tuple[Node, Node], EdgeFitResult],
    reconstruction: GraphReconstruction,
) -> Path:
    diagnostics_dir = output_root / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    path = diagnostics_dir / f"{stem}_bz_chain_reconstruction.txt"
    bz_sorted = tuple(sorted(bz_values))
    with path.open("w", encoding="utf-8") as handle:
        handle.write("bT\tbz\tstate\tchain_mean\tchain_err\tgraph_mean\tgraph_err\n")
        for bT in sorted(bT_values):
            base = reconstruction.samples_by_node.get((bT, 0))
            if base is None:
                continue
            chain_samples = np.array(base, copy=True)
            for bz in bz_sorted:
                if bz == 0:
                    current = chain_samples
                else:
                    previous_values = [candidate for candidate in bz_sorted if candidate < bz]
                    if not previous_values:
                        continue
                    previous = previous_values[-1]
                    edge = edge_results.get(((bT, previous), (bT, bz)))
                    if edge is None:
                        continue
                    chain_samples = chain_samples + edge.delta_samples
                    current = chain_samples
                graph = reconstruction.samples_by_node.get((bT, bz))
                if graph is None:
                    continue
                for state_idx in range(current.shape[1]):
                    chain_values = current[:, state_idx][np.isfinite(current[:, state_idx])]
                    graph_values = graph[:, state_idx][np.isfinite(graph[:, state_idx])]
                    chain_mean, chain_err = robust_mean_and_error(chain_values)
                    graph_mean, graph_err = robust_mean_and_error(graph_values)
                    handle.write(
                        f"{bT}\t{bz}\tm{state_idx}\t{chain_mean:.10e}\t{chain_err:.10e}\t"
                        f"{graph_mean:.10e}\t{graph_err:.10e}\n"
                    )
    return path


def run_tmdwf_nstate_diff_fit(
    input_file: str | Path,
    *,
    results_dir: str | Path | None = None,
) -> list[Path]:
    spec = parse_tmdwf_nstate_diff_input(input_file, results_dir=results_dir)
    outputs: list[Path] = []
    fit_windows = load_fit_window_table(spec.fit_window)
    nodes = tuple((bT, bz) for bT in sorted(spec.bTlist) for bz in sorted(spec.bzlist))
    edges = _build_edges(spec.bTlist, spec.bzlist)
    anchor = (0, 0)

    try:
        import h5py  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("h5py is required to load TMDWF HDF5 data") from exc

    for pz in spec.pzlist:
        title = expand_template(spec.title_pattern, pz=pz)
        c2pt_path = expand_template(spec.c2pt, pz=pz)
        _, c2pt_raw = load_correlator_csv(c2pt_path)
        c2pt_processed = apply_fold_t(c2pt_raw, spec.nt, spec.fold_t)
        t0, t1 = spec.tsrange
        c2pt_selected = c2pt_processed[:, t0 : t1 + 1]
        denominator_binned = bin_samples(c2pt_selected, binsize=spec.binsize)
        n_cfg = denominator_binned.shape[0]
        if n_cfg < 2:
            raise ValueError("bootstrap requires at least two samples")
        n_boot = n_cfg if spec.bootstrap_samples is None else spec.bootstrap_samples
        draw_size = n_cfg if spec.bootstrap_size is None else spec.bootstrap_size
        indices = common_bootstrap_indices(n_cfg, draw_size, seed=spec.seed, n_boot=n_boot)
        denominator_boot = bootstrap_means(denominator_binned, indices=indices)

        dataset_root = spec.results_dir / title
        dataset_root.mkdir(parents=True, exist_ok=True)
        two_point_window = spec.two_point_fit_window_by_pz.get(pz)
        if two_point_window is None:
            raise ValueError(f"missing two_point_fit_window_by_pz entry for pz={pz}")
        two_point_tmin, two_point_tmax = two_point_window
        fit_reference_cache = {}
        two_point_sample_cache: dict[tuple[Path, int], tuple[np.ndarray, np.ndarray]] = {}
        for nstates in spec.nstates:
            fit_reference = resolve_two_point_fit_reference(
                spec.two_point_fit_root,
                title=title,
                nstates=nstates,
                tmin=two_point_tmin,
                tmax=two_point_tmax,
            )
            fit_reference_cache[nstates] = (
                fit_reference.path,
                fit_reference.tmin,
                fit_reference.tmax,
                fit_reference.amplitudes,
                fit_reference.energies,
            )

        def select_two_point_fit_parameters(
            fit_table_path: Path,
            reference_tmin: int,
            nstates: int,
            amplitudes: np.ndarray,
            energies: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            if not spec.two_point_fit_sample_coupled:
                return amplitudes, energies
            cache_key = (fit_table_path, reference_tmin)
            if cache_key not in two_point_sample_cache:
                two_point_sample_cache[cache_key] = _load_two_point_sample_parameters(
                    _two_point_sample_table_path(fit_table_path),
                    tmin=reference_tmin,
                    nstates=nstates,
                    sample_count=n_boot,
                    fallback_amplitudes=amplitudes,
                    fallback_energies=energies,
                )
            return two_point_sample_cache[cache_key]

        for gm in spec.gmlist:
            qtmdwf_path = resolve_qtmdwf_h5_path(spec.qtmdwf_h5, pz=pz, gm=gm)
            with h5py.File(qtmdwf_path, "r") as qtmdwf_handle:
                for eta in spec.etalist:
                    ratio_by_node: dict[Node, np.ndarray] = {}
                    for node in nodes:
                        bT, bz = node
                        numerator_selected = _load_tmdwf_correlator_from_handle(
                            qtmdwf_handle,
                            spec.dataset_path_template,
                            gm=gm,
                            eta=eta,
                            pz=pz,
                            tdirs=spec.tdirlist,
                            bT=bT,
                            bz=bz,
                            nt=spec.nt,
                            ns=spec.ns,
                            file_label=qtmdwf_path,
                        )[:, t0 : t1 + 1]
                        numerator_binned = bin_samples(numerator_selected, binsize=spec.binsize)
                        if numerator_binned.shape != denominator_binned.shape:
                            raise ValueError("numerator and denominator must have matching post-binning shapes")
                        numerator_boot = bootstrap_means(numerator_binned, indices=indices)
                        ratio_by_node[node] = np.divide(
                            numerator_boot,
                            denominator_boot,
                            out=np.full_like(numerator_boot, np.nan + 0.0j),
                            where=denominator_boot != 0.0,
                        )

                    fit_window = fit_windows.get((gm, pz), fit_windows.get((None, pz)))
                    if fit_window is None:
                        raise ValueError(f"missing fit_window entry for gm={gm}, pz={pz}")
                    fit_tmin, fit_tmax = fit_window
                    ratio_stem_by_bT: dict[int, Path] = {}
                    for bT in sorted(spec.bTlist):
                        ratio_records = [
                            TMDWFRatioRecord(
                                bz=bz,
                                tmin=fit_tmin,
                                tmax=fit_tmax,
                                two_point_fit_table_resolved=str(fit_reference_cache[spec.nstates[0]][0]),
                                two_point_fit_tmax_source="config",
                                two_point_fit_tmax=two_point_tmax,
                                tsrange_start=t0,
                                tsrange_end=t1,
                                ratio_samples=ratio_by_node[(bT, bz)],
                            )
                            for bz in sorted(spec.bzlist)
                        ]
                        combo_stem = f"{title}_{sanitize_token(gm)}_{sanitize_token(eta)}_bT{bT}"
                        ratio_stem_by_bT[bT] = _write_ratio_outputs(dataset_root, combo_stem, tuple(ratio_records))
                        outputs.append(ratio_stem_by_bT[bT])

                    for nstates in spec.nstates:
                        fit_table_path, two_point_tmin, two_point_tmax, amplitudes, energies = fit_reference_cache[nstates]
                        fit_amplitudes, fit_energies = select_two_point_fit_parameters(
                            fit_table_path,
                            two_point_tmin,
                            nstates,
                            amplitudes,
                            energies,
                        )
                        for component in _component_list(spec.fit_component):
                            anchor_result, anchor_samples = fit_tmdwf_component(
                                ratio_by_node[anchor],
                                fit_amplitudes,
                                fit_energies,
                                spec.nt,
                                pz,
                                spec.ns,
                                gm,
                                fit_tmin,
                                fit_tmax,
                                component,
                            )
                            edge_results: dict[tuple[Node, Node], EdgeFitResult] = {}
                            for edge in edges:
                                delta_ratio_samples = ratio_by_node[edge.target] - ratio_by_node[edge.source]
                                delta_samples, sigma, chi2_dof_samples, pvalue_samples = fit_tmdwf_edge_delta(
                                    delta_ratio_samples,
                                    fit_amplitudes,
                                    fit_energies,
                                    spec.nt,
                                    pz,
                                    spec.ns,
                                    gm,
                                    fit_tmin,
                                    fit_tmax,
                                    component,
                                )
                                edge_results[(edge.source, edge.target)] = EdgeFitResult(
                                    edge=edge,
                                    delta_samples=delta_samples,
                                    sigma=sigma,
                                    chi2_dof_samples=chi2_dof_samples,
                                    pvalue_samples=pvalue_samples,
                                )

                            reconstruction = reconstruct_tmdwf_graph_samples(
                                nodes,
                                edges,
                                edge_results,
                                anchor_samples,
                                anchor=anchor,
                            )
                            diagnostic_stem = (
                                f"{title}_{sanitize_token(gm)}_{sanitize_token(eta)}_"
                                f"{component}_{nstates}state"
                            )
                            outputs.extend(_write_edge_diagnostics(dataset_root, diagnostic_stem, edge_results))
                            outputs.append(_write_graph_diagnostics(dataset_root, diagnostic_stem, reconstruction))
                            outputs.append(
                                _write_plaquette_diagnostics(
                                    dataset_root,
                                    diagnostic_stem,
                                    spec.bTlist,
                                    spec.bzlist,
                                    edge_results,
                                )
                            )
                            outputs.append(
                                _write_bz_chain_diagnostics(
                                    dataset_root,
                                    diagnostic_stem,
                                    spec.bTlist,
                                    spec.bzlist,
                                    edge_results,
                                    reconstruction,
                                )
                            )

                            records_by_bT: dict[int, list[TMDWFOutputRecord]] = {bT: [] for bT in spec.bTlist}
                            for node in nodes:
                                bT, bz = node
                                sample_params = reconstruction.samples_by_node[node]
                                fit_result = (
                                    anchor_result
                                    if node == anchor
                                    else _fit_result_from_reconstructed_samples(
                                        node,
                                        sample_params,
                                        ratio_by_node[node],
                                        fit_amplitudes,
                                        fit_energies,
                                        spec.nt,
                                        pz,
                                        spec.ns,
                                        gm,
                                        fit_tmin,
                                        fit_tmax,
                                        component,
                                    )
                                )
                                records_by_bT[bT].append(
                                    TMDWFOutputRecord(
                                        bz=bz,
                                        component=component,
                                        nstates=nstates,
                                        tmin=fit_tmin,
                                        tmax=fit_tmax,
                                        fit_result=fit_result,
                                        sample_params=sample_params,
                                        amplitudes=fit_amplitudes,
                                        energies=fit_energies,
                                        pz=pz,
                                        ns=spec.ns,
                                        gm=gm,
                                        two_point_fit_tmin=two_point_tmin,
                                        two_point_fit_tmax=two_point_tmax,
                                        two_point_fit_table_resolved=str(fit_table_path),
                                        two_point_fit_tmax_source="config",
                                        tsrange_start=t0,
                                        tsrange_end=t1,
                                        ratio_samples=ratio_by_node[node],
                                    )
                                )
                            for bT, records in records_by_bT.items():
                                combo_stem = f"{title}_{sanitize_token(gm)}_{sanitize_token(eta)}_bT{bT}"
                                outputs.extend(
                                    _write_component_outputs(
                                        dataset_root,
                                        combo_stem,
                                        tuple(records),
                                        spec.nt,
                                        make_plots=False,
                                    )
                                )
    return outputs
