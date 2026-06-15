import tempfile
import unittest
from pathlib import Path

import numpy as np

from lqcd_analysis.tmdwf.fourier import load_tmdwf_m0_fit_table, load_tmdwf_m0_sample_table
from lqcd_analysis.tmdwf.models import evaluate_tmdwf_numerator, evaluate_tmdwf_ratio, evaluate_two_point_symmetric
from lqcd_analysis.tmdwf.nstate_diff_fit import (
    DifferenceEdge,
    EdgeFitResult,
    _write_plaquette_diagnostics,
    fit_tmdwf_edge_delta,
    reconstruct_tmdwf_graph_samples,
    run_tmdwf_nstate_diff_fit,
)

try:
    import h5py
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    h5py = None


class TMDWFNStateDiffFitTests(unittest.TestCase):
    @staticmethod
    def _write_two_point_fit_reference(
        root: Path,
        title: str,
        pz: int,
        amplitudes: np.ndarray,
        energies: np.ndarray,
        *,
        nstates: int = 1,
        tmin: int = 2,
        tmax: int = 5,
    ) -> None:
        tables_dir = root / title / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        table_path = tables_dir / f"{title}_normal_{nstates}state_tmax{tmax}_fits.txt"
        if nstates != 1:
            raise NotImplementedError("test helper only writes one-state references")
        row = [
            tmin,
            tmax,
            1.0,
            0.5,
            0.1,
            0.2,
            0.3,
            0.4,
            1.0,
            float(amplitudes[0]),
            0.1,
            float(energies[0]),
            0.02,
        ]
        np.savetxt(table_path, np.asarray([row], dtype=float))

    def test_edge_delta_linear_fit_recovers_known_delta_matrix_element(self) -> None:
        nt = 12
        ns = 16
        times = np.arange(nt)
        amplitudes = np.asarray([2.8], dtype=float)
        energies = np.asarray([0.42], dtype=float)
        delta_m = 0.35
        delta_ratio = evaluate_tmdwf_ratio(times, amplitudes, energies, np.asarray([delta_m]), nt, gm="T5", pz=0, ns=ns)
        delta_ratio_samples = np.tile(delta_ratio, (5, 1))

        samples, _, _, _ = fit_tmdwf_edge_delta(
            delta_ratio_samples,
            amplitudes,
            energies,
            nt,
            0,
            ns,
            "T5",
            2,
            5,
            "real",
        )
        np.testing.assert_allclose(samples[:, 0], delta_m, rtol=0.0, atol=1e-12)

    def test_graph_reconstruction_recovers_2x2_nodes_from_edges(self) -> None:
        nodes = ((0, 0), (0, 1), (1, 0), (1, 1))
        edges = (
            DifferenceEdge("bz", (0, 0), (0, 1)),
            DifferenceEdge("bz", (1, 0), (1, 1)),
            DifferenceEdge("bT", (0, 0), (1, 0)),
            DifferenceEdge("bT", (0, 1), (1, 1)),
        )
        deltas = {
            ((0, 0), (0, 1)): 0.1,
            ((1, 0), (1, 1)): 0.1,
            ((0, 0), (1, 0)): 0.2,
            ((0, 1), (1, 1)): 0.2,
        }
        edge_results = {
            (edge.source, edge.target): EdgeFitResult(
                edge=edge,
                delta_samples=np.full((3, 1), deltas[(edge.source, edge.target)]),
                sigma=np.asarray([0.1]),
                chi2_dof_samples=np.zeros(3),
                pvalue_samples=np.ones(3),
            )
            for edge in edges
        }
        reconstruction = reconstruct_tmdwf_graph_samples(
            nodes,
            edges,
            edge_results,
            np.full((3, 1), 1.0),
        )
        expected = {
            (0, 0): 1.0,
            (0, 1): 1.1,
            (1, 0): 1.2,
            (1, 1): 1.3,
        }
        for node, value in expected.items():
            np.testing.assert_allclose(reconstruction.samples_by_node[node][:, 0], value, rtol=0.0, atol=1e-12)

    def test_plaquette_closure_reports_inconsistent_edges(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            edges = (
                DifferenceEdge("bz", (0, 0), (0, 1)),
                DifferenceEdge("bT", (0, 1), (1, 1)),
                DifferenceEdge("bz", (1, 0), (1, 1)),
                DifferenceEdge("bT", (0, 0), (1, 0)),
            )
            values = [0.1, 0.2, 0.4, 0.2]
            edge_results = {
                (edge.source, edge.target): EdgeFitResult(
                    edge=edge,
                    delta_samples=np.full((4, 1), value),
                    sigma=np.asarray([0.1]),
                    chi2_dof_samples=np.zeros(4),
                    pvalue_samples=np.ones(4),
                )
                for edge, value in zip(edges, values, strict=True)
            }
            path = _write_plaquette_diagnostics(tmp, "demo", (0, 1), (0, 1), edge_results)
            text = path.read_text(encoding="utf-8")
        self.assertIn("-3.0000000000e-01", text)

    @unittest.skipIf(h5py is None, "h5py is required for TMDWF HDF5 tests")
    def test_workflow_writes_downstream_compatible_reconstructed_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            nt = 12
            ns = 16
            pz = 0
            times = np.arange(nt)
            amplitudes = np.asarray([2.8], dtype=float)
            energies = np.asarray([0.42], dtype=float)
            matrix_values = {
                (0, 0): 1.0,
                (0, 1): 1.1,
                (1, 0): 1.2,
                (1, 1): 1.3,
            }
            denominator = evaluate_two_point_symmetric(times, amplitudes, energies, nt)
            c2pt_csv = tmp / "c2pt_pz0.csv"
            with c2pt_csv.open("w", encoding="utf-8") as handle:
                handle.write(",".join(["t"] + [f"cfg_{idx}" for idx in range(6)]) + "\n")
                for t in range(nt):
                    handle.write(str(t) + "," + ",".join(f"{denominator[t]:.12e}" for _ in range(6)) + "\n")

            h5_path = tmp / "tmdwf_pz0.h5"
            with h5py.File(h5_path, "w") as handle:
                for (bT, bz), matrix_value in matrix_values.items():
                    numerator = evaluate_tmdwf_numerator(
                        times,
                        amplitudes,
                        energies,
                        np.asarray([matrix_value]),
                        nt,
                        gm="T5",
                        pz=pz,
                        ns=ns,
                    )
                    for tdir in ("plus", "minus"):
                        for signed_bz in ((0,) if bz == 0 else (bz, -bz)):
                            handle.create_dataset(
                                f"T5/eta0/pz0/{tdir}/bT{bT}/bz{signed_bz}",
                                data=np.tile(numerator, (6, 1)),
                            )

            two_point_fit_window_path = tmp / "two_point_fit_windows.txt"
            two_point_fit_window_path.write_text("0 2 5\n", encoding="utf-8")
            fit_window_path = tmp / "fit_window.txt"
            fit_window_path.write_text("0 2 5\n", encoding="utf-8")
            two_point_fit_root = tmp / "two_point_fit_root"
            self._write_two_point_fit_reference(
                two_point_fit_root,
                "demo_pz0",
                pz,
                amplitudes,
                energies,
            )
            input_path = tmp / "input_tmdwf_diff.txt"
            input_path.write_text(
                "\n".join(
                    [
                        "demo_pz* 16 12 0.076",
                        "fit_component real",
                        "nstates 1",
                        "pzlist 0",
                        "gmlist T5",
                        "etalist eta0",
                        "Tdirlist plus minus",
                        "bTlist 0 1",
                        "bzlist 0 1",
                        "binsize 1",
                        "bootstrap_samples 6",
                        "bootstrap_size 6",
                        "seed 11",
                        f"fit_window {fit_window_path}",
                        f"two_point_fit_root {two_point_fit_root}",
                        f"qtmdwf_h5 {h5_path}",
                        "dataset_path_template {gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
                        f"two_point_fit_window_by_pz {two_point_fit_window_path}",
                        f"c2pt {tmp / 'c2pt_pz*.csv'}",
                        "fold_t periodic",
                        "tsrange 0 6",
                        f"results_dir {tmp / 'results'}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            outputs = run_tmdwf_nstate_diff_fit(input_path)
            fit_path = tmp / "results" / "demo_pz0" / "tables" / "demo_pz0_T5_eta0_bT1_real_1state_fit.txt"
            sample_path = tmp / "results" / "demo_pz0" / "samples" / "demo_pz0_T5_eta0_bT1_real_1state_samples.txt"
            bz_fit, m0_mean, _ = load_tmdwf_m0_fit_table(fit_path)
            bz_samples, m0_samples = load_tmdwf_m0_sample_table(sample_path)

        self.assertIn(fit_path, outputs)
        self.assertTrue(np.array_equal(bz_fit, np.asarray([0, 1])))
        np.testing.assert_allclose(m0_mean, np.asarray([1.2, 1.3]), rtol=0.0, atol=1e-10)
        self.assertTrue(np.array_equal(bz_samples, np.asarray([0, 1])))
        np.testing.assert_allclose(m0_samples[:, 1], 1.3, rtol=0.0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
