import tempfile
import unittest
from pathlib import Path

import numpy as np

from lqcd_analysis.tmdwf.fit_nstate import (
    build_bootstrap_ratio_samples,
    fit_tmdwf_component,
    parse_tmdwf_fit_input,
    run_tmdwf_nstate_fit,
)
from lqcd_analysis.tmdwf.io import (
    apply_tmdwf_preprocessing,
    expand_template,
    load_tmdwf_correlator,
    load_two_point_plateau_values,
)
from lqcd_analysis.tmdwf.models import (
    evaluate_tmdwf_numerator,
    evaluate_tmdwf_numerator_t5,
    evaluate_tmdwf_numerator_z5,
    evaluate_tmdwf_ratio,
    evaluate_two_point_symmetric,
)

try:
    import h5py
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    h5py = None


@unittest.skipIf(h5py is None, "h5py is required for TMDWF HDF5 tests")
class TMDWFFitTests(unittest.TestCase):
    def test_parse_input_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input_tmdwf.txt"
            input_file.write_text(
                "\n".join(
                    [
                        "demo_pz* 32 16 0.076",
                        "fit_target ratio",
                        "fit_component both",
                        "nstates 1 2",
                        "pzlist 0 1",
                        "gmlist T5",
                        "etalist eta0",
                        "Tdirlist plus minus",
                        "bTrange 0 1",
                        "bzlist 0 2",
                        "binsize 2",
                        "bootstrap_samples 12",
                        "bootstrap_size 8",
                        "seed 7",
                        "tmin 2",
                        "tmax 6",
                        "qtmdwf_h5 /tmp/tmdwf_pz*.h5",
                        "dataset_path_template {gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
                        "two_point_plateau_table /tmp/plateau_pz*.txt",
                        "c2pt /tmp/c2pt_pz*.csv",
                        "fold_t periodic",
                        "tsrange 0 8",
                        "plot false",
                        "results_dir /tmp/tmdwf_results",
                    ]
                ),
                encoding="utf-8",
            )
            parsed = parse_tmdwf_fit_input(input_file)

        self.assertEqual(parsed.title_pattern, "demo_pz*")
        self.assertEqual(parsed.nstates, (1, 2))
        self.assertEqual(parsed.pzlist, (0, 1))
        self.assertEqual(parsed.bTlist, (0, 1))
        self.assertEqual(parsed.bzlist, (0, 2))
        self.assertEqual(parsed.fit_component, "both")
        self.assertEqual(parsed.results_dir, Path("/tmp/tmdwf_results"))
        self.assertEqual(parsed.fold_t, "periodic")

    def test_parse_input_file_normalizes_fold_t_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input_tmdwf.txt"
            input_file.write_text(
                "\n".join(
                    [
                        "demo_pz* 32 16 0.076",
                        "fit_target ratio",
                        "fit_component both",
                        "nstates 1",
                        "pzlist 0",
                        "gmlist T5",
                        "etalist eta0",
                        "Tdirlist plus minus",
                        "bTlist 0",
                        "bzlist 0",
                        "tmin 2",
                        "tmax 6",
                        "qtmdwf_h5 /tmp/tmdwf_pz*.h5",
                        "dataset_path_template {gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
                        "two_point_plateau_table /tmp/plateau_pz*.txt",
                        "c2pt /tmp/c2pt_pz*.csv",
                        "fold_t true",
                        "tsrange 0 8",
                    ]
                ),
                encoding="utf-8",
            )

            parsed = parse_tmdwf_fit_input(input_file)

        self.assertEqual(parsed.fold_t, "periodic")

    def test_expand_template_and_load_hdf5_correlator(self) -> None:
        self.assertEqual(
            expand_template(
                "{gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
                gm="T5",
                eta="eta0",
                pz=1,
                Tdir="plus",
                bT=2,
                bz=-3,
            ),
            "T5/eta0/pz1/plus/bT2/bz-3",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "demo.h5"
            nt = 8
            base = np.arange(1, nt + 1, dtype=float)
            with h5py.File(h5_path, "w") as handle:
                for tdir in ("plus", "minus"):
                    for bz in (1, -1):
                        path = f"T5/eta0/pz1/{tdir}/bT0/bz{bz}"
                        values = np.tile(base + (0.1 if tdir == "minus" else 0.0), (3, 1))
                        handle.create_dataset(path, data=values)

            loaded = load_tmdwf_correlator(
                h5_path,
                "{gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
                gm="T5",
                eta="eta0",
                pz=1,
                tdirs=("plus", "minus"),
                bT=0,
                bz=1,
                nt=nt,
                ns=32,
            )

        self.assertEqual(loaded.shape, (3, nt // 2 + 1))
        self.assertTrue(np.iscomplexobj(loaded))

    def test_model_evaluation_ratio_formula(self) -> None:
        times = np.arange(2, 6)
        amplitudes = np.array([3.0])
        energies = np.array([0.4])
        matrix_elements = np.array([1.7])
        nt = 16
        pz = 2
        ns = 32

        denominator = evaluate_two_point_symmetric(times, amplitudes, energies, nt)
        numerator = evaluate_tmdwf_numerator_t5(times, amplitudes, energies, matrix_elements, nt)
        ratio = evaluate_tmdwf_ratio(times, amplitudes, energies, matrix_elements, nt, gm="T5", pz=pz, ns=ns)

        expected_numerator = 0.5 * np.sqrt(amplitudes[0] * 2.0 * energies[0]) * matrix_elements[0] * (
            np.exp(-energies[0] * times) - np.exp(-energies[0] * (nt - times))
        )
        self.assertTrue(np.allclose(numerator, expected_numerator))
        self.assertTrue(np.allclose(ratio, numerator / denominator))

    def test_z5_model_includes_momentum_over_energy_factor(self) -> None:
        times = np.arange(2, 6)
        amplitudes = np.array([3.0])
        energies = np.array([0.4])
        matrix_elements = np.array([1.7])
        nt = 16
        pz = 2
        ns = 32
        momentum = 2.0 * np.pi * pz / ns

        numerator = evaluate_tmdwf_numerator_z5(times, amplitudes, energies, matrix_elements, nt, pz=pz, ns=ns)
        t5_numerator = evaluate_tmdwf_numerator_t5(times, amplitudes, energies, matrix_elements, nt)
        self.assertTrue(np.allclose(numerator, t5_numerator * (momentum / energies[0])))

        ratio = evaluate_tmdwf_ratio(times, amplitudes, energies, matrix_elements, nt, gm="Z5", pz=pz, ns=ns)
        denominator = evaluate_two_point_symmetric(times, amplitudes, energies, nt)
        self.assertTrue(np.allclose(ratio, numerator / denominator))

    def test_operator_dependent_preprocessing(self) -> None:
        nt = 8
        values = np.ones((2, nt), dtype=np.complex128)
        t5 = apply_tmdwf_preprocessing(values, nt, "T5")
        z5 = apply_tmdwf_preprocessing(values, nt, "Z5")

        expected_t5 = np.ones((2, nt), dtype=np.complex128)
        expected_t5[:, nt // 2 :] = -1.0
        self.assertTrue(np.allclose(t5, expected_t5))
        self.assertTrue(np.allclose(z5, -1j * np.ones((2, nt), dtype=np.complex128)))

    def test_bootstrap_ratio_construction_preserves_complex_values(self) -> None:
        numerator = np.array(
            [
                [1.0 + 1.0j, 2.0 + 0.5j],
                [1.2 + 1.1j, 2.1 + 0.6j],
                [0.9 + 0.8j, 1.8 + 0.4j],
                [1.1 + 0.9j, 2.2 + 0.7j],
            ]
        )
        denominator = np.array(
            [
                [2.0, 4.0],
                [2.1, 4.2],
                [1.9, 3.8],
                [2.2, 4.4],
            ]
        )

        ratio_boot, numerator_boot, denominator_boot = build_bootstrap_ratio_samples(
            numerator,
            denominator,
            binsize=2,
            bootstrap_samples=6,
            bootstrap_size=2,
            seed=5,
        )

        self.assertEqual(ratio_boot.shape, (6, 2))
        self.assertEqual(numerator_boot.shape, (6, 2))
        self.assertEqual(denominator_boot.shape, (6, 2))
        self.assertTrue(np.any(np.abs(np.imag(ratio_boot)) > 0.0))

    def test_load_two_point_plateau_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "plateau.txt"
            row = np.array([[3.0, 1.1, 0.40, 0.90, 0.2, 0.3, 0.04, 0.09]])
            np.savetxt(path, row)
            amplitudes, energies = load_two_point_plateau_values(path, 2)
        self.assertTrue(np.allclose(amplitudes, [3.0, 1.1]))
        self.assertTrue(np.allclose(energies, [0.40, 0.90]))

    def test_fit_tmdwf_component_one_state(self) -> None:
        nt = 16
        amplitudes = np.array([2.5])
        energies = np.array([0.35])
        matrix_elements = np.array([1.2])
        times = np.arange(0, nt // 2 + 1)
        model = evaluate_tmdwf_ratio(times, amplitudes, energies, matrix_elements, nt, gm="T5", pz=0, ns=32)
        rng = np.random.default_rng(7)
        ratio_samples = model[None, :] + 0.002 * rng.normal(size=(24, times.size))

        meanfit, sample_params = fit_tmdwf_component(ratio_samples, amplitudes, energies, nt, 0, 32, "T5", 2, 6, "real")

        self.assertTrue(meanfit.success)
        self.assertAlmostEqual(meanfit.params[0], matrix_elements[0], delta=0.05)
        self.assertEqual(sample_params.shape, (24, 1))

    def test_fit_tmdwf_component_two_state(self) -> None:
        nt = 20
        amplitudes = np.array([3.5, 1.2])
        energies = np.array([0.30, 0.70])
        matrix_elements = np.array([0.8, -0.35])
        times = np.arange(0, nt // 2 + 1)
        real_model = evaluate_tmdwf_ratio(times, amplitudes, energies, matrix_elements, nt, gm="T5", pz=0, ns=32)
        rng = np.random.default_rng(11)
        ratio_samples = real_model[None, :] + 0.002 * rng.normal(size=(30, times.size))

        meanfit, sample_params = fit_tmdwf_component(ratio_samples, amplitudes, energies, nt, 0, 32, "T5", 2, 8, "real")

        self.assertTrue(meanfit.success)
        self.assertTrue(np.allclose(meanfit.params, matrix_elements, atol=0.08))
        self.assertEqual(sample_params.shape, (30, 2))

    def test_end_to_end_workflow_writes_outputs_for_t5_and_z5(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            nt = 12
            ns = 16
            times = np.arange(nt)
            folded_times = np.arange(nt // 2 + 1)
            amplitudes = np.array([2.8])
            energies = np.array([0.42])
            matrix_elements = np.array([1.1])
            denominator = evaluate_two_point_symmetric(times, amplitudes, energies, nt)

            c2pt_csv = tmp / "c2pt_pz0.csv"
            c2pt_data = np.column_stack(
                [denominator * (1.0 + 0.001 * np.cos(times + cfg)) for cfg in range(10)]
            )
            with c2pt_csv.open("w", encoding="utf-8") as handle:
                handle.write(",".join(["t"] + [f"cfg_{idx}" for idx in range(c2pt_data.shape[1])]) + "\n")
                for t in range(nt):
                    row = ",".join([str(t)] + [f"{value:.12e}" for value in c2pt_data[t]])
                    handle.write(row + "\n")

            plateau_path = tmp / "plateau_pz0.txt"
            np.savetxt(plateau_path, np.array([[amplitudes[0], energies[0], 0.05, 0.02]]))

            h5_path = tmp / "tmdwf_pz0.h5"
            with h5py.File(h5_path, "w") as handle:
                for gm in ("T5", "Z5"):
                    numerator = evaluate_tmdwf_numerator(times, amplitudes, energies, matrix_elements, nt, gm=gm, pz=0, ns=ns)
                    for tdir in ("plus", "minus"):
                        dataset = np.column_stack(
                            [numerator * (1.0 + 0.001 * np.sin(times + cfg)) for cfg in range(10)]
                        ).T
                        handle.create_dataset(
                            f"{gm}/eta0/pz0/{tdir}/bT0/bz0",
                            data=dataset,
                        )

            input_path = tmp / "input_tmdwf.txt"
            input_path.write_text(
                "\n".join(
                    [
                        "demo_pz* 16 12 0.076",
                        "fit_target ratio",
                        "fit_component both",
                        "nstates 1",
                        "pzlist 0",
                        "gmlist T5 Z5",
                        "etalist eta0",
                        "Tdirlist plus minus",
                        "bTlist 0",
                        "bzlist 0",
                        "binsize 1",
                        "bootstrap_samples 12",
                        "bootstrap_size 10",
                        "seed 9",
                        "tmin 2",
                        f"tmax {folded_times[-1]}",
                        f"qtmdwf_h5 {h5_path}",
                        "dataset_path_template {gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
                        f"two_point_plateau_table {tmp / 'plateau_pz*.txt'}",
                        f"c2pt {tmp / 'c2pt_pz*.csv'}",
                        "fold_t periodic",
                        f"tsrange 0 {folded_times[-1]}",
                        f"results_dir {tmp / 'results'}",
                    ]
                ),
                encoding="utf-8",
            )

            outputs = run_tmdwf_nstate_fit(input_path)

            self.assertTrue(outputs)
            for gm in ("T5", "Z5"):
                summary = tmp / "results" / "demo_pz0" / f"demo_pz0_{gm}_eta0_bT0_bz0_real_1state_summary.txt"
                imag_summary = tmp / "results" / "demo_pz0" / f"demo_pz0_{gm}_eta0_bT0_bz0_imag_1state_summary.txt"
                self.assertTrue(summary.exists())
                self.assertTrue(imag_summary.exists())
                self.assertIn("m0", summary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
