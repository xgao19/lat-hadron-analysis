import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from lqcd_analysis.tmdwf.fit_nstate import (
    _select_curve_component,
    _write_component_outputs,
    build_bootstrap_ratio_samples,
    fit_tmdwf_component,
    fit_tmdwf_mean_component,
    parse_tmdwf_fit_input,
    run_tmdwf_nstate_fit,
    scan_reference_tmin_rows,
    TMDWFFitResult,
    TMDWFOutputRecord,
)
from lqcd_analysis.tmdwf.io import (
    _load_tmdwf_correlator_from_handle,
    apply_tmdwf_preprocessing,
    expand_template,
    fold_symmetric_complex,
    load_tmdwf_correlator,
    load_two_point_plateau_values,
    resolve_qtmdwf_h5_path,
    resolve_two_point_plateau_table,
)
from lqcd_analysis.tmdwf.models import (
    evaluate_tmdwf_numerator,
    evaluate_tmdwf_numerator_t5,
    evaluate_tmdwf_numerator_z5,
    evaluate_tmdwf_ratio,
    evaluate_two_point_symmetric,
)
from lqcd_analysis.tmdwf.plotting import (
    RatioFitPlotSeries,
    TMDWFM0VsBZSeries,
    _build_m0_series_from_fit_tables,
    plot_tmdwf_grouped_outputs,
    plot_tmdwf_m0_from_fit_tables,
    plot_tmdwf_m0_vs_bz,
    plot_tmdwf_ratio_fit,
    write_tmdwf_plot_notebook,
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
        self.assertEqual(parsed.tmax, 6)
        self.assertFalse(parsed.shared_window_by_pz_gm)
        self.assertIsNone(parsed.decay_constant)
        self.assertEqual(parsed.min_fit_dof, 1)

    def test_parse_input_file_with_shared_window_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input_tmdwf.txt"
            input_file.write_text(
                "\n".join(
                    [
                        "demo_pz* 32 16 0.076",
                        "fit_target ratio",
                        "fit_component real",
                        "nstates 1",
                        "pzlist 0",
                        "gmlist T5",
                        "etalist eta0 eta1",
                        "Tdirlist plus minus",
                        "bTlist 0 1",
                        "bzlist 0",
                        "tmin 2",
                        "shared_window_by_pz_gm true",
                        "decay_constant 0.12 0.03",
                        "min_fit_dof 2",
                        "qtmdwf_h5 /tmp/tmdwf_pz*.h5",
                        "dataset_path_template {gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
                        "two_point_plateau_table /tmp/plateau_pz*.txt",
                        "c2pt /tmp/c2pt_pz*.csv",
                        "fold_t periodic",
                    ]
                ),
                encoding="utf-8",
            )
            parsed = parse_tmdwf_fit_input(input_file)
        self.assertTrue(parsed.shared_window_by_pz_gm)
        self.assertEqual(parsed.decay_constant, (0.12, 0.03))
        self.assertEqual(parsed.min_fit_dof, 2)

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

    def test_parse_input_file_defaults_tsrange_when_missing(self) -> None:
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
                    ]
                ),
                encoding="utf-8",
            )
            parsed = parse_tmdwf_fit_input(input_file)

        self.assertEqual(parsed.tsrange, (0, 7))

    def test_parse_input_file_defaults_tmax_to_auto_when_missing(self) -> None:
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
                        "qtmdwf_h5 /tmp/tmdwf_pz*.h5",
                        "dataset_path_template {gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
                        "two_point_plateau_table /tmp/plateau_pz*.txt",
                        "c2pt /tmp/c2pt_pz*.csv",
                        "fold_t true",
                    ]
                ),
                encoding="utf-8",
            )
            parsed = parse_tmdwf_fit_input(input_file)

        self.assertIsNone(parsed.tmax)

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

    def test_resolve_qtmdwf_h5_path_supports_gm_and_legacy_pz_wildcard(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gm_path = tmp / "qtmdwf_pz0_OT5.h5"
            gm_path.touch()
            legacy_path = tmp / "qtmdwf_pz0.h5"
            legacy_path.touch()

            self.assertEqual(
                resolve_qtmdwf_h5_path(tmp / "qtmdwf_pz{pz}_O{gm}.h5", pz=0, gm="T5"),
                gm_path,
            )
            self.assertEqual(
                resolve_qtmdwf_h5_path(tmp / "qtmdwf_pz*.h5", pz=0, gm="Z5"),
                legacy_path,
            )

    def test_resolve_qtmdwf_h5_path_missing_file_reports_fully_resolved_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with self.assertRaises(FileNotFoundError) as ctx:
                resolve_qtmdwf_h5_path(tmp / "qtmdwf_pz{pz}_O{gm}.h5", pz=0, gm="Z5")
        self.assertEqual(str(ctx.exception), str(tmp / "qtmdwf_pz0_OZ5.h5"))

    def test_handle_based_loader_matches_path_based_loader(self) -> None:
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

            path_loaded = load_tmdwf_correlator(
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
            with h5py.File(h5_path, "r") as handle:
                handle_loaded = _load_tmdwf_correlator_from_handle(
                    handle,
                    "{gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
                    gm="T5",
                    eta="eta0",
                    pz=1,
                    tdirs=("plus", "minus"),
                    bT=0,
                    bz=1,
                    nt=nt,
                    ns=32,
                    file_label=str(h5_path),
                )
        self.assertTrue(np.allclose(path_loaded, handle_loaded))

    def test_resolve_plateau_table_hash_selects_largest_tmax(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            first = tmp / "demo_pz0_normal_1state_tmax8_plateau.txt"
            second = tmp / "demo_pz0_normal_1state_tmax12_plateau.txt"
            np.savetxt(first, np.array([[1.0, 0.4, 0.1, 0.02]]))
            np.savetxt(second, np.array([[1.0, 0.4, 0.1, 0.02]]))
            resolved, inferred_tmax = resolve_two_point_plateau_table(
                tmp / "demo_pz*_normal_1state_tmax#_plateau.txt",
                pz=0,
            )
        self.assertEqual(resolved, second)
        self.assertEqual(inferred_tmax, 12)

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

    def test_tmdwf_loader_uses_symmetric_fold_after_preprocessing(self) -> None:
        nt = 8
        raw = np.array([[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]], dtype=float)
        processed = apply_tmdwf_preprocessing(raw, nt, "T5")
        expected = fold_symmetric_complex(processed, nt)

        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "demo.h5"
            with h5py.File(h5_path, "w") as handle:
                handle.create_dataset("T5/eta0/pz0/plus/bT0/bz0", data=raw)
            loaded = load_tmdwf_correlator(
                h5_path,
                "{gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
                gm="T5",
                eta="eta0",
                pz=0,
                tdirs=("plus",),
                bT=0,
                bz=0,
                nt=nt,
                ns=32,
            )

        self.assertTrue(np.allclose(loaded, expected))

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

    def test_select_curve_component_uses_requested_part(self) -> None:
        values = np.array([[1.0 + 3.0j, 2.0 + 4.0j], [5.0 + 7.0j, 6.0 + 8.0j]])
        self.assertTrue(np.allclose(_select_curve_component(values, "real"), np.array([[1.0, 2.0], [5.0, 6.0]])))
        self.assertTrue(np.allclose(_select_curve_component(values, "imag"), np.array([[3.0, 4.0], [7.0, 8.0]])))

    def test_load_two_point_plateau_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "plateau.txt"
            row = np.array([[3.0, 1.1, 0.40, 0.90, 0.2, 0.3, 0.04, 0.09]])
            np.savetxt(path, row)
            amplitudes, energies = load_two_point_plateau_values(path, 2)
        self.assertTrue(np.allclose(amplitudes, [3.0, 1.1]))
        self.assertTrue(np.allclose(energies, [0.40, 0.90]))

    def test_curve_output_uses_imag_component_for_imag_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            times = np.arange(0, 3, dtype=int)
            amplitudes = np.array([2.5])
            energies = np.array([0.35])
            sample_params = np.array([[1.0], [1.2]], dtype=float)
            record = TMDWFOutputRecord(
                bz=0,
                component="imag",
                nstates=1,
                tmin=0,
                tmax=2,
                fit_result=TMDWFFitResult(
                    params=np.array([1.1]),
                    chi2=0.0,
                    chi2_dof=0.0,
                    pvalue=1.0,
                    success=True,
                    message="ok",
                ),
                sample_params=sample_params,
                amplitudes=amplitudes,
                energies=energies,
                pz=1,
                ns=16,
                gm="Z5",
                shared_window_flag=0,
                reference_eta="none",
                reference_bT=-1,
                reference_bz=-1,
                plateau_tmax_used=2,
                two_point_plateau_table_resolved="plateau.txt",
                two_point_tmax_source="explicit",
                two_point_tmax_inferred="none",
                tsrange_start=0,
                tsrange_end=2,
                ratio_samples=np.ones((2, 3), dtype=np.complex128),
            )
            _write_component_outputs(output_root, "demo", (record,), 8, make_plots=False)
            curve_path = output_root / "tables" / "demo_imag_1state_curve.txt"
            rows = curve_path.read_text(encoding="utf-8").splitlines()[1:]
            written = np.array([float(line.split("\t")[3]) for line in rows])
            expected_curves = np.array(
                [
                    np.imag(evaluate_tmdwf_ratio(times, amplitudes, energies, params, 8, gm="Z5", pz=1, ns=16))
                    for params in sample_params
                ]
            )
            expected_mean = 0.5 * (
                np.percentile(expected_curves, 84.0, axis=0) + np.percentile(expected_curves, 16.0, axis=0)
            )
        self.assertTrue(np.allclose(written, expected_mean))

    def test_curve_output_uses_real_component_for_real_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            times = np.arange(0, 3, dtype=int)
            amplitudes = np.array([2.5])
            energies = np.array([0.35])
            sample_params = np.array([[1.0], [1.2]], dtype=float)
            record = TMDWFOutputRecord(
                bz=0,
                component="real",
                nstates=1,
                tmin=0,
                tmax=2,
                fit_result=TMDWFFitResult(
                    params=np.array([1.1]),
                    chi2=0.0,
                    chi2_dof=0.0,
                    pvalue=1.0,
                    success=True,
                    message="ok",
                ),
                sample_params=sample_params,
                amplitudes=amplitudes,
                energies=energies,
                pz=0,
                ns=16,
                gm="T5",
                shared_window_flag=0,
                reference_eta="none",
                reference_bT=-1,
                reference_bz=-1,
                plateau_tmax_used=2,
                two_point_plateau_table_resolved="plateau.txt",
                two_point_tmax_source="explicit",
                two_point_tmax_inferred="none",
                tsrange_start=0,
                tsrange_end=2,
                ratio_samples=np.ones((2, 3), dtype=np.complex128),
            )
            _write_component_outputs(output_root, "demo", (record,), 8, make_plots=False)
            curve_path = output_root / "tables" / "demo_real_1state_curve.txt"
            rows = curve_path.read_text(encoding="utf-8").splitlines()[1:]
            written = np.array([float(line.split("\t")[3]) for line in rows])
            expected_curves = np.array(
                [
                    np.real(evaluate_tmdwf_ratio(times, amplitudes, energies, params, 8, gm="T5", pz=0, ns=16))
                    for params in sample_params
                ]
            )
            expected_mean = 0.5 * (
                np.percentile(expected_curves, 84.0, axis=0) + np.percentile(expected_curves, 16.0, axis=0)
            )
        self.assertTrue(np.allclose(written, expected_mean))

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

    def test_fit_tmdwf_mean_component_runs_mean_fit_only(self) -> None:
        nt = 16
        amplitudes = np.array([2.5])
        energies = np.array([0.35])
        matrix_elements = np.array([1.2])
        times = np.arange(0, nt // 2 + 1)
        model = evaluate_tmdwf_ratio(times, amplitudes, energies, matrix_elements, nt, gm="T5", pz=0, ns=32)
        rng = np.random.default_rng(7)
        ratio_samples = model[None, :] + 0.002 * rng.normal(size=(24, times.size))

        meanfit = fit_tmdwf_mean_component(ratio_samples, amplitudes, energies, nt, 0, 32, "T5", 2, 6, "real")

        self.assertTrue(meanfit.success)
        self.assertAlmostEqual(meanfit.params[0], matrix_elements[0], delta=0.05)

    def test_scan_reference_tmin_rows_uses_mean_fits_only(self) -> None:
        nt = 16
        amplitudes = np.array([2.5])
        energies = np.array([0.35])
        matrix_elements = np.array([1.2])
        times = np.arange(0, nt // 2 + 1)
        model = evaluate_tmdwf_ratio(times, amplitudes, energies, matrix_elements, nt, gm="T5", pz=0, ns=32)
        rng = np.random.default_rng(7)
        ratio_samples = model[None, :] + 0.002 * rng.normal(size=(24, times.size))

        with patch("lqcd_analysis.tmdwf.fit_nstate.fit_tmdwf_component", side_effect=AssertionError("full fit should not be used")):
            rows = scan_reference_tmin_rows(
                ratio_samples,
                amplitudes,
                energies,
                nt=nt,
                pz=0,
                ns=32,
                gm="T5",
                tmin_start=2,
                tmax=6,
                nstates=1,
                min_fit_dof=1,
                component="real",
            )

        self.assertTrue(rows)
        self.assertTrue(all(np.isfinite(row.m0_mean) for row in rows))

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
                summary = tmp / "results" / "demo_pz0" / f"demo_pz0_{gm}_eta0_bT0_real_1state_summary.txt"
                imag_summary = tmp / "results" / "demo_pz0" / f"demo_pz0_{gm}_eta0_bT0_imag_1state_summary.txt"
                self.assertTrue(summary.exists())
                self.assertTrue(imag_summary.exists())
                self.assertIn("m0", summary.read_text(encoding="utf-8"))

    def test_end_to_end_workflow_selects_gm_specific_qtmdwf_files(self) -> None:
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

            for gm in ("T5", "Z5"):
                h5_path = tmp / f"tmdwf_pz0_O{gm}.h5"
                with h5py.File(h5_path, "w") as handle:
                    numerator = evaluate_tmdwf_numerator(
                        times,
                        amplitudes,
                        energies,
                        matrix_elements,
                        nt,
                        gm=gm,
                        pz=0,
                        ns=ns,
                    )
                    for tdir in ("plus", "minus"):
                        dataset = np.column_stack(
                            [numerator * (1.0 + 0.001 * np.sin(times + cfg)) for cfg in range(10)]
                        ).T
                        handle.create_dataset(
                            f"{gm}/eta0/pz0/{tdir}/bT0/bz0",
                            data=dataset,
                        )

            input_path = tmp / "input_tmdwf_gm_specific.txt"
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
                        f"qtmdwf_h5 {tmp / 'tmdwf_pz{pz}_O{gm}.h5'}",
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
                summary = tmp / "results" / "demo_pz0" / f"demo_pz0_{gm}_eta0_bT0_real_1state_summary.txt"
                imag_summary = tmp / "results" / "demo_pz0" / f"demo_pz0_{gm}_eta0_bT0_imag_1state_summary.txt"
                self.assertTrue(summary.exists())
                self.assertTrue(imag_summary.exists())
                self.assertIn("m0", summary.read_text(encoding="utf-8"))

    def test_missing_gm_specific_qtmdwf_file_raises_with_resolved_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            c2pt_csv = tmp / "c2pt_pz0.csv"
            c2pt_csv.write_text("t,cfg_0\n0,1.0\n1,0.9\n2,0.8\n3,0.7\n", encoding="utf-8")
            plateau_path = tmp / "plateau_pz0.txt"
            np.savetxt(plateau_path, np.array([[1.0, 0.5, 0.05, 0.02]]))

            input_path = tmp / "input_missing_qtmdwf.txt"
            input_path.write_text(
                "\n".join(
                    [
                        "demo_pz* 16 4 0.076",
                        "fit_target ratio",
                        "fit_component real",
                        "nstates 1",
                        "pzlist 0",
                        "gmlist Z5",
                        "etalist eta0",
                        "Tdirlist plus minus",
                        "bTlist 0",
                        "bzlist 0",
                        "tmin 0",
                        "tmax 1",
                        f"qtmdwf_h5 {tmp / 'missing_pz{pz}_O{gm}.h5'}",
                        "dataset_path_template {gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
                        f"two_point_plateau_table {tmp / 'plateau_pz*.txt'}",
                        f"c2pt {tmp / 'c2pt_pz*.csv'}",
                        "fold_t none",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(FileNotFoundError) as ctx:
                run_tmdwf_nstate_fit(input_path)

        self.assertEqual(str(ctx.exception), str(tmp / "missing_pz0_OZ5.h5"))

    def test_explicit_tmax_overrides_inferred_plateau_filename_tmax(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            nt = 12
            times = np.arange(nt)
            amplitudes = np.array([2.8])
            energies = np.array([0.42])
            matrix_elements = np.array([1.1])
            denominator = evaluate_two_point_symmetric(times, amplitudes, energies, nt)
            numerator = evaluate_tmdwf_numerator(times, amplitudes, energies, matrix_elements, nt, gm="T5", pz=0, ns=16)

            c2pt_csv = tmp / "c2pt_pz0.csv"
            c2pt_data = np.column_stack([denominator for _ in range(8)])
            with c2pt_csv.open("w", encoding="utf-8") as handle:
                handle.write(",".join(["t"] + [f"cfg_{idx}" for idx in range(c2pt_data.shape[1])]) + "\n")
                for t in range(nt):
                    handle.write(str(t) + "," + ",".join(f"{value:.12e}" for value in c2pt_data[t]) + "\n")

            plateau_path = tmp / "plateau_pz0_tmax9_plateau.txt"
            np.savetxt(plateau_path, np.array([[amplitudes[0], energies[0], 0.05, 0.02]]))
            h5_path = tmp / "tmdwf_pz0.h5"
            with h5py.File(h5_path, "w") as handle:
                for tdir in ("plus", "minus"):
                    handle.create_dataset(f"T5/eta0/pz0/{tdir}/bT0/bz0", data=np.column_stack([numerator for _ in range(8)]).T)

            input_path = tmp / "input_tmdwf.txt"
            input_path.write_text(
                "\n".join(
                    [
                        "demo_pz* 16 12 0.076",
                        "fit_target ratio",
                        "fit_component real",
                        "nstates 1",
                        "pzlist 0",
                        "gmlist T5",
                        "etalist eta0",
                        "Tdirlist plus minus",
                        "bTlist 0",
                        "bzlist 0",
                        "tmin 2",
                        "tmax 5",
                        f"qtmdwf_h5 {h5_path}",
                        "dataset_path_template {gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
                        f"two_point_plateau_table {tmp / 'plateau_pz*_tmax#_plateau.txt'}",
                        f"c2pt {tmp / 'c2pt_pz*.csv'}",
                        "fold_t periodic",
                        f"results_dir {tmp / 'results'}",
                    ]
                ),
                encoding="utf-8",
            )
            run_tmdwf_nstate_fit(input_path)
            summary = (tmp / "results" / "demo_pz0" / "demo_pz0_T5_eta0_bT0_real_1state_summary.txt").read_text(encoding="utf-8")
        self.assertIn("tfit 2 5", summary)
        self.assertIn("two_point_plateau_table_resolved", summary)
        self.assertIn("two_point_tmax_source explicit", summary)
        self.assertIn("two_point_tmax_inferred 9", summary)

    def test_auto_or_missing_tmax_uses_inferred_plateau_filename_tmax(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            nt = 12
            times = np.arange(nt)
            amplitudes = np.array([2.8])
            energies = np.array([0.42])
            matrix_elements = np.array([1.1])
            denominator = evaluate_two_point_symmetric(times, amplitudes, energies, nt)
            numerator = evaluate_tmdwf_numerator(times, amplitudes, energies, matrix_elements, nt, gm="T5", pz=0, ns=16)

            c2pt_csv = tmp / "c2pt_pz0.csv"
            c2pt_data = np.column_stack([denominator for _ in range(8)])
            with c2pt_csv.open("w", encoding="utf-8") as handle:
                handle.write(",".join(["t"] + [f"cfg_{idx}" for idx in range(c2pt_data.shape[1])]) + "\n")
                for t in range(nt):
                    handle.write(str(t) + "," + ",".join(f"{value:.12e}" for value in c2pt_data[t]) + "\n")

            plateau_path = tmp / "plateau_pz0_tmax5_plateau.txt"
            np.savetxt(plateau_path, np.array([[amplitudes[0], energies[0], 0.05, 0.02]]))
            h5_path = tmp / "tmdwf_pz0.h5"
            with h5py.File(h5_path, "w") as handle:
                for tdir in ("plus", "minus"):
                    handle.create_dataset(f"T5/eta0/pz0/{tdir}/bT0/bz0", data=np.column_stack([numerator for _ in range(8)]).T)

            for include_tmax_line in (True, False):
                input_path = tmp / ("input_auto.txt" if include_tmax_line else "input_missing.txt")
                lines = [
                    "demo_pz* 16 12 0.076",
                    "fit_target ratio",
                    "fit_component real",
                    "nstates 1",
                    "pzlist 0",
                    "gmlist T5",
                    "etalist eta0",
                    "Tdirlist plus minus",
                    "bTlist 0",
                    "bzlist 0",
                    "tmin 2",
                ]
                if include_tmax_line:
                    lines.append("tmax auto")
                lines.extend(
                    [
                        f"qtmdwf_h5 {h5_path}",
                        "dataset_path_template {gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
                        f"two_point_plateau_table {tmp / 'plateau_pz*_tmax#_plateau.txt'}",
                        f"c2pt {tmp / 'c2pt_pz*.csv'}",
                        "fold_t periodic",
                        f"results_dir {tmp / ('results_auto' if include_tmax_line else 'results_missing')}",
                    ]
                )
                input_path.write_text("\n".join(lines), encoding="utf-8")
                run_tmdwf_nstate_fit(input_path)

            auto_summary = (tmp / "results_auto" / "demo_pz0" / "demo_pz0_T5_eta0_bT0_real_1state_summary.txt").read_text(encoding="utf-8")
            missing_summary = (tmp / "results_missing" / "demo_pz0" / "demo_pz0_T5_eta0_bT0_real_1state_summary.txt").read_text(encoding="utf-8")
        self.assertIn("tfit 2 5", auto_summary)
        self.assertIn("tfit 2 5", missing_summary)

    def test_auto_or_missing_tmax_without_inferable_plateau_filename_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            nt = 12
            times = np.arange(nt)
            amplitudes = np.array([2.8])
            energies = np.array([0.42])
            matrix_elements = np.array([1.1])
            denominator = evaluate_two_point_symmetric(times, amplitudes, energies, nt)
            numerator = evaluate_tmdwf_numerator(times, amplitudes, energies, matrix_elements, nt, gm="T5", pz=0, ns=16)
            c2pt_csv = tmp / "c2pt_pz0.csv"
            c2pt_data = np.column_stack([denominator for _ in range(8)])
            with c2pt_csv.open("w", encoding="utf-8") as handle:
                handle.write(",".join(["t"] + [f"cfg_{idx}" for idx in range(c2pt_data.shape[1])]) + "\n")
                for t in range(nt):
                    handle.write(str(t) + "," + ",".join(f"{value:.12e}" for value in c2pt_data[t]) + "\n")

            plateau_path = tmp / "plateau_pz0_plateau.txt"
            np.savetxt(plateau_path, np.array([[amplitudes[0], energies[0], 0.05, 0.02]]))
            h5_path = tmp / "tmdwf_pz0.h5"
            with h5py.File(h5_path, "w") as handle:
                for tdir in ("plus", "minus"):
                    handle.create_dataset(f"T5/eta0/pz0/{tdir}/bT0/bz0", data=np.column_stack([numerator for _ in range(8)]).T)

            input_path = tmp / "input_tmdwf.txt"
            input_path.write_text(
                "\n".join(
                    [
                        "demo_pz* 16 12 0.076",
                        "fit_target ratio",
                        "fit_component real",
                        "nstates 1",
                        "pzlist 0",
                        "gmlist T5",
                        "etalist eta0",
                        "Tdirlist plus minus",
                        "bTlist 0",
                        "bzlist 0",
                        "tmin 2",
                        f"qtmdwf_h5 {h5_path}",
                        "dataset_path_template {gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
                        f"two_point_plateau_table {plateau_path}",
                        f"c2pt {tmp / 'c2pt_pz*.csv'}",
                        "fold_t periodic",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "could not infer tmax"):
                run_tmdwf_nstate_fit(input_path)

    def test_shared_window_by_pz_gm_reuses_reference_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            nt = 12
            ns = 16
            times = np.arange(nt)
            amplitudes = np.array([2.8])
            energies = np.array([0.42])
            matrix_elements = np.array([0.11])
            denominator = evaluate_two_point_symmetric(times, amplitudes, energies, nt)
            c2pt_csv = tmp / "c2pt_pz0.csv"
            c2pt_data = np.column_stack(
                [denominator * (1.0 + 0.001 * np.cos(times + cfg)) for cfg in range(10)]
            )
            with c2pt_csv.open("w", encoding="utf-8") as handle:
                handle.write(",".join(["t"] + [f"cfg_{idx}" for idx in range(c2pt_data.shape[1])]) + "\n")
                for t in range(nt):
                    handle.write(str(t) + "," + ",".join(f"{value:.12e}" for value in c2pt_data[t]) + "\n")

            plateau_path = tmp / "plateau_pz0_tmax5_plateau.txt"
            np.savetxt(plateau_path, np.array([[amplitudes[0], energies[0], 0.05, 0.02]]))
            h5_path = tmp / "tmdwf_pz0.h5"
            with h5py.File(h5_path, "w") as handle:
                for eta in ("eta0", "eta1"):
                    for bT in (0, 1):
                        numerator = evaluate_tmdwf_numerator(
                            times, amplitudes, energies, matrix_elements, nt, gm="T5", pz=0, ns=ns
                        )
                        for tdir in ("plus", "minus"):
                            dataset = np.column_stack(
                                [numerator * (1.0 + 0.001 * np.sin(times + cfg + bT)) for cfg in range(10)]
                            ).T
                            handle.create_dataset(
                                f"T5/{eta}/pz0/{tdir}/bT{bT}/bz0",
                                data=dataset,
                            )

            input_path = tmp / "input_tmdwf.txt"
            input_path.write_text(
                "\n".join(
                    [
                        "demo_pz* 16 12 0.076",
                        "fit_target ratio",
                        "fit_component real",
                        "nstates 1",
                        "pzlist 0",
                        "gmlist T5",
                        "etalist eta0 eta1",
                        "Tdirlist plus minus",
                        "bTlist 0 1",
                        "bzlist 0",
                        "bootstrap_samples 12",
                        "bootstrap_size 10",
                        "seed 9",
                        "tmin 2",
                        "shared_window_by_pz_gm true",
                        "decay_constant 0.11 0.03",
                        "min_fit_dof 1",
                        f"qtmdwf_h5 {h5_path}",
                        "dataset_path_template {gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
                        f"two_point_plateau_table {tmp / 'plateau_pz*_tmax#_plateau.txt'}",
                        f"c2pt {tmp / 'c2pt_pz*.csv'}",
                        "fold_t periodic",
                        f"results_dir {tmp / 'results'}",
                    ]
                ),
                encoding="utf-8",
            )
            run_tmdwf_nstate_fit(input_path)

            shared_summary = tmp / "results" / "demo_pz0" / "demo_pz0_T5_1state_shared_window.txt"
            first_summary = tmp / "results" / "demo_pz0" / "demo_pz0_T5_eta0_bT0_real_1state_summary.txt"
            second_summary = tmp / "results" / "demo_pz0" / "demo_pz0_T5_eta1_bT1_real_1state_summary.txt"
            shared_exists = shared_summary.exists()
            shared_text = shared_summary.read_text(encoding="utf-8")
            first_text = first_summary.read_text(encoding="utf-8")
            second_text = second_summary.read_text(encoding="utf-8")

        self.assertTrue(shared_exists)
        self.assertIn("selection_basis reference_mean_fits_only", shared_text)
        self.assertIn("reference_dataset gm=T5 eta=eta0 pz=0 bT=0 bz=0", shared_text)
        self.assertIn("shared_tfit", first_text)
        self.assertIn("shared_tfit", second_text)
        self.assertIn("m0 ", first_text)
        self.assertEqual(
            next(line for line in first_text.splitlines() if line.startswith("shared_tfit ")),
            next(line for line in second_text.splitlines() if line.startswith("shared_tfit ")),
        )

    def test_grouped_outputs_by_bt_include_multiple_bz_rows_and_fit_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            nt = 12
            ns = 16
            times = np.arange(nt)
            amplitudes = np.array([2.8])
            energies = np.array([0.42])
            matrix_elements = np.array([1.1])
            denominator = evaluate_two_point_symmetric(times, amplitudes, energies, nt)
            numerator = evaluate_tmdwf_numerator(times, amplitudes, energies, matrix_elements, nt, gm="T5", pz=0, ns=ns)

            c2pt_csv = tmp / "c2pt_pz0.csv"
            c2pt_data = np.column_stack([denominator for _ in range(8)])
            with c2pt_csv.open("w", encoding="utf-8") as handle:
                handle.write(",".join(["t"] + [f"cfg_{idx}" for idx in range(c2pt_data.shape[1])]) + "\n")
                for t in range(nt):
                    handle.write(str(t) + "," + ",".join(f"{value:.12e}" for value in c2pt_data[t]) + "\n")

            plateau_path = tmp / "plateau_pz0_tmax5_plateau.txt"
            np.savetxt(plateau_path, np.array([[amplitudes[0], energies[0], 0.05, 0.02]]))
            h5_path = tmp / "tmdwf_pz0.h5"
            with h5py.File(h5_path, "w") as handle:
                for tdir in ("plus", "minus"):
                    for bz in (0, 1, -1):
                        handle.create_dataset(
                            f"T5/eta0/pz0/{tdir}/bT0/bz{bz}",
                            data=np.column_stack([numerator for _ in range(8)]).T,
                        )

            input_path = tmp / "input_tmdwf.txt"
            input_path.write_text(
                "\n".join(
                    [
                        "demo_pz* 16 12 0.076",
                        "fit_target ratio",
                        "fit_component real",
                        "nstates 1",
                        "pzlist 0",
                        "gmlist T5",
                        "etalist eta0",
                        "Tdirlist plus minus",
                        "bTlist 0",
                        "bzlist 0 1",
                        "tmin 2",
                        "tmax auto",
                        f"qtmdwf_h5 {h5_path}",
                        "dataset_path_template {gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
                        f"two_point_plateau_table {tmp / 'plateau_pz*_tmax#_plateau.txt'}",
                        f"c2pt {tmp / 'c2pt_pz*.csv'}",
                        "fold_t periodic",
                        f"results_dir {tmp / 'results'}",
                    ]
                ),
                encoding="utf-8",
            )
            run_tmdwf_nstate_fit(input_path)
            summary_path = tmp / "results" / "demo_pz0" / "demo_pz0_T5_eta0_bT0_real_1state_summary.txt"
            fit_path = tmp / "results" / "demo_pz0" / "tables" / "demo_pz0_T5_eta0_bT0_real_1state_fit.txt"
            ratio_path = tmp / "results" / "demo_pz0" / "tables" / "demo_pz0_T5_eta0_bT0_ratio.txt"
            sample_path = tmp / "results" / "demo_pz0" / "samples" / "demo_pz0_T5_eta0_bT0_real_1state_samples.txt"
            curve_path = tmp / "results" / "demo_pz0" / "tables" / "demo_pz0_T5_eta0_bT0_real_1state_curve.txt"
            summary_text = summary_path.read_text(encoding="utf-8")
            fit_text = fit_path.read_text(encoding="utf-8")
            ratio_text = ratio_path.read_text(encoding="utf-8")
            sample_text = sample_path.read_text(encoding="utf-8")
            curve_text = curve_path.read_text(encoding="utf-8")

        self.assertIn("begin_bz 0", summary_text)
        self.assertIn("begin_bz 1", summary_text)
        self.assertIn("two_point_tmax_source inferred", summary_text)
        self.assertIn("shared_window_flag", fit_text.splitlines()[0])
        self.assertIn("reference_eta", fit_text.splitlines()[0])
        self.assertIn("plateau_tmax_used", fit_text.splitlines()[0])
        self.assertIn("tsrange 0 5", ratio_text)
        self.assertIn("tfit 2 5", ratio_text)
        self.assertIn("shared_window_flag 0", ratio_text)
        self.assertIn("two_point_plateau_table_resolved", ratio_text)
        self.assertIn("two_point_tmax_source inferred", ratio_text)
        self.assertIn("two_point_tmax_inferred 5", ratio_text)
        ratio_header = next(line for line in ratio_text.splitlines() if line.startswith("bz\t"))
        self.assertEqual(
            ratio_header.split("\t"),
            ["bz", "t", "in_fit_window", "ratio_real_mean", "ratio_real_err", "ratio_imag_mean", "ratio_imag_err"],
        )
        self.assertTrue(any(line.startswith("0\t") for line in ratio_text.splitlines() if "\t" in line))
        self.assertTrue(any(line.startswith("1\t") for line in ratio_text.splitlines() if "\t" in line))
        self.assertTrue(any(line.startswith("0\t") for line in fit_text.splitlines()[1:]))
        self.assertTrue(any(line.startswith("1\t") for line in fit_text.splitlines()[1:]))
        self.assertEqual(sample_text.splitlines()[0].split("\t")[:3], ["bz", "sample_id", "success"])
        self.assertTrue(any(line.startswith("0\t") for line in sample_text.splitlines()[1:]))
        self.assertTrue(any(line.startswith("1\t") for line in sample_text.splitlines()[1:]))
        self.assertEqual(
            curve_text.splitlines()[0].split("\t"),
            ["bz", "t", "in_fit_window", "fit_mean", "fit_p16", "fit_p84"],
        )
        self.assertTrue(any(line.startswith("0\t") for line in curve_text.splitlines()[1:]))
        self.assertTrue(any(line.startswith("1\t") for line in curve_text.splitlines()[1:]))

    def test_plot_outputs_are_written_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            nt = 12
            ns = 16
            times = np.arange(nt)
            amplitudes = np.array([2.8])
            energies = np.array([0.42])
            matrix_elements = np.array([1.1])
            denominator = evaluate_two_point_symmetric(times, amplitudes, energies, nt)
            numerator = evaluate_tmdwf_numerator(times, amplitudes, energies, matrix_elements, nt, gm="T5", pz=0, ns=ns)

            c2pt_csv = tmp / "c2pt_pz0.csv"
            c2pt_data = np.column_stack([denominator for _ in range(8)])
            with c2pt_csv.open("w", encoding="utf-8") as handle:
                handle.write(",".join(["t"] + [f"cfg_{idx}" for idx in range(c2pt_data.shape[1])]) + "\n")
                for t in range(nt):
                    handle.write(str(t) + "," + ",".join(f"{value:.12e}" for value in c2pt_data[t]) + "\n")

            plateau_path = tmp / "plateau_pz0_tmax5_plateau.txt"
            np.savetxt(plateau_path, np.array([[amplitudes[0], energies[0], 0.05, 0.02]]))
            h5_path = tmp / "tmdwf_pz0.h5"
            with h5py.File(h5_path, "w") as handle:
                for tdir in ("plus", "minus"):
                    for bz in (0, 1, -1):
                        handle.create_dataset(
                            f"T5/eta0/pz0/{tdir}/bT0/bz{bz}",
                            data=np.column_stack([numerator for _ in range(8)]).T,
                        )

            input_path = tmp / "input_tmdwf_plot.txt"
            input_path.write_text(
                "\n".join(
                    [
                        "demo_pz* 16 12 0.076",
                        "fit_target ratio",
                        "fit_component both",
                        "nstates 1",
                        "pzlist 0",
                        "gmlist T5",
                        "etalist eta0",
                        "Tdirlist plus minus",
                        "bTlist 0",
                        "bzlist 0 1",
                        "tmin 2",
                        "tmax auto",
                        f"qtmdwf_h5 {h5_path}",
                        "dataset_path_template {gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
                        f"two_point_plateau_table {tmp / 'plateau_pz*_tmax#_plateau.txt'}",
                        f"c2pt {tmp / 'c2pt_pz*.csv'}",
                        "fold_t periodic",
                        "plot true",
                        f"results_dir {tmp / 'results'}",
                    ]
                ),
                encoding="utf-8",
            )

            def fake_plot(output_path, *args, **kwargs):
                output_path.write_text("fake plot\n", encoding="utf-8")
                return output_path

            with patch("lqcd_analysis.tmdwf.fit_nstate.plot_tmdwf_ratio_fit", side_effect=fake_plot):
                with patch("lqcd_analysis.tmdwf.fit_nstate.plot_tmdwf_m0_from_fit_tables", side_effect=fake_plot):
                    run_tmdwf_nstate_fit(input_path)

            real_plot = tmp / "results" / "demo_pz0" / "plots" / "demo_pz0_T5_eta0_bT0_real_1state_ratio_fit.pdf"
            imag_plot = tmp / "results" / "demo_pz0" / "plots" / "demo_pz0_T5_eta0_bT0_imag_1state_ratio_fit.pdf"
            m0_plot = tmp / "results" / "demo_pz0" / "plots" / "demo_pz0_T5_eta0_real_1state_m0_vs_bz.pdf"
            real_exists = real_plot.exists()
            imag_exists = imag_plot.exists()
            m0_exists = m0_plot.exists()

        self.assertTrue(real_exists)
        self.assertTrue(imag_exists)
        self.assertTrue(m0_exists)

    def test_plot_tmdwf_ratio_fit_uses_log_for_positive_real_series(self) -> None:
        class FakeAxes:
            def __init__(self):
                self.scale_calls = []

            def axvspan(self, *args, **kwargs):
                return None

            def errorbar(self, *args, **kwargs):
                return None

            def fill_between(self, *args, **kwargs):
                return None

            def plot(self, *args, **kwargs):
                return None

            def set_yscale(self, *args, **kwargs):
                self.scale_calls.append((args, kwargs))

            def set_xlabel(self, *args, **kwargs):
                return None

            def set_ylabel(self, *args, **kwargs):
                return None

            def set_title(self, *args, **kwargs):
                return None

            def legend(self, *args, **kwargs):
                return None

        class FakeFigure:
            def tight_layout(self):
                return None

            def savefig(self, *args, **kwargs):
                return None

        fake_ax = FakeAxes()
        fake_fig = FakeFigure()
        fake_plt = type(
            "FakePlt",
            (),
            {
                "subplots": staticmethod(lambda **kwargs: (fake_fig, fake_ax)),
                "close": staticmethod(lambda fig: None),
            },
        )()
        series = (
            RatioFitPlotSeries(
                bz=0,
                times=np.array([0, 1, 2]),
                ratio_mean=np.array([1.0, 2.0, 3.0]),
                ratio_err=np.array([0.1, 0.1, 0.1]),
                fit_mean=np.array([1.1, 2.1, 3.1]),
                fit_p16=np.array([1.0, 2.0, 3.0]),
                fit_p84=np.array([1.2, 2.2, 3.2]),
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("lqcd_analysis.tmdwf.plotting.prepare_matplotlib", return_value=fake_plt):
                plot_tmdwf_ratio_fit(Path(tmpdir) / "plot.pdf", series, component="real", fit_window=(1, 2))
        self.assertEqual(fake_ax.scale_calls, [(("log",), {})])

    def test_plot_tmdwf_ratio_fit_uses_symlog_for_imag_or_nonpositive_real(self) -> None:
        class FakeAxes:
            def __init__(self):
                self.scale_calls = []

            def axvspan(self, *args, **kwargs):
                return None

            def errorbar(self, *args, **kwargs):
                return None

            def fill_between(self, *args, **kwargs):
                return None

            def plot(self, *args, **kwargs):
                return None

            def set_yscale(self, *args, **kwargs):
                self.scale_calls.append((args, kwargs))

            def set_xlabel(self, *args, **kwargs):
                return None

            def set_ylabel(self, *args, **kwargs):
                return None

            def set_title(self, *args, **kwargs):
                return None

            def legend(self, *args, **kwargs):
                return None

        class FakeFigure:
            def tight_layout(self):
                return None

            def savefig(self, *args, **kwargs):
                return None

        def build_fake_plot():
            fake_ax = FakeAxes()
            fake_fig = FakeFigure()
            fake_plt = type(
                "FakePlt",
                (),
                {
                    "subplots": staticmethod(lambda **kwargs: (fake_fig, fake_ax)),
                    "close": staticmethod(lambda fig: None),
                },
            )()
            return fake_ax, fake_plt

        imag_series = (
            RatioFitPlotSeries(
                bz=0,
                times=np.array([0, 1, 2]),
                ratio_mean=np.array([0.1, -0.2, 0.3]),
                ratio_err=np.array([0.01, 0.01, 0.01]),
                fit_mean=np.array([0.11, -0.21, 0.31]),
                fit_p16=np.array([0.09, -0.25, 0.29]),
                fit_p84=np.array([0.13, -0.18, 0.33]),
            ),
        )
        fake_ax, fake_plt = build_fake_plot()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("lqcd_analysis.tmdwf.plotting.prepare_matplotlib", return_value=fake_plt):
                plot_tmdwf_ratio_fit(Path(tmpdir) / "imag.pdf", imag_series, component="imag", fit_window=(1, 2))
        self.assertEqual(fake_ax.scale_calls[0][0][0], "symlog")
        self.assertIn("linthresh", fake_ax.scale_calls[0][1])

        nonpositive_real_series = (
            RatioFitPlotSeries(
                bz=0,
                times=np.array([0, 1, 2]),
                ratio_mean=np.array([1.0, 0.0, 3.0]),
                ratio_err=np.array([0.1, 0.1, 0.1]),
                fit_mean=np.array([1.1, 0.0, 3.1]),
                fit_p16=np.array([1.0, -0.1, 3.0]),
                fit_p84=np.array([1.2, 0.1, 3.2]),
            ),
        )
        fake_ax2, fake_plt2 = build_fake_plot()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("lqcd_analysis.tmdwf.plotting.prepare_matplotlib", return_value=fake_plt2):
                plot_tmdwf_ratio_fit(Path(tmpdir) / "real.pdf", nonpositive_real_series, component="real", fit_window=(1, 2))
        self.assertEqual(fake_ax2.scale_calls[0][0][0], "symlog")
        self.assertIn("linthresh", fake_ax2.scale_calls[0][1])

    def test_write_tmdwf_plot_notebook(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            notebook = write_tmdwf_plot_notebook(
                notebook_path=tmp / "notebook_plots" / "demo.ipynb",
                notebook_output_dir=tmp / "notebook_plots" / "generated",
                ratio_tables={0: tmp / "bT0_ratio.txt", 2: tmp / "bT2_ratio.txt"},
                curve_tables={
                    0: {"real": {1: tmp / "bT0_real1.txt", 2: tmp / "bT0_real2.txt"}, "imag": {1: tmp / "bT0_imag1.txt"}},
                    2: {"real": {1: tmp / "bT2_real1.txt", 2: tmp / "bT2_real2.txt"}, "imag": {1: tmp / "bT2_imag1.txt"}},
                },
                fit_tables={
                    0: {"real": {1: tmp / "bT0_real1_fit.txt", 2: tmp / "bT0_real2_fit.txt"}, "imag": {1: tmp / "bT0_imag1_fit.txt"}},
                    2: {"real": {1: tmp / "bT2_real1_fit.txt", 2: tmp / "bT2_real2_fit.txt"}, "imag": {1: tmp / "bT2_imag1_fit.txt"}},
                },
                title="demo_pz0",
                gm="T5",
                eta="eta0",
            )
            self.assertTrue(notebook.exists())
            payload = json.loads(notebook.read_text(encoding="utf-8"))
            joined = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
        self.assertEqual(payload["nbformat"], 4)
        self.assertIn("plot_tmdwf_grouped_outputs", joined)
        self.assertIn("plot_tmdwf_m0_from_fit_tables", joined)
        self.assertIn("ratio_tables =", joined)
        self.assertIn("curve_tables =", joined)
        self.assertIn("fit_tables =", joined)
        self.assertIn("chosen_bT = available_bT[0]", joined)
        self.assertIn("component = 'real'", joined)
        self.assertIn("bz_values = None", joined)
        self.assertIn("selected_bT_values = None", joined)
        self.assertIn("m0_output_name = None", joined)

    def test_plot_tmdwf_grouped_outputs_reads_grouped_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ratio_table = tmp / "ratio.txt"
            curve_table = tmp / "curve.txt"
            ratio_table.write_text(
                "\n".join(
                    [
                        "tsrange 0 2",
                        "tfit 1 2",
                        "shared_window_flag 0",
                        "two_point_plateau_table_resolved plateaus.txt",
                        "two_point_tmax_source explicit",
                        "two_point_tmax_inferred none",
                        "bz\tt\tin_fit_window\tratio_real_mean\tratio_real_err\tratio_imag_mean\tratio_imag_err",
                        "0\t0\t0\t1.0\t0.1\t0.2\t0.02",
                        "0\t1\t1\t1.1\t0.1\t0.3\t0.02",
                        "0\t2\t1\t1.2\t0.1\t0.4\t0.02",
                    ]
                ),
                encoding="utf-8",
            )
            curve_table.write_text(
                "\n".join(
                    [
                        "bz\tt\tin_fit_window\tfit_mean\tfit_p16\tfit_p84",
                        "0\t0\t0\t0.2\t0.1\t0.3",
                        "0\t1\t1\t0.3\t0.2\t0.4",
                        "0\t2\t1\t0.4\t0.3\t0.5",
                    ]
                ),
                encoding="utf-8",
            )

            captured = {}

            def fake_plot(output_path, series, **kwargs):
                captured["output_path"] = output_path
                captured["series"] = series
                captured["component"] = kwargs["component"]
                captured["fit_window"] = kwargs["fit_window"]
                captured["title"] = kwargs.get("title")
                output_path.write_text("ok\n", encoding="utf-8")
                return output_path

            with patch("lqcd_analysis.tmdwf.plotting.plot_tmdwf_ratio_fit", side_effect=fake_plot):
                output = plot_tmdwf_grouped_outputs(
                    tmp / "out.pdf",
                    ratio_table,
                    curve_table,
                    component="imag",
                    title="demo",
                )
                output_exists = output.exists()

        self.assertTrue(output_exists)
        self.assertEqual(captured["component"], "imag")
        self.assertEqual(captured["fit_window"], (1, 2))
        self.assertEqual(captured["title"], "demo")
        self.assertTrue(np.allclose(captured["series"][0].ratio_mean, np.array([0.2, 0.3, 0.4])))

    def test_build_m0_series_from_fit_tables_sorts_bz_and_reads_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fit_bt0 = tmp / "bT0_fit.txt"
            fit_bt2 = tmp / "bT2_fit.txt"
            header = "\t".join(
                [
                    "bz",
                    "tmin",
                    "tmax",
                    "success_meanfit",
                    "chi2_dof",
                    "pvalue",
                    "shared_window_flag",
                    "reference_eta",
                    "reference_bT",
                    "reference_bz",
                    "plateau_tmax_used",
                    "m0_mean",
                    "m0_err",
                ]
            )
            fit_bt0.write_text(
                "\n".join(
                    [
                        header,
                        "2\t2\t5\t1\t1.0\t0.5\t0\tnone\t-1\t-1\t5\t1.20\t0.12",
                        "0\t2\t5\t1\t1.0\t0.5\t0\tnone\t-1\t-1\t5\t1.00\t0.10",
                        "1\t2\t5\t1\t1.0\t0.5\t0\tnone\t-1\t-1\t5\t1.10\t0.11",
                    ]
                ),
                encoding="utf-8",
            )
            fit_bt2.write_text(
                "\n".join(
                    [
                        header,
                        "1\t2\t5\t1\t1.0\t0.5\t0\tnone\t-1\t-1\t5\t2.10\t0.21",
                        "0\t2\t5\t1\t1.0\t0.5\t0\tnone\t-1\t-1\t5\t2.00\t0.20",
                    ]
                ),
                encoding="utf-8",
            )
            series = _build_m0_series_from_fit_tables({2: fit_bt2, 0: fit_bt0})

        self.assertEqual([item.bT for item in series], [0, 2])
        self.assertTrue(np.array_equal(series[0].bz, np.array([0, 1, 2])))
        self.assertTrue(np.allclose(series[0].m0_mean, np.array([1.00, 1.10, 1.20])))
        self.assertTrue(np.allclose(series[0].m0_err, np.array([0.10, 0.11, 0.12])))
        self.assertTrue(np.array_equal(series[1].bz, np.array([0, 1])))
        self.assertTrue(np.allclose(series[1].m0_mean, np.array([2.00, 2.10])))
        self.assertTrue(np.allclose(series[1].m0_err, np.array([0.20, 0.21])))

    def test_plot_tmdwf_m0_from_fit_tables_builds_one_series_per_bt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fit_bt0 = tmp / "bT0_fit.txt"
            fit_bt2 = tmp / "bT2_fit.txt"
            header = "\t".join(
                [
                    "bz",
                    "tmin",
                    "tmax",
                    "success_meanfit",
                    "chi2_dof",
                    "pvalue",
                    "shared_window_flag",
                    "reference_eta",
                    "reference_bT",
                    "reference_bz",
                    "plateau_tmax_used",
                    "m0_mean",
                    "m0_err",
                ]
            )
            fit_bt0.write_text("\n".join([header, "1\t2\t5\t1\t1.0\t0.5\t0\tnone\t-1\t-1\t5\t1.10\t0.11"]), encoding="utf-8")
            fit_bt2.write_text("\n".join([header, "0\t2\t5\t1\t1.0\t0.5\t0\tnone\t-1\t-1\t5\t2.00\t0.20"]), encoding="utf-8")

            captured = {}

            def fake_plot(output_path, series, **kwargs):
                captured["output_path"] = output_path
                captured["series"] = series
                captured["kwargs"] = kwargs
                output_path.write_text("ok\n", encoding="utf-8")
                return output_path

            with patch("lqcd_analysis.tmdwf.plotting.plot_tmdwf_m0_vs_bz", side_effect=fake_plot):
                output = plot_tmdwf_m0_from_fit_tables(
                    tmp / "m0_vs_bz.pdf",
                    {0: fit_bt0, 2: fit_bt2},
                    component="real",
                    nstates=1,
                    title="demo",
                )
                output_exists = output.exists()

        self.assertTrue(output_exists)
        self.assertEqual([item.bT for item in captured["series"]], [0, 2])
        self.assertEqual(captured["kwargs"]["component"], "real")
        self.assertEqual(captured["kwargs"]["nstates"], 1)
        self.assertEqual(captured["kwargs"]["title"], "demo")

    def test_workflow_writes_tmdwf_plot_notebook(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            nt = 12
            ns = 16
            times = np.arange(nt)
            amplitudes = np.array([2.8])
            energies = np.array([0.42])
            matrix_elements = np.array([1.1])
            denominator = evaluate_two_point_symmetric(times, amplitudes, energies, nt)
            numerator = evaluate_tmdwf_numerator(times, amplitudes, energies, matrix_elements, nt, gm="T5", pz=0, ns=ns)

            c2pt_csv = tmp / "c2pt_pz0.csv"
            c2pt_data = np.column_stack([denominator for _ in range(8)])
            with c2pt_csv.open("w", encoding="utf-8") as handle:
                handle.write(",".join(["t"] + [f"cfg_{idx}" for idx in range(c2pt_data.shape[1])]) + "\n")
                for t in range(nt):
                    handle.write(str(t) + "," + ",".join(f"{value:.12e}" for value in c2pt_data[t]) + "\n")

            np.savetxt(tmp / "plateau_pz0_tmax5_plateau.txt", np.array([[amplitudes[0], energies[0], 0.05, 0.02]]))
            h5_path = tmp / "tmdwf_pz0.h5"
            with h5py.File(h5_path, "w") as handle:
                for tdir in ("plus", "minus"):
                    for bz in (0, 1, -1):
                        handle.create_dataset(
                            f"T5/eta0/pz0/{tdir}/bT0/bz{bz}",
                            data=np.column_stack([numerator for _ in range(8)]).T,
                        )

            input_path = tmp / "input_tmdwf_notebook.txt"
            input_path.write_text(
                "\n".join(
                    [
                        "demo_pz* 16 12 0.076",
                        "fit_target ratio",
                        "fit_component both",
                        "nstates 1",
                        "pzlist 0",
                        "gmlist T5",
                        "etalist eta0",
                        "Tdirlist plus minus",
                        "bTlist 0",
                        "bzlist 0 1",
                        "tmin 2",
                        "tmax auto",
                        f"qtmdwf_h5 {h5_path}",
                        "dataset_path_template {gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
                        f"two_point_plateau_table {tmp / 'plateau_pz*_tmax#_plateau.txt'}",
                        f"c2pt {tmp / 'c2pt_pz*.csv'}",
                        "fold_t periodic",
                        f"results_dir {tmp / 'results'}",
                    ]
                ),
                encoding="utf-8",
            )

            run_tmdwf_nstate_fit(input_path)
            notebook = tmp / "results" / "notebook_plots" / "demo_pz0" / "demo_pz0_T5_eta0_tmdwf_plots.ipynb"
            notebook_exists = notebook.exists()
            payload = json.loads(notebook.read_text(encoding="utf-8"))
            joined = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])

        self.assertTrue(notebook_exists)
        self.assertEqual(payload["nbformat"], 4)
        self.assertIn("plot_tmdwf_grouped_outputs", joined)
        self.assertIn("plot_tmdwf_m0_from_fit_tables", joined)
        self.assertIn("demo_pz0_T5_eta0_bT0_ratio.txt", joined)
        self.assertIn("demo_pz0_T5_eta0_bT0_real_1state_fit.txt", joined)
        self.assertIn("demo_pz0_T5_eta0_bT0_imag_1state_fit.txt", joined)
        self.assertIn("demo_pz0_T5_eta0_bT0_real_1state_curve.txt", joined)
        self.assertIn("demo_pz0_T5_eta0_bT0_imag_1state_curve.txt", joined)

    def test_workflow_groups_m0_vs_bz_plot_across_bt_fit_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            nt = 12
            ns = 16
            times = np.arange(nt)
            amplitudes = np.array([2.8])
            energies = np.array([0.42])
            matrix_elements = np.array([1.1])
            denominator = evaluate_two_point_symmetric(times, amplitudes, energies, nt)
            numerator = evaluate_tmdwf_numerator(times, amplitudes, energies, matrix_elements, nt, gm="T5", pz=0, ns=ns)

            c2pt_csv = tmp / "c2pt_pz0.csv"
            c2pt_data = np.column_stack([denominator for _ in range(8)])
            with c2pt_csv.open("w", encoding="utf-8") as handle:
                handle.write(",".join(["t"] + [f"cfg_{idx}" for idx in range(c2pt_data.shape[1])]) + "\n")
                for t in range(nt):
                    handle.write(str(t) + "," + ",".join(f"{value:.12e}" for value in c2pt_data[t]) + "\n")

            np.savetxt(tmp / "plateau_pz0_tmax5_plateau.txt", np.array([[amplitudes[0], energies[0], 0.05, 0.02]]))
            h5_path = tmp / "tmdwf_pz0.h5"
            with h5py.File(h5_path, "w") as handle:
                for bT in (0, 1):
                    for tdir in ("plus", "minus"):
                        for bz in (0, 1, -1):
                            handle.create_dataset(
                                f"T5/eta0/pz0/{tdir}/bT{bT}/bz{bz}",
                                data=np.column_stack([numerator for _ in range(8)]).T,
                            )

            input_path = tmp / "input_tmdwf_m0_group.txt"
            input_path.write_text(
                "\n".join(
                    [
                        "demo_pz* 16 12 0.076",
                        "fit_target ratio",
                        "fit_component real",
                        "nstates 1",
                        "pzlist 0",
                        "gmlist T5",
                        "etalist eta0",
                        "Tdirlist plus minus",
                        "bTlist 0 1",
                        "bzlist 0 1",
                        "tmin 2",
                        "tmax auto",
                        f"qtmdwf_h5 {h5_path}",
                        "dataset_path_template {gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
                        f"two_point_plateau_table {tmp / 'plateau_pz*_tmax#_plateau.txt'}",
                        f"c2pt {tmp / 'c2pt_pz*.csv'}",
                        "fold_t periodic",
                        "plot true",
                        f"results_dir {tmp / 'results'}",
                    ]
                ),
                encoding="utf-8",
            )

            captured = {}

            def fake_ratio_plot(output_path, *args, **kwargs):
                output_path.write_text("ratio\n", encoding="utf-8")
                return output_path

            def fake_m0_plot(output_path, fit_tables_by_bT, **kwargs):
                captured["keys"] = sorted(fit_tables_by_bT.keys())
                captured["paths"] = [Path(value).name for value in fit_tables_by_bT.values()]
                output_path.write_text("m0\n", encoding="utf-8")
                return output_path

            with patch("lqcd_analysis.tmdwf.fit_nstate.plot_tmdwf_ratio_fit", side_effect=fake_ratio_plot):
                with patch("lqcd_analysis.tmdwf.fit_nstate.plot_tmdwf_m0_from_fit_tables", side_effect=fake_m0_plot):
                    run_tmdwf_nstate_fit(input_path)

        self.assertEqual(captured["keys"], [0, 1])
        self.assertIn("demo_pz0_T5_eta0_bT0_real_1state_fit.txt", captured["paths"])
        self.assertIn("demo_pz0_T5_eta0_bT1_real_1state_fit.txt", captured["paths"])

    def test_plateau_resolution_prints_once_per_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            nt = 12
            ns = 16
            times = np.arange(nt)
            amplitudes = np.array([2.8, 1.1])
            energies = np.array([0.42, 0.8])
            matrix_elements = np.array([1.1])
            denominator = evaluate_two_point_symmetric(times, amplitudes[:1], energies[:1], nt)
            c2pt_csv = tmp / "c2pt_pz0.csv"
            c2pt_data = np.column_stack([denominator for _ in range(8)])
            with c2pt_csv.open("w", encoding="utf-8") as handle:
                handle.write(",".join(["t"] + [f"cfg_{idx}" for idx in range(c2pt_data.shape[1])]) + "\n")
                for t in range(nt):
                    handle.write(str(t) + "," + ",".join(f"{value:.12e}" for value in c2pt_data[t]) + "\n")
            np.savetxt(tmp / "plateau_pz0_tmax5_plateau.txt", np.array([[amplitudes[0], energies[0], 0.05, 0.02]]))
            np.savetxt(tmp / "plateau2_pz0_tmax5_plateau.txt", np.array([[amplitudes[0], amplitudes[1], energies[0], energies[1], 0.05, 0.03, 0.02, 0.04]]))
            h5_path = tmp / "tmdwf_pz0.h5"
            numerator = evaluate_tmdwf_numerator(times, amplitudes[:1], energies[:1], matrix_elements, nt, gm="T5", pz=0, ns=ns)
            with h5py.File(h5_path, "w") as handle:
                for eta in ("eta0", "eta1"):
                    for tdir in ("plus", "minus"):
                        handle.create_dataset(f"T5/{eta}/pz0/{tdir}/bT0/bz0", data=np.column_stack([numerator for _ in range(8)]).T)
            input_path = tmp / "input_tmdwf.txt"
            input_path.write_text(
                "\n".join(
                    [
                        "demo_pz* 16 12 0.076",
                        "fit_target ratio",
                        "fit_component real",
                        "nstates 1",
                        "pzlist 0",
                        "gmlist T5",
                        "etalist eta0 eta1",
                        "Tdirlist plus minus",
                        "bTlist 0",
                        "bzlist 0",
                        "tmin 2",
                        "tmax auto",
                        f"qtmdwf_h5 {h5_path}",
                        "dataset_path_template {gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
                        f"two_point_plateau_table {tmp / 'plateau_pz*_tmax#_plateau.txt'}",
                        f"c2pt {tmp / 'c2pt_pz*.csv'}",
                        "fold_t periodic",
                        f"results_dir {tmp / 'results'}",
                    ]
                ),
                encoding="utf-8",
            )
            with patch("builtins.print") as mocked_print:
                run_tmdwf_nstate_fit(input_path)
        selected_msgs = [call.args[0] for call in mocked_print.call_args_list if "[tmdwf-fit] Selected plateau table" in str(call.args[0])]
        self.assertEqual(len(selected_msgs), 1)


if __name__ == "__main__":
    unittest.main()
