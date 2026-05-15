import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from lqcd_analysis.DA.fit_nstate import (
    _select_curve_component,
    _write_component_outputs,
    fit_da_component,
    parse_da_fit_input,
    run_da_nstate_fit,
    DAFitResult,
    DAOutputRecord,
)
from lqcd_analysis.DA.io import (
    _load_da_correlator_from_handle,
    apply_da_preprocessing,
    expand_template,
    fold_symmetric_complex,
    load_da_correlator,
    resolve_qda_h5_path,
    resolve_two_point_fit_reference,
)
from lqcd_analysis.DA.models import (
    evaluate_da_numerator,
    evaluate_da_numerator_t5,
    evaluate_da_numerator_z5,
    evaluate_da_ratio,
    evaluate_two_point_symmetric,
)
from lqcd_analysis.DA.fourier import (
    compute_da_cosine_transform,
    load_da_m0_fit_table,
    load_da_m0_sample_table,
    run_da_fourier_from_fit_outputs,
    run_da_fourier_workflow,
    summarize_da_fourier_samples,
)
from lqcd_analysis.DA.normalize import run_da_normalization
from lqcd_analysis.DA.plotting import (
    RatioFitPlotSeries,
    DAM0VsBZSeries,
    _build_m0_series_from_fit_tables,
    plot_da_grouped_outputs,
    plot_da_m0_from_fit_tables,
    plot_da_m0_vs_bz,
    plot_da_ratio_fit,
    write_da_plot_notebook,
)
from lqcd_analysis.common.parsing import load_fit_window_table

try:
    import h5py
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    h5py = None


@unittest.skipIf(h5py is None, "h5py is required for DA HDF5 tests")
class DAFitTests(unittest.TestCase):
    @staticmethod
    def _write_da_m0_outputs(
        root: Path,
        title: str,
        gm: str,
        eta: str,
        bT: int,
        component: str,
        nstates: int,
        fit_rows: list[tuple[int, float, float]],
        sample_rows: list[tuple[int, int, int, float]],
    ) -> None:
        fit_dir = root / title / "tables"
        sample_dir = root / title / "samples"
        fit_dir.mkdir(parents=True, exist_ok=True)
        sample_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{title}_{gm}_{eta}_bT{bT}_{component}_{nstates}state"
        (fit_dir / f"{stem}_fit.txt").write_text(
            "\n".join(
                [
                    "bz\ttmin\ttmax\tsuccess_meanfit\tchi2_dof\tpvalue\ttwo_point_fit_tmax\tm0_mean\tm0_err",
                    *[
                        f"{bz}\t2\t6\t1\t1.0\t0.5\t6\t{mean:.10e}\t{err:.10e}"
                        for bz, mean, err in fit_rows
                    ],
                ]
            ),
            encoding="utf-8",
        )
        (sample_dir / f"{stem}_samples.txt").write_text(
            "\n".join(
                [
                    "bz\tsample_id\tsuccess\tm0",
                    *[
                        f"{bz}\t{sample_id}\t{success}\t{value:.10e}"
                        for bz, sample_id, success, value in sample_rows
                    ],
                ]
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _write_two_point_fit_reference(
        root: Path,
        title: str,
        fit_window_path: Path,
        pz: int,
        amplitudes: np.ndarray,
        energies: np.ndarray,
        *,
        nstates: int = 1,
        gm: str | None = None,
    ) -> Path:
        windows = load_fit_window_table(fit_window_path)
        window_key = (gm, pz) if (gm, pz) in windows else (None, pz)
        if window_key not in windows:
            raise ValueError(f"missing fit window for pz={pz}")
        tmin, tmax = windows[window_key]
        tables_dir = root / title / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        table_path = tables_dir / f"{title}_normal_{nstates}state_tmax{tmax}_fits.txt"
        if nstates == 1:
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
        else:
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
                1.0,
                float(amplitudes[0]),
                float(amplitudes[1]),
                0.1,
                0.2,
                float(energies[0]),
                float(energies[1]),
                0.02,
            ]
        np.savetxt(table_path, np.array([row], dtype=float))
        return table_path

    def test_parse_input_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            two_point_fit_window_path = Path(tmpdir) / "two_point_fit_windows.txt"
            two_point_fit_window_path.write_text("0 2 5\n1 2 5\n", encoding="utf-8")
            two_point_fit_root = str(Path(tmpdir) / "two_point_fit_root")
            input_file = Path(tmpdir) / "input_da.txt"
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
                        "fit_window /tmp/fit_windows.txt",
                        f"two_point_fit_root {two_point_fit_root}",
                        "qda_h5 /tmp/da_pz*.h5",
                        "dataset_path_template {gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
                        f"two_point_fit_window_by_pz {two_point_fit_window_path}",
                        "c2pt /tmp/c2pt_pz*.csv",
                        "fold_t periodic",
                        "tsrange 0 8",
                        "plot false",
                        "results_dir /tmp/da_results",
                    ]
                ),
                encoding="utf-8",
            )
            parsed = parse_da_fit_input(input_file)

        self.assertEqual(parsed.title_pattern, "demo_pz*")
        self.assertEqual(parsed.nstates, (1, 2))
        self.assertEqual(parsed.pzlist, (0, 1))
        self.assertEqual(parsed.bTlist, (0, 1))
        self.assertEqual(parsed.bzlist, (0, 2))
        self.assertEqual(parsed.fit_component, "both")
        self.assertEqual(parsed.results_dir, Path("/tmp/da_results"))
        self.assertEqual(parsed.fold_t, "periodic")
        self.assertEqual(parsed.fit_window, "/tmp/fit_windows.txt")

    def test_parse_input_file_supports_fit_window_without_tmin(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            two_point_fit_window_path = Path(tmpdir) / "two_point_fit_windows.txt"
            two_point_fit_window_path.write_text("0 2 5\n1 2 5\n", encoding="utf-8")
            two_point_fit_root = str(Path(tmpdir) / "two_point_fit_root")
            input_file = Path(tmpdir) / "input_da.txt"
            input_file.write_text(
                "\n".join(
                    [
                        "demo_pz* 32 16 0.076",
                        "fit_target ratio",
                        "fit_component real",
                        "nstates 1",
                        "pzlist 0",
                        "gmlist T5",
                        "etalist eta0",
                        "Tdirlist plus minus",
                        "bTlist 0",
                        "bzlist 0",
                        "fit_window /tmp/fit_windows.txt",
                        f"two_point_fit_root {two_point_fit_root}",
                        "qda_h5 /tmp/da_pz*.h5",
                        "dataset_path_template {gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
                        f"two_point_fit_window_by_pz {two_point_fit_window_path}",
                        "c2pt /tmp/c2pt_pz*.csv",
                        "fold_t periodic",
                    ]
                ),
                encoding="utf-8",
            )
            parsed = parse_da_fit_input(input_file)
        self.assertEqual(parsed.fit_window, "/tmp/fit_windows.txt")

    def test_parse_input_file_normalizes_fold_t_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            two_point_fit_window_path = Path(tmpdir) / "two_point_fit_windows.txt"
            two_point_fit_window_path.write_text("0 2 5\n1 2 5\n", encoding="utf-8")
            two_point_fit_root = str(Path(tmpdir) / "two_point_fit_root")
            input_file = Path(tmpdir) / "input_da.txt"
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
                        "fit_window /tmp/fit_windows.txt",
                        f"two_point_fit_root {two_point_fit_root}",
                        "qda_h5 /tmp/da_pz*.h5",
                        "dataset_path_template {gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
                        f"two_point_fit_window_by_pz {two_point_fit_window_path}",
                        "c2pt /tmp/c2pt_pz*.csv",
                        "fold_t true",
                        "tsrange 0 8",
                    ]
                ),
                encoding="utf-8",
            )

            parsed = parse_da_fit_input(input_file)

        self.assertEqual(parsed.fold_t, "periodic")

    def test_parse_input_file_defaults_tsrange_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            two_point_fit_window_path = Path(tmpdir) / "two_point_fit_windows.txt"
            two_point_fit_window_path.write_text("0 2 5\n1 2 5\n", encoding="utf-8")
            two_point_fit_root = str(Path(tmpdir) / "two_point_fit_root")
            input_file = Path(tmpdir) / "input_da.txt"
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
                        "fit_window /tmp/fit_windows.txt",
                        f"two_point_fit_root {two_point_fit_root}",
                        "qda_h5 /tmp/da_pz*.h5",
                        "dataset_path_template {gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
                        f"two_point_fit_window_by_pz {two_point_fit_window_path}",
                        "c2pt /tmp/c2pt_pz*.csv",
                        "fold_t true",
                    ]
                ),
                encoding="utf-8",
            )
            parsed = parse_da_fit_input(input_file)

        self.assertEqual(parsed.tsrange, (0, 7))

    def test_parse_input_file_accepts_fit_window_without_extra_t_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            two_point_fit_window_path = Path(tmpdir) / "two_point_fit_windows.txt"
            two_point_fit_window_path.write_text("0 2 5\n1 2 5\n", encoding="utf-8")
            two_point_fit_root = "/tmp/two_point_fit_root"
            input_file = Path(tmpdir) / "input_da.txt"
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
                        "fit_window /tmp/fit_windows.txt",
                        f"two_point_fit_root {two_point_fit_root}",
                        "qda_h5 /tmp/da_pz*.h5",
                        "dataset_path_template {gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
                        f"two_point_fit_window_by_pz {two_point_fit_window_path}",
                        "c2pt /tmp/c2pt_pz*.csv",
                        "fold_t true",
                    ]
                ),
                encoding="utf-8",
            )
            parsed = parse_da_fit_input(input_file)

        self.assertEqual(parsed.fit_window, "/tmp/fit_windows.txt")

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

            loaded = load_da_correlator(
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

    def test_resolve_qda_h5_path_supports_gm_and_legacy_pz_wildcard(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            two_point_fit_root = tmp / "two_point_fit_root"
            gm_path = tmp / "qda_pz0_OT5.h5"
            gm_path.touch()
            legacy_path = tmp / "qda_pz0.h5"
            legacy_path.touch()

            self.assertEqual(
                resolve_qda_h5_path(tmp / "qda_pz{pz}_O{gm}.h5", pz=0, gm="T5"),
                gm_path,
            )
            self.assertEqual(
                resolve_qda_h5_path(tmp / "qda_pz*.h5", pz=0, gm="Z5"),
                legacy_path,
            )

    def test_resolve_qda_h5_path_missing_file_reports_fully_resolved_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            two_point_fit_root = tmp / "two_point_fit_root"
            with self.assertRaises(FileNotFoundError) as ctx:
                resolve_qda_h5_path(tmp / "qda_pz{pz}_O{gm}.h5", pz=0, gm="Z5")
        self.assertEqual(str(ctx.exception), str(tmp / "qda_pz0_OZ5.h5"))

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

            path_loaded = load_da_correlator(
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
                handle_loaded = _load_da_correlator_from_handle(
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

    def test_resolve_two_point_fit_reference_selects_requested_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            two_point_fit_root = tmp / "two_point_fit_root"
            fit_root = tmp / "fit_root"
            tables_dir = fit_root / "demo_pz0" / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            table_path = tables_dir / "demo_pz0_normal_1state_tmax12_fits.txt"
            np.savetxt(
                table_path,
                np.array(
                    [
                        [8, 12, 1.0, 0.1, 0.4, 0.02, 0.3, 0.4, 1.0, 1.1, 0.1, 0.5, 0.02],
                        [12, 12, 1.0, 0.1, 0.4, 0.02, 0.3, 0.4, 1.0, 1.1, 0.1, 0.5, 0.02],
                    ]
                ),
            )
            resolved = resolve_two_point_fit_reference(fit_root, title="demo_pz0", nstates=1, tmin=12, tmax=12)
        self.assertEqual(resolved.path, table_path)
        self.assertEqual(resolved.tmin, 12)
        self.assertEqual(resolved.tmax, 12)
        self.assertTrue(np.allclose(resolved.amplitudes, [1.1]))
        self.assertTrue(np.allclose(resolved.energies, [0.5]))

    def test_model_evaluation_ratio_formula(self) -> None:
        times = np.arange(2, 6)
        amplitudes = np.array([3.0])
        energies = np.array([0.4])
        matrix_elements = np.array([1.7])
        nt = 16
        pz = 2
        ns = 32

        denominator = evaluate_two_point_symmetric(times, amplitudes, energies, nt)
        numerator = evaluate_da_numerator_t5(times, amplitudes, energies, matrix_elements, nt)
        ratio = evaluate_da_ratio(times, amplitudes, energies, matrix_elements, nt, gm="T5", pz=pz, ns=ns)

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

        numerator = evaluate_da_numerator_z5(times, amplitudes, energies, matrix_elements, nt, pz=pz, ns=ns)
        t5_numerator = evaluate_da_numerator_t5(times, amplitudes, energies, matrix_elements, nt)
        self.assertTrue(np.allclose(numerator, t5_numerator * (momentum / energies[0])))

        ratio = evaluate_da_ratio(times, amplitudes, energies, matrix_elements, nt, gm="Z5", pz=pz, ns=ns)
        denominator = evaluate_two_point_symmetric(times, amplitudes, energies, nt)
        self.assertTrue(np.allclose(ratio, numerator / denominator))

    def test_operator_dependent_preprocessing(self) -> None:
        nt = 8
        values = np.ones((2, nt), dtype=np.complex128)
        t5 = apply_da_preprocessing(values, nt, "T5")
        z5 = apply_da_preprocessing(values, nt, "Z5")

        expected_t5 = np.ones((2, nt), dtype=np.complex128)
        expected_t5[:, nt // 2 :] = -1.0
        self.assertTrue(np.allclose(t5, expected_t5))
        self.assertTrue(np.allclose(z5, -1j * np.ones((2, nt), dtype=np.complex128)))

    def test_da_loader_uses_symmetric_fold_after_preprocessing(self) -> None:
        nt = 8
        raw = np.array([[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]], dtype=float)
        processed = apply_da_preprocessing(raw, nt, "T5")
        expected = fold_symmetric_complex(processed, nt)

        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "demo.h5"
            with h5py.File(h5_path, "w") as handle:
                handle.create_dataset("T5/eta0/pz0/plus/bT0/bz0", data=raw)
            loaded = load_da_correlator(
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

    def test_select_curve_component_uses_requested_part(self) -> None:
        values = np.array([[1.0 + 3.0j, 2.0 + 4.0j], [5.0 + 7.0j, 6.0 + 8.0j]])
        self.assertTrue(np.allclose(_select_curve_component(values, "real"), np.array([[1.0, 2.0], [5.0, 6.0]])))
        self.assertTrue(np.allclose(_select_curve_component(values, "imag"), np.array([[3.0, 4.0], [7.0, 8.0]])))

    def test_resolve_two_point_fit_reference_reads_selected_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fit_root = Path(tmpdir) / "fit_root"
            tables_dir = fit_root / "demo_pz0" / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            table_path = tables_dir / "demo_pz0_normal_2state_tmax6_fits.txt"
            np.savetxt(
                table_path,
                np.array(
                    [
                        [
                            4,
                            6,
                            1.0,
                            0.2,
                            0.3,
                            0.4,
                            0.5,
                            0.6,
                            1.0,
                            3.0,
                            1.1,
                            0.2,
                            0.3,
                            0.4,
                            0.9,
                            0.04,
                            0.09,
                        ]
                    ]
                ),
            )
            resolved = resolve_two_point_fit_reference(fit_root, title="demo_pz0", nstates=2, tmin=4, tmax=6)
        self.assertEqual(resolved.path, table_path)
        self.assertTrue(np.allclose(resolved.amplitudes, [3.0, 1.1]))
        self.assertTrue(np.allclose(resolved.energies, [0.4, 0.9]))

    def test_curve_output_uses_imag_component_for_imag_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            times = np.arange(0, 3, dtype=int)
            amplitudes = np.array([2.5])
            energies = np.array([0.35])
            sample_params = np.array([[1.0], [1.2]], dtype=float)
            record = DAOutputRecord(
                bz=0,
                component="imag",
                nstates=1,
                tmin=0,
                tmax=2,
                two_point_fit_tmin=0,
                fit_result=DAFitResult(
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
                two_point_fit_tmax=2,
                two_point_fit_table_resolved="two_point_fit_windows.txt",
                two_point_fit_tmax_source="config",
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
                    np.imag(evaluate_da_ratio(times, amplitudes, energies, params, 8, gm="Z5", pz=1, ns=16))
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
            record = DAOutputRecord(
                bz=0,
                component="real",
                nstates=1,
                tmin=0,
                tmax=2,
                two_point_fit_tmin=0,
                fit_result=DAFitResult(
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
                two_point_fit_tmax=2,
                two_point_fit_table_resolved="two_point_fit_windows.txt",
                two_point_fit_tmax_source="config",
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
                    np.real(evaluate_da_ratio(times, amplitudes, energies, params, 8, gm="T5", pz=0, ns=16))
                    for params in sample_params
                ]
            )
            expected_mean = 0.5 * (
                np.percentile(expected_curves, 84.0, axis=0) + np.percentile(expected_curves, 16.0, axis=0)
            )
        self.assertTrue(np.allclose(written, expected_mean))

    def test_fit_da_component_one_state(self) -> None:
        nt = 16
        amplitudes = np.array([2.5])
        energies = np.array([0.35])
        matrix_elements = np.array([1.2])
        times = np.arange(0, nt // 2 + 1)
        model = evaluate_da_ratio(times, amplitudes, energies, matrix_elements, nt, gm="T5", pz=0, ns=32)
        rng = np.random.default_rng(7)
        ratio_samples = model[None, :] + 0.002 * rng.normal(size=(24, times.size))

        meanfit, sample_params = fit_da_component(ratio_samples, amplitudes, energies, nt, 0, 32, "T5", 2, 6, "real")

        self.assertTrue(meanfit.success)
        self.assertAlmostEqual(meanfit.params[0], matrix_elements[0], delta=0.05)
        self.assertEqual(sample_params.shape, (24, 1))

    def test_fit_da_component_two_state(self) -> None:
        nt = 20
        amplitudes = np.array([3.5, 1.2])
        energies = np.array([0.30, 0.70])
        matrix_elements = np.array([0.8, -0.35])
        times = np.arange(0, nt // 2 + 1)
        real_model = evaluate_da_ratio(times, amplitudes, energies, matrix_elements, nt, gm="T5", pz=0, ns=32)
        rng = np.random.default_rng(11)
        ratio_samples = real_model[None, :] + 0.002 * rng.normal(size=(30, times.size))

        meanfit, sample_params = fit_da_component(ratio_samples, amplitudes, energies, nt, 0, 32, "T5", 2, 8, "real")

        self.assertTrue(meanfit.success)
        self.assertTrue(np.allclose(meanfit.params, matrix_elements, atol=0.08))
        self.assertEqual(sample_params.shape, (30, 2))

    def test_end_to_end_workflow_writes_outputs_for_t5_and_z5(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            two_point_fit_root = tmp / "two_point_fit_root"
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

            two_point_fit_window_path = tmp / "two_point_fit_windows.txt"
            two_point_fit_window_path.write_text("0 2 5\n", encoding="utf-8")
            fit_window_path = tmp / "fit_window.txt"
            two_point_fit_root = tmp / "two_point_fit_root"
            fit_window_path.write_text("0 2 5\n", encoding="utf-8")
            self._write_two_point_fit_reference(
                two_point_fit_root,
                "demo_pz0",
                two_point_fit_window_path,
                0,
                amplitudes,
                energies,
            )
            h5_path = tmp / "da_pz0.h5"
            with h5py.File(h5_path, "w") as handle:
                for gm in ("T5", "Z5"):
                    numerator = evaluate_da_numerator(times, amplitudes, energies, matrix_elements, nt, gm=gm, pz=0, ns=ns)
                    for tdir in ("plus", "minus"):
                        dataset = np.column_stack(
                            [numerator * (1.0 + 0.001 * np.sin(times + cfg)) for cfg in range(10)]
                        ).T
                        handle.create_dataset(
                            f"{gm}/eta0/pz0/{tdir}/bT0/bz0",
                            data=dataset,
                        )

            input_path = tmp / "input_da.txt"
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
                        f"fit_window {fit_window_path}",
                        f"two_point_fit_root {two_point_fit_root}",
                        f"qda_h5 {h5_path}",
                        "dataset_path_template {gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
                        f"two_point_fit_window_by_pz {tmp / 'two_point_fit_windows.txt'}",
                        f"c2pt {tmp / 'c2pt_pz*.csv'}",
                        "fold_t periodic",
                        f"tsrange 0 {folded_times[-1]}",
                        f"results_dir {tmp / 'results'}",
                    ]
                ),
                encoding="utf-8",
            )

            outputs = run_da_nstate_fit(input_path)

            self.assertTrue(outputs)
            for gm in ("T5", "Z5"):
                summary = tmp / "results" / "demo_pz0" / f"demo_pz0_{gm}_eta0_bT0_real_1state_summary.txt"
                imag_summary = tmp / "results" / "demo_pz0" / f"demo_pz0_{gm}_eta0_bT0_imag_1state_summary.txt"
                self.assertTrue(summary.exists())
                self.assertTrue(imag_summary.exists())
                self.assertIn("m0", summary.read_text(encoding="utf-8"))

    def test_end_to_end_workflow_selects_gm_specific_qda_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            two_point_fit_root = tmp / "two_point_fit_root"
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

            two_point_fit_window_path = tmp / "two_point_fit_windows.txt"
            two_point_fit_window_path.write_text("0 2 5\n", encoding="utf-8")

            for gm in ("T5", "Z5"):
                h5_path = tmp / f"da_pz0_O{gm}.h5"
                with h5py.File(h5_path, "w") as handle:
                    numerator = evaluate_da_numerator(
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

            fit_window_path = tmp / "fit_window.txt"
            two_point_fit_root = tmp / "two_point_fit_root"
            fit_window_path.write_text("0 2 5\n", encoding="utf-8")
            self._write_two_point_fit_reference(
                two_point_fit_root,
                "demo_pz0",
                two_point_fit_window_path,
                0,
                amplitudes,
                energies,
            )
            input_path = tmp / "input_da_gm_specific.txt"
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
                        f"fit_window {fit_window_path}",
                        f"two_point_fit_root {two_point_fit_root}",
                        f"qda_h5 {tmp / 'da_pz{pz}_O{gm}.h5'}",
                        "dataset_path_template {gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
                        f"two_point_fit_window_by_pz {tmp / 'two_point_fit_windows.txt'}",
                        f"c2pt {tmp / 'c2pt_pz*.csv'}",
                        "fold_t periodic",
                        f"tsrange 0 {folded_times[-1]}",
                        f"results_dir {tmp / 'results'}",
                    ]
                ),
                encoding="utf-8",
            )

            outputs = run_da_nstate_fit(input_path)

            self.assertTrue(outputs)
            for gm in ("T5", "Z5"):
                summary = tmp / "results" / "demo_pz0" / f"demo_pz0_{gm}_eta0_bT0_real_1state_summary.txt"
                imag_summary = tmp / "results" / "demo_pz0" / f"demo_pz0_{gm}_eta0_bT0_imag_1state_summary.txt"
                self.assertTrue(summary.exists())
                self.assertTrue(imag_summary.exists())
                self.assertIn("m0", summary.read_text(encoding="utf-8"))

    def test_missing_gm_specific_qda_file_raises_with_resolved_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            two_point_fit_root = tmp / "two_point_fit_root"
            c2pt_csv = tmp / "c2pt_pz0.csv"
            c2pt_csv.write_text("t,cfg_0\n0,1.0\n1,0.9\n2,0.8\n3,0.7\n", encoding="utf-8")
            two_point_fit_window_path = tmp / "two_point_fit_windows.txt"
            two_point_fit_window_path.write_text("0 2 5\n", encoding="utf-8")
            fit_window_path = tmp / "fit_window.txt"
            two_point_fit_root = tmp / "two_point_fit_root"
            fit_window_path.write_text("0 0 1\n", encoding="utf-8")
            self._write_two_point_fit_reference(
                two_point_fit_root,
                "demo_pz0",
                two_point_fit_window_path,
                0,
                np.array([2.8]),
                np.array([0.42]),
            )
            input_path = tmp / "input_missing___QDA__.txt"
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
                        f"fit_window {fit_window_path}",
                        f"two_point_fit_root {two_point_fit_root}",
                        f"qda_h5 {tmp / 'missing_pz{pz}_O{gm}.h5'}",
                        "dataset_path_template {gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
                        f"two_point_fit_window_by_pz {tmp / 'two_point_fit_windows.txt'}",
                        f"c2pt {tmp / 'c2pt_pz*.csv'}",
                        "fold_t none",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(FileNotFoundError) as ctx:
                run_da_nstate_fit(input_path)

        self.assertEqual(str(ctx.exception), str(tmp / "missing_pz0_OZ5.h5"))

    def test_two_point_fit_tmax_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            two_point_fit_root = tmp / "two_point_fit_root"
            nt = 12
            times = np.arange(nt)
            amplitudes = np.array([2.8])
            energies = np.array([0.42])
            matrix_elements = np.array([1.1])
            denominator = evaluate_two_point_symmetric(times, amplitudes, energies, nt)
            numerator = evaluate_da_numerator(times, amplitudes, energies, matrix_elements, nt, gm="T5", pz=0, ns=16)

            c2pt_csv = tmp / "c2pt_pz0.csv"
            c2pt_data = np.column_stack([denominator for _ in range(8)])
            with c2pt_csv.open("w", encoding="utf-8") as handle:
                handle.write(",".join(["t"] + [f"cfg_{idx}" for idx in range(c2pt_data.shape[1])]) + "\n")
                for t in range(nt):
                    handle.write(str(t) + "," + ",".join(f"{value:.12e}" for value in c2pt_data[t]) + "\n")

            two_point_fit_window_path = tmp / "two_point_fit_windows.txt"
            two_point_fit_window_path.write_text("0 2 5\n", encoding="utf-8")
            fit_window_path = tmp / "fit_window.txt"
            two_point_fit_root = tmp / "two_point_fit_root"
            fit_window_path.write_text("0 2 5\n", encoding="utf-8")
            self._write_two_point_fit_reference(
                two_point_fit_root,
                "demo_pz0",
                two_point_fit_window_path,
                0,
                amplitudes,
                energies,
            )
            h5_path = tmp / "da_pz0.h5"
            with h5py.File(h5_path, "w") as handle:
                for tdir in ("plus", "minus"):
                    handle.create_dataset(f"T5/eta0/pz0/{tdir}/bT0/bz0", data=np.column_stack([numerator for _ in range(8)]).T)

            input_path = tmp / "input_inferred.txt"
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
                        f"fit_window {fit_window_path}",
                        f"two_point_fit_root {two_point_fit_root}",
                        f"qda_h5 {h5_path}",
                        "dataset_path_template {gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
                        f"two_point_fit_window_by_pz {tmp / 'two_point_fit_windows.txt'}",
                        f"c2pt {tmp / 'c2pt_pz*.csv'}",
                        "fold_t periodic",
                        f"results_dir {tmp / 'results'}",
                    ]
                ),
                encoding="utf-8",
            )
            run_da_nstate_fit(input_path)
            summary = (tmp / "results" / "demo_pz0" / "demo_pz0_T5_eta0_bT0_real_1state_summary.txt").read_text(encoding="utf-8")
        self.assertIn("tfit 2 5", summary)
        self.assertIn("two_point_fit_tmax_source config", summary)
        self.assertIn("two_point_fit_tmax 5", summary)

    def test_missing_two_point_fit_window_entry_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            two_point_fit_root = tmp / "two_point_fit_root"
            nt = 12
            times = np.arange(nt)
            amplitudes = np.array([2.8])
            energies = np.array([0.42])
            matrix_elements = np.array([1.1])
            denominator = evaluate_two_point_symmetric(times, amplitudes, energies, nt)
            numerator = evaluate_da_numerator(times, amplitudes, energies, matrix_elements, nt, gm="T5", pz=0, ns=16)
            c2pt_csv = tmp / "c2pt_pz0.csv"
            c2pt_data = np.column_stack([denominator for _ in range(8)])
            with c2pt_csv.open("w", encoding="utf-8") as handle:
                handle.write(",".join(["t"] + [f"cfg_{idx}" for idx in range(c2pt_data.shape[1])]) + "\n")
                for t in range(nt):
                    handle.write(str(t) + "," + ",".join(f"{value:.12e}" for value in c2pt_data[t]) + "\n")

            two_point_fit_window_path = tmp / "two_point_fit_windows.txt"
            two_point_fit_window_path.write_text("1 2 5\n", encoding="utf-8")
            fit_window_path = tmp / "fit_window.txt"
            two_point_fit_root = tmp / "two_point_fit_root"
            fit_window_path.write_text("0 2 5\n", encoding="utf-8")
            h5_path = tmp / "da_pz0.h5"
            with h5py.File(h5_path, "w") as handle:
                for tdir in ("plus", "minus"):
                    handle.create_dataset(f"T5/eta0/pz0/{tdir}/bT0/bz0", data=np.column_stack([numerator for _ in range(8)]).T)

            input_path = tmp / "input_da.txt"
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
                        f"fit_window {fit_window_path}",
                        f"two_point_fit_root {two_point_fit_root}",
                        f"qda_h5 {h5_path}",
                        "dataset_path_template {gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
                        f"two_point_fit_window_by_pz {two_point_fit_window_path}",
                        f"c2pt {tmp / 'c2pt_pz*.csv'}",
                        "fold_t periodic",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing entries for pz values: \\[0\\]"):
                run_da_nstate_fit(input_path)

    def test_fit_window_is_used_directly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            two_point_fit_root = tmp / "two_point_fit_root"
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

            two_point_fit_window_path = tmp / "two_point_fit_windows.txt"
            two_point_fit_window_path.write_text("0 2 5\n", encoding="utf-8")
            (tmp / "fit_windows.txt").write_text("0 2 5\n", encoding="utf-8")
            self._write_two_point_fit_reference(
                two_point_fit_root,
                "demo_pz0",
                two_point_fit_window_path,
                0,
                amplitudes,
                energies,
            )
            h5_path = tmp / "da_pz0.h5"
            with h5py.File(h5_path, "w") as handle:
                numerator = evaluate_da_numerator(
                    times, amplitudes, energies, matrix_elements, nt, gm="T5", pz=0, ns=ns
                )
                for eta in ("eta0", "eta1"):
                    for bT in (0, 1):
                        for tdir in ("plus", "minus"):
                            dataset = np.column_stack(
                                [numerator * (1.0 + 0.001 * np.sin(times + cfg + bT)) for cfg in range(10)]
                            ).T
                            handle.create_dataset(f"T5/{eta}/pz0/{tdir}/bT{bT}/bz0", data=dataset)

            override_path = tmp / "fit_windows.txt"
            two_point_fit_root = tmp / "two_point_fit_root"
            override_path.write_text("0 3 6\n", encoding="utf-8")
            self._write_two_point_fit_reference(
                two_point_fit_root,
                "demo_pz0",
                two_point_fit_window_path,
                0,
                amplitudes,
                energies,
            )
            input_path = tmp / "input_da_override.txt"
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
                        f"fit_window {override_path}",
                        f"two_point_fit_root {two_point_fit_root}",
                        f"qda_h5 {h5_path}",
                        "dataset_path_template {gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
                        f"two_point_fit_window_by_pz {tmp / 'two_point_fit_windows.txt'}",
                        f"c2pt {tmp / 'c2pt_pz*.csv'}",
                        "fold_t periodic",
                        "tsrange 0 6",
                        f"results_dir {tmp / 'results'}",
                    ]
                ),
                encoding="utf-8",
            )

            run_da_nstate_fit(input_path)

            summary_path = tmp / "results" / "demo_pz0" / "demo_pz0_T5_eta0_bT0_real_1state_summary.txt"
            ratio_path = tmp / "results" / "demo_pz0" / "tables" / "demo_pz0_T5_eta0_bT0_ratio.txt"
            summary_text = summary_path.read_text(encoding="utf-8")
            ratio_text = ratio_path.read_text(encoding="utf-8")

        self.assertIn("tfit 3 6", summary_text)
        self.assertIn("tfit 3 6", ratio_text)
        self.assertNotIn("fit_window_source", ratio_text)

    def test_grouped_outputs_by_bt_include_multiple_bz_rows_and_fit_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            two_point_fit_root = tmp / "two_point_fit_root"
            nt = 12
            ns = 16
            times = np.arange(nt)
            amplitudes = np.array([2.8])
            energies = np.array([0.42])
            matrix_elements = np.array([1.1])
            denominator = evaluate_two_point_symmetric(times, amplitudes, energies, nt)
            numerator = evaluate_da_numerator(times, amplitudes, energies, matrix_elements, nt, gm="T5", pz=0, ns=ns)

            c2pt_csv = tmp / "c2pt_pz0.csv"
            c2pt_data = np.column_stack([denominator for _ in range(8)])
            with c2pt_csv.open("w", encoding="utf-8") as handle:
                handle.write(",".join(["t"] + [f"cfg_{idx}" for idx in range(c2pt_data.shape[1])]) + "\n")
                for t in range(nt):
                    handle.write(str(t) + "," + ",".join(f"{value:.12e}" for value in c2pt_data[t]) + "\n")

            two_point_fit_window_path = tmp / "two_point_fit_windows.txt"
            two_point_fit_window_path.write_text("0 2 5\n", encoding="utf-8")
            (tmp / "fit_windows.txt").write_text("0 2 5\n", encoding="utf-8")
            self._write_two_point_fit_reference(
                two_point_fit_root,
                "demo_pz0",
                two_point_fit_window_path,
                0,
                amplitudes,
                energies,
            )
            h5_path = tmp / "da_pz0.h5"
            with h5py.File(h5_path, "w") as handle:
                for tdir in ("plus", "minus"):
                    for bz in (0, 1, -1):
                        handle.create_dataset(
                            f"T5/eta0/pz0/{tdir}/bT0/bz{bz}",
                            data=np.column_stack([numerator for _ in range(8)]).T,
                        )

            input_path = tmp / "input_da.txt"
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
                        f"fit_window {tmp / 'fit_windows.txt'}",
                        f"two_point_fit_root {two_point_fit_root}",
                        f"qda_h5 {h5_path}",
                        "dataset_path_template {gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
                        f"two_point_fit_window_by_pz {tmp / 'two_point_fit_windows.txt'}",
                        f"c2pt {tmp / 'c2pt_pz*.csv'}",
                        "fold_t periodic",
                        f"results_dir {tmp / 'results'}",
                    ]
                ),
                encoding="utf-8",
            )
            (tmp / "fit_windows.txt").write_text("0 2 5\n", encoding="utf-8")
            run_da_nstate_fit(input_path)
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
        self.assertIn("two_point_fit_tmax_source config", summary_text)
        self.assertIn("two_point_fit_tmax", fit_text.splitlines()[0])
        self.assertIn("tsrange 0 5", ratio_text)
        self.assertIn("tfit 2 5", ratio_text)
        self.assertIn("two_point_fit_table_resolved", ratio_text)
        self.assertIn("two_point_fit_tmax_source config", ratio_text)
        self.assertIn("two_point_fit_tmax 5", ratio_text)
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
            two_point_fit_root = tmp / "two_point_fit_root"
            nt = 12
            ns = 16
            times = np.arange(nt)
            amplitudes = np.array([2.8])
            energies = np.array([0.42])
            matrix_elements = np.array([1.1])
            denominator = evaluate_two_point_symmetric(times, amplitudes, energies, nt)
            numerator = evaluate_da_numerator(times, amplitudes, energies, matrix_elements, nt, gm="T5", pz=0, ns=ns)

            c2pt_csv = tmp / "c2pt_pz0.csv"
            c2pt_data = np.column_stack([denominator for _ in range(8)])
            with c2pt_csv.open("w", encoding="utf-8") as handle:
                handle.write(",".join(["t"] + [f"cfg_{idx}" for idx in range(c2pt_data.shape[1])]) + "\n")
                for t in range(nt):
                    handle.write(str(t) + "," + ",".join(f"{value:.12e}" for value in c2pt_data[t]) + "\n")

            two_point_fit_window_path = tmp / "two_point_fit_windows.txt"
            two_point_fit_window_path.write_text("0 2 5\n", encoding="utf-8")
            h5_path = tmp / "da_pz0.h5"
            with h5py.File(h5_path, "w") as handle:
                for tdir in ("plus", "minus"):
                    for bz in (0, 1, -1):
                        handle.create_dataset(
                            f"T5/eta0/pz0/{tdir}/bT0/bz{bz}",
                            data=np.column_stack([numerator for _ in range(8)]).T,
                        )

            fit_window_path = tmp / "fit_window.txt"
            two_point_fit_root = tmp / "two_point_fit_root"
            fit_window_path.write_text("0 2 5\n", encoding="utf-8")
            self._write_two_point_fit_reference(
                two_point_fit_root,
                "demo_pz0",
                two_point_fit_window_path,
                0,
                amplitudes,
                energies,
            )
            input_path = tmp / "input_da_plot.txt"
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
                        f"fit_window {fit_window_path}",
                        f"two_point_fit_root {two_point_fit_root}",
                        f"qda_h5 {h5_path}",
                        "dataset_path_template {gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
                        f"two_point_fit_window_by_pz {tmp / 'two_point_fit_windows.txt'}",
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

            with patch("lqcd_analysis.DA.fit_nstate.plot_da_ratio_fit", side_effect=fake_plot):
                with patch("lqcd_analysis.DA.fit_nstate.plot_da_m0_from_fit_tables", side_effect=fake_plot):
                    run_da_nstate_fit(input_path)

            real_plot = tmp / "results" / "demo_pz0" / "plots" / "demo_pz0_T5_eta0_bT0_real_1state_ratio_fit.pdf"
            imag_plot = tmp / "results" / "demo_pz0" / "plots" / "demo_pz0_T5_eta0_bT0_imag_1state_ratio_fit.pdf"
            m0_plot = tmp / "results" / "demo_pz0" / "plots" / "demo_pz0_T5_eta0_real_1state_m0_vs_bz.pdf"
            real_exists = real_plot.exists()
            imag_exists = imag_plot.exists()
            m0_exists = m0_plot.exists()

        self.assertTrue(real_exists)
        self.assertTrue(imag_exists)
        self.assertTrue(m0_exists)

    def test_plot_da_ratio_fit_uses_log_for_positive_real_series(self) -> None:
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
            with patch("lqcd_analysis.DA.plotting.prepare_matplotlib", return_value=fake_plt):
                plot_da_ratio_fit(Path(tmpdir) / "plot.pdf", series, component="real", fit_window=(1, 2))
        self.assertEqual(fake_ax.scale_calls, [(("log",), {})])

    def test_plot_da_ratio_fit_uses_symlog_for_imag_or_nonpositive_real(self) -> None:
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
            with patch("lqcd_analysis.DA.plotting.prepare_matplotlib", return_value=fake_plt):
                plot_da_ratio_fit(Path(tmpdir) / "imag.pdf", imag_series, component="imag", fit_window=(1, 2))
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
            with patch("lqcd_analysis.DA.plotting.prepare_matplotlib", return_value=fake_plt2):
                plot_da_ratio_fit(Path(tmpdir) / "real.pdf", nonpositive_real_series, component="real", fit_window=(1, 2))
        self.assertEqual(fake_ax2.scale_calls[0][0][0], "symlog")
        self.assertIn("linthresh", fake_ax2.scale_calls[0][1])

    def test_write_da_plot_notebook(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            two_point_fit_root = tmp / "two_point_fit_root"
            notebook = write_da_plot_notebook(
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
                sample_tables={
                    0: {"real": {1: tmp / "bT0_real1_samples.txt", 2: tmp / "bT0_real2_samples.txt"}, "imag": {1: tmp / "bT0_imag1_samples.txt"}},
                    2: {"real": {1: tmp / "bT2_real1_samples.txt", 2: tmp / "bT2_real2_samples.txt"}, "imag": {1: tmp / "bT2_imag1_samples.txt"}},
                },
                title="demo_pz0",
                gm="T5",
                eta="eta0",
                pz=2,
                ns=64,
                lattice_spacing_fm=0.076,
            )
            self.assertTrue(notebook.exists())
            payload = json.loads(notebook.read_text(encoding="utf-8"))
            joined = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
        self.assertEqual(payload["nbformat"], 4)
        self.assertIn("plot_da_grouped_outputs", joined)
        self.assertIn("plot_da_m0_from_fit_tables", joined)
        self.assertIn("run_da_fourier_from_fit_outputs", joined)
        self.assertIn("ratio_tables =", joined)
        self.assertIn("curve_tables =", joined)
        self.assertIn("fit_tables =", joined)
        self.assertIn("sample_tables =", joined)
        self.assertIn("chosen_bT = available_bT[0]", joined)
        self.assertIn("component = 'real'", joined)
        self.assertIn("bz_values = None", joined)
        self.assertIn("selected_bT_values = tuple(value for value in available_bT if value % 2 == 0)", joined)
        self.assertIn("m0_output_name = None", joined)
        self.assertIn("x_values = np.linspace(-0.5, 1.5, 201)", joined)
        self.assertIn("interpolation_kind = 'cubic'", joined)

    def test_plot_da_grouped_outputs_reads_grouped_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            two_point_fit_root = tmp / "two_point_fit_root"
            ratio_table = tmp / "ratio.txt"
            curve_table = tmp / "curve.txt"
            ratio_table.write_text(
                "\n".join(
                    [
                        "tsrange 0 2",
                        "tfit 1 2",
                        "two_point_fit_table_resolved two_point_fit_windows.txt",
                        "two_point_fit_tmax_source config",
                        "two_point_fit_tmax none",
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

            with patch("lqcd_analysis.DA.plotting.plot_da_ratio_fit", side_effect=fake_plot):
                output = plot_da_grouped_outputs(
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
            two_point_fit_root = tmp / "two_point_fit_root"
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
                    "two_point_fit_tmax",
                    "m0_mean",
                    "m0_err",
                ]
            )
            fit_bt0.write_text(
                "\n".join(
                    [
                        header,
                        "2\t2\t5\t1\t1.0\t0.5\t5\t1.20\t0.12",
                        "0\t2\t5\t1\t1.0\t0.5\t5\t1.00\t0.10",
                        "1\t2\t5\t1\t1.0\t0.5\t5\t1.10\t0.11",
                    ]
                ),
                encoding="utf-8",
            )
            fit_bt2.write_text(
                "\n".join(
                    [
                        header,
                        "1\t2\t5\t1\t1.0\t0.5\t5\t2.10\t0.21",
                        "0\t2\t5\t1\t1.0\t0.5\t5\t2.00\t0.20",
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

    def test_plot_da_m0_from_fit_tables_builds_one_series_per_bt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            two_point_fit_root = tmp / "two_point_fit_root"
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
                    "two_point_fit_tmax",
                    "m0_mean",
                    "m0_err",
                ]
            )
            fit_bt0.write_text("\n".join([header, "1\t2\t5\t1\t1.0\t0.5\t5\t1.10\t0.11"]), encoding="utf-8")
            fit_bt2.write_text("\n".join([header, "0\t2\t5\t1\t1.0\t0.5\t5\t2.00\t0.20"]), encoding="utf-8")

            captured = {}

            def fake_plot(output_path, series, **kwargs):
                captured["output_path"] = output_path
                captured["series"] = series
                captured["kwargs"] = kwargs
                output_path.write_text("ok\n", encoding="utf-8")
                return output_path

            with patch("lqcd_analysis.DA.plotting.plot_da_m0_vs_bz", side_effect=fake_plot):
                output = plot_da_m0_from_fit_tables(
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

    def test_load_da_m0_fit_and_sample_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            two_point_fit_root = tmp / "two_point_fit_root"
            fit_table = tmp / "fit.txt"
            sample_table = tmp / "samples.txt"
            fit_table.write_text(
                "\n".join(
                    [
                        "bz\ttmin\ttmax\tsuccess_meanfit\tchi2_dof\tpvalue\ttwo_point_fit_tmax\tm0_mean\tm0_err",
                        "2\t2\t6\t1\t1.0\t0.5\t6\t1.20\t0.12",
                        "0\t2\t6\t1\t1.0\t0.5\t6\t1.00\t0.10",
                        "1\t2\t6\t1\t1.0\t0.5\t6\t1.10\t0.11",
                    ]
                ),
                encoding="utf-8",
            )
            sample_table.write_text(
                "\n".join(
                    [
                        "bz\tsample_id\tsuccess\tm0",
                        "0\t0\t1\t1.00",
                        "1\t0\t1\t1.10",
                        "2\t0\t1\t1.20",
                        "0\t1\t1\t0.90",
                        "1\t1\t1\t1.00",
                        "2\t1\t1\t1.10",
                        "0\t2\t1\t1.10",
                        "1\t2\t0\tnan",
                        "2\t2\t1\t1.30",
                    ]
                ),
                encoding="utf-8",
            )
            bz_fit, m0_mean, m0_err = load_da_m0_fit_table(fit_table)
            bz_samples, m0_samples = load_da_m0_sample_table(sample_table)

        self.assertTrue(np.array_equal(bz_fit, np.array([0, 1, 2])))
        self.assertTrue(np.allclose(m0_mean, np.array([1.0, 1.1, 1.2])))
        self.assertTrue(np.allclose(m0_err, np.array([0.10, 0.11, 0.12])))
        self.assertTrue(np.array_equal(bz_samples, np.array([0, 1, 2])))
        self.assertEqual(m0_samples.shape, (2, 3))
        self.assertTrue(np.allclose(m0_samples[0], np.array([1.0, 1.1, 1.2])))

    def test_compute_da_cosine_transform_constant_function(self) -> None:
        bz_values = np.array([0, 1, 2], dtype=int)
        m0_samples = np.ones((2, 3), dtype=float)
        x_values = np.array([0.5], dtype=float)
        x_grid, transformed = compute_da_cosine_transform(
            bz_values,
            m0_samples,
            pz=1,
            ns=10,
            lattice_spacing_fm=1.0,
            x_values=x_values,
            zstep_fm=0.01,
            interpolation_kind="linear",
        )
        expected = np.array([[0.4], [0.4]])
        self.assertTrue(np.array_equal(x_grid, x_values))
        self.assertTrue(np.allclose(transformed, expected, atol=5e-3))

    def test_summarize_and_write_da_fourier_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            two_point_fit_root = tmp / "two_point_fit_root"
            fit_table = tmp / "fit.txt"
            sample_table = tmp / "samples.txt"
            fit_table.write_text(
                "\n".join(
                    [
                        "bz\ttmin\ttmax\tsuccess_meanfit\tchi2_dof\tpvalue\ttwo_point_fit_tmax\tm0_mean\tm0_err",
                        "0\t2\t6\t1\t1.0\t0.5\t6\t1.00\t0.10",
                        "1\t2\t6\t1\t1.0\t0.5\t6\t1.10\t0.11",
                        "2\t2\t6\t1\t1.0\t0.5\t6\t1.20\t0.12",
                    ]
                ),
                encoding="utf-8",
            )
            sample_table.write_text(
                "\n".join(
                    [
                        "bz\tsample_id\tsuccess\tm0",
                        "0\t0\t1\t1.00",
                        "1\t0\t1\t1.10",
                        "2\t0\t1\t1.20",
                        "0\t1\t1\t0.90",
                        "1\t1\t1\t1.00",
                        "2\t1\t1\t1.10",
                    ]
                ),
                encoding="utf-8",
            )
            outputs = run_da_fourier_from_fit_outputs(
                output_root=tmp,
                stem="demo_pz0_T5_eta0_bT0",
                fit_table=fit_table,
                sample_table=sample_table,
                pz=1,
                ns=10,
                lattice_spacing_fm=1.0,
                bT=0,
                component="real",
                nstates=1,
                x_values=np.array([0.0, 0.5, 1.0]),
                zstep_fm=0.05,
                interpolation_kind="linear",
                make_plots=False,
            )
            summary = (tmp / "tables" / "demo_pz0_T5_eta0_bT0_real_1state_fourier.txt").read_text(encoding="utf-8")
            samples = (tmp / "samples" / "demo_pz0_T5_eta0_bT0_real_1state_fourier_samples.txt").read_text(encoding="utf-8")

        self.assertEqual(len(outputs), 2)
        self.assertIn("zstep_fm 5.0000000000e-02", summary)
        self.assertIn("interpolation_kind linear", summary)
        self.assertIn("x\tq_mean\tq_err\tq_p16\tq_p84", summary)
        self.assertIn("sample_id\tx\tq_sample", samples)

    def test_da_fourier_batch_workflow_resolves_paths_for_multiple_pz_and_bt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            two_point_fit_root = tmp / "two_point_fit_root"
            input_root = tmp / "fit_outputs"
            for title in ("demo_pz0", "demo_pz2"):
                for bT in (0, 1):
                    self._write_da_m0_outputs(
                        input_root,
                        title,
                        "T5",
                        "eta0",
                        bT,
                        "real",
                        1,
                        fit_rows=[(0, 1.0 + 0.1 * bT, 0.1), (1, 1.2 + 0.1 * bT, 0.1)],
                        sample_rows=[
                            (0, 0, 1, 1.0 + 0.1 * bT),
                            (1, 0, 1, 1.2 + 0.1 * bT),
                            (0, 1, 1, 0.9 + 0.1 * bT),
                            (1, 1, 1, 1.1 + 0.1 * bT),
                        ],
                    )
            input_path = tmp / "fourier.txt"
            input_path.write_text(
                "\n".join(
                    [
                        "title_pattern demo_pz*",
                        f"input_root {input_root}",
                        "ns 64",
                        "lattice_spacing_fm 0.076",
                        "pzlist 0 2",
                        "gmlist T5",
                        "etalist eta0",
                        "bTlist 0 1",
                        "component real",
                        "nstates 1",
                        "normalization_mode raw",
                        "x_range -0.5 1.5",
                        "x_count 11",
                        "zstep_fm 0.02",
                        "interpolation_kind linear",
                        "plot false",
                        f"results_dir {tmp / 'fourier_outputs'}",
                    ]
                ),
                encoding="utf-8",
            )

            outputs = run_da_fourier_workflow(input_path)

            expected = [
                tmp / "fourier_outputs" / "demo_pz0" / "tables" / "demo_pz0_T5_eta0_bT0_real_1state_fourier.txt",
                tmp / "fourier_outputs" / "demo_pz0" / "tables" / "demo_pz0_T5_eta0_bT1_real_1state_fourier.txt",
                tmp / "fourier_outputs" / "demo_pz2" / "tables" / "demo_pz2_T5_eta0_bT0_real_1state_fourier.txt",
                tmp / "fourier_outputs" / "demo_pz2" / "tables" / "demo_pz2_T5_eta0_bT1_real_1state_fourier.txt",
            ]
            expected_exist = all(path.exists() for path in expected)
            fourier_txt_count = len([path for path in outputs if path.suffix == ".txt" and "fourier" in path.name])

        self.assertTrue(expected_exist)
        self.assertEqual(fourier_txt_count, 8)

    def test_da_fourier_resolves_raw_vs_normalized_outputs_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            two_point_fit_root = tmp / "two_point_fit_root"
            input_root = tmp / "fit_outputs"
            self._write_da_m0_outputs(
                input_root,
                "demo_pz2",
                "T5",
                "eta0",
                0,
                "real",
                1,
                fit_rows=[(0, 1.0, 0.1)],
                sample_rows=[(0, 0, 1, 1.0)],
            )
            normalized_root = tmp / "normalized"
            norm_title_root = normalized_root / "demo_pz2"
            (norm_title_root / "tables").mkdir(parents=True)
            (norm_title_root / "samples").mkdir(parents=True)
            (norm_title_root / "tables" / "demo_pz2_T5_eta0_bT0_mode1_real_1state_fit.txt").write_text(
                "normalization_mode mode1\nbz\tm0_mean\tm0_err\n0\t2.0\t0.1\n",
                encoding="utf-8",
            )
            (norm_title_root / "samples" / "demo_pz2_T5_eta0_bT0_mode1_real_1state_samples.txt").write_text(
                "normalization_mode mode1\nbz\tsample_id\tsuccess\tm0\n0\t0\t1\t2.0\n",
                encoding="utf-8",
            )

            raw_input = tmp / "fourier_raw.txt"
            raw_input.write_text(
                "\n".join(
                    [
                        "title_pattern demo_pz*",
                        f"input_root {input_root}",
                        "ns 64",
                        "lattice_spacing_fm 0.076",
                        "pzlist 2",
                        "gmlist T5",
                        "etalist eta0",
                        "bTlist 0",
                        "component real",
                        "nstates 1",
                        "normalization_mode raw",
                        "x_range -0.5 1.5",
                        "x_count 5",
                        "plot false",
                        f"results_dir {tmp / 'raw_outputs'}",
                    ]
                ),
                encoding="utf-8",
            )
            norm_input = tmp / "fourier_mode1.txt"
            norm_input.write_text(
                "\n".join(
                    [
                        "title_pattern demo_pz*",
                        f"input_root {normalized_root}",
                        "ns 64",
                        "lattice_spacing_fm 0.076",
                        "pzlist 2",
                        "gmlist T5",
                        "etalist eta0",
                        "bTlist 0",
                        "component real",
                        "nstates 1",
                        "normalization_mode mode1",
                        "x_range -0.5 1.5",
                        "x_count 5",
                        "plot false",
                        f"results_dir {tmp / 'norm_outputs'}",
                    ]
                ),
                encoding="utf-8",
            )

            run_da_fourier_workflow(raw_input)
            run_da_fourier_workflow(norm_input)

            raw_summary = (tmp / "raw_outputs" / "demo_pz2" / "tables" / "demo_pz2_T5_eta0_bT0_real_1state_fourier.txt").read_text(encoding="utf-8")
            norm_summary = (tmp / "norm_outputs" / "demo_pz2" / "tables" / "demo_pz2_T5_eta0_bT0_mode1_real_1state_fourier.txt").read_text(encoding="utf-8")

        self.assertIn("q_mean", raw_summary)
        self.assertIn("q_mean", norm_summary)

    def test_da_fourier_missing_requested_normalization_mode_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            two_point_fit_root = tmp / "two_point_fit_root"
            input_root = tmp / "normalized"
            title_root = input_root / "demo_pz2"
            (title_root / "tables").mkdir(parents=True)
            (title_root / "samples").mkdir(parents=True)
            (title_root / "tables" / "demo_pz2_T5_eta0_bT0_mode1_real_1state_fit.txt").write_text(
                "normalization_mode mode1\nbz\tm0_mean\tm0_err\n0\t2.0\t0.1\n",
                encoding="utf-8",
            )
            (title_root / "samples" / "demo_pz2_T5_eta0_bT0_mode1_real_1state_samples.txt").write_text(
                "normalization_mode mode1\nbz\tsample_id\tsuccess\tm0\n0\t0\t1\t2.0\n",
                encoding="utf-8",
            )
            input_path = tmp / "fourier_mode2.txt"
            input_path.write_text(
                "\n".join(
                    [
                        "title_pattern demo_pz*",
                        f"input_root {input_root}",
                        "ns 64",
                        "lattice_spacing_fm 0.076",
                        "pzlist 2",
                        "gmlist T5",
                        "etalist eta0",
                        "bTlist 0",
                        "component real",
                        "nstates 1",
                        "normalization_mode mode2",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaises(FileNotFoundError) as ctx:
                run_da_fourier_workflow(input_path)
        self.assertIn("mode2 fit/sample outputs do not exist", str(ctx.exception))

    def test_da_normalization_mode1_mode2_mode3(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            two_point_fit_root = tmp / "two_point_fit_root"
            input_root = tmp / "fit_outputs"
            self._write_da_m0_outputs(
                input_root,
                "demo_pz2",
                "T5",
                "eta0",
                1,
                "real",
                1,
                fit_rows=[(0, 5.0, 0.1), (1, 10.0, 0.2)],
                sample_rows=[
                    (0, 0, 1, 4.0),
                    (1, 0, 1, 8.0),
                    (0, 1, 1, 6.0),
                    (1, 1, 1, 12.0),
                ],
            )
            self._write_da_m0_outputs(
                input_root,
                "demo_pz2",
                "T5",
                "eta0",
                0,
                "real",
                1,
                fit_rows=[(0, 2.5, 0.1)],
                sample_rows=[(0, 0, 1, 2.0), (0, 1, 1, 3.0)],
            )
            self._write_da_m0_outputs(
                input_root,
                "demo_pz0",
                "T5",
                "eta0",
                1,
                "real",
                1,
                fit_rows=[(0, 1.5, 0.1)],
                sample_rows=[(0, 0, 1, 1.0), (0, 1, 1, 2.0)],
            )
            self._write_da_m0_outputs(
                input_root,
                "demo_pz0",
                "T5",
                "eta0",
                0,
                "real",
                1,
                fit_rows=[(0, 1.0, 0.1)],
                sample_rows=[(0, 0, 1, 0.5), (0, 1, 1, 1.0)],
            )

            def run_mode(mode: str) -> tuple[str, str]:
                input_path = tmp / f"{mode}.txt"
                input_path.write_text(
                    "\n".join(
                        [
                            "title_pattern demo_pz*",
                            f"input_root {input_root}",
                            "ns 64",
                            "lattice_spacing_fm 0.076",
                            "pzlist 2",
                            "gmlist T5",
                            "etalist eta0",
                            "bTlist 1",
                            "bzlist 0 1",
                            "component real",
                            "nstates 1",
                            f"normalization_mode {mode}",
                            f"results_dir {tmp / ('normalized_' + mode)}",
                        ]
                    ),
                    encoding="utf-8",
                )
                run_da_normalization(input_path)
                out_root = tmp / f"normalized_{mode}" / "demo_pz2"
                fit_text = (out_root / "tables" / f"demo_pz2_T5_eta0_bT1_{mode}_real_1state_fit.txt").read_text(encoding="utf-8")
                sample_text = (out_root / "samples" / f"demo_pz2_T5_eta0_bT1_{mode}_real_1state_samples.txt").read_text(encoding="utf-8")
                return fit_text, sample_text

            fit_mode1, samples_mode1 = run_mode("mode1")
            fit_mode2, samples_mode2 = run_mode("mode2")
            fit_mode3, samples_mode3 = run_mode("mode3")

        self.assertIn("normalization_mode mode1", fit_mode1)
        self.assertIn("2.0000000000e+00", fit_mode1)
        self.assertIn("4.0000000000e+00", fit_mode1)
        self.assertIn("3.5000000000e+00", fit_mode2)
        self.assertIn("7.0000000000e+00", fit_mode2)
        self.assertIn("1.0000000000e+00", fit_mode3)
        self.assertIn("2.0000000000e+00", fit_mode3)
        self.assertIn("0\t0\t1\t2.0000000000e+00", samples_mode1)
        self.assertIn("1\t1\t1\t6.0000000000e+00", samples_mode2)
        self.assertIn("1\t0\t1\t2.0000000000e+00", samples_mode3)

    def test_da_normalization_uses_complex_samples_when_both_components_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            two_point_fit_root = tmp / "two_point_fit_root"
            input_root = tmp / "fit_outputs"
            # target: (1 + i) / (1 - i) = i, so real output should be 0 and imag output should be 1
            self._write_da_m0_outputs(
                input_root,
                "demo_pz2",
                "T5",
                "eta0",
                1,
                "real",
                1,
                fit_rows=[(0, 1.0, 0.1)],
                sample_rows=[(0, 0, 1, 1.0)],
            )
            self._write_da_m0_outputs(
                input_root,
                "demo_pz2",
                "T5",
                "eta0",
                1,
                "imag",
                1,
                fit_rows=[(0, 1.0, 0.1)],
                sample_rows=[(0, 0, 1, 1.0)],
            )
            self._write_da_m0_outputs(
                input_root,
                "demo_pz0",
                "T5",
                "eta0",
                1,
                "real",
                1,
                fit_rows=[(0, 1.0, 0.1)],
                sample_rows=[(0, 0, 1, 1.0)],
            )
            self._write_da_m0_outputs(
                input_root,
                "demo_pz0",
                "T5",
                "eta0",
                1,
                "imag",
                1,
                fit_rows=[(0, -1.0, 0.1)],
                sample_rows=[(0, 0, 1, -1.0)],
            )

            def run_component(component: str) -> tuple[str, str]:
                input_path = tmp / f"normalize_{component}.txt"
                input_path.write_text(
                    "\n".join(
                        [
                            "title_pattern demo_pz*",
                            f"input_root {input_root}",
                            "ns 64",
                            "lattice_spacing_fm 0.076",
                            "pzlist 2",
                            "gmlist T5",
                            "etalist eta0",
                            "bTlist 1",
                            "bzlist 0",
                            f"component {component}",
                            "nstates 1",
                            "normalization_mode mode2",
                            f"results_dir {tmp / ('normalized_' + component)}",
                        ]
                    ),
                    encoding="utf-8",
                )
                run_da_normalization(input_path)
                out_root = tmp / f"normalized_{component}" / "demo_pz2"
                fit_text = (out_root / "tables" / f"demo_pz2_T5_eta0_bT1_mode2_{component}_1state_fit.txt").read_text(encoding="utf-8")
                sample_text = (out_root / "samples" / f"demo_pz2_T5_eta0_bT1_mode2_{component}_1state_samples.txt").read_text(encoding="utf-8")
                return fit_text, sample_text

            fit_real, sample_real = run_component("real")
            fit_imag, sample_imag = run_component("imag")

        self.assertIn("normalization_sample_domain complex", fit_real)
        self.assertIn("normalization_sample_domain complex", fit_imag)
        self.assertIn("0.0000000000e+00", fit_real)
        self.assertIn("1.0000000000e+00", fit_imag)
        self.assertIn("0\t0\t1\t0.0000000000e+00", sample_real)
        self.assertIn("0\t0\t1\t1.0000000000e+00", sample_imag)

    def test_da_normalization_missing_reference_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            two_point_fit_root = tmp / "two_point_fit_root"
            input_root = tmp / "fit_outputs"
            self._write_da_m0_outputs(
                input_root,
                "demo_pz2",
                "T5",
                "eta0",
                1,
                "real",
                1,
                fit_rows=[(0, 5.0, 0.1)],
                sample_rows=[(0, 0, 1, 4.0)],
            )
            input_path = tmp / "normalize.txt"
            input_path.write_text(
                "\n".join(
                    [
                        "title_pattern demo_pz*",
                        f"input_root {input_root}",
                        "ns 64",
                        "lattice_spacing_fm 0.076",
                        "pzlist 2",
                        "gmlist T5",
                        "etalist eta0",
                        "bTlist 1",
                        "bzlist 0",
                        "component real",
                        "nstates 1",
                        "normalization_mode mode1",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaises(FileNotFoundError) as ctx:
                run_da_normalization(input_path)
        self.assertIn("DA normalization fit table does not exist", str(ctx.exception))

    def test_da_fit_workflow_does_not_auto_run_fourier(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            two_point_fit_root = tmp / "two_point_fit_root"
            nt = 12
            ns = 16
            times = np.arange(nt)
            amplitudes = np.array([2.8])
            energies = np.array([0.42])
            matrix_elements = np.array([1.1])
            denominator = evaluate_two_point_symmetric(times, amplitudes, energies, nt)
            numerator = evaluate_da_numerator(times, amplitudes, energies, matrix_elements, nt, gm="T5", pz=0, ns=ns)
            c2pt_data = np.tile(denominator[:, None], (1, 4))
            numerator_data = np.column_stack([numerator for _ in range(4)]).T
            h5_path = tmp / "da_pz0.h5"
            with h5py.File(h5_path, "w") as handle:
                handle.create_dataset("T5/eta0/pz0/plus/bT0/bz0", data=numerator_data)
                handle.create_dataset("T5/eta0/pz0/minus/bT0/bz0", data=numerator_data)
            with (tmp / "c2pt_pz0.csv").open("w", encoding="utf-8") as handle:
                handle.write("," + ",".join(f"cfg{i}" for i in range(c2pt_data.shape[1])) + "\n")
                for t in range(nt):
                    handle.write(str(t) + "," + ",".join(f"{value:.12e}" for value in c2pt_data[t]) + "\n")
            two_point_fit_window_path = tmp / "two_point_fit_windows.txt"
            two_point_fit_window_path.write_text("0 2 5\n", encoding="utf-8")
            fit_window_path = tmp / "fit_window.txt"
            two_point_fit_root = tmp / "two_point_fit_root"
            fit_window_path.write_text("0 2 5\n", encoding="utf-8")
            self._write_two_point_fit_reference(
                two_point_fit_root,
                "demo_pz0",
                two_point_fit_window_path,
                0,
                amplitudes,
                energies,
            )
            input_path = tmp / "input_da.txt"
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
                        f"fit_window {fit_window_path}",
                        f"two_point_fit_root {two_point_fit_root}",
                        f"qda_h5 {h5_path}",
                        "dataset_path_template {gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
                        f"two_point_fit_window_by_pz {tmp / 'two_point_fit_windows.txt'}",
                        f"c2pt {tmp / 'c2pt_pz*.csv'}",
                        "fold_t periodic",
                        f"results_dir {tmp / 'results'}",
                    ]
                ),
                encoding="utf-8",
            )

            run_da_nstate_fit(input_path)

            fourier_table = tmp / "results" / "demo_pz0" / "tables" / "demo_pz0_T5_eta0_bT0_real_1state_fourier.txt"
            fourier_samples = tmp / "results" / "demo_pz0" / "samples" / "demo_pz0_T5_eta0_bT0_real_1state_fourier_samples.txt"
            fourier_plot = tmp / "results" / "demo_pz0" / "plots" / "demo_pz0_T5_eta0_bT0_real_1state_fourier.pdf"

        self.assertFalse(fourier_table.exists())
        self.assertFalse(fourier_samples.exists())
        self.assertFalse(fourier_plot.exists())

    def test_workflow_writes_da_plot_notebook(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            two_point_fit_root = tmp / "two_point_fit_root"
            nt = 12
            ns = 16
            times = np.arange(nt)
            amplitudes = np.array([2.8])
            energies = np.array([0.42])
            matrix_elements = np.array([1.1])
            denominator = evaluate_two_point_symmetric(times, amplitudes, energies, nt)
            numerator = evaluate_da_numerator(times, amplitudes, energies, matrix_elements, nt, gm="T5", pz=0, ns=ns)

            c2pt_csv = tmp / "c2pt_pz0.csv"
            c2pt_data = np.column_stack([denominator for _ in range(8)])
            with c2pt_csv.open("w", encoding="utf-8") as handle:
                handle.write(",".join(["t"] + [f"cfg_{idx}" for idx in range(c2pt_data.shape[1])]) + "\n")
                for t in range(nt):
                    handle.write(str(t) + "," + ",".join(f"{value:.12e}" for value in c2pt_data[t]) + "\n")

            two_point_fit_window_path = tmp / "two_point_fit_windows.txt"
            two_point_fit_window_path.write_text("0 2 5\n", encoding="utf-8")
            h5_path = tmp / "da_pz0.h5"
            with h5py.File(h5_path, "w") as handle:
                for tdir in ("plus", "minus"):
                    for bz in (0, 1, -1):
                        handle.create_dataset(
                            f"T5/eta0/pz0/{tdir}/bT0/bz{bz}",
                            data=np.column_stack([numerator for _ in range(8)]).T,
                        )

            fit_window_path = tmp / "fit_window.txt"
            two_point_fit_root = tmp / "two_point_fit_root"
            fit_window_path.write_text("0 2 5\n", encoding="utf-8")
            self._write_two_point_fit_reference(
                two_point_fit_root,
                "demo_pz0",
                two_point_fit_window_path,
                0,
                amplitudes,
                energies,
            )
            input_path = tmp / "input_da_notebook.txt"
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
                        f"fit_window {fit_window_path}",
                        f"two_point_fit_root {two_point_fit_root}",
                        f"qda_h5 {h5_path}",
                        "dataset_path_template {gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
                        f"two_point_fit_window_by_pz {tmp / 'two_point_fit_windows.txt'}",
                        f"c2pt {tmp / 'c2pt_pz*.csv'}",
                        "fold_t periodic",
                        f"results_dir {tmp / 'results'}",
                    ]
                ),
                encoding="utf-8",
            )

            run_da_nstate_fit(input_path)
            notebook = tmp / "results" / "notebook_plots" / "demo_pz0" / "demo_pz0_T5_eta0_da_plots.ipynb"
            notebook_exists = notebook.exists()
            payload = json.loads(notebook.read_text(encoding="utf-8"))
            joined = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])

        self.assertTrue(notebook_exists)
        self.assertEqual(payload["nbformat"], 4)
        self.assertIn("plot_da_grouped_outputs", joined)
        self.assertIn("plot_da_m0_from_fit_tables", joined)
        self.assertIn("run_da_fourier_from_fit_outputs", joined)
        self.assertIn("demo_pz0_T5_eta0_bT0_ratio.txt", joined)
        self.assertIn("demo_pz0_T5_eta0_bT0_real_1state_fit.txt", joined)
        self.assertIn("demo_pz0_T5_eta0_bT0_imag_1state_fit.txt", joined)
        self.assertIn("demo_pz0_T5_eta0_bT0_real_1state_samples.txt", joined)
        self.assertIn("demo_pz0_T5_eta0_bT0_real_1state_curve.txt", joined)
        self.assertIn("demo_pz0_T5_eta0_bT0_imag_1state_curve.txt", joined)

    def test_workflow_groups_m0_vs_bz_plot_across_bt_fit_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            two_point_fit_root = tmp / "two_point_fit_root"
            nt = 12
            ns = 16
            times = np.arange(nt)
            amplitudes = np.array([2.8])
            energies = np.array([0.42])
            matrix_elements = np.array([1.1])
            denominator = evaluate_two_point_symmetric(times, amplitudes, energies, nt)
            numerator = evaluate_da_numerator(times, amplitudes, energies, matrix_elements, nt, gm="T5", pz=0, ns=ns)

            c2pt_csv = tmp / "c2pt_pz0.csv"
            c2pt_data = np.column_stack([denominator for _ in range(8)])
            with c2pt_csv.open("w", encoding="utf-8") as handle:
                handle.write(",".join(["t"] + [f"cfg_{idx}" for idx in range(c2pt_data.shape[1])]) + "\n")
                for t in range(nt):
                    handle.write(str(t) + "," + ",".join(f"{value:.12e}" for value in c2pt_data[t]) + "\n")

            two_point_fit_window_path = tmp / "two_point_fit_windows.txt"
            two_point_fit_window_path.write_text("0 2 5\n", encoding="utf-8")
            h5_path = tmp / "da_pz0.h5"
            with h5py.File(h5_path, "w") as handle:
                for bT in (0, 1):
                    for tdir in ("plus", "minus"):
                        for bz in (0, 1, -1):
                            handle.create_dataset(
                                f"T5/eta0/pz0/{tdir}/bT{bT}/bz{bz}",
                                data=np.column_stack([numerator for _ in range(8)]).T,
                            )

            fit_window_path = tmp / "fit_window.txt"
            two_point_fit_root = tmp / "two_point_fit_root"
            fit_window_path.write_text("0 2 5\n", encoding="utf-8")
            self._write_two_point_fit_reference(
                two_point_fit_root,
                "demo_pz0",
                two_point_fit_window_path,
                0,
                amplitudes,
                energies,
            )
            input_path = tmp / "input_da_m0_group.txt"
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
                        f"fit_window {fit_window_path}",
                        f"two_point_fit_root {two_point_fit_root}",
                        f"qda_h5 {h5_path}",
                        "dataset_path_template {gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
                        f"two_point_fit_window_by_pz {tmp / 'two_point_fit_windows.txt'}",
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

            with patch("lqcd_analysis.DA.fit_nstate.plot_da_ratio_fit", side_effect=fake_ratio_plot):
                with patch("lqcd_analysis.DA.fit_nstate.plot_da_m0_from_fit_tables", side_effect=fake_m0_plot):
                    run_da_nstate_fit(input_path)

        self.assertEqual(captured["keys"], [0])
        self.assertIn("demo_pz0_T5_eta0_bT0_real_1state_fit.txt", captured["paths"])

    def test_two_point_fit_reference_is_resolved_once_per_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            two_point_fit_root = tmp / "two_point_fit_root"
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
            two_point_fit_window_path = tmp / "two_point_fit_windows.txt"
            two_point_fit_window_path.write_text("0 2 5\n", encoding="utf-8")
            self._write_two_point_fit_reference(
                two_point_fit_root,
                "demo_pz0",
                two_point_fit_window_path,
                0,
                amplitudes,
                energies,
            )
            h5_path = tmp / "da_pz0.h5"
            numerator = evaluate_da_numerator(times, amplitudes[:1], energies[:1], matrix_elements, nt, gm="T5", pz=0, ns=ns)
            with h5py.File(h5_path, "w") as handle:
                for eta in ("eta0", "eta1"):
                    for tdir in ("plus", "minus"):
                        handle.create_dataset(f"T5/{eta}/pz0/{tdir}/bT0/bz0", data=np.column_stack([numerator for _ in range(8)]).T)
            fit_window_path = tmp / "fit_window.txt"
            two_point_fit_root = tmp / "two_point_fit_root"
            fit_window_path.write_text("0 2 5\n", encoding="utf-8")
            self._write_two_point_fit_reference(
                two_point_fit_root,
                "demo_pz0",
                two_point_fit_window_path,
                0,
                amplitudes,
                energies,
            )
            input_path = tmp / "input_da.txt"
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
                        f"fit_window {fit_window_path}",
                        f"two_point_fit_root {two_point_fit_root}",
                        f"qda_h5 {h5_path}",
                        "dataset_path_template {gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
                        f"two_point_fit_window_by_pz {two_point_fit_window_path}",
                        f"c2pt {tmp / 'c2pt_pz*.csv'}",
                        "fold_t periodic",
                        f"results_dir {tmp / 'results'}",
                    ]
                ),
                encoding="utf-8",
            )
            with patch("lqcd_analysis.DA.fit_nstate.resolve_two_point_fit_reference", wraps=resolve_two_point_fit_reference) as mocked_resolve:
                run_da_nstate_fit(input_path)
        self.assertEqual(mocked_resolve.call_count, 1)


if __name__ == "__main__":
    unittest.main()
