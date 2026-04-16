import tempfile
import unittest
from pathlib import Path

import numpy as np

from lqcd_analysis.tmdwf.cs_kernel_extract import (
    compute_pairwise_type2_estimators,
    extract_cs_kernel_for_reference,
    load_cs_kernel_dataset,
    load_cs_kernel_observable,
    parse_tmdwf_cs_kernel_input,
    run_tmdwf_cs_kernel_workflow,
)
from lqcd_analysis.tmdwf.cs_kernel_matching import build_cs_dgamma, perturbative_order_from_label


class TMDWFCSKernelTests(unittest.TestCase):
    def test_perturbative_order_mapping_matches_legacy_labels(self) -> None:
        self.assertEqual(perturbative_order_from_label("LO"), 0)
        self.assertEqual(perturbative_order_from_label("NLO"), 1)
        self.assertEqual(perturbative_order_from_label("NLL"), 1)
        self.assertEqual(perturbative_order_from_label("NNLO"), 2)
        self.assertEqual(perturbative_order_from_label("NNLL"), 2)

    def test_load_cs_kernel_observable_accepts_fourier_sample_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            sample_path = tmp / "samples.txt"
            sample_path.write_text(
                "\n".join(
                    [
                        "sample_id\tx\tq_sample",
                        "0\t0.1\t1.0",
                        "0\t0.3\t1.2",
                        "0\t0.5\t1.4",
                        "1\t0.1\t1.1",
                        "1\t0.3\t1.3",
                        "1\t0.5\t1.5",
                    ]
                ),
                encoding="utf-8",
            )
            observable = load_cs_kernel_observable(sample_path)
        self.assertTrue(np.allclose(observable.x, [0.1, 0.3, 0.5]))
        self.assertEqual(observable.samples.shape, (2, 3))
        self.assertTrue(np.allclose(observable.samples[1], [1.1, 1.3, 1.5]))

    def test_pairwise_type2_estimator_matches_expected_log_ratio_when_lo(self) -> None:
        reference = np.array([4.0, 8.0, 16.0], dtype=float)
        comparison = [np.array([2.0, 4.0, 8.0], dtype=float)]
        estimators, sigmas = compute_pairwise_type2_estimators(
            reference,
            comparison,
            scheme="CG",
            kernel_label="LO",
            mu=2.0,
            x_value=0.3,
            p1_gev=2.0,
            p2_gevs=[4.0],
        )
        expected = 1.0 / np.log(2.0 / 4.0) * np.log(np.abs(reference / comparison[0]))
        self.assertTrue(np.allclose(estimators[0], expected))
        self.assertTrue(np.all(sigmas > 0.0))

    def test_extract_cs_kernel_for_reference_preserves_bootstrap_gamma_values(self) -> None:
        x_grid = np.array([0.1, 0.3, 0.5, 0.9], dtype=float)
        gamma_by_sample = np.array([0.2, 0.25, 0.3], dtype=float)
        dataset = {}
        for pz in (2, 3, 4):
            samples = []
            for sample_gamma, base_offset in zip(gamma_by_sample, (1.0, 1.5, 2.0), strict=True):
                base = base_offset + x_grid
                samples.append(base * (float(pz) ** sample_gamma))
            samples_array = np.asarray(samples, dtype=float)
            dataset[(0, pz)] = type("Obs", (), {"x": x_grid, "samples": samples_array})()
        x_values, gamma_samples, chi2_samples, p2_gevs = extract_cs_kernel_for_reference(
            dataset,
            bT=0,
            reference_pz=2,
            comparison_pz_list=[3, 4],
            kernel_label="LO",
            mu=2.0,
            scheme="CG",
            ns=64,
            lattice_spacing_fm=0.076,
            x_window=(0.2, 0.8),
        )
        self.assertTrue(np.allclose(x_values, [0.3, 0.5]))
        self.assertEqual(gamma_samples.shape, (2, 3))
        self.assertTrue(np.allclose(gamma_samples[0], gamma_by_sample, atol=1e-10))
        self.assertEqual(len(p2_gevs), 2)
        self.assertEqual(chi2_samples.shape, (2, 3))

    def test_parse_tmdwf_cs_kernel_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "input_cs.txt"
            input_path.write_text(
                "\n".join(
                    [
                        "title_pattern demo_cs",
                        f"input_root {tmp}",
                        "ns 64",
                        "lattice_spacing_fm 0.076",
                        "gmlist T5",
                        "etalist eta0",
                        "component real",
                        "nstates 1",
                        "normalization_mode raw",
                        "mu 2.0",
                        "scheme CG",
                        "extraction_type type2",
                        "kernel_labels LO NLL",
                        "bTrange 0 2",
                        "pzlist 2 3 4",
                        "x_window 0.2 0.8",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            parsed = parse_tmdwf_cs_kernel_input(input_path)
        self.assertEqual(parsed.kernel_labels, ("LO", "NLL"))
        self.assertEqual(parsed.bTlist, (0, 1, 2))
        self.assertEqual(parsed.pzlist, (2, 3, 4))
        self.assertEqual(parsed.scheme, "CG")
        self.assertEqual(parsed.normalization_mode, "raw")

    def test_run_tmdwf_cs_kernel_workflow_writes_batch_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_root = tmp / "inputs"
            x_grid = np.array([0.1, 0.3, 0.5, 0.9], dtype=float)
            gamma_by_sample = np.array([0.2, 0.25, 0.3], dtype=float)
            for bT in (0, 2):
                for pz in (2, 3, 4):
                    title = f"demo_pz{pz}"
                    tables_dir = input_root / title / "tables"
                    samples_dir = input_root / title / "samples"
                    tables_dir.mkdir(parents=True, exist_ok=True)
                    samples_dir.mkdir(parents=True, exist_ok=True)
                    samples = []
                    for x_value in x_grid:
                        base = np.array([1.0 + x_value + 0.1 * bT, 1.5 + x_value + 0.1 * bT, 2.0 + x_value + 0.1 * bT], dtype=float)
                        row = base * (float(pz) ** gamma_by_sample)
                        samples.append(row)
                    q_samples = np.asarray(samples, dtype=float).T
                    q_mean = np.mean(q_samples, axis=0)
                    q_p16 = np.percentile(q_samples, 16.0, axis=0)
                    q_p84 = np.percentile(q_samples, 84.0, axis=0)
                    q_err = 0.5 * (q_p84 - q_p16)
                    table_lines = [
                        "pz 0",
                        f"bT {bT}",
                        "component real",
                        "nstates 1",
                        "normalization_mode raw",
                        "lattice_spacing_fm 7.6000000000e-02",
                        "zstep_fm 1.0000000000e-02",
                        "interpolation_kind linear",
                        "x\tq_mean\tq_err\tq_p16\tq_p84",
                    ]
                    for x_value, mean_value, err_value, p16_value, p84_value in zip(x_grid, q_mean, q_err, q_p16, q_p84, strict=True):
                        table_lines.append(f"{x_value:.10e}\t{mean_value:.10e}\t{err_value:.10e}\t{p16_value:.10e}\t{p84_value:.10e}")
                    (tables_dir / f"{title}_T5_eta0_bT{bT}_real_1state_fourier.txt").write_text("\n".join(table_lines) + "\n", encoding="utf-8")
                    sample_lines = ["sample_id\tx\tq_sample"]
                    for sample_id, sample_values in enumerate(q_samples):
                        for x_value, q_value in zip(x_grid, sample_values, strict=True):
                            sample_lines.append(f"{sample_id}\t{x_value:.10e}\t{q_value:.10e}")
                    (samples_dir / f"{title}_T5_eta0_bT{bT}_real_1state_fourier_samples.txt").write_text("\n".join(sample_lines) + "\n", encoding="utf-8")

            input_path = tmp / "input_cs.txt"
            input_path.write_text(
                "\n".join(
                    [
                        "title_pattern demo_pz*",
                        f"input_root {input_root}",
                        "ns 64",
                        "lattice_spacing_fm 0.076",
                        "gmlist T5",
                        "etalist eta0",
                        "component real",
                        "nstates 1",
                        "normalization_mode raw",
                        "mu 2.0",
                        "scheme CG",
                        "extraction_type type2",
                        "kernel_labels LO",
                        "bTlist 0 2",
                        "pzlist 2 3 4",
                        "x_window 0.2 0.8",
                        "plot false",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            outputs = run_tmdwf_cs_kernel_workflow(input_path, results_dir=tmp / "results")

            self.assertEqual(len(outputs), 16)
            output_title = "demo_pzmultiPz"
            band_path = tmp / "results" / output_title / "tables" / f"{output_title}_T5_eta0_raw_real_1state_CG_LO_bT0_refpz2_type2_band.txt"
            diagnostics_path = tmp / "results" / output_title / "diagnostics" / f"{output_title}_T5_eta0_raw_real_1state_CG_LO_bT0_refpz2_type2_diagnostics.txt"
            sample_path = tmp / "results" / output_title / "samples" / f"{output_title}_T5_eta0_raw_real_1state_CG_LO_bT0_refpz2_type2_samples.txt"
            self.assertTrue(band_path.exists())
            self.assertTrue(diagnostics_path.exists())
            self.assertTrue(sample_path.exists())

            band_lines = [line.strip() for line in band_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            data_lines = [line for line in band_lines if "\t" in line][1:]
            self.assertEqual(len(data_lines), 2)
            first_band = data_lines[0].split("\t")
            self.assertAlmostEqual(float(first_band[1]), 0.2, places=8)
            self.assertAlmostEqual(float(first_band[2]), 0.25, places=8)
            self.assertAlmostEqual(float(first_band[3]), 0.3, places=8)

            sample_lines = [line.strip() for line in sample_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            sample_rows = [line for line in sample_lines if "\t" in line][1:]
            self.assertEqual(len(sample_rows), 6)
            first_sample = sample_rows[0].split("\t")
            self.assertEqual(first_sample[1], "0")
            self.assertAlmostEqual(float(first_sample[3]), 0.2, places=8)

    def test_load_cs_kernel_dataset_rejects_inconsistent_sample_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            for pz, sample_count in ((2, 2), (3, 3)):
                title = f"demo_pz{pz}"
                tables_dir = tmp / title / "tables"
                samples_dir = tmp / title / "samples"
                tables_dir.mkdir(parents=True)
                samples_dir.mkdir(parents=True)
                (tables_dir / f"{title}_T5_eta0_bT0_real_1state_fourier.txt").write_text(
                    "x\tq_mean\tq_err\tq_p16\tq_p84\n0.1\t1.0\t0.1\t0.9\t1.1\n0.3\t1.1\t0.1\t1.0\t1.2\n",
                    encoding="utf-8",
                )
                rows = ["sample_id\tx\tq_sample"]
                for sample_id in range(sample_count):
                    rows.append(f"{sample_id}\t1.0000000000e-01\t{1.0 + 0.1*sample_id:.10e}")
                    rows.append(f"{sample_id}\t3.0000000000e-01\t{1.1 + 0.1*sample_id:.10e}")
                (samples_dir / f"{title}_T5_eta0_bT0_real_1state_fourier_samples.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_cs_kernel_dataset(
                    input_root=tmp,
                    title_pattern="demo_pz*",
                    gm="T5",
                    eta="eta0",
                    component="real",
                    nstates=1,
                    normalization_mode="raw",
                    bTlist=(0,),
                    pzlist=(2, 3),
                )
        self.assertIn("inconsistent bootstrap sample count", str(ctx.exception))

    def test_build_cs_dgamma_exposes_legacy_running_coupling(self) -> None:
        correction = build_cs_dgamma(2.0, "NLO")
        self.assertAlmostEqual(correction.mu, 2.0)
        self.assertGreater(correction.alphas(2.0), 0.0)


if __name__ == "__main__":
    unittest.main()
