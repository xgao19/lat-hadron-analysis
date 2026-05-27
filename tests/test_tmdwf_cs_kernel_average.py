import tempfile
import unittest
from pathlib import Path

import numpy as np

from lqcd_analysis.tmdwf.cs_kernel_average import (
    discover_tmdwf_cs_kernel_sources,
    parse_tmdwf_cs_kernel_average_input,
    run_tmdwf_cs_kernel_average_workflow,
    summarize_tmdwf_cs_kernel_average,
)


class TMDWFCSKernelAverageTests(unittest.TestCase):
    @staticmethod
    def _write_source_outputs(
        root: Path,
        *,
        source_title: str,
        output_title: str,
        reference_bT: int,
        reference_pz: int,
        reference_pz_label: str,
        dP_gev: float,
        x_grid: list[float],
        sample_values: dict[int, list[float]],
        band_shift: float,
    ) -> None:
        tables_dir = root / source_title / "tables"
        samples_dir = root / source_title / "samples"
        tables_dir.mkdir(parents=True, exist_ok=True)
        samples_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{source_title}_T5_eta0_bT{reference_bT}_real_1state_CG_LO_band"
        band_lines = [
            "title_pattern demo_src*",
            f"output_title {output_title}",
            "gm T5",
            "eta eta0",
            "component real",
            "nstates 1",
            "normalization_mode raw",
            "scheme CG",
            "extraction_type type2",
            "kernel_label LO",
            f"reference_bT {reference_bT}",
            f"reference_pz {reference_pz}",
            f"reference_pz_label {reference_pz_label}",
            f"dP_GeV {dP_gev:.10e}",
            "x\tgamma_p16\tgamma_p50\tgamma_p84",
        ]
        for x_value in x_grid:
            center = x_value + band_shift
            band_lines.append(
                f"{x_value:.10e}\t{(center - 0.1):.10e}\t{center:.10e}\t{(center + 0.1):.10e}"
            )
        (tables_dir / f"{stem}_band.txt").write_text("\n".join(band_lines) + "\n", encoding="utf-8")

        sample_lines = [
            "title_pattern demo_src*",
            f"output_title {output_title}",
            "gm T5",
            "eta eta0",
            "component real",
            "nstates 1",
            "normalization_mode raw",
            "scheme CG",
            "extraction_type type2",
            "kernel_label LO",
            f"reference_bT {reference_bT}",
            f"reference_pz {reference_pz}",
            f"reference_pz_label {reference_pz_label}",
            f"dP_GeV {dP_gev:.10e}",
            "x sample_id success gamma_zeta chi2_dof",
        ]
        for sample_id, values in sample_values.items():
            for x_value, value in zip(x_grid, values, strict=True):
                sample_lines.append(f"{x_value:.10e}\t{sample_id}\t1\t{value:.10e}\t0.0")
        (samples_dir / f"{stem}_samples.txt").write_text("\n".join(sample_lines) + "\n", encoding="utf-8")

    def test_average_workflow_writes_bootstrap_samples_and_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_root = tmp / "cs_kernel"
            output_root = tmp / "averaged"
            x_grid = [0.25, 0.45, 0.55, 0.75]

            self._write_source_outputs(
                input_root,
                source_title="demo_src5",
                output_title="demo_pzmultiPz",
                reference_bT=2,
                reference_pz=5,
                reference_pz_label="5-6",
                dP_gev=0.25,
                x_grid=x_grid,
                sample_values={0: [1.0, 1.0, 3.0, 3.0], 1: [2.0, 2.0, 4.0, 4.0]},
                band_shift=0.0,
            )
            self._write_source_outputs(
                input_root,
                source_title="demo_src6",
                output_title="demo_pzmultiPz",
                reference_bT=2,
                reference_pz=6,
                reference_pz_label="6-7",
                dP_gev=0.25,
                x_grid=x_grid,
                sample_values={0: [5.0, 5.0, 7.0, 7.0], 1: [6.0, 6.0, 8.0, 8.0]},
                band_shift=1.0,
            )

            input_file = tmp / "input_avg.txt"
            input_file.write_text(
                "\n".join(
                    [
                        "title_pattern demo_src*",
                        f"input_root {input_root}",
                        "lattice_spacing_fm 0.076",
                        "gm T5",
                        "eta eta0",
                        "component real",
                        "nstates 1",
                        "normalization_mode raw",
                        "scheme CG",
                        "extraction_type type2",
                        "kernel_label LO",
                        "bTrange 2 2",
                        "reference_pz_labels 5-6 6-7",
                        f"results_dir {output_root}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            parsed = parse_tmdwf_cs_kernel_average_input(input_file)
            sources = discover_tmdwf_cs_kernel_sources(parsed)
            rows, sample_rows, selections = summarize_tmdwf_cs_kernel_average(parsed, sources)
            outputs = run_tmdwf_cs_kernel_average_workflow(input_file, results_dir=output_root)

            self.assertEqual(len(sources), 2)
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(sample_rows), 2)
            self.assertEqual(len(selections), 2)
            self.assertEqual(len(outputs), 5)
            self.assertTrue(outputs[-1].suffix == ".pdf")

            row = rows[0]
            self.assertEqual(row.bT, 2)
            self.assertEqual(row.n_selected_sources, 2)
            self.assertEqual(row.n_selected_points, 4)
            self.assertEqual(row.n_samples, 2)
            self.assertTrue(np.isclose(row.value, 4.5))

            sample_means = np.array([4.0, 5.0], dtype=float)
            sample_stds = np.array([np.std([1.0, 3.0, 5.0, 7.0], ddof=1), np.std([2.0, 4.0, 6.0, 8.0], ddof=1)], dtype=float)
            q16, q50, q84 = np.percentile(sample_means, [16.0, 50.0, 84.0])
            self.assertTrue(np.isclose(row.stat_err, 0.5 * (q84 - q16)))
            self.assertTrue(np.isclose(row.sys_err, np.mean(sample_stds)))
            self.assertTrue(np.isclose(row.total_err, np.sqrt(row.stat_err**2 + row.sys_err**2)))

            sample_row_means = [sample_row.mean for sample_row in sample_rows]
            self.assertTrue(np.allclose(sample_row_means, sample_means))
            sample_row_stds = [sample_row.std for sample_row in sample_rows]
            self.assertTrue(np.allclose(sample_row_stds, sample_stds))

            title_root = output_root / "demo_pzmultiPz"
            summary = (title_root / "demo_pzmultiPz_T5_eta0_raw_real_1state_CG_LO_type2_refpz5-6_6-7_xavg_summary.txt").read_text(
                encoding="utf-8"
            )
            self.assertIn("reference_pz_labels 5-6 6-7", summary)
            values_text = (title_root / "tables" / "demo_pzmultiPz_T5_eta0_raw_real_1state_CG_LO_type2_refpz5-6_6-7_xavg_values.txt").read_text(
                encoding="utf-8"
            )
            self.assertIn("bT_fm", values_text)
            self.assertIn("4.5000000000e+00", values_text)
            sample_text = (title_root / "samples" / "demo_pzmultiPz_T5_eta0_raw_real_1state_CG_LO_type2_refpz5-6_6-7_xavg_samples.txt").read_text(
                encoding="utf-8"
            )
            self.assertIn("\t0\t1\t4.0000000000e+00\t", sample_text)
            self.assertIn("\t1\t1\t5.0000000000e+00\t", sample_text)

    def test_average_workflow_uses_explicit_x_range_without_physical_cuts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_root = tmp / "cs_kernel"
            output_root = tmp / "averaged"
            x_grid = [0.05, 0.25, 0.75]

            self._write_source_outputs(
                input_root,
                source_title="demo_src5",
                output_title="demo_pzmultiPz",
                reference_bT=2,
                reference_pz=5,
                reference_pz_label="5-6",
                dP_gev=0.25,
                x_grid=x_grid,
                sample_values={0: [1.0, 2.0, 3.0], 1: [2.0, 3.0, 4.0]},
                band_shift=0.0,
            )

            input_file = tmp / "input_avg_xrange.txt"
            input_file.write_text(
                "\n".join(
                    [
                        "title_pattern demo_src*",
                        f"input_root {input_root}",
                        "lattice_spacing_fm 0.076",
                        "gm T5",
                        "eta eta0",
                        "component real",
                        "nstates 1",
                        "normalization_mode raw",
                        "scheme CG",
                        "extraction_type type2",
                        "kernel_label LO",
                        "bTrange 2 2",
                        "x_range 0.0 0.3",
                        "reference_pz_labels 5-6",
                        f"results_dir {output_root}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            parsed = parse_tmdwf_cs_kernel_average_input(input_file)
            sources = discover_tmdwf_cs_kernel_sources(parsed)
            rows, sample_rows, selections = summarize_tmdwf_cs_kernel_average(parsed, sources)
            outputs = run_tmdwf_cs_kernel_average_workflow(input_file, results_dir=output_root)

            self.assertEqual(len(sources), 1)
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(sample_rows), 2)
            self.assertEqual(len(selections), 1)
            self.assertEqual(len(outputs), 5)
            self.assertTrue(outputs[-1].suffix == ".pdf")

            row = rows[0]
            self.assertEqual(row.n_selected_points, 2)
            self.assertTrue(np.isclose(row.value, 2.0))
            q16, _, q84 = np.percentile([1.5, 2.5], [16.0, 50.0, 84.0])
            self.assertTrue(np.isclose(row.stat_err, 0.5 * (q84 - q16)))

            selection = selections[0]
            self.assertTrue(np.isclose(selection.x_min, 0.05))
            self.assertTrue(np.isclose(selection.x_max, 0.25))

            title_root = output_root / "demo_pzmultiPz"
            summary = (title_root / "demo_pzmultiPz_T5_eta0_raw_real_1state_CG_LO_type2_refpz5-6_xrange0p000_0p300_summary.txt").read_text(
                encoding="utf-8"
            )
            self.assertIn("x_range 0.0000000000e+00 3.0000000000e-01", summary)

    def test_average_central_value_uses_bootstrap_median(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_root = tmp / "cs_kernel"
            x_grid = [0.4, 0.6]

            self._write_source_outputs(
                input_root,
                source_title="demo_src5",
                output_title="demo_pzmultiPz",
                reference_bT=0,
                reference_pz=5,
                reference_pz_label="5-6",
                dP_gev=0.25,
                x_grid=x_grid,
                sample_values={
                    0: [-0.2, -0.2],
                    1: [-0.1, -0.1],
                    2: [-0.3, -0.3],
                    3: [-20.0, -20.0],
                    4: [-0.15, -0.15],
                },
                band_shift=0.0,
            )

            input_file = tmp / "input_avg_outlier.txt"
            input_file.write_text(
                "\n".join(
                    [
                        "title_pattern demo_src*",
                        f"input_root {input_root}",
                        "lattice_spacing_fm 0.076",
                        "gm T5",
                        "eta eta0",
                        "component real",
                        "nstates 1",
                        "normalization_mode raw",
                        "scheme CG",
                        "extraction_type type2",
                        "kernel_label LO",
                        "bTrange 0 0",
                        "x_range 0.3 0.7",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            parsed = parse_tmdwf_cs_kernel_average_input(input_file)
            sources = discover_tmdwf_cs_kernel_sources(parsed)
            rows, _, _ = summarize_tmdwf_cs_kernel_average(parsed, sources)

            sample_means = np.array([-0.2, -0.1, -0.3, -20.0, -0.15], dtype=float)
            q16, q50, q84 = np.percentile(sample_means, [16.0, 50.0, 84.0])
            self.assertTrue(np.isclose(rows[0].value, q50))
            self.assertTrue(np.isclose(rows[0].stat_err, 0.5 * (q84 - q16)))
            self.assertFalse(np.isclose(rows[0].value, np.mean(sample_means)))


if __name__ == "__main__":
    unittest.main()
