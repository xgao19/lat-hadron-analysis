import tempfile
import unittest
from pathlib import Path

import numpy as np

from lqcd_analysis.tmdwf.cs_kernel_extract import momentum_unit_gev
from lqcd_analysis.tmdwf.cs_kernel_joint import (
    JointCSObservation,
    _evaluate_correction_shape,
    fit_gamma_eff_at_x,
    parse_tmdwf_cs_kernel_joint_input,
    run_tmdwf_cs_kernel_joint_workflow,
)


def _write_fourier_samples(
    root: Path,
    *,
    title: str,
    bT: int,
    pz: int,
    x_grid: np.ndarray,
    q_samples: np.ndarray,
) -> None:
    tables_dir = root / title / "tables"
    samples_dir = root / title / "samples"
    tables_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    q_mean = np.mean(q_samples, axis=0)
    q_p16 = np.percentile(q_samples, 16.0, axis=0)
    q_p84 = np.percentile(q_samples, 84.0, axis=0)
    q_err = 0.5 * (q_p84 - q_p16)
    table_lines = ["x\tq_mean\tq_err\tq_p16\tq_p84"]
    for x_value, mean_value, err_value, p16_value, p84_value in zip(
        x_grid,
        q_mean,
        q_err,
        q_p16,
        q_p84,
        strict=True,
    ):
        table_lines.append(
            f"{x_value:.10e}\t{mean_value:.10e}\t{err_value:.10e}\t"
            f"{p16_value:.10e}\t{p84_value:.10e}"
        )
    (tables_dir / f"{title}_T5_eta0_bT{bT}_mode3_real_2state_fourier.txt").write_text(
        "\n".join(table_lines) + "\n",
        encoding="utf-8",
    )
    sample_lines = ["sample_id\tx\tq_sample"]
    for sample_id, sample_values in enumerate(q_samples):
        for x_value, q_value in zip(x_grid, sample_values, strict=True):
            sample_lines.append(f"{sample_id}\t{x_value:.10e}\t{q_value:.10e}")
    (samples_dir / f"{title}_T5_eta0_bT{bT}_mode3_real_2state_fourier_samples.txt").write_text(
        "\n".join(sample_lines) + "\n",
        encoding="utf-8",
    )


class TMDWFCSKernelJointTests(unittest.TestCase):
    def test_parse_joint_input_reads_repeated_ensemble_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            root = tmp / "root"
            root.mkdir()
            input_path = tmp / "joint.txt"
            input_path.write_text(
                "\n".join(
                    [
                        "gm T5",
                        "eta eta0",
                        "component real",
                        "nstates 2",
                        "normalization_mode mode3",
                        "mu 2.0",
                        "scheme CG",
                        "kernel_label LO",
                        "reference_p1_gev 1.0",
                        "spline_kind cubic",
                        "plot false",
                        "progress false",
                        f"ensemble ens {root} demo_pz* 64 0.076 pz=2,3 bT=0:1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            parsed = parse_tmdwf_cs_kernel_joint_input(input_path)
        self.assertEqual(parsed.ensembles[0].label, "ens")
        self.assertEqual(parsed.ensembles[0].pzlist, (2, 3))
        self.assertEqual(parsed.ensembles[0].bTlist, (0, 1))
        self.assertEqual(parsed.kernel_label, "LO")
        self.assertEqual(parsed.spline_kind, "cubic")
        self.assertFalse(parsed.make_plots)
        self.assertFalse(parsed.show_progress)

    def test_analytic_corrections_use_two_parameters_per_enabled_channel(self) -> None:
        observations = []
        for sample_id in range(2):
            for group_id, bT_fm in enumerate((0.1, 0.2)):
                for pz_gev in (1.2, 1.8, 2.4):
                    observations.append(
                        JointCSObservation(
                            group_id=group_id,
                            sample_id=sample_id,
                            x=0.4,
                            bT_fm=bT_fm,
                            pz_gev=pz_gev,
                            value=1.0 + 0.05 * sample_id + 0.02 * group_id + 0.01 * pz_gev,
                            sigma=0.05,
                            ensemble_label="ens",
                            a_fm=0.06,
                            fv_prefactor=0.01,
                            fv_exp_m_pi_bT=np.exp(0.2 * bT_fm),
                            spatial_extent_fm=3.0,
                        )
                    )

        result = fit_gamma_eff_at_x(
            observations,
            sample_count=2,
            x_value=0.4,
            bT_knots_fm=np.asarray([0.1, 0.2]),
            spline_kind="linear",
            reference_p1_gev=1.0,
            scheme="CG",
            kernel_label="LO",
            mu=2.0,
            component="real",
            show_progress=False,
            progress_every=None,
            fit_a2_correction=True,
            fit_fv_correction=True,
            fit_pz2_correction=True,
            fit_apz2_correction=True,
        )

        self.assertEqual(result.n_gamma_knots, 2)
        self.assertEqual(result.n_correction_params, 7)
        self.assertEqual(result.coeff_samples.shape, (2, 9))
        self.assertEqual(result.alpha_samples.shape, (2, 2))
        self.assertEqual(result.beta_samples.shape, (2, 2))
        self.assertEqual(result.kappa_samples.shape, (2, 2))
        self.assertEqual(result.lambda_samples.shape, (2, 1))

    def test_inverse_bT_corrections_reject_zero_bT(self) -> None:
        observations = [
            JointCSObservation(
                group_id=0,
                sample_id=0,
                x=0.4,
                bT_fm=0.0,
                pz_gev=1.2,
                value=1.0,
                sigma=0.05,
                ensemble_label="ens",
                a_fm=0.06,
                fv_prefactor=0.01,
                fv_exp_m_pi_bT=1.0,
                spatial_extent_fm=3.0,
            )
        ]

        with self.assertRaisesRegex(ValueError, "strictly positive bT"):
            fit_gamma_eff_at_x(
                observations,
                sample_count=1,
                x_value=0.4,
                bT_knots_fm=np.asarray([0.0, 0.1]),
                spline_kind="linear",
                reference_p1_gev=1.0,
                scheme="CG",
                kernel_label="LO",
                mu=2.0,
                component="real",
                show_progress=False,
                progress_every=None,
                fit_a2_correction=True,
            )

    def test_fv_shape_uses_exp_mpi_bT_and_increases_with_bT(self) -> None:
        coeffs = np.asarray([0.2, 0.4], dtype=float)
        values = _evaluate_correction_shape(
            "fv",
            coeffs,
            np.asarray([0.1, 0.3], dtype=float),
            fv_exp_m_pi_bT=np.asarray([np.exp(0.1), np.exp(0.3)], dtype=float),
        )
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertGreater(values[1], values[0])

    def test_fv_correction_rejects_bT_at_box_size(self) -> None:
        observations = []
        for pz_gev in (1.2, 1.8):
            observations.append(
                JointCSObservation(
                    group_id=0,
                    sample_id=0,
                    x=0.4,
                    bT_fm=3.0,
                    pz_gev=pz_gev,
                    value=1.0,
                    sigma=0.05,
                    ensemble_label="ens",
                    a_fm=0.06,
                    fv_prefactor=0.01,
                    fv_exp_m_pi_bT=np.exp(0.3),
                    spatial_extent_fm=3.0,
                )
            )

        with self.assertRaisesRegex(ValueError, "below the spatial box size"):
            fit_gamma_eff_at_x(
                observations,
                sample_count=1,
                x_value=0.4,
                bT_knots_fm=np.asarray([3.0]),
                spline_kind="linear",
                reference_p1_gev=1.0,
                scheme="CG",
                kernel_label="LO",
                mu=2.0,
                component="real",
                show_progress=False,
                progress_every=None,
                fit_fv_correction=True,
            )

    def test_joint_workflow_recovers_linear_gamma_eff_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            x_grid = np.array([0.25, 0.55], dtype=float)
            sample_count = 4
            ensembles = [
                ("coarse", tmp / "coarse", 48, 0.060, (3, 4, 5), (1, 2)),
                ("fine", tmp / "fine", 64, 0.050, (4, 5, 6), (1, 3)),
            ]
            reference_p1 = 1.0
            for label, root, ns, lattice_spacing_fm, pzlist, bTlist in ensembles:
                d_p = momentum_unit_gev(ns, lattice_spacing_fm)
                for bT in bTlist:
                    bT_fm = bT * lattice_spacing_fm
                    for pz in pzlist:
                        pz_gev = pz * d_p
                        title = f"{label}_pz{pz}"
                        rows = []
                        for sample_id in range(sample_count):
                            amplitude = np.array(
                                [
                                    1.0 + 0.2 * sample_id + 0.3 * x_value + 0.1 * bT
                                    for x_value in x_grid
                                ],
                                dtype=float,
                            )
                            gamma = 0.15 + 0.2 * x_grid + 0.7 * bT_fm + 0.01 * sample_id
                            rows.append(amplitude * np.exp(np.log(pz_gev / reference_p1) * gamma))
                        _write_fourier_samples(
                            root,
                            title=title,
                            bT=bT,
                            pz=pz,
                            x_grid=x_grid,
                            q_samples=np.asarray(rows, dtype=float),
                        )

            input_path = tmp / "joint_input.txt"
            input_path.write_text(
                "\n".join(
                    [
                        "gm T5",
                        "eta eta0",
                        "component real",
                        "nstates 2",
                        "normalization_mode mode3",
                        "mu 2.0",
                        "scheme CG",
                        "kernel_label LO",
                        f"reference_p1_gev {reference_p1}",
                        "x_window 0.2 0.8",
                        "x_knots 0.25 0.55",
                        "bT_knots_fm 0.05 0.06 0.12 0.15",
                        "plot false",
                        "progress false",
                        (
                            f"ensemble coarse {tmp / 'coarse'} coarse_pz* 48 0.060 "
                            "pz=3,4,5 bT=1,2"
                        ),
                        (
                            f"ensemble fine {tmp / 'fine'} fine_pz* 64 0.050 "
                            "pz=4,5,6 bT=1,3"
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            outputs = run_tmdwf_cs_kernel_joint_workflow(input_path, results_dir=tmp / "results")

            surface_path = (
                tmp
                / "results"
                / "joint_gamma_eff"
                / "tables"
                / "joint_T5_eta0_mode3_real_2state_CG_LO_gamma_eff_surface.txt"
            )
            self.assertIn(surface_path, outputs)
            lines = [
                line.strip()
                for line in surface_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            rows = [line.split("\t") for line in lines[1:]]
            by_point = {(float(row[0]), float(row[1])): float(row[3]) for row in rows}
            expected = 0.15 + 0.2 * 0.25 + 0.7 * 0.12 + 0.015
            self.assertAlmostEqual(by_point[(0.25, 0.12)], expected, places=5)

            coeff_path = (
                tmp
                / "results"
                / "joint_gamma_eff"
                / "samples"
                / "joint_T5_eta0_mode3_real_2state_CG_LO_gamma_eff_coefficients.txt"
            )
            self.assertIn(coeff_path, outputs)
            coeff_lines = [
                line.strip()
                for line in coeff_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            coeff_header = coeff_lines[0].split("\t")
            self.assertEqual(coeff_header[0], "x")
            self.assertEqual(coeff_header[1], "sample_id")
            self.assertTrue(all(col.startswith("c") for col in coeff_header[2:]))
            # 2 x-points × 4 samples = 8 data rows + header
            self.assertEqual(len(coeff_lines), 9)

            coeff_data = {
                (float(parts[0]), int(parts[1])): [float(c) for c in parts[2:]]
                for line in coeff_lines[1:]
                for parts in [line.split("\t")]
            }

            # Verify coefficients reconstruct surface values at knot points.
            # At a knot point the coefficient equals gamma_eff for that sample.
            samples_path = (
                tmp
                / "results"
                / "joint_gamma_eff"
                / "samples"
                / "joint_T5_eta0_mode3_real_2state_CG_LO_gamma_eff_samples.txt"
            )
            self.assertIn(samples_path, outputs)
            sample_lines = [
                line.strip()
                for line in samples_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            gamma_sample_0 = None
            for line in sample_lines[1:]:
                parts = line.split("\t")
                if (
                    abs(float(parts[0]) - 0.25) < 1e-6
                    and abs(float(parts[1]) - 0.12) < 1e-6
                    and int(parts[2]) == 0
                ):
                    gamma_sample_0 = float(parts[3])
                    break
            self.assertIsNotNone(gamma_sample_0)
            self.assertAlmostEqual(
                coeff_data[(0.25, 0)][2],
                gamma_sample_0,
                places=10,
            )


if __name__ == "__main__":
    unittest.main()
