import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from lqcd_analysis.two_point.fit_nstate import (
    SHRINKAGE_LAMBDAS,
    build_residual_model,
    build_fallback_fit_attempts,
    compute_fit_window_parameter_summary,
    build_energy_priors_from_fit_window_summary,
    compute_bootstrap_covariance,
    evaluate_model,
    extract_shrinkage_lambda_from_message,
    find_first_usable_correlated_residual_model,
    FitResult,
    fit_residuals,
    fit_nstate_sample,
    FitSummaryRow,
    EnergyPrior,
    pack_fit_parameters,
    parse_nstate_fit_input,
    FitWindowParameterSummary,
    FitWindowSummary,
    run_single_dataset,
    run_sliding_fits,
    run_nstate_fit,
    shrink_covariance_to_diagonal,
    target_ground_energy_from_pz0,
)
from lqcd_analysis.two_point.effective_mass import (
    compute_effective_mass_antisymmetric_root,
    compute_effective_mass_cosh_root,
    effective_mass_single,
    solve_antisymmetric_effective_mass,
    solve_cosh_effective_mass,
)
from lqcd_analysis.two_point.plotting import (
    build_reconstruction_band,
    plot_nstate_outputs,
    select_scan_state_indices,
    write_nstate_plot_notebook,
)


class NStateFitTests(unittest.TestCase):
    def test_normal_effective_mass(self) -> None:
        correlator = np.exp(-0.4 * np.arange(8))
        meff = effective_mass_single(correlator, "normal")
        self.assertTrue(np.allclose(meff, 0.4))

    def test_symmetric_model(self) -> None:
        times = np.arange(5)
        values = evaluate_model(
            times,
            amplitudes=np.array([2.0]),
            energies=np.array([0.5]),
            nt=16,
            model="symmetric",
        )
        self.assertTrue(np.all(values > 0.0))

    def test_cosh_root_effective_mass_recovers_known_mass(self) -> None:
        nt = 32
        mass = 0.42
        times = np.arange(nt)
        correlator = np.exp(-mass * times) + np.exp(-mass * (nt - times))
        meff = compute_effective_mass_cosh_root(correlator, nt)
        self.assertTrue(np.allclose(meff[2 : nt // 2 - 2], mass, atol=1e-6, equal_nan=False))

    def test_cosh_root_invalid_ratio_returns_nan(self) -> None:
        self.assertTrue(np.isnan(solve_cosh_effective_mass(t=3, ratio=-1.0, nt=32)))
        self.assertTrue(np.isnan(solve_cosh_effective_mass(t=3, ratio=np.nan, nt=32)))

    def test_cosh_root_effective_mass_boundary_alignment(self) -> None:
        nt = 16
        correlator = np.exp(-0.3 * np.arange(nt)) + np.exp(-0.3 * (nt - np.arange(nt)))
        meff = compute_effective_mass_cosh_root(correlator, nt)
        self.assertEqual(len(meff), len(correlator))
        self.assertTrue(np.isnan(meff[-1]))
        self.assertTrue(np.isfinite(meff[0]))

    def test_symmetric_effective_mass_interface_uses_root_solver(self) -> None:
        nt = 24
        mass = 0.35
        times = np.arange(nt)
        correlator = 2.0 * (np.exp(-mass * times) + np.exp(-mass * (nt - times)))
        meff = effective_mass_single(correlator, "symmetric", nt=nt)
        self.assertTrue(np.allclose(meff[2 : nt // 2 - 2], mass, atol=1e-6, equal_nan=False))

    def test_antisymmetric_root_effective_mass_recovers_known_mass(self) -> None:
        nt = 32
        mass = 0.37
        times = np.arange(nt)
        correlator = np.exp(-mass * times) - np.exp(-mass * (nt - times))
        meff = compute_effective_mass_antisymmetric_root(correlator, nt)
        self.assertTrue(np.allclose(meff[2 : nt // 2 - 3], mass, atol=1e-6, equal_nan=False))

    def test_antisymmetric_root_invalid_ratio_returns_nan(self) -> None:
        self.assertTrue(np.isnan(solve_antisymmetric_effective_mass(t=3, ratio=-1.0, nt=32)))
        self.assertTrue(np.isnan(solve_antisymmetric_effective_mass(t=3, ratio=np.nan, nt=32)))

    def test_antisymmetric_effective_mass_dispatch_uses_root_solver(self) -> None:
        correlator = np.linspace(1.0, 0.2, 8)
        with patch(
            "lqcd_analysis.two_point.effective_mass.compute_effective_mass_antisymmetric_root",
            return_value=np.full(len(correlator), 0.123),
        ) as mock_solver:
            meff = effective_mass_single(correlator, "antisymmetric", nt=16)
        mock_solver.assert_called_once()
        self.assertTrue(np.allclose(meff, 0.123))

    def test_prior_residuals_are_appended_and_scaled_by_sqrt_lambda(self) -> None:
        times = np.array([2.0, 3.0, 4.0])
        data = np.array([1.0, 0.7, 0.5])
        sigma = np.array([0.1, 0.1, 0.1])
        theta = np.array([np.log(2.0), np.log(0.4), np.log(0.8)])
        priors = (EnergyPrior(energy_index=0, center=0.5, sigma=0.2),)
        residual = fit_residuals(
            theta,
            times,
            data,
            sigma,
            nt=32,
            model="normal",
            nstates=1,
            priors=priors,
            lambda_prior=4.0,
        )
        self.assertEqual(len(residual), len(times) + 1)
        amplitude, energy = np.exp(theta[0]), np.exp(theta[1])
        expected_prior = np.sqrt(4.0) * (energy - 0.5) / 0.2
        self.assertAlmostEqual(residual[-1], expected_prior)

    def test_bad_prior_width_is_skipped_safely(self) -> None:
        times = np.array([2.0, 3.0])
        data = np.array([1.0, 0.7])
        sigma = np.array([0.1, 0.1])
        theta = np.array([np.log(2.0), np.log(0.4), np.log(0.8)])
        priors = (
            EnergyPrior(energy_index=0, center=0.5, sigma=0.0),
            EnergyPrior(energy_index=0, center=0.5, sigma=np.nan),
        )
        residual = fit_residuals(
            theta,
            times,
            data,
            sigma,
            nt=32,
            model="normal",
            nstates=1,
            priors=priors,
            lambda_prior=1.0,
        )
        self.assertEqual(len(residual), len(times))

    def test_backward_compatibility_when_priors_disabled(self) -> None:
        times = np.array([2.0, 3.0, 4.0])
        data = np.array([1.0, 0.7, 0.5])
        sigma = np.array([0.1, 0.1, 0.1])
        theta = np.array([np.log(2.0), np.log(0.4), np.log(0.8)])
        priors = (EnergyPrior(energy_index=0, center=0.5, sigma=0.2),)
        residual_with_disabled_prior = fit_residuals(
            theta,
            times,
            data,
            sigma,
            nt=32,
            model="normal",
            nstates=1,
            priors=priors,
            lambda_prior=0.0,
        )
        residual_without_prior = fit_residuals(
            theta,
            times,
            data,
            sigma,
            nt=32,
            model="normal",
            nstates=1,
            priors=(),
            lambda_prior=1.0,
        )
        self.assertTrue(np.allclose(residual_with_disabled_prior, residual_without_prior))

    def test_default_fit_mode_is_uncorrelated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "input.txt"
            path.write_text(
                "\n".join(
                    [
                        "demo_pz* 64 64 0.076",
                        "c2pt /tmp/c2pt_pz*.csv",
                        "pzlist 0",
                        "fold_t none",
                        "tmax 24",
                        "model normal",
                        "nstates 1",
                        "tmin_window /tmp/tmin_windows.txt",
                    ]
                ),
                encoding="utf-8",
            )
            parsed = parse_nstate_fit_input(path)
        self.assertEqual(parsed.fit_mode, "uncorrelated")
        self.assertIsNone(parsed.pz0_ground_energy)
        self.assertFalse(parsed.fix_ground_energy_from_dispersion)

    def test_parse_optional_pz0_ground_energy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "input.txt"
            path.write_text(
                "\n".join(
                    [
                        "demo_pz* 64 64 0.076",
                        "c2pt /tmp/c2pt_pz*.csv",
                        "pzlist 0 1",
                        "fold_t none",
                        "tmax 24",
                        "model normal",
                        "pz0_ground_energy 0.42",
                        "nstates 1",
                        "tmin_window /tmp/tmin_windows.txt",
                    ]
                ),
                encoding="utf-8",
            )
            parsed = parse_nstate_fit_input(path)
        self.assertAlmostEqual(parsed.pz0_ground_energy, 0.42)

    def test_parse_fit_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "input.txt"
            path.write_text(
                "\n".join(
                    [
                        "demo_pz* 64 64 0.076",
                        "c2pt /tmp/c2pt_pz*.csv",
                        "pzlist 0 1",
                        "fold_t none",
                        "model normal",
                        "nstates 1",
                        "tmax 24",
                        "tmin_window /tmp/tmin_windows.txt",
                    ]
                ),
                encoding="utf-8",
            )
            parsed = parse_nstate_fit_input(path)
        self.assertEqual(parsed.tmin_window, "/tmp/tmin_windows.txt")
        self.assertEqual(parsed.tmax, 24)

    def test_parse_fixed_ground_energy_requires_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "input.txt"
            path.write_text(
                "\n".join(
                    [
                        "demo_pz* 64 64 0.076",
                        "c2pt /tmp/c2pt_pz*.csv",
                        "pzlist 0 1",
                        "fold_t none",
                        "model normal",
                        "fix_ground_energy_from_dispersion true",
                        "nstates 1",
                        "tmax 24",
                        "tmin_window /tmp/tmin_windows.txt",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "pz0_ground_energy"):
                parse_nstate_fit_input(path)

    def test_fit_nstate_sample_can_fix_ground_energy(self) -> None:
        nt = 32
        times = np.arange(2, 10)
        amplitudes = np.array([3.0, 0.8])
        energies = np.array([0.42, 0.9])
        data = evaluate_model(times, amplitudes, energies, nt, "normal")
        sigma = np.full_like(data, 1e-3, dtype=float)
        result = fit_nstate_sample(
            times,
            data,
            sigma,
            nt,
            "normal",
            amplitudes,
            np.array([0.5, 1.0]),
            2,
            fixed_ground_energy=energies[0],
        )
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.params[2], energies[0], places=10)

    def test_correlated_residual_uses_shared_covariance_whitening(self) -> None:
        theta = pack_fit_parameters(np.array([2.0]), np.array([0.4]))
        times = np.array([2.0, 3.0])
        data = np.array([0.5, 0.4])
        sigma = np.array([0.1, 0.2])
        covariance = np.array([[4.0, 1.0], [1.0, 3.0]])
        residual_model = build_residual_model("correlated", sigma, covariance)
        residual = fit_residuals(
            theta,
            times,
            data,
            sigma,
            nt=32,
            model="normal",
            nstates=1,
            residual_model=residual_model,
        )
        amplitudes, energies = np.array([2.0]), np.array([0.4])
        delta = evaluate_model(times, amplitudes, energies, 32, "normal") - data
        expected = np.linalg.solve(np.linalg.cholesky(covariance), delta)
        self.assertTrue(np.allclose(residual, expected))

    def test_correlated_residual_and_prior_terms_coexist(self) -> None:
        theta = pack_fit_parameters(np.array([2.0]), np.array([0.4]))
        times = np.array([2.0, 3.0])
        data = np.array([0.5, 0.4])
        sigma = np.array([0.1, 0.2])
        covariance = np.array([[4.0, 1.0], [1.0, 3.0]])
        residual_model = build_residual_model("correlated", sigma, covariance)
        residual = fit_residuals(
            theta,
            times,
            data,
            sigma,
            nt=32,
            model="normal",
            nstates=1,
            priors=(EnergyPrior(energy_index=0, center=0.5, sigma=0.2),),
            lambda_prior=1.0,
            residual_model=residual_model,
        )
        self.assertEqual(len(residual), len(times) + 1)

    def test_shrinkage_scan_finds_first_factorizable_lambda(self) -> None:
        sigma = np.array([0.1, 0.1])
        covariance = np.array([[1.0, 1.5], [1.5, 1.0]])
        residual_model, chosen_lambda = find_first_usable_correlated_residual_model(sigma, covariance)
        self.assertIsNotNone(residual_model)
        self.assertAlmostEqual(chosen_lambda, 0.5)
        self.assertAlmostEqual(residual_model.shrinkage_lambda, 0.5)

    def test_lambda_one_is_diagonal_limit(self) -> None:
        sigma = np.array([0.1, 0.2])
        covariance = np.array([[1.0, 2.0], [2.0, 4.0]])
        diagonal_limit = shrink_covariance_to_diagonal(covariance, 1.0)
        self.assertTrue(np.array_equal(diagonal_limit, np.diag(np.diag(covariance))))
        residual_model = build_residual_model("correlated", sigma, covariance, shrinkage_lambda=1.0)
        expected_cholesky = np.linalg.cholesky(np.diag(np.diag(covariance)))
        self.assertTrue(np.allclose(residual_model.cholesky_factor, expected_cholesky))

    def test_compute_bootstrap_covariance_shape(self) -> None:
        bootstrap_means = np.array(
            [
                [1.0, 2.0, 3.0],
                [1.2, 2.1, 2.9],
                [0.8, 1.9, 3.1],
            ]
        )
        covariance = compute_bootstrap_covariance(bootstrap_means)
        self.assertEqual(covariance.shape, (3, 3))

    def test_extract_shrinkage_lambda_from_message(self) -> None:
        self.assertAlmostEqual(
            extract_shrinkage_lambda_from_message("correlated fit failed; retried with shrinkage_lambda=0.30; ok"),
            0.30,
        )
        self.assertIsNone(extract_shrinkage_lambda_from_message("plain success message"))

    def test_run_sliding_fits_reuses_shared_covariance_and_slices_windows(self) -> None:
        bootstrap_means = np.array(
            [
                [5.0, 4.0, 3.0, 2.0, 1.5],
                [5.2, 4.1, 3.1, 2.1, 1.6],
            ],
            dtype=float,
        )
        sigma = np.full(5, 0.1, dtype=float)
        covariance = np.arange(25, dtype=float).reshape(5, 5) + 5.0 * np.eye(5)
        recorded_covariances: list[np.ndarray] = []

        def fake_build_residual_model(fit_mode, sigma_slice, covariance_slice, shrinkage_lambda=0.0):
            recorded_covariances.append(np.array(covariance_slice, copy=True))
            return SimpleNamespace(
                fit_mode=fit_mode,
                sigma=np.array(sigma_slice, copy=True),
                cholesky_factor=np.eye(len(sigma_slice)),
                shrinkage_lambda=shrinkage_lambda,
            )

        with patch("lqcd_analysis.two_point.fit_nstate.build_residual_model", side_effect=fake_build_residual_model), patch(
            "lqcd_analysis.two_point.fit_nstate.fit_nstate_sample",
            return_value=FitResult(
                params=np.array([1.0, 0.4]),
                chi2=1.0,
                chi2_dof=1.0,
                pvalue=0.5,
                success=True,
                message="ok",
            ),
        ) as mock_fit:
            run_sliding_fits(
                bootstrap_means=bootstrap_means,
                sigma=sigma,
                fit_mode="correlated",
                nt=32,
                model="normal",
                nstates=1,
                tmin_values=range(1, 3),
                tmax=3,
                initial_amplitudes=np.array([1.0]),
                initial_energies=np.array([0.4]),
                covariance=covariance,
            )

        self.assertEqual(len(recorded_covariances), 2)
        self.assertTrue(np.array_equal(recorded_covariances[0], covariance[1:4, 1:4]))
        self.assertTrue(np.array_equal(recorded_covariances[1], covariance[2:4, 2:4]))
        for call in mock_fit.call_args_list:
            self.assertEqual(call.kwargs["residual_model"].fit_mode, "correlated")

    def test_correlated_window_factorization_failure_uses_shrinkage_schedule(self) -> None:
        bootstrap_means = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=float,
        )
        sigma = np.full(3, 0.1, dtype=float)
        covariance = np.array(
            [
                [1.0, 1.5, 0.0],
                [1.5, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        recorded_lambdas: list[float | None] = []

        def fake_fit_nstate_sample(*args, **kwargs):
            residual_model = kwargs["residual_model"]
            recorded_lambdas.append(residual_model.shrinkage_lambda)
            return FitResult(
                params=np.array([1.0, 0.4]),
                chi2=1.0,
                chi2_dof=1.0,
                pvalue=0.5,
                success=True,
                message="ok",
            )

        with patch("lqcd_analysis.two_point.fit_nstate.fit_nstate_sample", side_effect=fake_fit_nstate_sample):
            rows, sample_tables, _ = run_sliding_fits(
                bootstrap_means=bootstrap_means,
                sigma=sigma,
                fit_mode="correlated",
                nt=32,
                model="normal",
                nstates=1,
                tmin_values=range(0, 1),
                tmax=1,
                initial_amplitudes=np.array([1.0]),
                initial_energies=np.array([0.4]),
                covariance=covariance,
            )
        self.assertTrue(all(abs(value - 0.5) < 1e-12 for value in recorded_lambdas if value is not None))
        self.assertEqual(rows[0].bootstrap_total, len(bootstrap_means))
        self.assertEqual(sample_tables[0].shape[0], len(bootstrap_means))

    def test_correlated_optimizer_failure_uses_shrinkage_schedule(self) -> None:
        times = np.array([2.0, 3.0])
        data = np.array([0.5, 0.4])
        sigma = np.array([0.1, 0.2])
        covariance = np.array([[4.0, 1.0], [1.0, 3.0]])
        residual_model = build_residual_model("correlated", sigma, covariance)
        failure_theta = pack_fit_parameters(np.array([1.0]), np.array([0.4]))
        success_theta = pack_fit_parameters(np.array([1.1]), np.array([0.42]))
        with patch(
            "lqcd_analysis.two_point.fit_nstate.least_squares",
            side_effect=[
                *[SimpleNamespace(success=False, x=failure_theta, message="fail") for _ in range(9)],
                SimpleNamespace(success=True, x=success_theta, message="ok"),
            ],
        ) as mock_least_squares:
            result = fit_nstate_sample(
                times=times,
                data=data,
                sigma=sigma,
                nt=32,
                model="normal",
                initial_amplitudes=np.array([1.0]),
                initial_energies=np.array([0.4]),
                nstates=1,
                residual_model=residual_model,
                covariance=covariance,
            )
        self.assertTrue(result.success)
        self.assertIn("shrinkage_lambda=0.20", result.message)
        self.assertEqual(mock_least_squares.call_count, 10)
        self.assertFalse(result.used_uncorrelated_fallback)

    def test_correlated_optimizer_failure_can_reach_diagonal_limit_lambda_one(self) -> None:
        times = np.array([2.0, 3.0])
        data = np.array([0.5, 0.4])
        sigma = np.array([0.1, 0.2])
        covariance = np.array([[4.0, 1.0], [1.0, 3.0]])
        residual_model = build_residual_model("correlated", sigma, covariance)
        failure_theta = pack_fit_parameters(np.array([1.0]), np.array([0.4]))
        success_theta = pack_fit_parameters(np.array([1.1]), np.array([0.42]))
        n_failures = 9 * len(SHRINKAGE_LAMBDAS)
        with patch(
            "lqcd_analysis.two_point.fit_nstate.least_squares",
            side_effect=[
                *[SimpleNamespace(success=False, x=failure_theta, message="fail") for _ in range(n_failures)],
                SimpleNamespace(success=True, x=success_theta, message="ok"),
            ],
        ) as mock_least_squares:
            result = fit_nstate_sample(
                times=times,
                data=data,
                sigma=sigma,
                nt=32,
                model="normal",
                initial_amplitudes=np.array([1.0]),
                initial_energies=np.array([0.4]),
                nstates=1,
                residual_model=residual_model,
                covariance=covariance,
            )
        self.assertTrue(result.success)
        self.assertIn("shrinkage_lambda=1.00", result.message)
        self.assertTrue(result.used_uncorrelated_fallback)
        self.assertEqual(mock_least_squares.call_count, n_failures + 1)

    def test_two_state_priors_use_one_state_fit_window_summary_e0_only(self) -> None:
        previous = FitWindowParameterSummary(params_mean=(2.5, 0.41), params_err=(0.15, 0.02))
        priors = build_energy_priors_from_fit_window_summary(previous, nstates=2)
        self.assertEqual(len(priors), 1)
        self.assertEqual(priors[0].energy_index, 0)
        self.assertAlmostEqual(priors[0].center, 0.41)
        self.assertAlmostEqual(priors[0].sigma, 0.02)

    def test_three_state_priors_use_two_state_fit_window_summary_e0_and_e1_only(self) -> None:
        previous = FitWindowParameterSummary(
            params_mean=(2.0, 1.0, 0.40, 0.82),
            params_err=(0.2, 0.2, 0.02, 0.05),
        )
        priors = build_energy_priors_from_fit_window_summary(previous, nstates=3)
        self.assertEqual(len(priors), 2)
        self.assertEqual([prior.energy_index for prior in priors], [0, 1])
        self.assertTrue(np.allclose([prior.center for prior in priors], [0.40, 0.82]))

    def test_single_attempt_success_skips_fallback_attempts(self) -> None:
        theta = pack_fit_parameters(np.array([1.5]), np.array([0.4]))
        with patch(
            "lqcd_analysis.two_point.fit_nstate.least_squares",
            return_value=SimpleNamespace(success=True, x=theta, message="ok"),
        ) as mock_least_squares, patch(
            "lqcd_analysis.two_point.fit_nstate.build_fallback_fit_attempts",
        ) as mock_fallback:
            result = fit_nstate_sample(
                times=np.array([2.0, 3.0, 4.0]),
                data=np.array([1.0, 0.7, 0.5]),
                sigma=np.array([0.1, 0.1, 0.1]),
                nt=32,
                model="normal",
                initial_amplitudes=np.array([1.5]),
                initial_energies=np.array([0.4]),
                nstates=1,
            )
        self.assertTrue(result.success)
        self.assertEqual(mock_least_squares.call_count, 1)
        mock_fallback.assert_not_called()

    def test_failed_primary_attempt_runs_fallback_attempts(self) -> None:
        failure_theta = pack_fit_parameters(np.array([1.5]), np.array([0.4]))
        success_theta = pack_fit_parameters(np.array([1.4]), np.array([0.38]))
        fallback_attempts = [
            (np.array([0.75]), np.array([0.4])),
            (np.array([1.4]), np.array([0.38])),
        ]
        with patch(
            "lqcd_analysis.two_point.fit_nstate.least_squares",
            side_effect=[
                SimpleNamespace(success=False, x=failure_theta, message="fail"),
                SimpleNamespace(success=True, x=success_theta, message="ok"),
            ],
        ) as mock_least_squares, patch(
            "lqcd_analysis.two_point.fit_nstate.build_fallback_fit_attempts",
            return_value=fallback_attempts,
        ) as mock_fallback:
            result = fit_nstate_sample(
                times=np.array([2.0, 3.0, 4.0]),
                data=np.array([1.0, 0.7, 0.5]),
                sigma=np.array([0.1, 0.1, 0.1]),
                nt=32,
                model="normal",
                initial_amplitudes=np.array([1.5]),
                initial_energies=np.array([0.4]),
                nstates=1,
            )
        self.assertTrue(result.success)
        self.assertEqual(mock_least_squares.call_count, 2)
        mock_fallback.assert_called_once()

    def test_warm_start_propagates_previous_window_solution(self) -> None:
        bootstrap_means = np.array(
            [
                [5.0, 4.0, 3.0, 2.0, 1.5],
                [5.2, 4.1, 3.1, 2.1, 1.6],
            ],
            dtype=float,
        )
        sigma = np.full(5, 0.1, dtype=float)
        successful_previous = FitResult(
            params=np.array([1.2, 0.45]),
            chi2=1.0,
            chi2_dof=1.0,
            pvalue=0.5,
            success=True,
            message="ok",
        )
        later_result = FitResult(
            params=np.array([1.1, 0.44]),
            chi2=1.0,
            chi2_dof=1.0,
            pvalue=0.5,
            success=True,
            message="ok",
        )
        recorded_initial_guesses: list[tuple[np.ndarray, np.ndarray]] = []

        def fake_fit_nstate_sample(*args, **kwargs):
            recorded_initial_guesses.append((np.array(args[5], copy=True), np.array(args[6], copy=True)))
            return successful_previous if len(recorded_initial_guesses) == 1 else later_result

        with patch("lqcd_analysis.two_point.fit_nstate.fit_nstate_sample", side_effect=fake_fit_nstate_sample):
            run_sliding_fits(
                bootstrap_means=bootstrap_means,
                sigma=sigma,
                fit_mode="uncorrelated",
                nt=32,
                model="normal",
                nstates=1,
                tmin_values=range(2, 4),
                tmax=4,
                initial_amplitudes=np.array([9.9]),
                initial_energies=np.array([0.9]),
            )

        self.assertTrue(np.allclose(recorded_initial_guesses[0][0], [9.9]))
        self.assertTrue(np.allclose(recorded_initial_guesses[0][1], [0.9]))
        self.assertTrue(np.allclose(recorded_initial_guesses[3][0], successful_previous.params[:1]))
        self.assertTrue(np.allclose(recorded_initial_guesses[3][1], successful_previous.params[1:]))

    def test_failed_previous_window_falls_back_to_original_initial_guess(self) -> None:
        bootstrap_means = np.array(
            [
                [5.0, 4.0, 3.0, 2.0, 1.5],
                [5.2, 4.1, 3.1, 2.1, 1.6],
            ],
            dtype=float,
        )
        sigma = np.full(5, 0.1, dtype=float)
        failed = FitResult(
            params=np.array([np.nan, np.nan]),
            chi2=np.nan,
            chi2_dof=np.nan,
            pvalue=np.nan,
            success=False,
            message="fail",
        )
        later_result = FitResult(
            params=np.array([1.1, 0.44]),
            chi2=1.0,
            chi2_dof=1.0,
            pvalue=0.5,
            success=True,
            message="ok",
        )
        recorded_initial_guesses: list[tuple[np.ndarray, np.ndarray]] = []

        def fake_fit_nstate_sample(*args, **kwargs):
            recorded_initial_guesses.append((np.array(args[5], copy=True), np.array(args[6], copy=True)))
            return failed if len(recorded_initial_guesses) == 1 else later_result

        with patch("lqcd_analysis.two_point.fit_nstate.fit_nstate_sample", side_effect=fake_fit_nstate_sample):
            run_sliding_fits(
                bootstrap_means=bootstrap_means,
                sigma=sigma,
                fit_mode="uncorrelated",
                nt=32,
                model="normal",
                nstates=1,
                tmin_values=range(2, 4),
                tmax=4,
                initial_amplitudes=np.array([9.9]),
                initial_energies=np.array([0.9]),
            )

        self.assertTrue(np.allclose(recorded_initial_guesses[0][0], [9.9]))
        self.assertTrue(np.allclose(recorded_initial_guesses[0][1], [0.9]))
        self.assertTrue(np.allclose(recorded_initial_guesses[3][0], [9.9]))
        self.assertTrue(np.allclose(recorded_initial_guesses[3][1], [0.9]))

    def test_parse_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "input.txt"
            path.write_text(
                "\n".join(
                    [
                        "demo_pz* 64 64 0.076",
                        "c2pt /tmp/c2pt_pz*.csv",
                        "pzlist 0 1",
                        "fold_t antiperiodic",
                        "tmax 24",
                        "model symmetric",
                        "fit_mode correlated",
                        "nstates 1 2 3",
                        "tmin_window /tmp/tmin_windows.txt",
                        "results_dir /tmp/nstate_results",
                    ]
                ),
                encoding="utf-8",
            )
            parsed = parse_nstate_fit_input(path)
        self.assertEqual(parsed.model, "symmetric")
        self.assertEqual(parsed.fit_mode, "correlated")
        self.assertEqual(parsed.nstates, (1, 2, 3))
        self.assertEqual(parsed.fold_t, "antiperiodic")
        self.assertEqual(parsed.results_dir, Path("/tmp/nstate_results"))

    def test_parse_input_requires_tmax(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "input.txt"
            path.write_text(
                "\n".join(
                    [
                        "demo_pz* 64 64 0.076",
                        "c2pt /tmp/c2pt_pz*.csv",
                        "pzlist 0 1",
                        "fold_t antiperiodic",
                        "model symmetric",
                        "fit_mode correlated",
                        "nstates 1 2 3",
                        "tmax 24",
                        "tmin_window /tmp/tmin_windows.txt",
                    ]
                ),
                encoding="utf-8",
            )
            parsed = parse_nstate_fit_input(path)
        self.assertEqual(parsed.tmax, 24)

    def test_end_to_end_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "c2pt_5_5_k0_pz0_real.csv"
            input_path = tmp / "input_nstate.txt"
            fit_window_path = tmp / "fit_window.txt"

            times = np.arange(18)
            base = 3.0 * np.exp(-0.35 * times) + 0.8 * np.exp(-0.8 * times)
            configs = []
            for cfg in range(64):
                noise = 0.001 * np.sin(times + cfg)
                configs.append(base * (1.0 + noise))
            data = np.array(configs).T

            with csv_path.open("w", encoding="utf-8") as handle:
                handle.write("t," + ",".join(f"cfg_{idx}" for idx in range(data.shape[1])) + "\n")
                for t in range(len(times)):
                    handle.write(
                        str(t)
                        + ","
                        + ",".join(f"{value:.12e}" for value in data[t])
                        + "\n"
                    )

            fit_window_path.write_text("0 0 2\n", encoding="utf-8")
            input_path.write_text(
                "\n".join(
                    [
                        "demo_pz* 16 18 0.076",
                        f"c2pt {csv_path.as_posix().replace('pz0', 'pz*')}",
                        "pzlist 0",
                        "fold_t none",
                        "tmax 10",
                        "model normal",
                        "nstates 1 2",
                        f"tmin_window {fit_window_path}",
                        "bootstrap_samples 16",
                        "bootstrap_size 16",
                        "plot false",
                    ]
                ),
                encoding="utf-8",
            )

            outputs = run_nstate_fit(input_path)
            self.assertTrue(outputs)
            summary = tmp / "results_nstate_fit" / "demo_pz0" / "demo_pz0_normal_summary.txt"
            self.assertTrue(summary.exists())
            self.assertIn(
                "1state fit_window_start_fallback_uncorrelated_successes",
                summary.read_text(encoding="utf-8"),
            )
            one_state = (
                tmp / "results_nstate_fit" / "demo_pz0" / "tables" / "demo_pz0_normal_1state_tmax10_fits.txt"
            )
            self.assertTrue(one_state.exists())
            notebook = tmp / "results_nstate_fit" / "notebook_plots" / "demo_pz0" / "demo_pz0_normal_nstate_plots.ipynb"
            self.assertTrue(notebook.exists())

    def test_notebook_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            notebook = write_nstate_plot_notebook(
                notebook_path=tmp / "notebook_plots" / "demo.ipynb",
                notebook_output_dir=tmp / "notebook_plots" / "generated",
                correlator_table=tmp / "corr.txt",
                meff_table=tmp / "meff.txt",
                fit_tables={1: tmp / "fit1.txt", 2: tmp / "fit2.txt"},
                model="normal",
                title="demo",
                nt=64,
                lattice_spacing_fm=0.076,
            )
            self.assertTrue(notebook.exists())
            text = notebook.read_text(encoding="utf-8")
            self.assertIn("plot_nstate_outputs", text)
            self.assertIn("lattice_spacing_fm = 0.076", text)

    def test_reconstruction_band_brackets_center_curve(self) -> None:
        times = np.arange(3, 8)
        amplitudes = np.array([2.0, 0.7])
        amplitude_errs = np.array([0.1, 0.05])
        energies = np.array([0.35, 0.8])
        energy_errs = np.array([0.02, 0.04])
        center, lower, upper = build_reconstruction_band(
            times,
            amplitudes,
            amplitude_errs,
            energies,
            energy_errs,
            nt=32,
            model="normal",
        )
        expected_center = evaluate_model(times, amplitudes, energies, 32, "normal")
        self.assertTrue(np.allclose(center, expected_center))
        self.assertTrue(np.all(lower <= center))
        self.assertTrue(np.all(center <= upper))

    def test_scan_state_indices_show_only_new_state_for_multistate_plots(self) -> None:
        self.assertTrue(np.array_equal(select_scan_state_indices(1), np.array([0])))
        self.assertTrue(np.array_equal(select_scan_state_indices(2), np.array([1])))
        self.assertTrue(np.array_equal(select_scan_state_indices(3), np.array([2])))

    def test_fit_window_parameter_summary_uses_bootstrap_window_averages_for_means(self) -> None:
        rows = [
            FitSummaryRow(1, 2, 10, 1, 2, 2, 1.0, 0, 1.0, 0.5, 1, (2.0, 0.40), (0.2, 0.10)),
            FitSummaryRow(1, 3, 10, 1, 2, 2, 1.0, 0, 1.0, 0.5, 1, (4.0, 0.80), (0.1, 0.20)),
        ]
        sample_tables = {
            2: np.array([[2, 0, 1, 1.0, 0.5, 1.8, 0.36], [2, 1, 1, 1.0, 0.5, 2.2, 0.44]], dtype=float),
            3: np.array([[3, 0, 1, 1.0, 0.5, 3.8, 0.76], [3, 1, 1, 1.0, 0.5, 4.2, 0.84]], dtype=float),
        }
        fit_window = FitWindowSummary(start_tmin=2, end_tmin=3, representative_tmin=2, energy_mean=0.0, amplitude_mean=0.0)
        summary = compute_fit_window_parameter_summary(rows, sample_tables, fit_window)
        amp_boot = np.array([3.4, 3.8])
        energy_boot = np.array([0.44, 0.52])
        amp_expected = 0.5 * np.sum(np.percentile(amp_boot, [16.0, 84.0]))
        energy_expected = 0.5 * np.sum(np.percentile(energy_boot, [16.0, 84.0]))
        self.assertAlmostEqual(summary.params_mean[0], amp_expected)
        self.assertAlmostEqual(summary.params_mean[1], energy_expected)

    def test_fit_window_parameter_summary_errors_use_bootstrap_window_averages(self) -> None:
        rows = [
            FitSummaryRow(1, 2, 10, 1, 2, 2, 1.0, 0, 1.0, 0.5, 1, (2.0, 0.40), (0.2, 0.10)),
            FitSummaryRow(1, 3, 10, 1, 2, 2, 1.0, 0, 1.0, 0.5, 1, (4.0, 0.80), (0.1, 0.20)),
        ]
        sample_tables = {
            2: np.array([[2, 0, 1, 1.0, 0.5, 1.8, 0.36], [2, 1, 1, 1.0, 0.5, 2.2, 0.44]], dtype=float),
            3: np.array([[3, 0, 1, 1.0, 0.5, 3.8, 0.76], [3, 1, 1, 1.0, 0.5, 4.2, 0.84]], dtype=float),
        }
        fit_window = FitWindowSummary(start_tmin=2, end_tmin=3, representative_tmin=2, energy_mean=0.0, amplitude_mean=0.0)
        summary = compute_fit_window_parameter_summary(rows, sample_tables, fit_window)
        w_amp = np.array([1.0 / 0.2**2, 1.0 / 0.1**2])
        amp_boot = np.array(
            [
                np.sum(w_amp * np.array([1.8, 3.8])) / np.sum(w_amp),
                np.sum(w_amp * np.array([2.2, 4.2])) / np.sum(w_amp),
            ]
        )
        p16, p84 = np.percentile(amp_boot, [16.0, 84.0])
        self.assertAlmostEqual(summary.params_err[0], 0.5 * (p84 - p16))

    def test_plotting_uses_best_chi2_selected_row_for_band(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            correlator = np.column_stack([np.arange(6), np.linspace(1.0, 0.2, 6), np.full(6, 0.01)])
            meff = np.column_stack([np.arange(6), np.full(6, 0.4), np.full(6, 0.02)])
            fit = np.array(
                [
                    [2, 5, 1, 2, 2, 1.0, 0, 2.0, 0.5, 1, 10.0, 0.2, 1.0, 0.1],
                    [3, 5, 1, 2, 2, 1.0, 0, 0.5, 0.5, 1, 20.5, 0.2, 2.05, 0.1],
                ],
                dtype=float,
            )
            corr_path = tmp / "corr.txt"
            meff_path = tmp / "meff.txt"
            fit_path = tmp / "fit.txt"
            np.savetxt(corr_path, correlator)
            np.savetxt(meff_path, meff)
            np.savetxt(fit_path, fit)

            with patch("lqcd_analysis.two_point.plotting.prepare_matplotlib", return_value=None):
                with patch("lqcd_analysis.two_point.plotting.plot_parameter_scan") as mock_scan:
                    with patch(
                        "lqcd_analysis.two_point.plotting.plot_effective_mass",
                        return_value=tmp / "meff.pdf",
                    ) as mock_meff:
                        with patch(
                            "lqcd_analysis.two_point.plotting.plot_best_fit_reconstruction",
                            return_value=tmp / "recon.pdf",
                        ):
                            plot_nstate_outputs(
                                output_dir=tmp,
                                correlator_table=corr_path,
                                meff_table=meff_path,
                                fit_table=fit_path,
                                nstates=1,
                                model="normal",
                                title="demo",
                                nt=16,
                                lattice_spacing_fm=0.1,
            )
            energy_call = mock_scan.call_args_list[0]
            amplitude_call = mock_scan.call_args_list[1]
            self.assertIsNone(energy_call.kwargs["selected_values"])
            self.assertIsNone(energy_call.kwargs["selected_errors"])
            self.assertAlmostEqual(amplitude_call.kwargs["selected_values"][0], 20.5)
            self.assertAlmostEqual(amplitude_call.kwargs["selected_errors"][0], 0.2)
            self.assertIsNone(mock_meff.call_args.kwargs["reference_value"])

    def test_plotting_uses_dispersion_reference_for_energy_and_meff_band(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            correlator = np.column_stack([np.arange(6), np.linspace(1.0, 0.2, 6), np.full(6, 0.01)])
            meff = np.column_stack([np.arange(6), np.full(6, 0.4), np.full(6, 0.02)])
            fit = np.array([[2, 5, 1, 2, 2, 1.0, 0, 0.5, 0.5, 1, 10.0, 0.2, 1.0, 0.1]], dtype=float)
            tables_dir = tmp / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            corr_path = tables_dir / "corr.txt"
            meff_path = tables_dir / "meff.txt"
            fit_path = tables_dir / "fit.txt"
            summary_path = tmp / "demo_pz1_normal_summary.txt"
            np.savetxt(corr_path, correlator)
            np.savetxt(meff_path, meff)
            np.savetxt(fit_path, fit)
            summary_path.write_text(
                "\n".join(
                    [
                        "title demo_pz1",
                        "model normal",
                        "1state_target_energy_pz1 3.0000000000e-01",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("lqcd_analysis.two_point.plotting.prepare_matplotlib", return_value=None):
                with patch("lqcd_analysis.two_point.plotting.plot_parameter_scan") as mock_scan:
                    with patch(
                        "lqcd_analysis.two_point.plotting.plot_effective_mass",
                        return_value=tmp / "meff.pdf",
                    ) as mock_meff:
                        with patch(
                            "lqcd_analysis.two_point.plotting.plot_best_fit_reconstruction",
                            return_value=tmp / "recon.pdf",
                        ):
                            plot_nstate_outputs(
                                output_dir=tmp,
                                correlator_table=corr_path,
                                meff_table=meff_path,
                                fit_table=fit_path,
                                nstates=1,
                                model="normal",
                                title="demo_pz1",
                                nt=16,
                                lattice_spacing_fm=0.1,
                            )
            energy_call = mock_scan.call_args_list[0]
            dispersion_mev = 0.3 * 197.3269804 / 0.1
            self.assertAlmostEqual(energy_call.kwargs["selected_values"][0], dispersion_mev)
            self.assertAlmostEqual(mock_meff.call_args.kwargs["reference_value"], dispersion_mev)

    def test_plotting_reconstruction_starts_at_fit_window_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            correlator = np.column_stack([np.arange(7), np.linspace(1.0, 0.2, 7), np.full(7, 0.01)])
            meff = np.column_stack([np.arange(7), np.full(7, 0.4), np.full(7, 0.02)])
            fit = np.array(
                [
                    [2, 5, 1, 2, 2, 1.0, 0, 1.0, 0.5, 0, 10.0, 0.2, 1.0, 0.1],
                    [3, 5, 1, 2, 2, 1.0, 0, 1.0, 0.5, 1, 10.5, 0.2, 1.05, 0.1],
                    [4, 5, 1, 2, 2, 1.0, 0, 1.0, 0.5, 1, 11.0, 0.2, 1.10, 0.1],
                ],
                dtype=float,
            )
            fit_window = np.array([[3.0, 0.30, 0.4, 0.04]], dtype=float)
            corr_path = tmp / "corr.txt"
            meff_path = tmp / "meff.txt"
            fit_path = tmp / "fit.txt"
            fit_window_path = tmp / "fit_window.txt"
            np.savetxt(corr_path, correlator)
            np.savetxt(meff_path, meff)
            np.savetxt(fit_path, fit)
            np.savetxt(fit_window_path, fit_window)

            with patch("lqcd_analysis.two_point.plotting.prepare_matplotlib", return_value=None):
                with patch("lqcd_analysis.two_point.plotting.plot_effective_mass", return_value=tmp / "meff.pdf"):
                    with patch("lqcd_analysis.two_point.plotting.plot_parameter_scan", return_value=tmp / "scan.pdf"):
                        with patch(
                            "lqcd_analysis.two_point.plotting.plot_best_fit_reconstruction",
                            return_value=tmp / "recon.pdf",
                        ) as mock_reconstruction:
                            plot_nstate_outputs(
                                output_dir=tmp,
                                correlator_table=corr_path,
                                meff_table=meff_path,
                                fit_table=fit_path,
                                nstates=1,
                                model="normal",
                                title="demo",
                                nt=16,
                                lattice_spacing_fm=0.1,
            )
            reconstruction_args = mock_reconstruction.call_args.args
            self.assertTrue(np.array_equal(reconstruction_args[1], np.arange(3, 6)))
            self.assertEqual(len(reconstruction_args[2]), 3)
            self.assertEqual(len(reconstruction_args[3]), 3)

    def test_summary_reports_fit_window_start_shrinkage_lambda(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "c2pt_5_5_k0_pz0_real.csv"
            input_path = tmp / "input_nstate.txt"
            fit_window_path = tmp / "fit_window.txt"

            times = np.arange(12)
            base = 2.0 * np.exp(-0.4 * times)
            data = np.array([base * (1.0 + 0.001 * np.cos(times + cfg)) for cfg in range(12)]).T

            with csv_path.open("w", encoding="utf-8") as handle:
                handle.write("t," + ",".join(f"cfg_{idx}" for idx in range(data.shape[1])) + "\n")
                for t in range(len(times)):
                    handle.write(str(t) + "," + ",".join(f"{value:.12e}" for value in data[t]) + "\n")

            fit_window_path.write_text("0 0 2\n", encoding="utf-8")
            input_path.write_text(
                "\n".join(
                    [
                        "demo_pz* 16 12 0.076",
                        f"c2pt {csv_path.as_posix().replace('pz0', 'pz*')}",
                        "pzlist 0",
                        "fold_t none",
                        "tmax 10",
                        "model normal",
                        "nstates 1",
                        f"tmin_window {fit_window_path}",
                        "bootstrap_samples 8",
                        "bootstrap_size 8",
                        "plot false",
                    ]
                ),
                encoding="utf-8",
            )

            def fake_run_sliding_fits(*args, **kwargs):
                del args, kwargs
                rows = [
                    FitSummaryRow(1, 0, 5, 1, 8, 8, 1.0, 0, 1.0, 0.5, 0, (2.0, 0.4), (0.2, 0.05)),
                    FitSummaryRow(1, 1, 5, 1, 8, 8, 1.0, 0, 1.1, 0.5, 0, (2.1, 0.41), (0.2, 0.05)),
                    FitSummaryRow(1, 2, 5, 1, 8, 8, 1.0, 0, 1.2, 0.5, 0, (2.2, 0.42), (0.2, 0.05)),
                ]
                sample_tables = {
                    row.tmin: np.array([[row.tmin, 0, 1, 1.0, 0.5, *row.params_mean]], dtype=float)
                    for row in rows
                }
                meanfits = {
                    0: FitResult(np.array([2.0, 0.4]), 1.0, 1.0, 0.5, True, "shrinkage_lambda=0.30; ok"),
                    1: FitResult(np.array([2.1, 0.41]), 1.0, 1.0, 0.5, True, "shrinkage_lambda=0.10; ok"),
                    2: FitResult(np.array([2.2, 0.42]), 1.0, 1.0, 0.5, True, "ok"),
                }
                return rows, sample_tables, meanfits

            with patch("lqcd_analysis.two_point.fit_nstate.run_sliding_fits", side_effect=fake_run_sliding_fits):
                run_nstate_fit(input_path)

            summary = tmp / "results_nstate_fit" / "demo_pz0" / "demo_pz0_normal_summary.txt"
            text = summary.read_text(encoding="utf-8")
            self.assertIn("1state fit_window_start_shrinkage_lambda 0.30", text)
            self.assertNotIn("representative_shrinkage_lambda", text)

    def test_fit_window_fixes_two_point_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "c2pt_5_5_k0_pz0_real.csv"
            input_path = tmp / "input_nstate.txt"
            fit_window_path = tmp / "fit_window.txt"
            override_path = tmp / "fit_windows.txt"

            times = np.arange(14)
            base = 2.5 * np.exp(-0.35 * times) + 0.5 * np.exp(-0.9 * times)
            data = np.array([base * (1.0 + 0.001 * np.sin(times + cfg)) for cfg in range(12)]).T

            with csv_path.open("w", encoding="utf-8") as handle:
                handle.write("t," + ",".join(f"cfg_{idx}" for idx in range(data.shape[1])) + "\n")
                for t in range(len(times)):
                    handle.write(str(t) + "," + ",".join(f"{value:.12e}" for value in data[t]) + "\n")

            override_path.write_text("0 0 4\n", encoding="utf-8")
            fit_window_path.write_text("0 0 2\n", encoding="utf-8")
            input_path.write_text(
                "\n".join(
                    [
                        "demo_pz* 16 14 0.076",
                        f"c2pt {csv_path.as_posix().replace('pz0', 'pz*')}",
                        "pzlist 0",
                        "fold_t none",
                        "tmax 6",
                        "model normal",
                        "nstates 1 2",
                        "bootstrap_samples 8",
                        "bootstrap_size 8",
                        f"tmin_window {override_path}",
                        "plot false",
                    ]
                ),
                encoding="utf-8",
            )

            run_nstate_fit(input_path)

            summary = tmp / "results_nstate_fit" / "demo_pz0" / "demo_pz0_normal_summary.txt"
            fit_table = tmp / "results_nstate_fit" / "demo_pz0" / "tables" / "demo_pz0_normal_1state_tmax6_fits.txt"
            summary_text = summary.read_text(encoding="utf-8")
            fit_rows = np.loadtxt(fit_table, ndmin=2)

        self.assertIn("tmin_window 0 4", summary_text)
        self.assertIn("1state fit_window_tmin 0 4", summary_text)
        self.assertEqual(fit_rows.shape[0], 5)
        self.assertTrue(np.array_equal(fit_rows[:, 0].astype(int), np.array([0, 1, 2, 3, 4])))
        self.assertTrue(np.all(fit_rows[:, 1].astype(int) == 6))

    def test_auto_computed_lower_states_are_included_in_plots_and_notebook(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "c2pt_5_5_k0_pz0_real.csv"
            input_path = tmp / "input_nstate.txt"
            fit_window_path = tmp / "fit_window.txt"

            times = np.arange(14)
            base = 2.5 * np.exp(-0.35 * times) + 0.5 * np.exp(-0.9 * times)
            data = np.array([base * (1.0 + 0.001 * np.sin(times + cfg)) for cfg in range(12)]).T

            with csv_path.open("w", encoding="utf-8") as handle:
                handle.write("t," + ",".join(f"cfg_{idx}" for idx in range(data.shape[1])) + "\n")
                for t in range(len(times)):
                    handle.write(str(t) + "," + ",".join(f"{value:.12e}" for value in data[t]) + "\n")

            fit_window_path.write_text("0 0 2\n", encoding="utf-8")
            input_path.write_text(
                "\n".join(
                    [
                        "demo_pz* 16 14 0.076",
                        f"c2pt {csv_path.as_posix().replace('pz0', 'pz*')}",
                        "pzlist 0",
                        "fold_t none",
                        "tmax 6",
                        "model normal",
                        "nstates 2",
                        f"tmin_window {fit_window_path}",
                        "bootstrap_samples 8",
                        "bootstrap_size 8",
                        "plot true",
                    ]
                ),
                encoding="utf-8",
            )

            plotted_states: list[int] = []
            notebook_fit_table_keys: list[int] = []

            def fake_plot_nstate_outputs(*args, **kwargs):
                del args
                plotted_states.append(kwargs["nstates"])
                return []

            def fake_write_nstate_plot_notebook(*args, **kwargs):
                del args
                notebook_fit_table_keys.extend(sorted(kwargs["fit_tables"].keys()))
                path = tmp / "notebook_plots" / "demo.ipynb"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
                return path

            with patch("lqcd_analysis.two_point.fit_nstate.plot_nstate_outputs", side_effect=fake_plot_nstate_outputs):
                with patch(
                    "lqcd_analysis.two_point.fit_nstate.write_nstate_plot_notebook",
                    side_effect=fake_write_nstate_plot_notebook,
                ):
                    run_nstate_fit(input_path)

            self.assertEqual(sorted(plotted_states), [1, 2])
            self.assertEqual(notebook_fit_table_keys, [1, 2])


if __name__ == "__main__":
    unittest.main()
