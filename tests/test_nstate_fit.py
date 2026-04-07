import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from lqcd_analysis.nstate_fit import (
    _try_load_previous_plateau,
    build_residual_model,
    build_fallback_fit_attempts,
    build_energy_priors_from_previous_row,
    compute_bootstrap_covariance,
    compute_effective_mass_antisymmetric_root,
    compute_effective_mass_cosh_root,
    effective_mass_single,
    evaluate_model,
    FitResult,
    fit_residuals,
    fit_nstate_sample,
    FitSummaryRow,
    EnergyPrior,
    pack_fit_parameters,
    parse_nstate_fit_input,
    run_sliding_fits,
    run_nstate_fit,
    solve_antisymmetric_effective_mass,
    solve_cosh_effective_mass,
    suggest_plateau,
)
from lqcd_analysis.plotting_2pt import build_reconstruction_band, write_nstate_plot_notebook


class NStateFitTests(unittest.TestCase):
    @staticmethod
    def _make_fit_row(tmin: int, energy: float, error: float = 0.05, chi2_dof: float = 1.0) -> FitSummaryRow:
        return FitSummaryRow(
            nstates=1,
            tmin=tmin,
            tmax=12,
            success_meanfit=1,
            bootstrap_successes=0,
            bootstrap_total=0,
            bootstrap_success_fraction=0.0,
            fallback_uncorrelated_successes=0,
            chi2_dof=chi2_dof,
            pvalue=0.5,
            plateau_flag=0,
            params_mean=(2.0, energy),
            params_err=(0.2, error),
        )

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
            "lqcd_analysis.nstate_fit.compute_effective_mass_antisymmetric_root",
            return_value=np.full(len(correlator), 0.123),
        ) as mock_solver:
            meff = effective_mass_single(correlator, "antisymmetric", nt=16)
        mock_solver.assert_called_once()
        self.assertTrue(np.allclose(meff, 0.123))

    def test_plateau_accepts_constant_fluctuating_window(self) -> None:
        rows = [
            self._make_fit_row(2, 1.02),
            self._make_fit_row(3, 0.97),
            self._make_fit_row(4, 1.03),
            self._make_fit_row(5, 0.99),
            self._make_fit_row(6, 1.01),
        ]
        plateau = suggest_plateau(rows)
        self.assertEqual((plateau.start_tmin, plateau.end_tmin), (2, 6))

    def test_plateau_rejects_upward_trend(self) -> None:
        rows = [self._make_fit_row(t, 0.5 + 0.15 * (t - 2), error=0.03) for t in range(2, 7)]
        with self.assertRaises(ValueError):
            suggest_plateau(rows)

    def test_plateau_rejects_downward_trend(self) -> None:
        rows = [self._make_fit_row(t, 1.4 - 0.15 * (t - 2), error=0.03) for t in range(2, 7)]
        with self.assertRaises(ValueError):
            suggest_plateau(rows)

    def test_plateau_accepts_noisy_local_oscillations_without_global_slope(self) -> None:
        energies = [1.0, 1.08, 0.94, 1.06, 0.96, 1.02]
        rows = [self._make_fit_row(t, energy, error=0.08) for t, energy in enumerate(energies, start=2)]
        plateau = suggest_plateau(rows)
        self.assertEqual((plateau.start_tmin, plateau.end_tmin), (2, 7))

    def test_plateau_prefers_longest_then_later_window(self) -> None:
        rows = [
            self._make_fit_row(2, 1.00, error=0.04),
            self._make_fit_row(3, 1.03, error=0.04),
            self._make_fit_row(4, 0.98, error=0.04),
            self._make_fit_row(6, 1.01, error=0.04),
            self._make_fit_row(7, 0.99, error=0.04),
            self._make_fit_row(8, 1.02, error=0.04),
        ]
        plateau = suggest_plateau(rows)
        self.assertEqual((plateau.start_tmin, plateau.end_tmin), (6, 8))

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
                        "tsrange 0 24",
                        "model normal",
                        "nstates 1",
                    ]
                ),
                encoding="utf-8",
            )
            parsed = parse_nstate_fit_input(path)
        self.assertEqual(parsed.fit_mode, "uncorrelated")

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

        def fake_build_residual_model(fit_mode, sigma_slice, covariance_slice):
            recorded_covariances.append(np.array(covariance_slice, copy=True))
            return SimpleNamespace(fit_mode=fit_mode, sigma=np.array(sigma_slice, copy=True), cholesky_factor=np.eye(len(sigma_slice)))

        with patch("lqcd_analysis.nstate_fit.build_residual_model", side_effect=fake_build_residual_model), patch(
            "lqcd_analysis.nstate_fit.fit_nstate_sample",
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

    def test_correlated_window_factorization_failure_falls_back_to_diagonal_fit(self) -> None:
        bootstrap_means = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=float,
        )
        sigma = np.full(3, 0.1, dtype=float)
        covariance = np.zeros((3, 3), dtype=float)
        rows, sample_tables, _ = run_sliding_fits(
            bootstrap_means=bootstrap_means,
            sigma=sigma,
            fit_mode="correlated",
            nt=32,
            model="normal",
            nstates=1,
            tmin_values=range(0, 1),
            tmax=2,
            initial_amplitudes=np.array([1.0]),
            initial_energies=np.array([0.4]),
            covariance=covariance,
        )
        self.assertEqual(rows[0].bootstrap_total, len(bootstrap_means))
        self.assertEqual(rows[0].fallback_uncorrelated_successes, 0)
        self.assertEqual(sample_tables[0].shape[0], len(bootstrap_means))

    def test_correlated_optimizer_failure_falls_back_to_diagonal_fit(self) -> None:
        times = np.array([2.0, 3.0])
        data = np.array([0.5, 0.4])
        sigma = np.array([0.1, 0.2])
        covariance = np.array([[4.0, 1.0], [1.0, 3.0]])
        residual_model = build_residual_model("correlated", sigma, covariance)
        failure_theta = pack_fit_parameters(np.array([1.0]), np.array([0.4]))
        success_theta = pack_fit_parameters(np.array([1.1]), np.array([0.42]))
        with patch(
            "lqcd_analysis.nstate_fit.least_squares",
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
            )
        self.assertTrue(result.success)
        self.assertIn("fell back to diagonal covariance fit", result.message)
        self.assertEqual(mock_least_squares.call_count, 10)
        self.assertTrue(result.used_uncorrelated_fallback)

    def test_two_state_priors_use_one_state_e0_only(self) -> None:
        previous = FitSummaryRow(
            nstates=1,
            tmin=4,
            tmax=12,
            success_meanfit=1,
            bootstrap_successes=10,
            bootstrap_total=10,
            bootstrap_success_fraction=1.0,
            fallback_uncorrelated_successes=0,
            chi2_dof=1.0,
            pvalue=0.5,
            plateau_flag=1,
            params_mean=(2.0, 0.45),
            params_err=(0.2, 0.03),
        )
        priors = build_energy_priors_from_previous_row(previous, nstates=2)
        self.assertEqual(len(priors), 1)
        self.assertEqual(priors[0].energy_index, 0)
        self.assertAlmostEqual(priors[0].center, 0.45)
        self.assertAlmostEqual(priors[0].sigma, 0.03)

    def test_three_state_priors_use_two_state_e0_and_e1_only(self) -> None:
        previous = FitSummaryRow(
            nstates=2,
            tmin=5,
            tmax=12,
            success_meanfit=1,
            bootstrap_successes=10,
            bootstrap_total=10,
            bootstrap_success_fraction=1.0,
            fallback_uncorrelated_successes=0,
            chi2_dof=1.0,
            pvalue=0.5,
            plateau_flag=1,
            params_mean=(2.0, 1.0, 0.40, 0.82),
            params_err=(0.2, 0.2, 0.02, 0.05),
        )
        priors = build_energy_priors_from_previous_row(previous, nstates=3)
        self.assertEqual(len(priors), 2)
        self.assertEqual([prior.energy_index for prior in priors], [0, 1])
        self.assertTrue(np.allclose([prior.center for prior in priors], [0.40, 0.82]))

    def test_single_attempt_success_skips_fallback_attempts(self) -> None:
        theta = pack_fit_parameters(np.array([1.5]), np.array([0.4]))
        with patch(
            "lqcd_analysis.nstate_fit.least_squares",
            return_value=SimpleNamespace(success=True, x=theta, message="ok"),
        ) as mock_least_squares, patch(
            "lqcd_analysis.nstate_fit.build_fallback_fit_attempts",
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
            "lqcd_analysis.nstate_fit.least_squares",
            side_effect=[
                SimpleNamespace(success=False, x=failure_theta, message="fail"),
                SimpleNamespace(success=True, x=success_theta, message="ok"),
            ],
        ) as mock_least_squares, patch(
            "lqcd_analysis.nstate_fit.build_fallback_fit_attempts",
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

        with patch("lqcd_analysis.nstate_fit.fit_nstate_sample", side_effect=fake_fit_nstate_sample):
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

        with patch("lqcd_analysis.nstate_fit.fit_nstate_sample", side_effect=fake_fit_nstate_sample):
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
                        "tsrange 0 24",
                        "model symmetric",
                        "fit_mode correlated",
                        "nstates 1 2 3",
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

    def test_end_to_end_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "c2pt_5_5_k0_pz0_real.csv"
            input_path = tmp / "input_nstate.txt"

            times = np.arange(18)
            base = 3.0 * np.exp(-0.35 * times) + 0.8 * np.exp(-0.8 * times)
            configs = []
            for cfg in range(24):
                noise = 0.01 * np.sin(times + cfg)
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

            input_path.write_text(
                "\n".join(
                    [
                        "demo_pz* 16 18 0.076",
                        f"c2pt {csv_path.as_posix().replace('pz0', 'pz*')}",
                        "pzlist 0",
                        "fold_t none",
                        "tsrange 0 12",
                        "model normal",
                        "nstates 1 2",
                        "tmax 8",
                        "bootstrap_samples 12",
                        "bootstrap_size 12",
                        "plot false",
                    ]
                ),
                encoding="utf-8",
            )

            outputs = run_nstate_fit(input_path)
            self.assertTrue(outputs)
            summary = tmp / "results_nstate_fit" / "demo_pz0" / "demo_pz0_normal_summary.txt"
            self.assertTrue(summary.exists())
            self.assertIn("1state source computed_fresh", summary.read_text(encoding="utf-8"))
            self.assertIn(
                "1state representative_fallback_uncorrelated_successes",
                summary.read_text(encoding="utf-8"),
            )
            one_state = (
                tmp / "results_nstate_fit" / "demo_pz0" / "tables" / "demo_pz0_normal_1state_tmax8_fits.txt"
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

    def test_try_load_previous_plateau(self) -> None:
        spec = parse_nstate_fit_input("templates/input_files/nstate_fit_example_realdata.txt")
        spec = type(spec)(**{**spec.__dict__, "results_dir": Path("examples/outputs/nstate_fit_realdata")})
        plateau = _try_load_previous_plateau(spec, "l64c64a076_m140_fit_k0_pz0", 2, 12)
        self.assertIsNotNone(plateau)
        assert plateau is not None
        self.assertEqual(plateau.start_tmin, 2)
        self.assertGreater(plateau.end_tmin, plateau.start_tmin)
        self.assertTrue(np.isfinite(plateau.energy_mean))

    def test_two_state_only_run_bootstraps_missing_lower_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "c2pt_5_5_k0_pz0_real.csv"
            input_path = tmp / "input_nstate.txt"

            times = np.arange(18)
            base = 3.0 * np.exp(-0.35 * times) + 0.8 * np.exp(-0.8 * times)
            configs = []
            for cfg in range(24):
                noise = 0.01 * np.sin(times + cfg)
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

            input_path.write_text(
                "\n".join(
                    [
                        "demo_pz* 16 18 0.076",
                        f"c2pt {csv_path.as_posix().replace('pz0', 'pz*')}",
                        "pzlist 0",
                        "fold_t none",
                        "tsrange 0 12",
                        "model normal",
                        "nstates 2",
                        "tmax 8",
                        "bootstrap_samples 12",
                        "bootstrap_size 12",
                        "plot false",
                    ]
                ),
                encoding="utf-8",
            )

            outputs = run_nstate_fit(input_path)
            self.assertTrue(outputs)
            one_state = (
                tmp / "results_nstate_fit" / "demo_pz0" / "tables" / "demo_pz0_normal_1state_tmax8_fits.txt"
            )
            two_state = (
                tmp / "results_nstate_fit" / "demo_pz0" / "tables" / "demo_pz0_normal_2state_tmax8_fits.txt"
            )
            self.assertTrue(one_state.exists())
            self.assertTrue(two_state.exists())


if __name__ == "__main__":
    unittest.main()
