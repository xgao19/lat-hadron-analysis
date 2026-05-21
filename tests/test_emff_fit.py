from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from lqcd_analysis.emff.io import (
    EMFFNStateInput,
    expand_template,
    load_emff_c2pt_correlator,
    parse_emff_fit_input,
)
from lqcd_analysis.emff.models import (
    compute_tau_range_for_tsep,
    evaluate_emff_plateau,
    evaluate_emff_ratio_2state,
    evaluate_emff_summed_ratio,
)
from lqcd_analysis.emff.fit_nstate import (
    EMFFFitterResult,
    _build_fit_data_2state,
    _build_fit_data_summation,
    _build_q_groups,
    _compute_emff_ratio,
    _hadron_energy_from_dispersion,
    fit_emff_2state,
    fit_emff_plateau,
    fit_emff_summation,
    summarize_parameter_samples,
)


class TestExpandTemplate:
    def test_simple_substitution(self):
        result = expand_template("test_{a}_{b}", a=1, b="x")
        assert result == "test_1_x"

    def test_no_placeholders(self):
        result = expand_template("no_placeholders")
        assert result == "no_placeholders"


class TestLoadEmffC2ptCorrelator:
    def test_loads_h5_dataset_as_config_by_time_array(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        h5_path = tmp_path / "c2pt.src5.h5"
        data = (
            np.arange(12).reshape(4, 3)
            + 1j * np.arange(12, 24).reshape(4, 3)
        )
        with h5py.File(h5_path, "w") as handle:
            handle.create_dataset("SS/5/PX0PY0PZ1", data=data)

        times, correlators = load_emff_c2pt_correlator(
            h5_path,
            sink_gamma="5",
            px=0,
            py=0,
            pz=1,
        )

        np.testing.assert_array_equal(times, np.arange(4))
        np.testing.assert_allclose(correlators, data.T)


class TestComputeTauRange:
    def test_basic(self):
        tau_vals = compute_tau_range_for_tsep(8, tau_min=1, tau_offset=-1)
        np.testing.assert_array_equal(tau_vals, np.arange(1, 8))

    def test_tau_offset_minus_2(self):
        tau_vals = compute_tau_range_for_tsep(10, tau_min=2, tau_offset=-2)
        np.testing.assert_array_equal(tau_vals, np.arange(2, 9))

    def test_positive_tau_offset(self):
        tau_vals = compute_tau_range_for_tsep(6, tau_min=0, tau_offset=5)
        np.testing.assert_array_equal(tau_vals, np.arange(0, 6))


class TestModels:
    def test_ratio_2state_1state(self):
        tsep = np.array([8.0, 8.0, 10.0, 10.0])
        tau = np.array([2.0, 4.0, 3.0, 5.0])
        delta_e = 0.5
        r1 = 0.1
        params = np.array([0.85])  # just M_00

        result = evaluate_emff_ratio_2state(tsep, tau, delta_e, r1, params)
        assert result.shape == (4,)
        assert np.all(np.isfinite(result))
        # Should be near 0.85 / (1 + r1 * exp(-delta_E * tsep))
        expected_denom = 1.0 + r1 * np.exp(-delta_e * tsep)
        expected = 0.85 / expected_denom
        np.testing.assert_allclose(result, expected)

    def test_ratio_2state_2state(self):
        tsep = np.array([8.0, 10.0])
        tau = np.array([3.0, 4.0])
        delta_e = 0.5
        r1 = 0.1
        params = np.array([0.85, 0.02, 0.01, 0.005])

        result = evaluate_emff_ratio_2state(tsep, tau, delta_e, r1, params)
        assert result.shape == (2,)
        assert np.all(np.isfinite(result))

        # Verify result contains excited-state contributions
        flat_result = evaluate_emff_ratio_2state(tsep, tau, delta_e, 0.0, params)
        # With r1=0, denominator = 1, so numerator-only
        expected_num = (
            0.85
            + 0.02 * np.exp(-delta_e * tau)
            + 0.01 * np.exp(-delta_e * (tsep - tau))
            + 0.005 * np.exp(-delta_e * tsep)
        )
        np.testing.assert_allclose(flat_result, expected_num)

    def test_ratio_2state_invalid_params(self):
        with pytest.raises(ValueError, match="params must have 1 or 4"):
            evaluate_emff_ratio_2state(
                np.array([8.0]), np.array([3.0]),
                0.5, 0.1, np.array([1.0, 2.0, 3.0]),
            )

    def test_summation_no_intercept(self):
        tsep = np.array([4.0, 6.0, 8.0, 10.0])
        params = np.array([0.85])
        result = evaluate_emff_summed_ratio(tsep, params)
        assert result.shape == (4,)
        np.testing.assert_allclose(result, tsep * 0.85)

    def test_summation_with_intercept(self):
        tsep = np.array([4.0, 6.0, 8.0])
        params = np.array([0.85, -0.5])
        result = evaluate_emff_summed_ratio(tsep, params)
        np.testing.assert_allclose(result, tsep * 0.85 - 0.5)

    def test_plateau(self):
        params = np.array([0.78])
        result = evaluate_emff_plateau(5, params)
        assert result.shape == (5,)
        np.testing.assert_allclose(result, 0.78)

    def test_plateau_invalid(self):
        with pytest.raises(ValueError, match="params must have exactly 1"):
            evaluate_emff_plateau(3, np.array([1.0, 2.0]))


class TestSummarizeParameterSamples:
    def test_basic(self):
        samples = np.array([
            [0.8, 0.02],
            [0.9, 0.03],
            [0.85, 0.025],
            [0.87, 0.028],
        ])
        means, errors = summarize_parameter_samples(samples)
        assert len(means) == 2
        assert len(errors) == 2
        assert means[0] > 0
        assert errors[0] > 0

    def test_with_nans(self):
        samples = np.array([
            [0.8, np.nan],
            [0.9, 0.03],
            [np.nan, 0.025],
        ])
        means, errors = summarize_parameter_samples(samples)
        assert np.isfinite(means[0])
        assert np.isfinite(means[1])


class TestBuildFitData:
    def test_2state(self):
        # Create mock ratio data: 2 tsep values, 5 bootstrap samples
        ratio_by_tsep = {
            6: np.random.randn(5, 7).astype(complex),  # 5 boot, 7 tau (0..6)
            8: np.random.randn(5, 9).astype(complex),  # 5 boot, 9 tau (0..8)
        }
        tsep_fit_list = [6, 8]
        tau_range = (1, -1)

        tsep_arr, tau_arr, data = _build_fit_data_2state(
            ratio_by_tsep, tau_range, tsep_fit_list
        )

        # tsep=6: tau=1..5 (5 values), tsep=8: tau=1..7 (7 values)
        assert len(tsep_arr) == 5 + 7
        assert len(tau_arr) == 5 + 7
        assert data.shape == (5, 12)

        # Check tsep values
        np.testing.assert_array_equal(tsep_arr[:5], np.full(5, 6))
        np.testing.assert_array_equal(tsep_arr[5:], np.full(7, 8))

    def test_summation(self):
        ratio_by_tsep = {
            4: np.full((3, 5), 0.5 + 0j),  # 3 boot, 5 tau
            6: np.full((3, 7), 0.5 + 0j),
        }
        tsep_fit_list = [4, 6]
        tau_range = (1, -1)

        tsep_arr, summed = _build_fit_data_summation(
            ratio_by_tsep, tau_range, tsep_fit_list
        )

        assert len(tsep_arr) == 2
        assert summed.shape == (3, 2)
        # tsep=4, tau=1..3: sum = 3 * 0.5 = 1.5
        # tsep=6, tau=1..5: sum = 5 * 0.5 = 2.5
        np.testing.assert_allclose(summed[:, 0], 1.5)
        np.testing.assert_allclose(summed[:, 1], 2.5)


class TestEmffRatio:
    def test_hadron_energy_uses_input_mass_dispersion(self):
        energy = _hadron_energy_from_dispersion(
            (1, 0, 0),
            ns=48,
            lattice_spacing_fm=0.060,
            hadron_mass_gev=0.300,
        )
        momentum_unit = 2.0 * np.pi * 0.1973269804 / (48 * 0.060)
        expected = np.sqrt(0.300**2 + momentum_unit**2)
        np.testing.assert_allclose(energy, expected)

    def test_compute_emff_ratio_matches_eq_6(self):
        c3pt_tau = np.array([
            [6.0 + 0.0j, 12.0 + 0.0j],
            [8.0 + 0.0j, 16.0 + 0.0j],
            [10.0 + 0.0j, 20.0 + 0.0j],
        ])
        c2pt_initial = np.array([
            [2.0, 3.0, 5.0],
            [4.0, 6.0, 10.0],
        ], dtype=complex)
        c2pt_final = np.array([
            [7.0, 11.0, 13.0],
            [14.0, 22.0, 26.0],
        ], dtype=complex)

        ratio = _compute_emff_ratio(
            c3pt_tau,
            c2pt_initial,
            c2pt_final,
            tsep=2,
            energy_initial=2.0,
            energy_final=8.0,
        )

        energy_factor = 2.0 * np.sqrt(8.0 * 2.0) / (8.0 + 2.0)
        expected = []
        for cfg in range(2):
            row = []
            for tau in range(3):
                sqrt_factor = (
                    c2pt_final[cfg, 2 - tau]
                    * c2pt_initial[cfg, tau]
                    * c2pt_initial[cfg, 2]
                    / (
                        c2pt_initial[cfg, 2 - tau]
                        * c2pt_final[cfg, tau]
                        * c2pt_final[cfg, 2]
                    )
                )
                row.append(
                    energy_factor
                    * c3pt_tau[tau, cfg]
                    / c2pt_initial[cfg, 2]
                    * np.sqrt(sqrt_factor)
                )
            expected.append(row)
        np.testing.assert_allclose(ratio, np.array(expected))

    def test_transverse_orbit_groups(self):
        groups = _build_q_groups(
            (-2, -1, 0, 1, 2),
            (-2, -1, 0, 1, 2),
            (0,),
            average_transverse_orbits=True,
            final_momentum=(0, 0, 0),
        )
        assert len(groups) == 6
        assert groups[(2, 1, 0)] == (
            (-2, -1, 0),
            (-2, 1, 0),
            (-1, -2, 0),
            (-1, 2, 0),
            (1, -2, 0),
            (1, 2, 0),
            (2, -1, 0),
            (2, 1, 0),
        )

    def test_transverse_orbit_requires_no_final_transverse_momentum(self):
        with pytest.raises(ValueError, match="no transverse component"):
            _build_q_groups(
                (-1, 0, 1),
                (-1, 0, 1),
                (0,),
                average_transverse_orbits=True,
                final_momentum=(1, 0, 0),
            )


class TestFits:
    def test_2state_fit(self):
        """Synthetic data: R(tsep,tau) = M_00 with small noise."""
        np.random.seed(42)
        m00_true = 0.78
        delta_e = 0.4
        r1 = 0.09

        ratio_by_tsep = {}
        for tsep in [6, 8, 10]:
            n_tau = tsep + 1
            boot_samples = np.zeros((100, n_tau), dtype=complex)
            for tau_idx in range(n_tau):
                denom = 1.0 + r1 * np.exp(-delta_e * float(tsep))
                true_val = m00_true / denom
                boot_samples[:, tau_idx] = (
                    true_val + 0.001 * np.random.randn(100)
                    + 1j * 0.001 * np.random.randn(100)
                )
            ratio_by_tsep[tsep] = boot_samples

        fit_result, sample_params = fit_emff_2state(
            ratio_by_tsep, delta_e, r1,
            tau_range=(1, -1), tsep_fit_list=[6, 8, 10], nstates=1,
        )

        assert fit_result.success
        assert fit_result.params.shape == (1,)
        assert np.abs(fit_result.params[0] - m00_true) < 0.01
        assert fit_result.chi2_dof < 5.0

    def test_summation_fit(self):
        """Synthetic data: S(tsep) = tsep * M_00 + B."""
        np.random.seed(123)
        m00_true = 0.65
        b_true = 0.3

        ratio_by_tsep = {}
        for tsep in [4, 6, 8, 10]:
            n_tau = tsep + 1
            # Each tau value: (tsep * M_00 + B) / n_tau_on_average
            n_tau_used = tsep - 1  # tau from 1 to tsep-1
            per_tau_val = (tsep * m00_true + b_true) / n_tau_used
            boot_samples = np.full((100, n_tau), per_tau_val + 0j, dtype=complex)
            boot_samples += 0.002 * (np.random.randn(100, n_tau) + 1j * np.random.randn(100, n_tau))
            ratio_by_tsep[tsep] = boot_samples

        fit_result, sample_params = fit_emff_summation(
            ratio_by_tsep, tau_range=(1, -1),
            tsep_fit_list=[4, 6, 8, 10], fit_intercept=True,
        )

        assert fit_result.success
        assert fit_result.params.shape == (2,)
        assert np.abs(fit_result.params[0] - m00_true) < 0.05

    def test_plateau_fit(self):
        """Synthetic data: constant value in plateau."""
        np.random.seed(456)
        m00_true = 0.72

        ratio_by_tsep = {}
        for tsep in [8, 10, 12]:
            n_tau = tsep + 1
            boot_samples = np.full((100, n_tau), m00_true + 0j, dtype=complex)
            boot_samples += 0.001 * (np.random.randn(100, n_tau) + 1j * np.random.randn(100, n_tau))
            ratio_by_tsep[tsep] = boot_samples

        fit_result, sample_params = fit_emff_plateau(
            ratio_by_tsep, tau_range=(2, -2), tsep_fit_list=[8, 10, 12],
        )

        assert fit_result.success
        assert fit_result.params.shape == (1,)
        assert np.abs(fit_result.params[0] - m00_true) < 0.01


class TestParseEmffFitInput:
    def make_control_file(self, **overrides: str) -> Path:
        defaults = {
            "first_line": "test_EMFF 48 64 0.060",
            "hadron_mass_gev": "0.140",
            "c2pt": "/data/c2pt.src{src_gamma}.h5",
            "c3pt_h5": "/data/*src{src_gamma}.*PX{pfx}PY{pfy}PZ{pfz}dt{tsep}.h5",
            "c3pt_dataset_path": "SS/{insert_gamma}/PX{qx}PY{qy}PZ{qz}",
            "src_gamma": "5",
            "sink_gamma": "5",
            "insert_gamma": "T",
            "nstates": "2",
            "pflist": "0 0 0",
            "qxlist": "-2 -1 0 1 2",
            "qylist": "-2 -1 0 1 2",
            "qzlist": "0",
            "tslist": "4 6 8 10 12",
            "fit_method": "2state",
            "tau_range": "1 -1",
            "tsep_range": "4 12",
            "two_point_fit_root": "/data/2pt_results",
            "two_point_fit_window_by_pz": "",
        }

        # Create a temp window file for two_point_fit_window_by_pz
        tmpdir = Path(tempfile.mkdtemp(prefix="test_emff_"))
        window_path = tmpdir / "window.txt"
        window_path.write_text("0 8 20\n")
        defaults["two_point_fit_window_by_pz"] = str(window_path)

        params = {**defaults, **overrides}
        first_line = params.pop("first_line")

        lines = [first_line]
        for key, value in params.items():
            lines.append(f"{key} {value}")

        ctrl_path = tmpdir / "ctrl_emff.txt"
        ctrl_path.write_text("\n".join(lines))
        return ctrl_path

    def test_basic_parsing(self):
        ctrl_path = self.make_control_file()
        spec = parse_emff_fit_input(ctrl_path)

        assert spec.title_pattern == "test_EMFF"
        assert spec.ns == 48
        assert spec.nt == 64
        assert spec.lattice_spacing_fm == 0.060
        assert spec.hadron_mass_gev == 0.140
        assert spec.src_gamma == "5"
        assert spec.sink_gamma == "5"
        assert spec.insert_gamma == "T"
        assert spec.nstates == (2,)
        assert spec.pflist == (0, 0, 0)
        assert spec.qxlist == (-2, -1, 0, 1, 2)
        assert spec.qylist == (-2, -1, 0, 1, 2)
        assert spec.qzlist == (0,)
        assert spec.tslist == (4, 6, 8, 10, 12)
        assert spec.average_transverse_orbits is True
        assert spec.fit_method == "2state"
        assert spec.tau_range == (1, -1)
        assert spec.tsep_range == (4, 12)

    def test_q_range(self):
        # Create control file with range keys instead of list keys
        tmpdir = Path(tempfile.mkdtemp(prefix="test_emff_"))
        window_path = tmpdir / "window.txt"
        window_path.write_text("0 8 20\n")
        ctrl_path = tmpdir / "ctrl_emff.txt"
        ctrl_path.write_text(
            "test_EMFF 48 64 0.060\n"
            "hadron_mass_gev 0.140\n"
            "c2pt /data/c2pt.src{src_gamma}.h5\n"
            "c3pt_h5 /data/file.h5\n"
            "c3pt_dataset_path SS/{insert_gamma}/PX{qx}PY{qy}PZ{qz}\n"
            "src_gamma 5\n"
            "sink_gamma 5\n"
            "insert_gamma T\n"
            "nstates 1 2\n"
            "pflist 0 0 0\n"
            "qxrange -1 1\n"
            "qyrange -1 1\n"
            "qzrange 0 2\n"
            "tslist 4 6 8\n"
            "fit_method summation\n"
            "tau_range 1 -1\n"
            "tsep_range 4 8\n"
            "two_point_fit_root /data/2pt\n"
            f"two_point_fit_window_by_pz {window_path}\n"
        )
        spec = parse_emff_fit_input(ctrl_path)
        assert spec.qxlist == (-1, 0, 1)
        assert spec.qylist == (-1, 0, 1)
        assert spec.qzlist == (0, 1, 2)
        assert spec.nstates == (1, 2)
        assert spec.fit_method == "summation"

    def test_invalid_fit_method(self):
        ctrl_path = self.make_control_file(fit_method="invalid")
        with pytest.raises(ValueError, match="fit_method must be"):
            parse_emff_fit_input(ctrl_path)

    def test_invalid_nstates(self):
        ctrl_path = self.make_control_file(nstates="3")
        with pytest.raises(ValueError, match="nstates must contain only"):
            parse_emff_fit_input(ctrl_path)

    def test_missing_required_key(self):
        ctrl_path = self.make_control_file()
        # Remove c2pt key by rewriting
        tmpdir = Path(tempfile.mkdtemp(prefix="test_emff_"))
        window_path = tmpdir / "window.txt"
        window_path.write_text("0 8 20\n")
        ctrl_path = tmpdir / "ctrl_emff.txt"
        ctrl_path.write_text(
            "test_EMFF 48 64 0.060\n"
            "hadron_mass_gev 0.140\n"
            "c3pt_h5 /data/file.h5\n"
            "c3pt_dataset_path SS/T/PX0PY0PZ0\n"
            "src_gamma 5\n"
            "sink_gamma 5\n"
            "insert_gamma T\n"
            "nstates 2\n"
            "pflist 0 0 0\n"
            "qxlist 0\nqylist 0\nqzlist 0\n"
            "tslist 4 6 8\n"
            "fit_method 2state\n"
            "tau_range 1 -1\n"
            "tsep_range 4 8\n"
            "two_point_fit_root /data/2pt\n"
            f"two_point_fit_window_by_pz {window_path}\n"
        )
        with pytest.raises(ValueError, match="missing required keys"):
            parse_emff_fit_input(ctrl_path)


class TestNotebookWorkflowFunctions:
    def test_emff_keys_exist(self):
        from lqcd_analysis.notebook_workflows import (
            EMFF_INPUT_KEYS,
            EMFF_RUN_KEYS,
        )
        assert "title_pattern" in EMFF_INPUT_KEYS
        assert "hadron_mass_gev" in EMFF_INPUT_KEYS
        assert "fit_method" in EMFF_INPUT_KEYS
        assert "tslist" in EMFF_INPUT_KEYS
        assert "results_dir" in EMFF_INPUT_KEYS
        assert "results_dir" in EMFF_RUN_KEYS

    def test_render_emff_input_text(self):
        from lqcd_analysis.notebook_workflows import render_emff_fit_input_text

        config = {
            "title_pattern": "test",
            "ns": 48,
            "nt": 64,
            "lattice_spacing_fm": 0.060,
            "hadron_mass_gev": 0.140,
            "src_gamma": "5",
            "sink_gamma": "5",
            "insert_gamma": "T",
            "nstates": 2,
            "c2pt": "/data/c2pt.h5",
            "c3pt_h5": "/data/file.h5",
            "c3pt_dataset_path": "SS/T/PX0PY0PZ0",
            "pflist": [0, 0, 0],
            "qxlist": [0],
            "qylist": [0],
            "qzlist": [0],
            "tslist": [4, 6, 8],
            "fit_method": "2state",
            "tau_range": [1, -1],
            "tsep_range": [4, 8],
            "two_point_fit_root": "/data/2pt",
            "two_point_fit_window_by_pz": "/data/window.txt",
            "results_dir": "/tmp/test_emff",
        }

        text = render_emff_fit_input_text(config)
        assert "test 48 64 0.06" in text
        assert "hadron_mass_gev 0.14" in text
        assert "src_gamma 5" in text
        assert "average_transverse_orbits true" in text
        assert "fit_method 2state" in text
        assert "tau_range 1 -1" in text
        assert "results_dir /tmp/test_emff" in text

    def test_render_missing_required(self):
        from lqcd_analysis.notebook_workflows import render_emff_fit_input_text

        with pytest.raises(ValueError, match="missing EMFF notebook config"):
            render_emff_fit_input_text({"title_pattern": "test"})


class TestCLIIntegration:
    def test_emff_subcommand_registered(self):
        from lqcd_analysis.cli import build_parser

        parser = build_parser()
        # Verify the subcommand exists
        subcommands = [
            action.dest
            for action in parser._actions
            if hasattr(action, "dest") and action.dest == "command"
        ]
        assert len(subcommands) > 0
