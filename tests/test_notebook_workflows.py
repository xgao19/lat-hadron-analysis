import json
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from lqcd_analysis.notebook_workflows import (
    _guess_notebook_dir,
    render_effective_mass_input_text,
    render_nstate_fit_input_text,
    render_da_fourier_input_text,
    render_da_ratio_fourier_t_input_text,
    render_da_fit_input_text,
    render_da_normalize_input_text,
    render_da_x_nstate_fit_input_text,
    render_da_xfit_normalize_input_text,
    render_tgevp_input_text,
    run_effective_mass_from_notebook,
    run_nstate_fit_from_notebook,
    run_da_fourier_from_notebook,
    run_da_ratio_fourier_t_from_notebook,
    run_da_fit_from_notebook,
    run_da_normalize_from_notebook,
    run_da_x_nstate_fit_from_notebook,
    run_da_xfit_normalize_from_notebook,
    validate_effective_mass_notebook_config,
    run_tgevp_from_notebook,
    validate_nstate_notebook_config,
    validate_plot_2pt_notebook_config,
    validate_da_fourier_notebook_config,
    validate_da_ratio_fourier_t_notebook_config,
    validate_da_notebook_config,
    validate_da_normalize_notebook_config,
    validate_da_x_nstate_fit_notebook_config,
    validate_da_xfit_normalize_notebook_config,
)
from lqcd_analysis.DA.fit_nstate import parse_da_fit_input
from lqcd_analysis.two_point.fit_nstate import parse_nstate_fit_input
from lqcd_analysis.two_point.tgevp import parse_tgevp_input


class NotebookWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tgevp_config = {
            "title_pattern": "demo_pz*",
            "ns": 64,
            "nt": 64,
            "lattice_spacing_fm": 0.076,
            "c2pt": "/tmp/c2pt.csv",
            "pzlist": [0],
            "fold_t": "periodic",
            "tsrange": [0, 20],
        }
        self.nstate_config = {
            "title_pattern": "demo_pz*",
            "ns": 64,
            "nt": 64,
            "lattice_spacing_fm": 0.076,
            "c2pt": "/tmp/c2pt.csv",
            "pzlist": [0],
            "fold_t": "none",
            "model": "normal",
            "fit_mode": "uncorrelated",
            "fix_ground_energy_from_dispersion": False,
            "nstates": 1,
            "tmin_window": {0: [0, 4]},
            "tmax": {0: 12},
        }
        self.da_config = {
            "title_pattern": "demo_pz*",
            "ns": 64,
            "nt": 64,
            "lattice_spacing_fm": 0.076,
            "fit_target": "ratio",
            "fit_component": "both",
            "nstates": [1, 2],
            "pzlist": [0],
            "gmlist": ["T5"],
            "etalist": ["eta0"],
            "Tdirlist": ["plus", "minus"],
            "bTlist": [0],
            "bzlist": [0],
            "binsize": 1,
            "bootstrap_samples": 16,
            "bootstrap_size": 16,
            "seed": 2026,
            "fit_window": {0: [3, 6]},
            "qda_h5": "/tmp/qda_pz*.h5",
            "dataset_path_template": "{gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
            "two_point_fit_root": "/tmp/two_point_fit_root",
            "two_point_fit_window_by_pz": {0: [3, 6]},
            "c2pt": "/tmp/c2pt.csv",
            "fold_t": "periodic",
            "tsrange": [0, 20],
        }
        self.da_normalize_config = {
            "title_pattern": "demo_pz*",
            "input_root": "/tmp/da_fit",
            "ns": 64,
            "lattice_spacing_fm": 0.076,
            "pzlist": [0],
            "gmlist": ["T5"],
            "etalist": ["eta0"],
            "bTlist": [0],
            "bzlist": [0],
            "component": "real",
            "nstates": 1,
            "normalization_mode": "mode1",
        }
        self.da_ratio_fourier_t_config = {
            "title_pattern": "demo_pz*",
            "ns": 64,
            "nt": 64,
            "lattice_spacing_fm": 0.076,
            "pzlist": [0],
            "gmlist": ["T5"],
            "etalist": ["eta0"],
            "Tdirlist": ["plus", "minus"],
            "bTlist": [0],
            "bzlist": [0, 2],
            "component": "real",
            "qda_h5": "/tmp/qda_pz*.h5",
            "dataset_path_template": "{gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
            "c2pt": "/tmp/c2pt.csv",
            "fold_t": "none",
            "tsrange": [0, 12],
            "x_range": [-0.5, 1.5],
            "x_count": 11,
        }
        self.da_x_nstate_fit_config = {
            "title_pattern": "demo_pz*",
            "input_root": "/tmp/da_ratio_fourier_t",
            "ns": 64,
            "nt": 64,
            "lattice_spacing_fm": 0.076,
            "pzlist": [0],
            "gmlist": ["T5"],
            "etalist": ["eta0"],
            "bTlist": [0],
            "component": "real",
            "nstates": [1, 2],
            "fit_window": {0: [3, 6]},
            "two_point_fit_root": "/tmp/two_point_fit_root",
            "two_point_fit_window_by_pz": {0: [3, 6]},
        }
        self.da_xfit_normalize_config = {
            "title_pattern": "demo_pz*",
            "input_root": "/tmp/da_xfit",
            "bare_matrix_root": "/tmp/da_fit",
            "ns": 64,
            "lattice_spacing_fm": 0.076,
            "pzlist": [0],
            "gmlist": ["T5"],
            "etalist": ["eta0"],
            "bTlist": [0],
            "bzlist": [0, 2],
            "component": "real",
            "nstates": 1,
            "normalization_mode": "mode1",
        }

    def test_render_tgevp_text(self) -> None:
        text = render_tgevp_input_text(
            {
                "title_pattern": "demo_pz*",
                "ns": 64,
                "nt": 64,
                "lattice_spacing_fm": 0.076,
                "c2pt": "/tmp/c2pt.csv",
                "pzlist": [0, 1],
                "fold_t": "periodic",
                "tsrange": [0, 20],
                "binsize": 2,
                "seed": 2026,
            }
        )
        self.assertIn("fold_t periodic", text)
        self.assertIn("pzlist 0 1", text)
        self.assertIn("binsize 2", text)
        self.assertIn("seed 2026", text)

    def test_render_effective_mass_text(self) -> None:
        text = render_effective_mass_input_text(
            {
                "title_pattern": "demo_pz*",
                "ns": 64,
                "nt": 64,
                "lattice_spacing_fm": 0.076,
                "c2pt": "/tmp/c2pt.csv",
                "pzlist": [0, 1],
                "fold_t": "periodic",
                "tsrange": [0, 20],
                "model": "symmetric",
                "binsize": 2,
                "bootstrap_samples": 32,
                "bootstrap_size": 32,
                "seed": 2026,
                "results_dir": "/tmp/effective_mass",
            }
        )
        self.assertTrue(text.startswith("demo_pz* 64 64 0.076\n"))
        self.assertIn("c2pt /tmp/c2pt.csv", text)
        self.assertIn("pzlist 0 1", text)
        self.assertIn("model symmetric", text)
        self.assertIn("results_dir /tmp/effective_mass", text)

    def test_validate_effective_mass_notebook_config(self) -> None:
        parsed = validate_effective_mass_notebook_config(
            {
                "title_pattern": "demo_pz*",
                "ns": 64,
                "nt": 64,
                "lattice_spacing_fm": 0.076,
                "c2pt": "/tmp/c2pt.csv",
                "pzlist": [0, 1],
                "fold_t": "periodic",
                "tsrange": [0, 20],
                "model": "symmetric",
            }
        )
        self.assertEqual(parsed.title_pattern, "demo_pz*")
        self.assertEqual(parsed.pzlist, (0, 1))

    def test_run_effective_mass_from_notebook_dispatches(self) -> None:
        fake_outputs = [Path("/tmp/meff.txt")]
        with patch("lqcd_analysis.notebook_workflows.run_effective_mass_workflow", return_value=fake_outputs) as mock_run:
            outputs = run_effective_mass_from_notebook(
                {
                    "title_pattern": "demo_pz*",
                    "ns": 64,
                    "nt": 64,
                    "lattice_spacing_fm": 0.076,
                    "c2pt": "/tmp/c2pt.csv",
                    "pzlist": [0],
                    "fold_t": "periodic",
                    "tsrange": [0, 20],
                    "model": "symmetric",
                    "results_dir": "/tmp/effective_mass",
                }
            )
        self.assertEqual(outputs, fake_outputs)
        self.assertEqual(Path(mock_run.call_args.kwargs["results_dir"]), Path("/tmp/effective_mass"))

    def test_render_nstate_text(self) -> None:
        text = render_nstate_fit_input_text(
            {
                "title_pattern": "demo_pz*",
                "ns": 64,
                "nt": 64,
                "lattice_spacing_fm": 0.076,
                "c2pt": "/tmp/c2pt.csv",
                "pzlist": [0, 1],
                "fold_t": "none",
                "model": "normal",
                "fit_mode": "correlated",
                "pz0_ground_energy": 0.42,
                "fix_ground_energy_from_dispersion": False,
                "nstates": 2,
                "tmin_window": {0: [0, 4]},
                "tmax": {0: 12},
                "low_state_prior_tmin": {0: 3, 1: 4},
                "lambda_prior": 0.5,
                "plot": True,
                "results_dir": "examples/outputs/demo",
            }
        )
        self.assertIn("model normal", text)
        self.assertIn("fit_mode correlated", text)
        self.assertIn("pz0_ground_energy 0.42", text)
        self.assertIn("fix_ground_energy_from_dispersion false", text)
        self.assertIn("nstates 2", text)
        self.assertIn("low_state_prior_tmin ", text)
        match = re.search(r"tmin_window (.+)", text)
        self.assertIsNotNone(match)
        override_path = Path(match.group(1).strip())
        self.assertEqual(override_path.read_text(encoding="utf-8"), "0 0 4\n")
        match = re.search(r"tmax (.+)", text)
        self.assertIsNotNone(match)
        tmax_path = Path(match.group(1).strip())
        self.assertEqual(tmax_path.read_text(encoding="utf-8"), "0 12\n")
        match = re.search(r"low_state_prior_tmin (.+)", text)
        self.assertIsNotNone(match)
        prior_path = Path(match.group(1).strip())
        self.assertEqual(prior_path.read_text(encoding="utf-8"), "0 3\n1 4\n")
        self.assertIn("lambda_prior 0.5", text)
        self.assertIn("results_dir examples/outputs/demo", text)

    def test_render_nstate_text_without_tsrange(self) -> None:
        config = dict(self.nstate_config)
        text = render_nstate_fit_input_text(config)
        self.assertNotIn("tsrange", text)

    def test_validate_nstate_notebook_config_supports_tmin_window_dict(self) -> None:
        parsed = validate_nstate_notebook_config(self.nstate_config)
        override_path = Path(parsed.tmin_window)
        self.assertTrue(override_path.exists())
        self.assertEqual(override_path.read_text(encoding="utf-8"), "0 0 4\n")
        tmax_path = Path(parsed.tmax)
        self.assertTrue(tmax_path.exists())
        self.assertEqual(tmax_path.read_text(encoding="utf-8"), "0 12\n")
        parsed = parse_nstate_fit_input(
            Path("templates/input_files/two_point/nstate_2.txt")
        )
        self.assertEqual(parsed.nt, 64)

    def test_validate_plot_config(self) -> None:
        config = {
            "output_dir": "examples/outputs/plot_2pt_notebook",
            "correlator_table": "corr.txt",
            "meff_table": "meff.txt",
            "fit_table": "fits.txt",
            "nstates": 2,
            "model": "normal",
            "title": "demo",
            "nt": 64,
            "lattice_spacing_fm": 0.076,
            "ignored": "extra",
        }
        validated = validate_plot_2pt_notebook_config(config)
        self.assertNotIn("ignored", validated)
        self.assertEqual(validated["nstates"], 2)

    def test_render_da_text(self) -> None:
        text = render_da_fit_input_text(
            {
                **self.da_config,
                "results_dir": "examples/outputs/da_demo",
            }
        )
        self.assertIn("fit_target ratio", text)
        self.assertIn("fit_component both", text)
        self.assertIn("gmlist T5", text)
        self.assertIn("bTlist 0", text)
        self.assertIn("bzlist 0", text)
        match = re.search(r"fit_window (.+)", text)
        self.assertIsNotNone(match)
        override_path = Path(match.group(1).strip())
        self.assertTrue(override_path.exists())
        self.assertEqual(override_path.read_text(encoding="utf-8"), "0 3 6\n")
        self.assertIn("results_dir examples/outputs/da_demo", text)

    def test_render_da_text_supports_nested_gm_fit_window_dict(self) -> None:
        text = render_da_fit_input_text(
            {
                **self.da_config,
                "fit_window": {"T5": {0: [4, 7]}},
            }
        )
        match = re.search(r"fit_window (.+)", text)
        self.assertIsNotNone(match)
        override_path = Path(match.group(1).strip())
        self.assertEqual(override_path.read_text(encoding="utf-8"), "T5 0 4 7\n")

    def test_render_da_text_preserves_qda_gm_placeholder(self) -> None:
        config = dict(self.da_config)
        config["qda_h5"] = "/tmp/qda_pz{pz}_O{gm}.h5"
        text = render_da_fit_input_text(config)
        self.assertIn("qda_h5 /tmp/qda_pz{pz}_O{gm}.h5", text)

    def test_render_da_text_without_tsrange(self) -> None:
        config = dict(self.da_config)
        config.pop("tsrange")
        text = render_da_fit_input_text(config)
        self.assertNotIn("tsrange", text)

    def test_validate_da_notebook_config_supports_fit_window_dict(self) -> None:
        parsed = validate_da_notebook_config(self.da_config)
        override_path = Path(parsed.fit_window)
        self.assertTrue(override_path.exists())
        self.assertEqual(override_path.read_text(encoding="utf-8"), "0 3 6\n")

    def test_validate_da_notebook_config(self) -> None:
        parsed = validate_da_notebook_config(self.da_config)
        self.assertEqual(parsed.fit_target, "ratio")
        self.assertEqual(parsed.nstates, (1, 2))

    def test_validate_nstate_and_da_configs_without_tsrange(self) -> None:
        nstate_config = dict(self.nstate_config)
        parsed_nstate = validate_nstate_notebook_config(nstate_config)
        self.assertTrue(Path(parsed_nstate.tmin_window).exists())
        self.assertTrue(Path(parsed_nstate.tmax).exists())

        da_config = dict(self.da_config)
        da_config.pop("tsrange")
        parsed_da = validate_da_notebook_config(da_config)
        self.assertEqual(parsed_da.tsrange, (0, 31))

    def test_render_da_fourier_text(self) -> None:
        text = render_da_fourier_input_text(
            {
                "title_pattern": "demo_pz*",
                "input_root": "/tmp/da_fit",
                "ns": 64,
                "lattice_spacing_fm": 0.076,
                "pzlist": [0],
                "gmlist": ["T5"],
                "etalist": ["eta0"],
                "bTlist": [0],
                "component": "real",
                "nstates": 1,
                "normalization_mode": "raw",
                "x_range": [-0.5, 1.5],
                "x_count": 101,
                "zstep_fm": 0.01,
                "interpolation_kind": "cubic",
                "plot": True,
                "results_dir": "/tmp/fourier",
            }
        )
        self.assertIn("title_pattern demo_pz*", text)
        self.assertIn("input_root /tmp/da_fit", text)
        self.assertIn("pzlist 0", text)
        self.assertIn("gmlist T5", text)
        self.assertIn("bTlist 0", text)
        self.assertIn("normalization_mode raw", text)
        self.assertIn("x_range -0.5 1.5", text)
        self.assertIn("x_count 101", text)
        self.assertIn("interpolation_kind cubic", text)

    def test_validate_da_fourier_notebook_config(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fit_dir = tmp / "demo_pz1" / "tables"
            sample_dir = tmp / "demo_pz1" / "samples"
            fit_dir.mkdir(parents=True)
            sample_dir.mkdir(parents=True)
            (fit_dir / "demo_pz1_T5_eta0_bT0_real_2state_fit.txt").write_text("bz\tm0_mean\tm0_err\n0\t1.0\t0.1\n", encoding="utf-8")
            (sample_dir / "demo_pz1_T5_eta0_bT0_real_2state_samples.txt").write_text("bz\tsample_id\tsuccess\tm0\n0\t0\t1\t1.0\n", encoding="utf-8")
            validated = validate_da_fourier_notebook_config(
                {
                    "title_pattern": "demo_pz*",
                    "input_root": str(tmp),
                    "ns": 64,
                    "lattice_spacing_fm": 0.076,
                    "pzlist": [1],
                    "gmlist": ["T5"],
                    "etalist": ["eta0"],
                    "bTlist": [0],
                    "component": "real",
                    "nstates": 2,
                    "normalization_mode": "raw",
                    "x_range": [-0.25, 1.25],
                    "x_count": 51,
                    "zstep_fm": 0.02,
                }
            )
        self.assertEqual(validated["title_pattern"], "demo_pz*")
        self.assertEqual(validated["input_root"], tmp)
        self.assertEqual(validated["pzlist"], (1,))
        self.assertEqual(validated["bTlist"], (0,))
        self.assertEqual(validated["component"], "real")
        self.assertEqual(validated["normalization_mode"], "raw")
        self.assertEqual(validated["x_values"].shape, (51,))
        self.assertAlmostEqual(validated["zstep_fm"], 0.02)

    def test_render_da_ratio_fourier_t_text(self) -> None:
        text = render_da_ratio_fourier_t_input_text(
            {
                **self.da_ratio_fourier_t_config,
                "results_dir": "/tmp/qxt",
            }
        )
        self.assertIn("demo_pz* 64 64 0.076", text)
        self.assertIn("Tdirlist plus minus", text)
        self.assertIn("bzlist 0 2", text)
        self.assertIn("component real", text)
        self.assertIn("x_count 11", text)
        self.assertIn("results_dir /tmp/qxt", text)

    def test_validate_da_ratio_fourier_t_notebook_config(self) -> None:
        validated = validate_da_ratio_fourier_t_notebook_config(self.da_ratio_fourier_t_config)
        self.assertEqual(validated["title_pattern"], "demo_pz*")
        self.assertEqual(validated["pzlist"], (0,))
        self.assertEqual(validated["bTlist"], (0,))
        self.assertEqual(validated["bzlist"], (0, 2))
        self.assertEqual(validated["x_values"].shape, (11,))

    def test_render_da_x_nstate_fit_text(self) -> None:
        text = render_da_x_nstate_fit_input_text(
            {
                **self.da_x_nstate_fit_config,
                "results_dir": "/tmp/xfit",
            }
        )
        self.assertIn("title_pattern demo_pz*", text)
        self.assertIn("input_root /tmp/da_ratio_fourier_t", text)
        self.assertIn("nstates 1 2", text)
        self.assertIn("two_point_fit_root /tmp/two_point_fit_root", text)
        self.assertIn("results_dir /tmp/xfit", text)

    def test_validate_da_x_nstate_fit_notebook_config(self) -> None:
        validated = validate_da_x_nstate_fit_notebook_config(self.da_x_nstate_fit_config)
        self.assertEqual(validated["title_pattern"], "demo_pz*")
        self.assertEqual(validated["input_root"], Path("/tmp/da_ratio_fourier_t"))
        self.assertEqual(validated["nstates"], (1, 2))
        self.assertTrue(Path(validated["fit_window"]).exists())

    def test_render_da_xfit_normalize_text(self) -> None:
        text = render_da_xfit_normalize_input_text(
            {
                **self.da_xfit_normalize_config,
                "results_dir": "/tmp/xfit_norm",
            }
        )
        self.assertIn("input_root /tmp/da_xfit", text)
        self.assertIn("bare_matrix_root /tmp/da_fit", text)
        self.assertIn("normalization_mode mode1", text)
        self.assertIn("results_dir /tmp/xfit_norm", text)

    def test_validate_da_xfit_normalize_notebook_config(self) -> None:
        validated = validate_da_xfit_normalize_notebook_config(self.da_xfit_normalize_config)
        self.assertEqual(validated["bare_matrix_root"], Path("/tmp/da_fit"))
        self.assertEqual(validated["normalization_mode"], "mode1")
        self.assertEqual(validated["bzlist"], (0, 2))

    def test_da_fourier_notebook_config_rejects_old_single_job_shape(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            render_da_fourier_input_text(
                {
                    "pz": 0,
                    "ns": 64,
                    "lattice_spacing_fm": 0.076,
                    "component": "real",
                    "nstates": 1,
                    "bT": 0,
                    "fit_table": "/tmp/fit.txt",
                    "sample_table": "/tmp/samples.txt",
                }
            )
        self.assertIn("missing DA Fourier notebook config keys", str(ctx.exception))

    def test_render_da_normalize_text(self) -> None:
        text = render_da_normalize_input_text(
            {
                **self.da_normalize_config,
                "results_dir": "/tmp/da_norm",
            }
        )
        self.assertIn("title_pattern demo_pz*", text)
        self.assertIn("input_root /tmp/da_fit", text)
        self.assertIn("normalization_mode mode1", text)
        self.assertIn("results_dir /tmp/da_norm", text)

    def test_validate_da_normalize_notebook_config(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            target_root = tmp / "inputs"
            title = "demo_pz0"
            fit_dir = target_root / title / "tables"
            sample_dir = target_root / title / "samples"
            fit_dir.mkdir(parents=True)
            sample_dir.mkdir(parents=True)
            (fit_dir / "demo_pz0_T5_eta0_bT0_real_1state_fit.txt").write_text(
                "bz\tm0_mean\tm0_err\n0\t1.0\t0.1\n",
                encoding="utf-8",
            )
            (sample_dir / "demo_pz0_T5_eta0_bT0_real_1state_samples.txt").write_text(
                "bz\tsample_id\tsuccess\tm0\n0\t0\t1\t1.0\n",
                encoding="utf-8",
            )
            validated = validate_da_normalize_notebook_config(
                {
                    **self.da_normalize_config,
                    "input_root": str(target_root),
                }
            )
        self.assertEqual(validated["normalization_mode"], "mode1")
        self.assertEqual(validated["bTlist"], (0,))
        self.assertEqual(validated["bzlist"], (0,))

    def test_guess_notebook_dir_uses_vscode_notebook_path(self) -> None:
        shell = SimpleNamespace(user_ns={"__vsc_ipynb_file__": "/tmp/demo/notebook.ipynb"})
        fake_ipython = SimpleNamespace(get_ipython=lambda: shell)
        with patch.dict(sys.modules, {"IPython": fake_ipython}):
            self.assertEqual(_guess_notebook_dir(), Path("/tmp/demo").resolve())

    def test_guess_notebook_dir_falls_back_to_cwd(self) -> None:
        fake_ipython = SimpleNamespace(get_ipython=lambda: SimpleNamespace(user_ns={}))
        with patch.dict(sys.modules, {"IPython": fake_ipython}):
            with patch("lqcd_analysis.notebook_workflows.Path.cwd", return_value=Path("/tmp/fallback")):
                self.assertEqual(_guess_notebook_dir(), Path("/tmp/fallback").resolve())

    def test_explicit_results_dir_override_wins_for_notebook_runners(self) -> None:
        with patch("lqcd_analysis.notebook_workflows.run_ss_2pt_tgevp", return_value=[] ) as mock_tgevp:
            run_tgevp_from_notebook(self.tgevp_config, results_dir="/tmp/explicit_tgevp")
        self.assertEqual(Path(mock_tgevp.call_args.kwargs["results_dir"]), Path("/tmp/explicit_tgevp"))

        with patch("lqcd_analysis.notebook_workflows.run_nstate_fit", return_value=[] ) as mock_nstate:
            run_nstate_fit_from_notebook(self.nstate_config, results_dir="/tmp/explicit_nstate")
        self.assertEqual(Path(mock_nstate.call_args.kwargs["results_dir"]), Path("/tmp/explicit_nstate"))

        with patch("lqcd_analysis.notebook_workflows.run_da_nstate_fit", return_value=[] ) as mock_da:
            run_da_fit_from_notebook(self.da_config, results_dir="/tmp/explicit_da")
        self.assertEqual(Path(mock_da.call_args.kwargs["results_dir"]), Path("/tmp/explicit_da"))

        with patch("lqcd_analysis.notebook_workflows.run_da_fourier_workflow", return_value=[] ) as mock_fourier:
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                fit_dir = tmp / "demo_pz0" / "tables"
                sample_dir = tmp / "demo_pz0" / "samples"
                fit_dir.mkdir(parents=True)
                sample_dir.mkdir(parents=True)
                (fit_dir / "demo_pz0_T5_eta0_bT0_real_1state_fit.txt").write_text("bz\tm0_mean\tm0_err\n0\t1.0\t0.1\n", encoding="utf-8")
                (sample_dir / "demo_pz0_T5_eta0_bT0_real_1state_samples.txt").write_text("bz\tsample_id\tsuccess\tm0\n0\t0\t1\t1.0\n", encoding="utf-8")
                run_da_fourier_from_notebook(
                    {
                        "title_pattern": "demo_pz*",
                        "input_root": str(tmp),
                        "ns": 64,
                        "lattice_spacing_fm": 0.076,
                        "pzlist": [0],
                        "gmlist": ["T5"],
                        "etalist": ["eta0"],
                        "bTlist": [0],
                        "component": "real",
                        "nstates": 1,
                        "normalization_mode": "raw",
                    },
                    results_dir="/tmp/explicit_da_fourier",
                )
        self.assertEqual(Path(mock_fourier.call_args.kwargs["results_dir"]), Path("/tmp/explicit_da_fourier"))

        with patch("lqcd_analysis.notebook_workflows.run_da_normalization", return_value=[] ) as mock_normalize:
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                fit_dir = tmp / "demo_pz0" / "tables"
                sample_dir = tmp / "demo_pz0" / "samples"
                fit_dir.mkdir(parents=True)
                sample_dir.mkdir(parents=True)
                (fit_dir / "demo_pz0_T5_eta0_bT0_real_1state_fit.txt").write_text(
                    "bz\tm0_mean\tm0_err\n0\t1.0\t0.1\n",
                    encoding="utf-8",
                )
                (sample_dir / "demo_pz0_T5_eta0_bT0_real_1state_samples.txt").write_text(
                    "bz\tsample_id\tsuccess\tm0\n0\t0\t1\t1.0\n",
                    encoding="utf-8",
                )
                run_da_normalize_from_notebook(
                    {
                        **self.da_normalize_config,
                        "input_root": str(tmp),
                    },
                    results_dir="/tmp/explicit_da_normalize",
                )
        self.assertEqual(Path(mock_normalize.call_args.kwargs["results_dir"]), Path("/tmp/explicit_da_normalize"))

    def test_notebook_runners_use_vscode_notebook_dir_when_available(self) -> None:
        shell = SimpleNamespace(user_ns={"__vsc_ipynb_file__": "/tmp/vscode/session.ipynb"})
        fake_ipython = SimpleNamespace(get_ipython=lambda: shell)
        with patch.dict(sys.modules, {"IPython": fake_ipython}):
            with patch("lqcd_analysis.notebook_workflows.run_ss_2pt_tgevp", return_value=[] ) as mock_tgevp:
                run_tgevp_from_notebook(self.tgevp_config)
            self.assertEqual(Path(mock_tgevp.call_args.kwargs["results_dir"]), Path("/tmp/vscode").resolve())

        with patch.dict(sys.modules, {"IPython": fake_ipython}):
            with patch("lqcd_analysis.notebook_workflows.run_nstate_fit", return_value=[] ) as mock_nstate:
                run_nstate_fit_from_notebook(self.nstate_config)
            self.assertEqual(Path(mock_nstate.call_args.kwargs["results_dir"]), Path("/tmp/vscode").resolve())

        with patch.dict(sys.modules, {"IPython": fake_ipython}):
            with patch("lqcd_analysis.notebook_workflows.run_da_nstate_fit", return_value=[] ) as mock_da:
                run_da_fit_from_notebook(self.da_config)
            self.assertEqual(Path(mock_da.call_args.kwargs["results_dir"]), Path("/tmp/vscode").resolve())

        with patch.dict(sys.modules, {"IPython": fake_ipython}):
            with patch("lqcd_analysis.notebook_workflows.run_da_fourier_workflow", return_value=[] ) as mock_fourier:
                import tempfile

                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp = Path(tmpdir)
                    fit_dir = tmp / "demo_pz0" / "tables"
                    sample_dir = tmp / "demo_pz0" / "samples"
                    fit_dir.mkdir(parents=True)
                    sample_dir.mkdir(parents=True)
                    (fit_dir / "demo_pz0_T5_eta0_bT0_real_1state_fit.txt").write_text("bz\tm0_mean\tm0_err\n0\t1.0\t0.1\n", encoding="utf-8")
                    (sample_dir / "demo_pz0_T5_eta0_bT0_real_1state_samples.txt").write_text("bz\tsample_id\tsuccess\tm0\n0\t0\t1\t1.0\n", encoding="utf-8")
                    run_da_fourier_from_notebook(
                        {
                            "title_pattern": "demo_pz*",
                            "input_root": str(tmp),
                            "ns": 64,
                            "lattice_spacing_fm": 0.076,
                            "pzlist": [0],
                            "gmlist": ["T5"],
                            "etalist": ["eta0"],
                            "bTlist": [0],
                            "component": "real",
                            "nstates": 1,
                            "normalization_mode": "raw",
                        }
                    )
                self.assertEqual(Path(mock_fourier.call_args.kwargs["results_dir"]), Path("/tmp/vscode").resolve())

        with patch.dict(sys.modules, {"IPython": fake_ipython}):
            with patch("lqcd_analysis.notebook_workflows.run_da_normalization", return_value=[] ) as mock_normalize:
                import tempfile

                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp = Path(tmpdir)
                    fit_dir = tmp / "demo_pz0" / "tables"
                    sample_dir = tmp / "demo_pz0" / "samples"
                    fit_dir.mkdir(parents=True)
                    sample_dir.mkdir(parents=True)
                    (fit_dir / "demo_pz0_T5_eta0_bT0_real_1state_fit.txt").write_text(
                        "bz\tm0_mean\tm0_err\n0\t1.0\t0.1\n",
                        encoding="utf-8",
                    )
                    (sample_dir / "demo_pz0_T5_eta0_bT0_real_1state_samples.txt").write_text(
                        "bz\tsample_id\tsuccess\tm0\n0\t0\t1\t1.0\n",
                        encoding="utf-8",
                    )
                    run_da_normalize_from_notebook(
                        {
                            **self.da_normalize_config,
                            "input_root": str(tmp),
                        }
                    )
                self.assertEqual(Path(mock_normalize.call_args.kwargs["results_dir"]), Path("/tmp/vscode").resolve())

    def test_template_notebooks_exist_and_are_valid_json(self) -> None:
        for relative in (
            "templates/two_point/tgevp.ipynb",
            "templates/two_point/nstate_1.ipynb",
            "templates/two_point/nstate_2.ipynb",
            "templates/da/da_nstate_template.ipynb",
            "templates/da/da_fourier_template.ipynb",
            "templates/da/da_normalize_template.ipynb",
            "templates/da/da_ratio_fourier_t_template.ipynb",
            "templates/da/da_x_nstate_fit_template.ipynb",
            "templates/da/da_xfit_normalize_template.ipynb",
            "templates/two_point/plot_2pt.ipynb",
        ):
            path = Path(relative)
            self.assertTrue(path.exists(), path)
            notebook = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(notebook["nbformat"], 4)
            self.assertTrue(notebook["cells"])

    def test_da_fourier_template_contains_expected_workflow_hooks(self) -> None:
        path = Path("templates/da/da_fourier_template.ipynb")
        notebook = json.loads(path.read_text(encoding="utf-8"))
        joined = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("render_da_fourier_input_text", joined)
        self.assertIn("validate_da_fourier_notebook_config", joined)
        self.assertIn("run_da_fourier_from_notebook", joined)
        self.assertIn("\"input_root\":", joined)
        self.assertIn("\"title_pattern\":", joined)
        self.assertIn("\"pzlist\": [0]", joined)
        self.assertIn("\"bTlist\": [0]", joined)
        self.assertIn("\"normalization_mode\": \"raw\"", joined)
        self.assertIn("\"x_range\": [-0.5, 1.5]", joined)
        self.assertIn("# Data settings", joined)
        self.assertIn("# Fourier-transform parameter settings", joined)

    def test_da_normalize_template_contains_expected_workflow_hooks(self) -> None:
        path = Path("templates/da/da_normalize_template.ipynb")
        notebook = json.loads(path.read_text(encoding="utf-8"))
        joined = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("render_da_normalize_input_text", joined)
        self.assertIn("validate_da_normalize_notebook_config", joined)
        self.assertIn("run_da_normalize_from_notebook", joined)
        self.assertIn("\"normalization_mode\": \"mode1\"", joined)
        self.assertIn("# Data settings", joined)
        self.assertIn("# Normalization settings", joined)

    def test_da_ratio_fourier_t_template_contains_expected_workflow_hooks(self) -> None:
        path = Path("templates/da/da_ratio_fourier_t_template.ipynb")
        notebook = json.loads(path.read_text(encoding="utf-8"))
        joined = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("render_da_ratio_fourier_t_input_text", joined)
        self.assertIn("validate_da_ratio_fourier_t_notebook_config", joined)
        self.assertIn("run_da_ratio_fourier_t_from_notebook", joined)
        self.assertIn("\"bzlist\": [0, 2, 4, 6, 8, 10, 12]", joined)

    def test_da_x_nstate_fit_template_contains_expected_workflow_hooks(self) -> None:
        path = Path("templates/da/da_x_nstate_fit_template.ipynb")
        notebook = json.loads(path.read_text(encoding="utf-8"))
        joined = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("render_da_x_nstate_fit_input_text", joined)
        self.assertIn("validate_da_x_nstate_fit_notebook_config", joined)
        self.assertIn("run_da_x_nstate_fit_from_notebook", joined)
        self.assertIn("\"fit_window\": {0: [4, 10]}", joined)

    def test_da_xfit_normalize_template_contains_expected_workflow_hooks(self) -> None:
        path = Path("templates/da/da_xfit_normalize_template.ipynb")
        notebook = json.loads(path.read_text(encoding="utf-8"))
        joined = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("render_da_xfit_normalize_input_text", joined)
        self.assertIn("validate_da_xfit_normalize_notebook_config", joined)
        self.assertIn("run_da_xfit_normalize_from_notebook", joined)
        self.assertIn("\"bare_matrix_root\":", joined)
        self.assertIn("\"normalization_mode\": \"mode1\"", joined)

    def test_example_input_files_parse(self) -> None:
        tgevp_input = Path("templates/input_files/two_point/tgevp.txt")
        nstate_1_input = Path("templates/input_files/two_point/nstate_1.txt")
        nstate_2_input = Path("templates/input_files/two_point/nstate_2.txt")
        da_input = Path("templates/input_files/da/da_nstate_example.txt")
        da_fourier_input = Path("templates/input_files/da/da_fourier_example.txt")
        da_fourier_annotated = Path("templates/input_files/da/da_fourier_example_annotated.txt")
        da_normalize_input = Path("templates/input_files/da/da_normalize_example.txt")
        da_normalize_annotated = Path("templates/input_files/da/da_normalize_example_annotated.txt")
        da_ratio_fourier_t_input = Path("templates/input_files/da/da_ratio_fourier_t_example.txt")
        da_ratio_fourier_t_annotated = Path("templates/input_files/da/da_ratio_fourier_t_example_annotated.txt")
        da_x_nstate_fit_input = Path("templates/input_files/da/da_x_nstate_fit_example.txt")
        da_x_nstate_fit_annotated = Path("templates/input_files/da/da_x_nstate_fit_example_annotated.txt")
        da_xfit_normalize_input = Path("templates/input_files/da/da_xfit_normalize_example.txt")
        da_xfit_normalize_annotated = Path("templates/input_files/da/da_xfit_normalize_example_annotated.txt")
        self.assertTrue(tgevp_input.exists())
        self.assertTrue(nstate_1_input.exists())
        self.assertTrue(nstate_2_input.exists())
        self.assertTrue(da_input.exists())
        self.assertTrue(da_fourier_input.exists())
        self.assertTrue(da_fourier_annotated.exists())
        self.assertTrue(da_normalize_input.exists())
        self.assertTrue(da_normalize_annotated.exists())
        self.assertTrue(da_ratio_fourier_t_input.exists())
        self.assertTrue(da_ratio_fourier_t_annotated.exists())
        self.assertTrue(da_x_nstate_fit_input.exists())
        self.assertTrue(da_x_nstate_fit_annotated.exists())
        self.assertTrue(da_xfit_normalize_input.exists())
        self.assertTrue(da_xfit_normalize_annotated.exists())
        parsed_tgevp = parse_tgevp_input(tgevp_input)
        parsed_nstate = parse_nstate_fit_input(nstate_2_input)
        parsed_da = parse_da_fit_input(da_input)
        self.assertEqual(parsed_tgevp.pzlist, (0,))
        self.assertEqual(parsed_nstate.pzlist, (0,))
        self.assertEqual(parsed_da.pzlist, (0,))

    def test_example_data_files_exist(self) -> None:
        base = Path("examples/data/l64c64a076_m140/comb_c2pt_csv")
        files = sorted(base.glob("c2pt_5_5_k0_pz*_real.csv"))
        self.assertGreaterEqual(len(files), 1)


if __name__ == "__main__":
    unittest.main()
