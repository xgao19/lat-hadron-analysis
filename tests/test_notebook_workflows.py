import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from lqcd_analysis.notebook_workflows import (
    _guess_notebook_dir,
    render_nstate_fit_input_text,
    render_tmdwf_fit_input_text,
    render_tgevp_input_text,
    run_nstate_fit_from_notebook,
    run_tmdwf_fit_from_notebook,
    run_tgevp_from_notebook,
    validate_nstate_notebook_config,
    validate_plot_2pt_notebook_config,
    validate_tmdwf_notebook_config,
)
from lqcd_analysis.tmdwf.fit_nstate import parse_tmdwf_fit_input
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
            "tsrange": [0, 24],
            "model": "normal",
            "fit_mode": "uncorrelated",
            "nstates": [1, 2],
        }
        self.tmdwf_config = {
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
            "tmin": 2,
            "tmax": 12,
            "qtmdwf_h5": "/tmp/qtmdwf_pz*.h5",
            "dataset_path_template": "{gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
            "two_point_plateau_table": "/tmp/plateau_pz*.txt",
            "c2pt": "/tmp/c2pt.csv",
            "fold_t": "periodic",
            "tsrange": [0, 20],
            "shared_window_by_pz_gm": True,
            "decay_constant": [0.1, 0.02],
            "min_fit_dof": 2,
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
                "tsrange": [0, 24],
                "model": "normal",
                "fit_mode": "correlated",
                "pz0_ground_energy": 0.42,
                "nstates": [1, 2],
                "tmax": "auto",
                "lambda_prior": 0.5,
                "plot": True,
                "results_dir": "examples/outputs/demo",
            }
        )
        self.assertIn("model normal", text)
        self.assertIn("fit_mode correlated", text)
        self.assertIn("pz0_ground_energy 0.42", text)
        self.assertIn("nstates 1 2", text)
        self.assertIn("lambda_prior 0.5", text)
        self.assertIn("results_dir examples/outputs/demo", text)

    def test_render_nstate_text_without_tsrange(self) -> None:
        config = dict(self.nstate_config)
        config.pop("tsrange")
        text = render_nstate_fit_input_text(config)
        self.assertNotIn("tsrange", text)
        parsed = parse_nstate_fit_input(
            Path("templates/input_files/two_point/nstate_fit_example_realdata.txt")
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

    def test_render_tmdwf_text(self) -> None:
        text = render_tmdwf_fit_input_text(
            {
                **self.tmdwf_config,
                "results_dir": "examples/outputs/tmdwf_demo",
            }
        )
        self.assertIn("fit_target ratio", text)
        self.assertIn("fit_component both", text)
        self.assertIn("gmlist T5", text)
        self.assertIn("bTlist 0", text)
        self.assertIn("bzlist 0", text)
        self.assertIn("tmax 12", text)
        self.assertIn("shared_window_by_pz_gm true", text)
        self.assertIn("decay_constant 0.1 0.02", text)
        self.assertIn("min_fit_dof 2", text)
        self.assertIn("results_dir examples/outputs/tmdwf_demo", text)

    def test_render_tmdwf_text_preserves_qtmdwf_gm_placeholder(self) -> None:
        config = dict(self.tmdwf_config)
        config["qtmdwf_h5"] = "/tmp/qtmdwf_pz{pz}_O{gm}.h5"
        text = render_tmdwf_fit_input_text(config)
        self.assertIn("qtmdwf_h5 /tmp/qtmdwf_pz{pz}_O{gm}.h5", text)

    def test_render_tmdwf_text_without_tsrange(self) -> None:
        config = dict(self.tmdwf_config)
        config.pop("tsrange")
        text = render_tmdwf_fit_input_text(config)
        self.assertNotIn("tsrange", text)

    def test_render_tmdwf_text_without_tmax(self) -> None:
        config = dict(self.tmdwf_config)
        config.pop("tmax")
        text = render_tmdwf_fit_input_text(config)
        self.assertNotIn("tmax", text)

    def test_render_tmdwf_text_with_auto_tmax(self) -> None:
        config = dict(self.tmdwf_config)
        config["tmax"] = "auto"
        text = render_tmdwf_fit_input_text(config)
        self.assertIn("tmax auto", text)

    def test_validate_tmdwf_notebook_config(self) -> None:
        parsed = validate_tmdwf_notebook_config(self.tmdwf_config)
        self.assertEqual(parsed.fit_target, "ratio")
        self.assertEqual(parsed.nstates, (1, 2))

    def test_validate_nstate_and_tmdwf_configs_without_tsrange(self) -> None:
        nstate_config = dict(self.nstate_config)
        nstate_config.pop("tsrange")
        parsed_nstate = validate_nstate_notebook_config(nstate_config)
        self.assertEqual(parsed_nstate.tsrange, (0, 31))

        tmdwf_config = dict(self.tmdwf_config)
        tmdwf_config.pop("tsrange")
        parsed_tmdwf = validate_tmdwf_notebook_config(tmdwf_config)
        self.assertEqual(parsed_tmdwf.tsrange, (0, 31))
        self.assertEqual(parsed_tmdwf.tmax, 12)

    def test_validate_tmdwf_config_without_tmax(self) -> None:
        tmdwf_config = dict(self.tmdwf_config)
        tmdwf_config.pop("tmax")
        parsed_tmdwf = validate_tmdwf_notebook_config(tmdwf_config)
        self.assertIsNone(parsed_tmdwf.tmax)

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

        with patch("lqcd_analysis.notebook_workflows.run_tmdwf_nstate_fit", return_value=[] ) as mock_tmdwf:
            run_tmdwf_fit_from_notebook(self.tmdwf_config, results_dir="/tmp/explicit_tmdwf")
        self.assertEqual(Path(mock_tmdwf.call_args.kwargs["results_dir"]), Path("/tmp/explicit_tmdwf"))

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
            with patch("lqcd_analysis.notebook_workflows.run_tmdwf_nstate_fit", return_value=[] ) as mock_tmdwf:
                run_tmdwf_fit_from_notebook(self.tmdwf_config)
            self.assertEqual(Path(mock_tmdwf.call_args.kwargs["results_dir"]), Path("/tmp/vscode").resolve())

    def test_template_notebooks_exist_and_are_valid_json(self) -> None:
        for relative in (
            "templates/two_point/tgevp_template.ipynb",
            "templates/two_point/nstate_fit_template.ipynb",
            "templates/tmdwf/tmdwf_nstate_template.ipynb",
            "templates/two_point/plot_2pt_template.ipynb",
        ):
            path = Path(relative)
            self.assertTrue(path.exists(), path)
            notebook = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(notebook["nbformat"], 4)
            self.assertTrue(notebook["cells"])

    def test_example_input_files_parse(self) -> None:
        tgevp_input = Path("templates/input_files/two_point/tgevp_example_realdata.txt")
        nstate_input = Path("templates/input_files/two_point/nstate_fit_example_realdata.txt")
        tmdwf_input = Path("templates/input_files/tmdwf/tmdwf_nstate_example.txt")
        self.assertTrue(tgevp_input.exists())
        self.assertTrue(nstate_input.exists())
        self.assertTrue(tmdwf_input.exists())
        parsed_tgevp = parse_tgevp_input(tgevp_input)
        parsed_nstate = parse_nstate_fit_input(nstate_input)
        parsed_tmdwf = parse_tmdwf_fit_input(tmdwf_input)
        self.assertEqual(parsed_tgevp.pzlist, (0,))
        self.assertEqual(parsed_nstate.pzlist, (0,))
        self.assertEqual(parsed_tmdwf.pzlist, (0,))

    def test_example_data_files_exist(self) -> None:
        base = Path("examples/data/l64c64a076_m140/comb_c2pt_csv")
        files = sorted(base.glob("c2pt_5_5_k0_pz*_real.csv"))
        self.assertGreaterEqual(len(files), 4)


if __name__ == "__main__":
    unittest.main()
