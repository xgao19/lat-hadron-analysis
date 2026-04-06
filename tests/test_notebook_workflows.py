import json
import unittest
from pathlib import Path

from lqcd_analysis.notebook_workflows import (
    render_nstate_fit_input_text,
    render_tgevp_input_text,
    validate_plot_2pt_notebook_config,
)
from lqcd_analysis.nstate_fit import parse_nstate_fit_input
from lqcd_analysis.tgevp import parse_tgevp_input


class NotebookWorkflowTests(unittest.TestCase):
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
                "nstates": [1, 2],
                "tmax": "auto",
                "plot": True,
                "results_dir": "examples/outputs/demo",
            }
        )
        self.assertIn("model normal", text)
        self.assertIn("nstates 1 2", text)
        self.assertIn("results_dir examples/outputs/demo", text)

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
            "ignored": "extra",
        }
        validated = validate_plot_2pt_notebook_config(config)
        self.assertNotIn("ignored", validated)
        self.assertEqual(validated["nstates"], 2)

    def test_template_notebooks_exist_and_are_valid_json(self) -> None:
        for relative in (
            "templates/tgevp_template.ipynb",
            "templates/nstate_fit_template.ipynb",
            "templates/plot_2pt_template.ipynb",
        ):
            path = Path(relative)
            self.assertTrue(path.exists(), path)
            notebook = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(notebook["nbformat"], 4)
            self.assertTrue(notebook["cells"])

    def test_example_input_files_parse(self) -> None:
        tgevp_input = Path("templates/input_files/tgevp_example_realdata.txt")
        nstate_input = Path("templates/input_files/nstate_fit_example_realdata.txt")
        self.assertTrue(tgevp_input.exists())
        self.assertTrue(nstate_input.exists())
        parsed_tgevp = parse_tgevp_input(tgevp_input)
        parsed_nstate = parse_nstate_fit_input(nstate_input)
        self.assertEqual(parsed_tgevp.pzlist, (0,))
        self.assertEqual(parsed_nstate.pzlist, (0,))

    def test_example_data_files_exist(self) -> None:
        base = Path("examples/data/l64c64a076_m140/comb_c2pt_csv")
        files = sorted(base.glob("c2pt_5_5_k0_pz*_real.csv"))
        self.assertGreaterEqual(len(files), 4)


if __name__ == "__main__":
    unittest.main()
