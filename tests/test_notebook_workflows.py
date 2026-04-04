import json
import unittest
from pathlib import Path

from lqcd_analysis.notebook_workflows import (
    render_nstate_fit_input_text,
    render_tgevp_input_text,
)


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
            }
        )
        self.assertIn("fold_t periodic", text)
        self.assertIn("pzlist 0 1", text)

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
            }
        )
        self.assertIn("model normal", text)
        self.assertIn("nstates 1 2", text)

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


if __name__ == "__main__":
    unittest.main()
