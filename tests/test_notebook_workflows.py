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
    render_tmdwf_cs_kernel_input_text,
    render_tmdwf_cs_kernel_average_input_text,
    render_tmdwf_cs_kernel_joint_input_text,
    render_tmdwf_fourier_input_text,
    render_tmdwf_fit_input_text,
    render_tmdwf_normalize_input_text,
    render_tgevp_input_text,
    run_effective_mass_from_notebook,
    run_nstate_fit_from_notebook,
    run_tmdwf_cs_kernel_from_notebook,
    run_tmdwf_cs_kernel_average_from_notebook,
    run_tmdwf_cs_kernel_joint_from_notebook,
    run_tmdwf_fourier_from_notebook,
    run_tmdwf_fit_from_notebook,
    run_tmdwf_normalize_from_notebook,
    validate_effective_mass_notebook_config,
    run_tgevp_from_notebook,
    validate_nstate_notebook_config,
    validate_plot_2pt_notebook_config,
    validate_tmdwf_cs_kernel_notebook_config,
    validate_tmdwf_cs_kernel_average_notebook_config,
    validate_tmdwf_cs_kernel_joint_notebook_config,
    validate_tmdwf_fourier_notebook_config,
    validate_tmdwf_notebook_config,
    validate_tmdwf_normalize_notebook_config,
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
            "model": "normal",
            "fit_mode": "uncorrelated",
            "fix_ground_energy_from_dispersion": False,
            "nstates": 1,
            "tmin_window": {0: [0, 4]},
            "tmax": {0: 12},
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
            "fit_window": {0: [3, 6]},
            "qtmdwf_h5": "/tmp/qtmdwf_pz*.h5",
            "dataset_path_template": "{gm}/{eta}/pz{pz}/{Tdir}/bT{bT}/bz{bz}",
            "two_point_fit_root": "/tmp/two_point_fit_root",
            "two_point_fit_window_by_pz": {0: [3, 6]},
            "c2pt": "/tmp/c2pt.csv",
            "fold_t": "periodic",
            "tsrange": [0, 20],
        }
        self.tmdwf_normalize_config = {
            "title_pattern": "demo_pz*",
            "input_root": "/tmp/tmdwf_fit",
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
        self.tmdwf_cs_kernel_config = {
            "title_pattern": "demo_pz*",
            "input_root": "/tmp/tmdwf_fourier",
            "ns": 64,
            "lattice_spacing_fm": 0.076,
            "gmlist": ["T5"],
            "etalist": ["eta0"],
            "component": "real",
            "nstates": 1,
            "normalization_mode": "raw",
            "mu": 2.0,
            "scheme": "CG",
            "extraction_type": "type2",
            "pair_mode": "all",
            "kernel_labels": ["LO", "NLO"],
            "bTlist": [0, 2],
            "pzlist": [2, 3, 4],
            "x_window": [0.2, 0.8],
        }
        self.tmdwf_cs_kernel_average_config = {
            "title_pattern": "demo_pz*",
            "input_root": "/tmp/tmdwf_cs_kernel",
            "lattice_spacing_fm": 0.076,
            "gm": "T5",
            "eta": "eta0",
            "component": "real",
            "nstates": 1,
            "normalization_mode": "raw",
            "scheme": "CG",
            "extraction_type": "type2",
            "kernel_label": "LO",
            "bTrange": [0, 2],
            "x_range": [0.25, 0.75],
            "reference_pz_labels": ["5-6", "6-7"],
        }
        self.tmdwf_cs_kernel_joint_config = {
            "ensembles": [
                {
                    "label": "coarse",
                    "input_root": "/tmp/tmdwf_joint_coarse",
                    "title_pattern": "coarse_pz*",
                    "ns": 48,
                    "lattice_spacing_fm": 0.060,
                    "pzlist": [3, 4, 5],
                    "bTlist": [0, 1, 2],
                },
                {
                    "label": "fine",
                    "input_root": "/tmp/tmdwf_joint_fine",
                    "title_pattern": "fine_pz*",
                    "ns": 64,
                    "lattice_spacing_fm": 0.050,
                    "pzrange": [4, 6],
                    "bTrange": [0, 3],
                },
            ],
            "gm": "T5",
            "eta": "eta0",
            "component": "real",
            "nstates": 2,
            "normalization_mode": "mode3",
            "mu": 2.0,
            "scheme": "CG",
            "kernel_label": "LO",
            "reference_p1_gev": 1.0,
            "x_window": [0.2, 0.8],
            "x_knots": [0.2, 0.4, 0.6, 0.8],
            "bT_knots_fm": [0.05, 0.10, 0.15],
            "spline_kind": "linear",
            "plot": True,
            "progress": True,
            "progress_every": 10,
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
            Path("templates/input_files/two_point/nstate_fit_2state_example_realdata.txt")
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
        match = re.search(r"fit_window (.+)", text)
        self.assertIsNotNone(match)
        override_path = Path(match.group(1).strip())
        self.assertTrue(override_path.exists())
        self.assertEqual(override_path.read_text(encoding="utf-8"), "0 3 6\n")
        self.assertIn("results_dir examples/outputs/tmdwf_demo", text)

    def test_render_tmdwf_text_supports_nested_gm_fit_window_dict(self) -> None:
        text = render_tmdwf_fit_input_text(
            {
                **self.tmdwf_config,
                "fit_window": {"T5": {0: [4, 7]}},
            }
        )
        match = re.search(r"fit_window (.+)", text)
        self.assertIsNotNone(match)
        override_path = Path(match.group(1).strip())
        self.assertEqual(override_path.read_text(encoding="utf-8"), "T5 0 4 7\n")

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

    def test_validate_tmdwf_notebook_config_supports_fit_window_dict(self) -> None:
        parsed = validate_tmdwf_notebook_config(self.tmdwf_config)
        override_path = Path(parsed.fit_window)
        self.assertTrue(override_path.exists())
        self.assertEqual(override_path.read_text(encoding="utf-8"), "0 3 6\n")

    def test_validate_tmdwf_notebook_config(self) -> None:
        parsed = validate_tmdwf_notebook_config(self.tmdwf_config)
        self.assertEqual(parsed.fit_target, "ratio")
        self.assertEqual(parsed.nstates, (1, 2))

    def test_validate_nstate_and_tmdwf_configs_without_tsrange(self) -> None:
        nstate_config = dict(self.nstate_config)
        parsed_nstate = validate_nstate_notebook_config(nstate_config)
        self.assertTrue(Path(parsed_nstate.tmin_window).exists())
        self.assertTrue(Path(parsed_nstate.tmax).exists())

        tmdwf_config = dict(self.tmdwf_config)
        tmdwf_config.pop("tsrange")
        parsed_tmdwf = validate_tmdwf_notebook_config(tmdwf_config)
        self.assertEqual(parsed_tmdwf.tsrange, (0, 31))

    def test_render_tmdwf_fourier_text(self) -> None:
        text = render_tmdwf_fourier_input_text(
            {
                "title_pattern": "demo_pz*",
                "input_root": "/tmp/tmdwf_fit",
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
        self.assertIn("input_root /tmp/tmdwf_fit", text)
        self.assertIn("pzlist 0", text)
        self.assertIn("gmlist T5", text)
        self.assertIn("bTlist 0", text)
        self.assertIn("normalization_mode raw", text)
        self.assertIn("x_range -0.5 1.5", text)
        self.assertIn("x_count 101", text)
        self.assertIn("interpolation_kind cubic", text)

    def test_render_tmdwf_cs_kernel_text(self) -> None:
        text = render_tmdwf_cs_kernel_input_text(
            {
                **self.tmdwf_cs_kernel_config,
                "results_dir": "/tmp/tmdwf_cs",
                "plot": True,
            }
        )
        self.assertIn("title_pattern demo_pz*", text)
        self.assertIn("scheme CG", text)
        self.assertIn("extraction_type type2", text)
        self.assertIn("pair_mode all", text)
        self.assertIn("kernel_labels LO NLO", text)
        self.assertIn("gmlist T5", text)
        self.assertIn("etalist eta0", text)
        self.assertIn("component real", text)
        self.assertIn("normalization_mode raw", text)
        self.assertIn("x_window 0.2 0.8", text)
        self.assertIn("results_dir /tmp/tmdwf_cs", text)

    def test_validate_tmdwf_fourier_notebook_config(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fit_dir = tmp / "demo_pz1" / "tables"
            sample_dir = tmp / "demo_pz1" / "samples"
            fit_dir.mkdir(parents=True)
            sample_dir.mkdir(parents=True)
            (fit_dir / "demo_pz1_T5_eta0_bT0_real_2state_fit.txt").write_text("bz\tm0_mean\tm0_err\n0\t1.0\t0.1\n", encoding="utf-8")
            (sample_dir / "demo_pz1_T5_eta0_bT0_real_2state_samples.txt").write_text("bz\tsample_id\tsuccess\tm0\n0\t0\t1\t1.0\n", encoding="utf-8")
            validated = validate_tmdwf_fourier_notebook_config(
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

    def test_validate_tmdwf_cs_kernel_notebook_config(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            sample_text = "sample_id\tx\tq_sample\n0\t1.0000000000e-01\t1.0\n0\t3.0000000000e-01\t1.1\n1\t1.0000000000e-01\t1.2\n1\t3.0000000000e-01\t1.3\n"
            for bT in (0, 2):
                for pz in (2, 3, 4):
                    title = f"demo_pz{pz}"
                    tables_dir = tmp / title / "tables"
                    samples_dir = tmp / title / "samples"
                    tables_dir.mkdir(parents=True, exist_ok=True)
                    samples_dir.mkdir(parents=True, exist_ok=True)
                    (tables_dir / f"{title}_T5_eta0_bT{bT}_real_1state_fourier.txt").write_text(
                        "normalization_mode raw\nx\tq_mean\tq_err\tq_p16\tq_p84\n1.0000000000e-01\t1.0\t0.1\t0.9\t1.1\n3.0000000000e-01\t1.1\t0.1\t1.0\t1.2\n",
                        encoding="utf-8",
                    )
                    (samples_dir / f"{title}_T5_eta0_bT{bT}_real_1state_fourier_samples.txt").write_text(sample_text, encoding="utf-8")
            validated = validate_tmdwf_cs_kernel_notebook_config(
                {
                    **self.tmdwf_cs_kernel_config,
                    "input_root": str(tmp),
                }
            )
        self.assertEqual(validated["scheme"], "CG")
        self.assertEqual(validated["kernel_labels"], ("LO", "NLO"))
        self.assertEqual(validated["pair_mode"], "all")
        self.assertEqual(validated["bTlist"], (0, 2))
        self.assertEqual(validated["pzlist"], (2, 3, 4))

    def test_render_tmdwf_cs_kernel_average_text(self) -> None:
        text = render_tmdwf_cs_kernel_average_input_text(
            {
                **self.tmdwf_cs_kernel_average_config,
                "results_dir": "/tmp/tmdwf_cs_kernel_average",
            }
        )
        self.assertIn("title_pattern demo_pz*", text)
        self.assertIn("input_root /tmp/tmdwf_cs_kernel", text)
        self.assertIn("lattice_spacing_fm 0.076", text)
        self.assertIn("kernel_label LO", text)
        self.assertIn("reference_pz_labels 5-6 6-7", text)
        self.assertIn("x_range 0.25 0.75", text)
        self.assertIn("results_dir /tmp/tmdwf_cs_kernel_average", text)

    def test_render_tmdwf_cs_kernel_joint_text(self) -> None:
        text = render_tmdwf_cs_kernel_joint_input_text(
            {
                **self.tmdwf_cs_kernel_joint_config,
                "results_dir": "/tmp/tmdwf_cs_kernel_joint",
            }
        )
        self.assertIn("gm T5", text)
        self.assertIn("eta eta0", text)
        self.assertIn("normalization_mode mode3", text)
        self.assertIn("kernel_label LO", text)
        self.assertIn("reference_p1_gev 1.0", text)
        self.assertIn("x_knots 0.2 0.4 0.6 0.8", text)
        self.assertIn("bT_knots_fm 0.05 0.1 0.15", text)
        self.assertIn("spline_kind linear", text)
        self.assertIn("plot true", text)
        self.assertIn("progress true", text)
        self.assertIn("progress_every 10", text)
        self.assertIn("ensemble coarse /tmp/tmdwf_joint_coarse coarse_pz*", text)
        self.assertIn("pz=3,4,5 bT=0,1,2", text)
        self.assertIn("ensemble fine /tmp/tmdwf_joint_fine fine_pz*", text)
        self.assertIn("pz=4:6 bT=0:3", text)
        self.assertIn("results_dir /tmp/tmdwf_cs_kernel_joint", text)

    def test_validate_tmdwf_cs_kernel_joint_notebook_config(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            coarse = tmp / "coarse"
            fine = tmp / "fine"
            coarse.mkdir()
            fine.mkdir()
            config = {
                **self.tmdwf_cs_kernel_joint_config,
                "ensembles": [
                    {
                        **self.tmdwf_cs_kernel_joint_config["ensembles"][0],
                        "input_root": str(coarse),
                    },
                    {
                        **self.tmdwf_cs_kernel_joint_config["ensembles"][1],
                        "input_root": str(fine),
                    },
                ],
            }
            validated = validate_tmdwf_cs_kernel_joint_notebook_config(config)
        self.assertEqual(validated["gm"], "T5")
        self.assertEqual(validated["kernel_label"], "LO")
        self.assertEqual(validated["reference_p1_gev"], 1.0)
        self.assertEqual(validated["spline_kind"], "linear")
        self.assertTrue(validated["plot"])
        self.assertTrue(validated["progress"])
        self.assertEqual(validated["progress_every"], 10)
        self.assertEqual(validated["ensembles"][0].label, "coarse")
        self.assertEqual(validated["ensembles"][0].pzlist, (3, 4, 5))
        self.assertEqual(validated["ensembles"][1].bTlist, (0, 1, 2, 3))

    def test_validate_tmdwf_cs_kernel_average_notebook_config(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_root = tmp / "inputs"
            title = "demo_src5"
            tables_dir = input_root / title / "tables"
            samples_dir = input_root / title / "samples"
            tables_dir.mkdir(parents=True)
            samples_dir.mkdir(parents=True)
            band_lines = [
                "title_pattern demo_src*",
                "output_title demo_pzmultiPz",
                "gm T5",
                "eta eta0",
                "lattice_spacing_fm 0.076",
                "component real",
                "nstates 1",
                "normalization_mode raw",
                "scheme CG",
                "extraction_type type2",
                "kernel_label LO",
                "reference_bT 2",
                "reference_pz 5",
                "reference_pz_label 5-6",
                "dP_GeV 2.5000000000e-01",
                "x\tgamma_p16\tgamma_p50\tgamma_p84",
                "2.5000000000e-01\t1.0\t1.1\t1.2",
            ]
            sample_lines = [
                "title_pattern demo_src*",
                "output_title demo_pzmultiPz",
                "gm T5",
                "eta eta0",
                "lattice_spacing_fm 0.076",
                "component real",
                "nstates 1",
                "normalization_mode raw",
                "scheme CG",
                "extraction_type type2",
                "kernel_label LO",
                "reference_bT 2",
                "reference_pz 5",
                "reference_pz_label 5-6",
                "dP_GeV 2.5000000000e-01",
                "x sample_id success gamma_zeta chi2_dof",
                "2.5000000000e-01\t0\t1\t1.0\t0.0",
            ]
            (tables_dir / f"{title}_T5_eta0_bT2_real_1state_CG_LO_band.txt").write_text("\n".join(band_lines) + "\n", encoding="utf-8")
            (samples_dir / f"{title}_T5_eta0_bT2_real_1state_CG_LO_samples.txt").write_text("\n".join(sample_lines) + "\n", encoding="utf-8")
            validated = validate_tmdwf_cs_kernel_average_notebook_config(
                {
                    **self.tmdwf_cs_kernel_average_config,
                    "input_root": str(input_root),
                }
            )
        self.assertEqual(validated["kernel_label"], "LO")
        self.assertEqual(validated["bTlist"], (0, 1, 2))
        self.assertAlmostEqual(float(validated["lattice_spacing_fm"]), 0.076)
        self.assertEqual(validated["x_range"], (0.25, 0.75))
        self.assertEqual(validated["reference_pz_labels"], ("5-6", "6-7"))

    def test_explicit_results_dir_override_wins_for_tmdwf_cs_kernel_average_notebook_runner(self) -> None:
        with patch("lqcd_analysis.notebook_workflows.run_tmdwf_cs_kernel_average_workflow", return_value=[]) as mock_avg:
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                tables_dir = tmp / "demo_src5" / "tables"
                samples_dir = tmp / "demo_src5" / "samples"
                tables_dir.mkdir(parents=True, exist_ok=True)
                samples_dir.mkdir(parents=True, exist_ok=True)
                (tables_dir / "demo_src5_T5_eta0_bT2_real_1state_CG_LO_band.txt").write_text(
                    "\n".join(
                        [
                            "title_pattern demo_src*",
                            "output_title demo_pzmultiPz",
                            "gm T5",
                            "eta eta0",
                            "lattice_spacing_fm 0.076",
                            "component real",
                            "nstates 1",
                            "normalization_mode raw",
                            "scheme CG",
                            "extraction_type type2",
                            "pair_mode all",
                            "kernel_label LO",
                            "reference_bT 2",
                            "reference_pz 5",
                            "reference_pz_label 5-6",
                            "dP_GeV 2.5000000000e-01",
                            "x\tgamma_p16\tgamma_p50\tgamma_p84",
                            "2.5000000000e-01\t1.0\t1.1\t1.2",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                (samples_dir / "demo_src5_T5_eta0_bT2_real_1state_CG_LO_samples.txt").write_text(
                    "\n".join(
                        [
                            "title_pattern demo_src*",
                            "output_title demo_pzmultiPz",
                            "gm T5",
                            "eta eta0",
                            "lattice_spacing_fm 0.076",
                            "component real",
                            "nstates 1",
                            "normalization_mode raw",
                            "scheme CG",
                            "extraction_type type2",
                            "pair_mode all",
                            "kernel_label LO",
                            "reference_bT 2",
                            "reference_pz 5",
                            "reference_pz_label 5-6",
                            "dP_GeV 2.5000000000e-01",
                            "x sample_id success gamma_zeta chi2_dof",
                            "2.5000000000e-01\t0\t1\t1.0\t0.0",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                run_tmdwf_cs_kernel_average_from_notebook(
                    {
                        **self.tmdwf_cs_kernel_average_config,
                        "input_root": str(tmp),
                    },
                    results_dir="/tmp/explicit_tmdwf_cs_average",
                )
        self.assertEqual(Path(mock_avg.call_args.kwargs["results_dir"]), Path("/tmp/explicit_tmdwf_cs_average"))

    def test_tmdwf_fourier_notebook_config_rejects_old_single_job_shape(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            render_tmdwf_fourier_input_text(
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
        self.assertIn("missing TMDWF Fourier notebook config keys", str(ctx.exception))

    def test_render_tmdwf_normalize_text(self) -> None:
        text = render_tmdwf_normalize_input_text(
            {
                **self.tmdwf_normalize_config,
                "results_dir": "/tmp/tmdwf_norm",
            }
        )
        self.assertIn("title_pattern demo_pz*", text)
        self.assertIn("input_root /tmp/tmdwf_fit", text)
        self.assertIn("normalization_mode mode1", text)
        self.assertIn("results_dir /tmp/tmdwf_norm", text)

    def test_validate_tmdwf_normalize_notebook_config(self) -> None:
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
            validated = validate_tmdwf_normalize_notebook_config(
                {
                    **self.tmdwf_normalize_config,
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

        with patch("lqcd_analysis.notebook_workflows.run_tmdwf_nstate_fit", return_value=[] ) as mock_tmdwf:
            run_tmdwf_fit_from_notebook(self.tmdwf_config, results_dir="/tmp/explicit_tmdwf")
        self.assertEqual(Path(mock_tmdwf.call_args.kwargs["results_dir"]), Path("/tmp/explicit_tmdwf"))

        with patch("lqcd_analysis.notebook_workflows.run_tmdwf_fourier_workflow", return_value=[] ) as mock_fourier:
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                fit_dir = tmp / "demo_pz0" / "tables"
                sample_dir = tmp / "demo_pz0" / "samples"
                fit_dir.mkdir(parents=True)
                sample_dir.mkdir(parents=True)
                (fit_dir / "demo_pz0_T5_eta0_bT0_real_1state_fit.txt").write_text("bz\tm0_mean\tm0_err\n0\t1.0\t0.1\n", encoding="utf-8")
                (sample_dir / "demo_pz0_T5_eta0_bT0_real_1state_samples.txt").write_text("bz\tsample_id\tsuccess\tm0\n0\t0\t1\t1.0\n", encoding="utf-8")
                run_tmdwf_fourier_from_notebook(
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
                    results_dir="/tmp/explicit_tmdwf_fourier",
                )
        self.assertEqual(Path(mock_fourier.call_args.kwargs["results_dir"]), Path("/tmp/explicit_tmdwf_fourier"))

        with patch("lqcd_analysis.notebook_workflows.run_tmdwf_cs_kernel_workflow", return_value=[] ) as mock_cs:
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                sample_text = "sample_id\tx\tq_sample\n0\t1.0000000000e-01\t1.0\n0\t3.0000000000e-01\t1.1\n1\t1.0000000000e-01\t1.2\n1\t3.0000000000e-01\t1.3\n"
                for bT in (0, 2):
                    for pz in (2, 3, 4):
                        title = f"demo_pz{pz}"
                        tables_dir = tmp / title / "tables"
                        samples_dir = tmp / title / "samples"
                        tables_dir.mkdir(parents=True, exist_ok=True)
                        samples_dir.mkdir(parents=True, exist_ok=True)
                        (tables_dir / f"{title}_T5_eta0_bT{bT}_real_1state_fourier.txt").write_text(
                            "normalization_mode raw\nx\tq_mean\tq_err\tq_p16\tq_p84\n1.0000000000e-01\t1.0\t0.1\t0.9\t1.1\n3.0000000000e-01\t1.1\t0.1\t1.0\t1.2\n",
                            encoding="utf-8",
                        )
                        (samples_dir / f"{title}_T5_eta0_bT{bT}_real_1state_fourier_samples.txt").write_text(sample_text, encoding="utf-8")
                run_tmdwf_cs_kernel_from_notebook(
                    {
                        **self.tmdwf_cs_kernel_config,
                        "input_root": str(tmp),
                    },
                    results_dir="/tmp/explicit_tmdwf_cs",
                )
        self.assertEqual(Path(mock_cs.call_args.kwargs["results_dir"]), Path("/tmp/explicit_tmdwf_cs"))

        with patch("lqcd_analysis.notebook_workflows.run_tmdwf_normalization", return_value=[] ) as mock_normalize:
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
                run_tmdwf_normalize_from_notebook(
                    {
                        **self.tmdwf_normalize_config,
                        "input_root": str(tmp),
                    },
                    results_dir="/tmp/explicit_tmdwf_normalize",
                )
        self.assertEqual(Path(mock_normalize.call_args.kwargs["results_dir"]), Path("/tmp/explicit_tmdwf_normalize"))

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

        with patch.dict(sys.modules, {"IPython": fake_ipython}):
            with patch("lqcd_analysis.notebook_workflows.run_tmdwf_fourier_workflow", return_value=[] ) as mock_fourier:
                import tempfile

                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp = Path(tmpdir)
                    fit_dir = tmp / "demo_pz0" / "tables"
                    sample_dir = tmp / "demo_pz0" / "samples"
                    fit_dir.mkdir(parents=True)
                    sample_dir.mkdir(parents=True)
                    (fit_dir / "demo_pz0_T5_eta0_bT0_real_1state_fit.txt").write_text("bz\tm0_mean\tm0_err\n0\t1.0\t0.1\n", encoding="utf-8")
                    (sample_dir / "demo_pz0_T5_eta0_bT0_real_1state_samples.txt").write_text("bz\tsample_id\tsuccess\tm0\n0\t0\t1\t1.0\n", encoding="utf-8")
                    run_tmdwf_fourier_from_notebook(
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
            with patch("lqcd_analysis.notebook_workflows.run_tmdwf_cs_kernel_workflow", return_value=[] ) as mock_cs:
                import tempfile

                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp = Path(tmpdir)
                    sample_text = "sample_id\tx\tq_sample\n0\t1.0000000000e-01\t1.0\n0\t3.0000000000e-01\t1.1\n1\t1.0000000000e-01\t1.2\n1\t3.0000000000e-01\t1.3\n"
                    for bT in (0, 2):
                        for pz in (2, 3, 4):
                            title = f"demo_pz{pz}"
                            tables_dir = tmp / title / "tables"
                            samples_dir = tmp / title / "samples"
                            tables_dir.mkdir(parents=True, exist_ok=True)
                            samples_dir.mkdir(parents=True, exist_ok=True)
                            (tables_dir / f"{title}_T5_eta0_bT{bT}_real_1state_fourier.txt").write_text(
                                "normalization_mode raw\nx\tq_mean\tq_err\tq_p16\tq_p84\n1.0000000000e-01\t1.0\t0.1\t0.9\t1.1\n3.0000000000e-01\t1.1\t0.1\t1.0\t1.2\n",
                                encoding="utf-8",
                            )
                            (samples_dir / f"{title}_T5_eta0_bT{bT}_real_1state_fourier_samples.txt").write_text(sample_text, encoding="utf-8")
                    run_tmdwf_cs_kernel_from_notebook(
                        {
                            **self.tmdwf_cs_kernel_config,
                            "input_root": str(tmp),
                        }
                    )
                self.assertEqual(Path(mock_cs.call_args.kwargs["results_dir"]), Path("/tmp/vscode").resolve())

        with patch.dict(sys.modules, {"IPython": fake_ipython}):
            with patch("lqcd_analysis.notebook_workflows.run_tmdwf_normalization", return_value=[] ) as mock_normalize:
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
                    run_tmdwf_normalize_from_notebook(
                        {
                            **self.tmdwf_normalize_config,
                            "input_root": str(tmp),
                        }
                    )
                self.assertEqual(Path(mock_normalize.call_args.kwargs["results_dir"]), Path("/tmp/vscode").resolve())

    def test_template_notebooks_exist_and_are_valid_json(self) -> None:
        for relative in (
            "templates/two_point/tgevp_template.ipynb",
            "templates/two_point/nstate_fit_1state_template.ipynb",
            "templates/two_point/nstate_fit_2state_template.ipynb",
            "templates/tmdwf/tmdwf_nstate_template.ipynb",
            "templates/tmdwf/tmdwf_fourier_template.ipynb",
            "templates/tmdwf/tmdwf_normalize_template.ipynb",
            "templates/tmdwf/tmdwf_cs_kernel_template.ipynb",
            "templates/tmdwf/tmdwf_cs_kernel_average_template.ipynb",
            "templates/two_point/plot_2pt_template.ipynb",
        ):
            path = Path(relative)
            self.assertTrue(path.exists(), path)
            notebook = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(notebook["nbformat"], 4)
            self.assertTrue(notebook["cells"])

    def test_tmdwf_fourier_template_contains_expected_workflow_hooks(self) -> None:
        path = Path("templates/tmdwf/tmdwf_fourier_template.ipynb")
        notebook = json.loads(path.read_text(encoding="utf-8"))
        joined = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("render_tmdwf_fourier_input_text", joined)
        self.assertIn("validate_tmdwf_fourier_notebook_config", joined)
        self.assertIn("run_tmdwf_fourier_from_notebook", joined)
        self.assertIn("\"input_root\":", joined)
        self.assertIn("\"title_pattern\":", joined)
        self.assertIn("\"pzlist\": [0]", joined)
        self.assertIn("\"bTlist\": [0]", joined)
        self.assertIn("\"normalization_mode\": \"raw\"", joined)
        self.assertIn("\"x_range\": [-0.5, 1.5]", joined)
        self.assertIn("# Data settings", joined)
        self.assertIn("# Fourier-transform parameter settings", joined)

    def test_tmdwf_normalize_template_contains_expected_workflow_hooks(self) -> None:
        path = Path("templates/tmdwf/tmdwf_normalize_template.ipynb")
        notebook = json.loads(path.read_text(encoding="utf-8"))
        joined = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("render_tmdwf_normalize_input_text", joined)
        self.assertIn("validate_tmdwf_normalize_notebook_config", joined)
        self.assertIn("run_tmdwf_normalize_from_notebook", joined)
        self.assertIn("\"normalization_mode\": \"mode1\"", joined)
        self.assertIn("# Data settings", joined)
        self.assertIn("# Normalization settings", joined)

    def test_tmdwf_cs_kernel_template_contains_expected_workflow_hooks(self) -> None:
        path = Path("templates/tmdwf/tmdwf_cs_kernel_template.ipynb")
        notebook = json.loads(path.read_text(encoding="utf-8"))
        joined = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("render_tmdwf_cs_kernel_input_text", joined)
        self.assertIn("validate_tmdwf_cs_kernel_notebook_config", joined)
        self.assertIn("run_tmdwf_cs_kernel_from_notebook", joined)
        self.assertIn("\"scheme\": \"CG\"", joined)
        self.assertIn("\"kernel_labels\": [\"LO\", \"NLO\", \"NLL\"]", joined)
        self.assertIn("## Option Guide", joined)
        self.assertIn("\"normalization_mode\": \"raw\"", joined)
        self.assertIn("\"component\": \"real\"", joined)
        self.assertIn("Expected input data shape", joined)
        self.assertIn("# Data settings", joined)
        self.assertIn("# CS-kernel extraction settings", joined)

    def test_tmdwf_cs_kernel_average_template_contains_expected_workflow_hooks(self) -> None:
        path = Path("templates/tmdwf/tmdwf_cs_kernel_average_template.ipynb")
        notebook = json.loads(path.read_text(encoding="utf-8"))
        joined = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("validate_tmdwf_cs_kernel_average_notebook_config", joined)
        self.assertIn("run_tmdwf_cs_kernel_average_from_notebook", joined)
        self.assertIn("\"kernel_label\": \"LO\"", joined)
        self.assertIn("\"reference_pz_labels\": [\"5-6\", \"6-7\", \"7-8\"]", joined)
        self.assertIn("# Data settings", joined)
        self.assertIn("# User Inputs", joined)

    def test_tmdwf_cs_kernel_joint_template_contains_expected_workflow_hooks(self) -> None:
        path = Path("templates/tmdwf/tmdwf_cs_kernel_joint_template.ipynb")
        notebook = json.loads(path.read_text(encoding="utf-8"))
        joined = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("render_tmdwf_cs_kernel_joint_input_text", joined)
        self.assertIn("validate_tmdwf_cs_kernel_joint_notebook_config", joined)
        self.assertIn("run_tmdwf_cs_kernel_joint_from_notebook", joined)
        self.assertIn("\"kernel_label\": \"LO\"", joined)
        self.assertIn("\"reference_p1_gev\": 1.0", joined)
        self.assertIn("\"spline_kind\": \"linear\"", joined)
        self.assertIn("\"plot\": True", joined)
        self.assertIn("\"progress\": True", joined)
        self.assertIn("\"normalization_mode\": \"mode3\"", joined)
        self.assertIn("\"ensembles\": [", joined)
        self.assertIn("gamma_eff(x, bT)", joined)
        self.assertIn("## Option Guide", joined)

    def test_example_input_files_parse(self) -> None:
        tgevp_input = Path("templates/input_files/two_point/tgevp_example_realdata.txt")
        nstate_1_input = Path("templates/input_files/two_point/nstate_fit_1state_example_realdata.txt")
        nstate_2_input = Path("templates/input_files/two_point/nstate_fit_2state_example_realdata.txt")
        tmdwf_input = Path("templates/input_files/tmdwf/tmdwf_nstate_example.txt")
        tmdwf_fourier_input = Path("templates/input_files/tmdwf/tmdwf_fourier_example.txt")
        tmdwf_fourier_annotated = Path("templates/input_files/tmdwf/tmdwf_fourier_example_annotated.txt")
        tmdwf_cs_input = Path("templates/input_files/tmdwf/tmdwf_cs_kernel_example.txt")
        tmdwf_cs_annotated = Path("templates/input_files/tmdwf/tmdwf_cs_kernel_example_annotated.txt")
        tmdwf_normalize_input = Path("templates/input_files/tmdwf/tmdwf_normalize_example.txt")
        tmdwf_normalize_annotated = Path("templates/input_files/tmdwf/tmdwf_normalize_example_annotated.txt")
        self.assertTrue(tgevp_input.exists())
        self.assertTrue(nstate_1_input.exists())
        self.assertTrue(nstate_2_input.exists())
        self.assertTrue(tmdwf_input.exists())
        self.assertTrue(tmdwf_fourier_input.exists())
        self.assertTrue(tmdwf_fourier_annotated.exists())
        self.assertTrue(tmdwf_cs_input.exists())
        self.assertTrue(tmdwf_cs_annotated.exists())
        self.assertTrue(tmdwf_normalize_input.exists())
        self.assertTrue(tmdwf_normalize_annotated.exists())
        parsed_tgevp = parse_tgevp_input(tgevp_input)
        parsed_nstate = parse_nstate_fit_input(nstate_2_input)
        parsed_tmdwf = parse_tmdwf_fit_input(tmdwf_input)
        self.assertEqual(parsed_tgevp.pzlist, (0,))
        self.assertEqual(parsed_nstate.pzlist, (0,))
        self.assertEqual(parsed_tmdwf.pzlist, (0,))

    def test_example_data_files_exist(self) -> None:
        base = Path("examples/data/l64c64a076_m140/comb_c2pt_csv")
        files = sorted(base.glob("c2pt_5_5_k0_pz*_real.csv"))
        self.assertGreaterEqual(len(files), 1)


if __name__ == "__main__":
    unittest.main()
