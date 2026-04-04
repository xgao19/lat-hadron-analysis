import tempfile
import unittest
from pathlib import Path

import numpy as np

from lqcd_analysis.nstate_fit import (
    effective_mass_single,
    evaluate_model,
    parse_nstate_fit_input,
    run_nstate_fit,
)
from lqcd_analysis.plotting_2pt import write_nstate_plot_notebook


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
                        "nstates 1 2 3",
                    ]
                ),
                encoding="utf-8",
            )
            parsed = parse_nstate_fit_input(path)
        self.assertEqual(parsed.model, "symmetric")
        self.assertEqual(parsed.nstates, (1, 2, 3))
        self.assertEqual(parsed.fold_t, "antiperiodic")

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
            one_state = (
                tmp / "results_nstate_fit" / "demo_pz0" / "tables" / "demo_pz0_normal_1state_tmax8_fits.txt"
            )
            self.assertTrue(one_state.exists())
            notebook = tmp / "notebook_plots" / "demo_pz0" / "demo_pz0_normal_nstate_plots.ipynb"
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
            )
            self.assertTrue(notebook.exists())
            text = notebook.read_text(encoding="utf-8")
            self.assertIn("plot_nstate_outputs", text)


if __name__ == "__main__":
    unittest.main()
