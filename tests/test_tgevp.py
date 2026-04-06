import tempfile
import unittest
from pathlib import Path

import numpy as np

from lqcd_analysis.tgevp import parse_tgevp_input, run_ss_2pt_tgevp, solve_tgevp
from lqcd_analysis.utils import apply_antiperiodic_fold


class TGEVPTests(unittest.TestCase):
    def test_solve_tgevp_recovers_two_state_energies(self) -> None:
        times = np.arange(12, dtype=float)
        correlator = 4.0 * np.exp(-0.35 * times) + 1.2 * np.exp(-0.8 * times)
        eigvals, _, overlaps = solve_tgevp(correlator, ts=1, n_states=2)
        energies = -np.log(eigvals)

        self.assertTrue(np.allclose(energies, [0.35, 0.8], atol=1e-10))
        self.assertTrue(np.all(overlaps > 0.0))

    def test_parse_input_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input_k0_SS.txt"
            input_file.write_text(
                "\n".join(
                    [
                        "demo_pz* 64 64 0.076",
                        "c2pt /tmp/c2pt_pz*.csv",
                        "pzlist 0 2",
                        "fold_t true",
                        "tsrange 0 20",
                        "binsize 2",
                        "bootstrap_samples 32",
                        "bootstrap_size 24",
                        "seed 2026",
                        "results_dir /tmp/tgevp_results",
                    ]
                ),
                encoding="utf-8",
            )
            parsed = parse_tgevp_input(input_file)

        self.assertEqual(parsed.title_pattern, "demo_pz*")
        self.assertEqual(parsed.pzlist, (0, 2))
        self.assertEqual(parsed.fold_t, "periodic")
        self.assertEqual(parsed.tsrange, (0, 20))
        self.assertEqual(parsed.binsize, 2)
        self.assertEqual(parsed.bootstrap_samples, 32)
        self.assertEqual(parsed.bootstrap_size, 24)
        self.assertEqual(parsed.seed, 2026)
        self.assertEqual(parsed.results_dir, Path("/tmp/tgevp_results"))

    def test_antiperiodic_fold(self) -> None:
        data = np.array([[10.0, 7.0, 5.0, 4.0, 2.0, 1.0]])
        folded = apply_antiperiodic_fold(data, 6)
        expected = np.array([[10.0, 3.0, 1.5, 4.0]])
        self.assertTrue(np.allclose(folded, expected))

    def test_end_to_end_analysis_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "c2pt_5_5_k0_pz0_real.csv"
            input_file = tmp / "input_k0_SS.txt"
            results_dir = tmp / "results"

            times = np.arange(10)
            base = 3.0 * np.exp(-0.45 * times) + 0.8 * np.exp(-0.9 * times)
            configs = []
            for cfg in range(12):
                noise = 0.002 * np.cos(times + cfg)
                configs.append(base * (1.0 + noise))
            data = np.array(configs).T

            with csv_path.open("w", encoding="utf-8") as handle:
                header = ",".join(["t"] + [f"cfg_{idx}" for idx in range(data.shape[1])])
                handle.write(header + "\n")
                for t in range(len(times)):
                    row = ",".join([str(t)] + [f"{value:.12e}" for value in data[t]])
                    handle.write(row + "\n")

            input_file.write_text(
                "\n".join(
                    [
                        "demo_pz* 16 10 0.076",
                        f"c2pt {csv_path.as_posix().replace('pz0', 'pz*')}",
                        "pzlist 0",
                        "fold_t false",
                        "tsrange 0 7",
                    ]
                ),
                encoding="utf-8",
            )

            outputs = run_ss_2pt_tgevp(
                input_file,
                bootstrap_samples=16,
                bootstrap_size=12,
                seed=7,
                results_dir=results_dir,
            )

            self.assertEqual(len(outputs), 3)
            for output in outputs:
                self.assertTrue(output.exists(), output)

            summary = np.loadtxt(results_dir / "demo_pz0_tgevp_summary.txt")
            corr = np.loadtxt(results_dir / "demo_pz0_tgevp_correlation.txt")
            samples = np.loadtxt(results_dir / "samples" / "demo_pz0_tgevp_samples.txt")

            self.assertEqual(summary.shape[1], 11)
            self.assertEqual(corr.shape[1], 4)
            self.assertEqual(samples.shape[1], 7)
            self.assertTrue(np.all(summary[:, 0] >= 2))


if __name__ == "__main__":
    unittest.main()
