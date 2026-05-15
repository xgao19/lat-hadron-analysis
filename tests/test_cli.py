import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from lqcd_analysis.cli import build_parser, main


class CLITests(unittest.TestCase):
    def test_build_parser_includes_new_da_subcommands(self) -> None:
        parser = build_parser()
        help_text = parser.format_help()
        self.assertIn("2pt-effective-mass", help_text)
        self.assertIn("da-normalize", help_text)
        self.assertIn("da-fourier", help_text)
        self.assertIn("da-ratio-fourier-t", help_text)
        self.assertIn("da-x-nstate-fit", help_text)
        self.assertIn("da-xfit-normalize", help_text)

    def test_two_point_effective_mass_cli_dispatches_to_workflow(self) -> None:
        output_buffer = io.StringIO()
        fake_outputs = [Path("/tmp/meff.txt")]
        with patch("lqcd_analysis.cli.run_effective_mass_workflow", return_value=fake_outputs) as mock_run:
            with redirect_stdout(output_buffer):
                main(["2pt-effective-mass", "meff_input.txt", "--results-dir", "/tmp/meff_results"])

        self.assertEqual(mock_run.call_args.args[0], "meff_input.txt")
        self.assertEqual(mock_run.call_args.kwargs["results_dir"], "/tmp/meff_results")
        printed = output_buffer.getvalue()
        self.assertIn("/tmp/meff.txt", printed)

    def test_da_normalize_cli_dispatches_to_workflow(self) -> None:
        output_buffer = io.StringIO()
        fake_outputs = [Path("/tmp/norm_fit.txt"), Path("/tmp/norm_samples.txt")]
        with patch("lqcd_analysis.cli.run_da_normalization", return_value=fake_outputs) as mock_run:
            with redirect_stdout(output_buffer):
                main(["da-normalize", "normalize_input.txt", "--results-dir", "/tmp/norm_results"])

        self.assertEqual(mock_run.call_args.args[0], "normalize_input.txt")
        self.assertEqual(mock_run.call_args.kwargs["results_dir"], "/tmp/norm_results")
        printed = output_buffer.getvalue()
        self.assertIn("/tmp/norm_fit.txt", printed)
        self.assertIn("/tmp/norm_samples.txt", printed)

    def test_da_fourier_cli_dispatches_to_workflow(self) -> None:
        output_buffer = io.StringIO()
        fake_outputs = [Path("/tmp/fourier.txt"), Path("/tmp/fourier.pdf")]
        with patch("lqcd_analysis.cli.run_da_fourier_workflow", return_value=fake_outputs) as mock_run:
            with redirect_stdout(output_buffer):
                main(["da-fourier", "fourier_input.txt", "--results-dir", "/tmp/fourier_results"])

        self.assertEqual(mock_run.call_args.args[0], "fourier_input.txt")
        self.assertEqual(mock_run.call_args.kwargs["results_dir"], "/tmp/fourier_results")
        printed = output_buffer.getvalue()
        self.assertIn("/tmp/fourier.txt", printed)
        self.assertIn("/tmp/fourier.pdf", printed)

    def test_da_ratio_fourier_t_cli_dispatches_to_workflow(self) -> None:
        output_buffer = io.StringIO()
        fake_outputs = [Path("/tmp/qxt.txt"), Path("/tmp/qxt_samples.txt")]
        with patch("lqcd_analysis.cli.run_da_ratio_fourier_t_workflow", return_value=fake_outputs) as mock_run:
            with redirect_stdout(output_buffer):
                main(["da-ratio-fourier-t", "ratio_fourier_t_input.txt", "--results-dir", "/tmp/qxt_results"])

        self.assertEqual(mock_run.call_args.args[0], "ratio_fourier_t_input.txt")
        self.assertEqual(mock_run.call_args.kwargs["results_dir"], "/tmp/qxt_results")
        printed = output_buffer.getvalue()
        self.assertIn("/tmp/qxt.txt", printed)
        self.assertIn("/tmp/qxt_samples.txt", printed)

    def test_da_x_nstate_fit_cli_dispatches_to_workflow(self) -> None:
        output_buffer = io.StringIO()
        fake_outputs = [Path("/tmp/xfit.txt"), Path("/tmp/xfit_samples.txt")]
        with patch("lqcd_analysis.cli.run_da_x_nstate_fit_workflow", return_value=fake_outputs) as mock_run:
            with redirect_stdout(output_buffer):
                main(["da-x-nstate-fit", "xfit_input.txt", "--results-dir", "/tmp/xfit_results"])

        self.assertEqual(mock_run.call_args.args[0], "xfit_input.txt")
        self.assertEqual(mock_run.call_args.kwargs["results_dir"], "/tmp/xfit_results")
        printed = output_buffer.getvalue()
        self.assertIn("/tmp/xfit.txt", printed)
        self.assertIn("/tmp/xfit_samples.txt", printed)

    def test_da_xfit_normalize_cli_dispatches_to_workflow(self) -> None:
        output_buffer = io.StringIO()
        fake_outputs = [Path("/tmp/xfit_norm.txt"), Path("/tmp/xfit_norm_samples.txt")]
        with patch("lqcd_analysis.cli.run_da_xfit_normalization", return_value=fake_outputs) as mock_run:
            with redirect_stdout(output_buffer):
                main(["da-xfit-normalize", "xfit_norm_input.txt", "--results-dir", "/tmp/xfit_norm_results"])

        self.assertEqual(mock_run.call_args.args[0], "xfit_norm_input.txt")
        self.assertEqual(mock_run.call_args.kwargs["results_dir"], "/tmp/xfit_norm_results")
        printed = output_buffer.getvalue()
        self.assertIn("/tmp/xfit_norm.txt", printed)
        self.assertIn("/tmp/xfit_norm_samples.txt", printed)



if __name__ == "__main__":
    unittest.main()
