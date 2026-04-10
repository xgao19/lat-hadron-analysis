import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from lqcd_analysis.cli import build_parser, main


class CLITests(unittest.TestCase):
    def test_build_parser_includes_new_tmdwf_subcommands(self) -> None:
        parser = build_parser()
        help_text = parser.format_help()
        self.assertIn("tmdwf-normalize", help_text)
        self.assertIn("tmdwf-fourier", help_text)

    def test_tmdwf_normalize_cli_dispatches_to_workflow(self) -> None:
        output_buffer = io.StringIO()
        fake_outputs = [Path("/tmp/norm_fit.txt"), Path("/tmp/norm_samples.txt")]
        with patch("lqcd_analysis.cli.run_tmdwf_normalization", return_value=fake_outputs) as mock_run:
            with redirect_stdout(output_buffer):
                main(["tmdwf-normalize", "normalize_input.txt", "--results-dir", "/tmp/norm_results"])

        self.assertEqual(mock_run.call_args.args[0], "normalize_input.txt")
        self.assertEqual(mock_run.call_args.kwargs["results_dir"], "/tmp/norm_results")
        printed = output_buffer.getvalue()
        self.assertIn("/tmp/norm_fit.txt", printed)
        self.assertIn("/tmp/norm_samples.txt", printed)

    def test_tmdwf_fourier_cli_dispatches_to_workflow(self) -> None:
        output_buffer = io.StringIO()
        fake_outputs = [Path("/tmp/fourier.txt"), Path("/tmp/fourier.pdf")]
        with patch("lqcd_analysis.cli.run_tmdwf_fourier_workflow", return_value=fake_outputs) as mock_run:
            with redirect_stdout(output_buffer):
                main(["tmdwf-fourier", "fourier_input.txt", "--results-dir", "/tmp/fourier_results"])

        self.assertEqual(mock_run.call_args.args[0], "fourier_input.txt")
        self.assertEqual(mock_run.call_args.kwargs["results_dir"], "/tmp/fourier_results")
        printed = output_buffer.getvalue()
        self.assertIn("/tmp/fourier.txt", printed)
        self.assertIn("/tmp/fourier.pdf", printed)


if __name__ == "__main__":
    unittest.main()
