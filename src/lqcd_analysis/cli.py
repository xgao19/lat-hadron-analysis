from __future__ import annotations

import argparse
from pathlib import Path

from .tmdwf.cs_kernel_extract import run_tmdwf_cs_kernel_workflow
from .tmdwf.cs_kernel_average import run_tmdwf_cs_kernel_average_workflow
from .tmdwf.fourier import run_tmdwf_fourier_workflow
from .tmdwf.fit_nstate import run_tmdwf_nstate_fit
from .tmdwf.normalize import run_tmdwf_normalization
from .two_point.effective_mass import run_effective_mass_workflow
from .two_point.fit_nstate import run_nstate_fit
from .two_point.tgevp import run_ss_2pt_tgevp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lqcd-analysis")
    subparsers = parser.add_subparsers(dest="command")

    tgevp_parser = subparsers.add_parser(
        "ss-2pt-tgevp",
        help="Run SS two-point TGEVP analysis from a text input file.",
    )
    tgevp_parser.add_argument("input_file", help="Input control file, e.g. input_k0_SS.txt")
    tgevp_parser.add_argument(
        "--binsize",
        type=int,
        default=None,
        help="Bin size before bootstrap; defaults to the input file value or 1",
    )
    tgevp_parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=None,
        help="Number of bootstrap samples; defaults to the number of binned configurations",
    )
    tgevp_parser.add_argument(
        "--bootstrap-size",
        type=int,
        default=None,
        help="Bootstrap draw size; defaults to the number of binned configurations",
    )
    tgevp_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for bootstrap; defaults to the input file value or 2026",
    )
    tgevp_parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory for summaries, correlations, and bootstrap samples",
    )

    nstate_parser = subparsers.add_parser(
        "2pt-nstate-fit",
        help="Run bootstrap-based traditional N-state fits for two-point correlators.",
    )
    nstate_parser.add_argument("input_file", help="Input control file for the N-state fit workflow")
    nstate_parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory for fit tables, sample outputs, and plots",
    )

    effective_mass_parser = subparsers.add_parser(
        "2pt-effective-mass",
        help="Run standalone effective-mass extraction for two-point correlators.",
    )
    effective_mass_parser.add_argument(
        "input_file",
        help="Input control file for the effective-mass workflow",
    )
    effective_mass_parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory for effective-mass tables and outputs",
    )

    tmdwf_parser = subparsers.add_parser(
        "tmdwf-nstate-fit",
        help="Run bootstrap-based TMDWF N-state fits from a text input file.",
    )
    tmdwf_parser.add_argument("input_file", help="Input control file for the TMDWF fit workflow")
    tmdwf_parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory for TMDWF fit tables, sample outputs, and summaries",
    )

    tmdwf_normalize_parser = subparsers.add_parser(
        "tmdwf-normalize",
        help="Run downstream TMDWF matrix-element normalization from a text input file.",
    )
    tmdwf_normalize_parser.add_argument(
        "input_file",
        help="Input control file for the TMDWF normalization workflow",
    )
    tmdwf_normalize_parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory for normalized TMDWF tables, sample outputs, and summaries",
    )

    tmdwf_fourier_parser = subparsers.add_parser(
        "tmdwf-fourier",
        help="Run downstream TMDWF Fourier postprocessing from a text input file.",
    )
    tmdwf_fourier_parser.add_argument(
        "input_file",
        help="Input control file for the TMDWF Fourier workflow",
    )
    tmdwf_fourier_parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory for TMDWF Fourier tables, sample outputs, and plots",
    )

    tmdwf_cs_kernel_parser = subparsers.add_parser(
        "tmdwf-cs-kernel",
        help="Run downstream TMDWF Collins-Soper kernel extraction from a text input file.",
    )
    tmdwf_cs_kernel_parser.add_argument(
        "input_file",
        help="Input control file for the TMDWF CS-kernel extraction workflow",
    )
    tmdwf_cs_kernel_parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory for TMDWF CS-kernel tables, bootstrap samples, diagnostics, and plots",
    )

    tmdwf_cs_kernel_average_parser = subparsers.add_parser(
        "tmdwf-cs-kernel-average",
        help="Average CS-kernel results over selected x-ranges and reference momenta.",
    )
    tmdwf_cs_kernel_average_parser.add_argument(
        "input_file",
        help="Input control file for the TMDWF CS-kernel averaging workflow",
    )
    tmdwf_cs_kernel_average_parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory for averaged CS-kernel tables and samples",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ss-2pt-tgevp":
        outputs = run_ss_2pt_tgevp(
            args.input_file,
            binsize=args.binsize,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_size=args.bootstrap_size,
            seed=args.seed,
            results_dir=args.results_dir,
        )
        for output in outputs:
            print(output)
        return

    if args.command == "2pt-nstate-fit":
        outputs = run_nstate_fit(
            args.input_file,
            results_dir=args.results_dir,
        )
        for output in outputs:
            print(output)
        return

    if args.command == "2pt-effective-mass":
        outputs = run_effective_mass_workflow(
            args.input_file,
            results_dir=args.results_dir,
        )
        for output in outputs:
            print(output)
        return

    if args.command == "tmdwf-nstate-fit":
        outputs = run_tmdwf_nstate_fit(
            args.input_file,
            results_dir=args.results_dir,
        )
        for output in outputs:
            print(output)
        return

    if args.command == "tmdwf-normalize":
        outputs = run_tmdwf_normalization(
            args.input_file,
            results_dir=args.results_dir,
        )
        for output in outputs:
            print(output)
        return

    if args.command == "tmdwf-fourier":
        outputs = run_tmdwf_fourier_workflow(
            args.input_file,
            results_dir=args.results_dir,
        )
        for output in outputs:
            print(output)
        return

    if args.command == "tmdwf-cs-kernel":
        outputs = run_tmdwf_cs_kernel_workflow(
            args.input_file,
            results_dir=args.results_dir,
        )
        for output in outputs:
            print(output)
        return

    if args.command == "tmdwf-cs-kernel-average":
        outputs = run_tmdwf_cs_kernel_average_workflow(
            args.input_file,
            results_dir=args.results_dir,
        )
        for output in outputs:
            print(output)
        return

    root = Path.cwd()
    print(f"lqcd-analysis scaffold ready at {root}")


if __name__ == "__main__":
    main()
