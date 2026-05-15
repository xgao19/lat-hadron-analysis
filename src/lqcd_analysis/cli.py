from __future__ import annotations

import argparse
from pathlib import Path

from .DA.fourier import run_da_fourier_workflow
from .DA.fit_nstate import run_da_nstate_fit
from .DA.normalize import run_da_normalization
from .DA.ratio_fourier_t import run_da_ratio_fourier_t_workflow
from .DA.x_nstate_fit import run_da_x_nstate_fit_workflow
from .DA.xfit_normalize import run_da_xfit_normalization
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

    da_parser = subparsers.add_parser(
        "da-nstate-fit",
        help="Run bootstrap-based DA N-state fits from a text input file.",
    )
    da_parser.add_argument("input_file", help="Input control file for the DA fit workflow")
    da_parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory for DA fit tables, sample outputs, and summaries",
    )

    da_normalize_parser = subparsers.add_parser(
        "da-normalize",
        help="Run downstream DA matrix-element normalization from a text input file.",
    )
    da_normalize_parser.add_argument(
        "input_file",
        help="Input control file for the DA normalization workflow",
    )
    da_normalize_parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory for normalized DA tables, sample outputs, and summaries",
    )

    da_fourier_parser = subparsers.add_parser(
        "da-fourier",
        help="Run downstream DA Fourier postprocessing from a text input file.",
    )
    da_fourier_parser.add_argument(
        "input_file",
        help="Input control file for the DA Fourier workflow",
    )
    da_fourier_parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory for DA Fourier tables, sample outputs, and plots",
    )

    da_ratio_fourier_t_parser = subparsers.add_parser(
        "da-ratio-fourier-t",
        help="Build DA ratio Fourier data q(x,t) from raw ratio inputs.",
    )
    da_ratio_fourier_t_parser.add_argument(
        "input_file",
        help="Input control file for the DA ratio-to-Fourier-per-t workflow.",
    )
    da_ratio_fourier_t_parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory for q(x,t) tables and bootstrap samples.",
    )

    da_x_nstate_fit_parser = subparsers.add_parser(
        "da-x-nstate-fit",
        help="Fit q(x,t) with the DA N-state time model at each x.",
    )
    da_x_nstate_fit_parser.add_argument(
        "input_file",
        help="Input control file for the DA x-space N-state fit workflow.",
    )
    da_x_nstate_fit_parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory for x-space fit tables and bootstrap samples.",
    )

    da_xfit_normalize_parser = subparsers.add_parser(
        "da-xfit-normalize",
        help="Normalize x-space DA fit outputs using old bare matrix-element outputs.",
    )
    da_xfit_normalize_parser.add_argument(
        "input_file",
        help="Input control file for the DA x-fit normalization workflow.",
    )
    da_xfit_normalize_parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory for normalized x-space fit tables and bootstrap samples.",
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

    if args.command == "da-nstate-fit":
        outputs = run_da_nstate_fit(
            args.input_file,
            results_dir=args.results_dir,
        )
        for output in outputs:
            print(output)
        return

    if args.command == "da-normalize":
        outputs = run_da_normalization(
            args.input_file,
            results_dir=args.results_dir,
        )
        for output in outputs:
            print(output)
        return

    if args.command == "da-fourier":
        outputs = run_da_fourier_workflow(
            args.input_file,
            results_dir=args.results_dir,
        )
        for output in outputs:
            print(output)
        return

    if args.command == "da-ratio-fourier-t":
        outputs = run_da_ratio_fourier_t_workflow(
            args.input_file,
            results_dir=args.results_dir,
        )
        for output in outputs:
            print(output)
        return

    if args.command == "da-x-nstate-fit":
        outputs = run_da_x_nstate_fit_workflow(
            args.input_file,
            results_dir=args.results_dir,
        )
        for output in outputs:
            print(output)
        return

    if args.command == "da-xfit-normalize":
        outputs = run_da_xfit_normalization(
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
