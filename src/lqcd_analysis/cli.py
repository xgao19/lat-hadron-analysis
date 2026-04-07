from __future__ import annotations

import argparse
from pathlib import Path

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

    root = Path.cwd()
    print(f"lqcd-analysis scaffold ready at {root}")


if __name__ == "__main__":
    main()
