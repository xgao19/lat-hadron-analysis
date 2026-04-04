from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lqcd_analysis.plotting_2pt import plot_nstate_outputs


def main() -> None:
    parser = argparse.ArgumentParser(prog="plot-2pt-results")
    parser.add_argument("output_dir")
    parser.add_argument("correlator_table")
    parser.add_argument("meff_table")
    parser.add_argument("fit_table")
    parser.add_argument("--nstates", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--nt", type=int, required=True)
    args = parser.parse_args()

    outputs = plot_nstate_outputs(
        output_dir=args.output_dir,
        correlator_table=args.correlator_table,
        meff_table=args.meff_table,
        fit_table=args.fit_table,
        nstates=args.nstates,
        model=args.model,
        title=args.title,
        nt=args.nt,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
