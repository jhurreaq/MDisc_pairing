"""Command-line entry points for the three original experiments."""

import argparse

from .runner import CASES, run_case


def main(argv=None, *, default_case=None):
    parser = argparse.ArgumentParser(description="Discover constitutive models using the original MDisc workflows.")
    if default_case is None:
        parser.add_argument("case", choices=[*CASES, "all"], help="Experiment to run; all runs the three cases sequentially.")
    parser.add_argument("--project-root", help="Root containing Anisotropic/ and Isotropic/ input and output directories (defaults to the package's project root).")
    args = parser.parse_args(argv)
    case = default_case if default_case is not None else args.case
    for name in CASES if case == "all" else (case,):
        run_case(name, project_root=args.project_root)
    return 0
