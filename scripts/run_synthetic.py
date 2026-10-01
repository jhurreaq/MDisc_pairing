#!/usr/bin/env python3
"""Run the synthetic experiment using the shared MDisc project."""

from pathlib import Path
import sys

# Support execution from a checkout as well as an installed package.
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from mdisc.cli import main

if __name__ == "__main__":
    raise SystemExit(main(default_case="synthetic"))
