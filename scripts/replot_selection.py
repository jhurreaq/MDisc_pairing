#!/usr/bin/env python3
"""Redraw criteria/heatmap PDFs from results.json, leaving numerical files intact."""
import argparse
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from mdisc.reporting import replot_selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path, help='Path to a saved results.json')
    parser.add_argument('--output-dir', type=Path,
                        help='Write into this folder instead of replacing existing selection PDFs')
    args = parser.parse_args()
    destination = replot_selection(args.report, args.output_dir)
    print(f'Selection plots saved in {destination}')


if __name__ == '__main__':
    main()
