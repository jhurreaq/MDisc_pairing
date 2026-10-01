#!/usr/bin/env python3
"""Compare complete workflows with their original Git versions, without changing local outputs.

Run with the same pinned scientific environment as the experiments. Validation
controls timestamps, PDF metadata dates, initial NumPy randomness, Python's hash
seed, and native numerical thread counts. Production runs retain original behavior.
"""

import argparse
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASE_FILES = {
    "synthetic": Path("Isotropic/Benchmarks/MDisc_synthetic_data_rel_noise_20251004_UT_PS_EBT_pos.py"),
    "treloar": Path("Isotropic/Treloar/MDisc_TreloarKawabata_datasets_combined_20250827.py"),
    "cardiac": Path("Anisotropic/HeartCANN_discovered_lasso_lars_omp_two_stages_20250827.py"),
}
EXPECTED_PDFS = {"synthetic": 192, "treloar": 16, "cardiac": 45}


def _canonical(value):
    """Keep full numerical precision, including exact array bytes, in result snapshots."""
    import numpy as np

    if isinstance(value, np.ndarray):
        return {"array_dtype": value.dtype.str, "shape": value.shape,
                "bytes": base64.b64encode(value.tobytes()).decode("ascii")}
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    return value


def _worker(case, mode, work_root):
    # Load C extensions before replacing datetime.datetime: replacing it before
    # importing NumPy/pandas can violate those extensions' datetime ABI checks.
    import numpy as np
    import pandas
    import sklearn
    import matplotlib.pyplot
    import scipy.optimize
    import openpyxl

    class FixedDateTime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 3, 4, 12, 0, 0, tzinfo=tz)

    datetime.datetime = FixedDateTime
    time.time = lambda: 1772625600.0
    np.random.seed(42)
    work_root = Path(work_root)
    relative = CASE_FILES[case]
    case_dir = work_root / relative.parent
    os.chdir(case_dir)
    with (work_root / f"{mode}_stdout.txt").open("w", encoding="utf-8") as stream:
        previous_stdout = sys.stdout
        sys.stdout = stream
        try:
            if mode == "reference":
                namespace = runpy.run_path(str(case_dir / relative.name), run_name="__main__")
                if case == "cardiac":
                    result = {"noise_results": namespace["all_results_by_noise"],
                              "summary": namespace["summary_table_data"]}
                else:
                    result = {"initial_results": namespace["all_initial_results"],
                              "refit_summaries": namespace["all_refit_summaries"]}
            else:
                sys.path.insert(0, str(PROJECT_ROOT))
                from mdisc.runner import run_case
                result = run_case(case, project_root=work_root)
        finally:
            sys.stdout = previous_stdout
    (work_root / f"{mode}_results.json").write_text(
        json.dumps(_canonical(result), sort_keys=True, indent=2), encoding="utf-8")
    versions = {m.__name__: m.__version__ for m in (np, pandas, sklearn, matplotlib, scipy, openpyxl)}
    (work_root / "environment.json").write_text(
        json.dumps({"python": sys.version, "libraries": versions}, indent=2), encoding="utf-8")


def _hashes(directory):
    return {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob("*") if p.is_file()}


def _run_worker(case, mode, work_root, environment):
    with (work_root / f"{mode}_stderr.txt").open("w", encoding="utf-8") as errors:
        process = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", case, mode, str(work_root)],
            env=environment, stdout=subprocess.PIPE, stderr=errors, text=True,
        )
    if process.returncode:
        raise RuntimeError(f"{case} {mode} failed (exit {process.returncode}); see {errors.name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", default="eaa8c20", help="Git revision containing the original standalone scripts.")
    parser.add_argument("--cases", nargs="+", choices=list(CASE_FILES), default=list(CASE_FILES))
    parser.add_argument("--work-dir", type=Path, help="New directory in which to retain reference/new outputs and the report.")
    args = parser.parse_args(argv)
    reference = subprocess.check_output(
        ["git", "rev-parse", args.reference], cwd=PROJECT_ROOT, text=True).strip()
    if args.work_dir is None:
        workspace = Path(tempfile.mkdtemp(prefix="mdisc-equivalence-"))
    else:
        workspace = args.work_dir.expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    environment.update({
        "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(workspace / "matplotlib"),
        "SOURCE_DATE_EPOCH": "1772625600", "PYTHONHASHSEED": "0",
        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
    })
    report = {"reference": reference, "cases": {}, "passed": True,
              "controls": "Fixed clock/PDF date, initial NumPy seed 42, hash seed 0, native threads 1"}
    print(f"Validation workspace: {workspace}", flush=True)
    for case in args.cases:
        relative = CASE_FILES[case]
        work_root = workspace / case
        case_dir = work_root / relative.parent
        case_dir.mkdir(parents=True)
        source = subprocess.check_output(["git", "show", f"{reference}:{relative.as_posix()}"], cwd=PROJECT_ROOT)
        (case_dir / relative.name).write_bytes(source)
        inputs = PROJECT_ROOT / relative.parent / "input"
        if inputs.exists():
            shutil.copytree(inputs, case_dir / "input")
        print(f"{case}: running complete original workflow…", flush=True)
        _run_worker(case, "reference", work_root, environment)
        outputs = case_dir / "outputs"
        reference_outputs = work_root / "reference_outputs"
        outputs.rename(reference_outputs)
        print(f"{case}: running complete refactored workflow…", flush=True)
        _run_worker(case, "refactored", work_root, environment)
        expected = _hashes(reference_outputs)
        actual = _hashes(outputs)
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        different = sorted(name for name in expected.keys() & actual.keys() if expected[name] != actual[name])
        reference_stdout = (work_root / "reference_stdout.txt").read_bytes()
        refactored_stdout = (work_root / "refactored_stdout.txt").read_bytes()
        reference_results = (work_root / "reference_results.json").read_bytes()
        refactored_results = (work_root / "refactored_results.json").read_bytes()
        stdout_equal = reference_stdout == refactored_stdout
        results_equal = reference_results == refactored_results
        pdf_count = sum(name.endswith(".pdf") for name in expected)
        passed = not (missing or extra or different) and stdout_equal and results_equal and pdf_count == EXPECTED_PDFS[case]
        report["cases"][case] = {
            "passed": passed, "pdfs": pdf_count, "output_files": len(expected),
            "stdout_equal": stdout_equal, "numerical_results_equal": results_equal,
            "reference_stdout_sha256": hashlib.sha256(reference_stdout).hexdigest(),
            "refactored_stdout_sha256": hashlib.sha256(refactored_stdout).hexdigest(),
            "reference_results_sha256": hashlib.sha256(reference_results).hexdigest(),
            "refactored_results_sha256": hashlib.sha256(refactored_results).hexdigest(),
            "missing": missing, "extra": extra, "different": different,
            "reference_hashes": expected, "refactored_hashes": actual,
            "environment": json.loads((work_root / "environment.json").read_text()),
            "input_hashes": _hashes(case_dir / "input") if inputs.exists() else {},
        }
        report["passed"] &= passed
        print(f"{case}: {'PASS' if passed else 'FAIL'} — {pdf_count} PDFs, logs, and exact numerical results", flush=True)
        (workspace / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {workspace / 'report.json'}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        _worker(*sys.argv[2:])
    else:
        raise SystemExit(main())
