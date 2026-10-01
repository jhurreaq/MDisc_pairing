"""Resolve case paths and run experiments independently of the caller's directory."""

from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
import os
import warnings


CASES = {
    "synthetic": ("Isotropic.Benchmarks.workflow", Path("Isotropic/Benchmarks")),
    "treloar": ("Isotropic.Treloar.workflow", Path("Isotropic/Treloar")),
    "cardiac": ("Anisotropic.workflow", Path("Anisotropic")),
}


@contextmanager
def _working_directory(directory):
    previous = Path.cwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)


def run_case(case, *, project_root=None):
    """Run a complete original experiment with its original inputs and output paths.

    Each invocation creates fresh case state. Experiments run sequentially because
    the original workflows use process-wide NumPy randomness and stdout logging.
    Use separate processes for concurrent runs, with separate project roots.
    """
    if case not in CASES:
        raise ValueError(f"Unknown case {case!r}; choose from {', '.join(CASES)}")
    root = Path(project_root).expanduser().resolve() if project_root is not None else Path(__file__).resolve().parent.parent
    module_name, relative = CASES[case]
    case_dir = root / relative
    workflow = import_module(module_name)
    with _working_directory(case_dir), warnings.catch_warnings():
        return workflow.run(case_dir=case_dir)
