"""Explicit case paths; no process-global cwd, stdout, warning or RNG mutation."""
from importlib import import_module
from pathlib import Path

CASES = {'synthetic': ('Isotropic.Benchmarks', Path('Isotropic/Benchmarks')),
         'treloar': ('Isotropic.Treloar', Path('Isotropic/Treloar')),
         'cardiac': ('Anisotropic', Path('Anisotropic'))}


def run_case(case, *, project_root=None, input_dir=None, output_dir=None, configuration=None, plots=True):
    if case not in CASES:
        raise ValueError(f'Unknown case: {case}')
    module, relative = CASES[case]
    # Installed packages use the caller's workspace, never site-packages.
    source_root = Path(__file__).resolve().parent.parent
    root = Path(project_root).expanduser().resolve() if project_root is not None else (
        source_root if (source_root / '.git').exists() else Path.cwd())
    input_dir = Path(input_dir).expanduser().resolve() if input_dir is not None else root / relative / 'input'
    output_dir = Path(output_dir).expanduser().resolve() if output_dir is not None else root / relative / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    workflow = import_module(f'{module}.workflow')
    configuration = configuration or import_module(f'{module}.configuration').Configuration()
    filename = 'log_clean.txt' if case == 'synthetic' else 'log.txt'
    with (output_dir / filename).open('w') as log:
        print(f'Starting {case} model discovery', file=log)
        try:
            return workflow.run(input_dir=input_dir, output_dir=output_dir, configuration=configuration, log=log, plots=plots)
        except Exception as error:
            print(f'FAILED: {type(error).__name__}: {error}', file=log, flush=True)
            raise
