"""Centralized command-line interface for all three configured experiments."""
import argparse
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from .runner import CASES, run_case


def main(argv=None, *, default_case=None):
    import matplotlib
    matplotlib.use('Agg')  # CLI exports files; it never needs a GUI backend.
    parser = argparse.ArgumentParser(description='Mechanics-based sparse constitutive model discovery')
    if default_case is None:
        parser.add_argument('case', choices=[*CASES, 'all'])
    parser.add_argument('--project-root', type=Path)
    parser.add_argument('--input-dir', type=Path, help='Case input directory; use --project-root with all')
    parser.add_argument('--output-dir', type=Path, help='Case output directory; with all, contains case subdirectories')
    parser.add_argument('--folds', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--alpha-count', type=int)
    parser.add_argument('--path-count', type=int)
    parser.add_argument('--noise-levels', nargs='+', type=float)
    parser.add_argument('--points', type=int, help='Synthetic samples per mode')
    parser.add_argument('--models', nargs='+', help='Synthetic ground-truth model names')
    parser.add_argument('--fit-datasets', nargs='+', choices=('Treloar','Kawabata'))
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)
    case = default_case or args.case
    if case == 'all' and args.input_dir is not None:
        parser.error('--input-dir requires one case; use --project-root for all')
    if args.noise_levels and any(level < 0 or level >= 1 for level in args.noise_levels):
        parser.error('Noise levels must satisfy 0 <= noise < 1')
    for name in CASES if case == 'all' else (case,):
        module, _ = CASES[name]
        cfg = import_module(f'{module}.configuration').Configuration()
        discovery_changes = {key: getattr(args,key) for key in ('folds','seed') if getattr(args,key) is not None}
        path_changes = {key: getattr(args,key) for key in ('alpha_count','path_count') if getattr(args,key) is not None}
        cfg = replace(cfg, discovery=replace(cfg.discovery, **discovery_changes,
                                            paths=replace(cfg.discovery.paths, **path_changes)))
        if name in ('synthetic','cardiac') and args.noise_levels is not None:
            cfg = replace(cfg, noise_levels=tuple(args.noise_levels))
        if name == 'synthetic':
            if args.points is not None:
                cfg = replace(cfg, points=args.points)
            if args.models is not None:
                unknown = set(args.models) - {truth.name for truth in cfg.ground_truths}
                if unknown:
                    parser.error(f'Unknown synthetic models: {sorted(unknown)}')
                cfg = replace(cfg, ground_truths=tuple(t for t in cfg.ground_truths if t.name in args.models))
        if name == 'treloar' and args.fit_datasets is not None:
            cfg = replace(cfg, fit_datasets=tuple(args.fit_datasets))
        output_dir = args.output_dir / name if case == 'all' and args.output_dir is not None else args.output_dir
        try:
            run_case(name, project_root=args.project_root, input_dir=args.input_dir,
                     output_dir=output_dir, configuration=cfg, plots=not args.no_plots)
        except (ValueError, FileNotFoundError, RuntimeError, FloatingPointError) as error:
            parser.exit(1, f'{name} failed: {error}\n')
    return 0
