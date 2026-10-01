# MDisc_pairing

Discover incompressible hyperelastic constitutive models with LASSO, least-angle regression (LAR), and nonnegative orthogonal matching pursuit (OMP). Three experiments share one mechanics and discovery implementation:

- **Isotropic/Benchmarks**: synthetic Mooney–Rivlin/Ogden rediscovery.
- **Isotropic/Treloar**: Treloar fitting and optional Kawabata fitting; untrained datasets are reported separately as held-out evaluations.
- **Anisotropic**: cardiac shear/biaxial measurements and nonlinear refinement of exponential terms.

Version 0.2 corrects the numerical and mechanical inconsistencies of the historical scripts. Results intentionally differ from `eaa8c20` and the preservation refactor `ea0a3c6`. `main` is unchanged; development is on `dev_MDisc_app`. [CHANGES.md](docs/CHANGES.md) maps the audit issues to their fixes.

## Install

Use Python 3.9–3.11. The validation environment is Python 3.9.6 with the pinned scientific dependencies in `pyproject.toml`.

```bash
python3 -m venv .venvMDisc
source .venvMDisc/bin/activate
python -m pip install -e .
```

`requirements.txt` delegates to the package metadata; dependency versions are defined once. Experimental inputs are external and ignored by Git. Place the files listed in [data_manifest.json](data_manifest.json) in their indicated case input folders. The manifest records the local validation files' hashes, units, stress measures and source information. Synthetic runs need no external inputs.

## Run

```bash
python scripts/run_synthetic.py
python scripts/run_treloar.py
python scripts/run_cardiac.py
# Equivalent installed entry points:
mdisc synthetic
mdisc treloar
mdisc cardiac
mdisc all
```

Runs default to the checkout's original case `outputs/` directories. An installed wheel defaults to the caller's workspace, never its installation directory. Explicit paths are supported:

```bash
mdisc cardiac --input-dir /path/to/cardiac/input --output-dir /path/to/new/results
mdisc all --project-root /path/to/MDisc_pairing --output-dir /path/to/new/results
```

All runs write a log, selection panels, activation heatmaps, prediction figures and `results.json`. Synthetic retains `log_clean.txt`; the other cases use `log.txt`, inside their output directories. The original plasma/Blues palettes, font defaults and case-specific prediction layouts are retained. Criteria labels and contents now describe the actual computations. PDF timestamps are omitted for reproducibility.

Heatmaps show **binary activation of the final refitted terms**: blue means active, white means inactive. Rows retain the original term IDs and show only terms active somewhere along the path. Columns use the same penalty/path-fraction grid as the criteria panels. [PLOTTING.md](docs/PLOTTING.md) explains the axes, selection markers, CV band and cardiac expressions.

Redraw the heatmaps and criteria panels from an existing numerical report without rerunning discovery:

```bash
python scripts/replot_selection.py Anisotropic/output_mdisc_app/results.json
# Keep existing plots and write the redrawn selection PDFs elsewhere:
python scripts/replot_selection.py Anisotropic/output_mdisc_app/results.json --output-dir /tmp/cardiac-plots
```

This command leaves logs, numerical reports and prediction plots unchanged. The numerical report's source hash continues to identify the implementation that originally computed its results.

Use `--no-plots` for numerical runs. Settings can be overridden without editing scripts:

```bash
mdisc synthetic --models MR2 MR1O1 --noise-levels 0 0.05 --points 60
mdisc treloar --fit-datasets Treloar Kawabata
mdisc cardiac --folds 5 --seed 42 --alpha-count 40 --path-count 21
```

Use the centralized `scripts/run_*.py` launchers or the installed `mdisc` command. The dated compatibility launchers have been removed.

## Project structure

```text
mdisc/
  mechanics.py       # Typed energy terms, analytical derivatives, stress measures, pressure
  data.py            # Independent isotropic stress observations and measurement groups
  preprocessing.py   # Training-only loss weights and origin-preserving scaling
  paths.py           # LASSO, documented LAR, nonnegative OMP
  regression.py      # Checked bounded least squares and effective dimension
  discovery.py       # Shared grouped CV, support refits and model selection
  criteria.py        # One conditional Gaussian working likelihood
  evaluation.py      # Metrics and mechanical admissibility diagnostics
  rendering.py       # Shared selection/heatmap rendering
  reporting.py       # Result contracts, logs and provenance
  runner.py, cli.py  # Explicit paths and centralized execution
Isotropic/Benchmarks/ # Configuration, generated data and prediction template
Isotropic/Treloar/    # Configuration, CSV loader and prediction templates
Anisotropic/          # Configuration, Excel loader, nonlinear refiner and prediction template
scripts/              # Centralized entry points and reproducibility validation
```

Case configuration dataclasses contain the effective settings. Runtime data and Matplotlib palettes are separate. Mathematical evaluation uses term metadata; display strings are not parsed to reconstruct equations. Plotting modules receive already-computed results and never fit models.

## Mechanics and statistical contracts

Stress derives from a reference-volume strain energy. The data determine whether nominal stress or Cauchy stress is required. Incompressibility is enforced through the deformation and pressure reaction; a free stress intercept is never fitted. [MECHANICS.md](docs/MECHANICS.md) documents equations, pressure conditions, cardiac exclusions, weighting, validation scope and limitations.

The cardiac registry preserves all original **32 positions**. The anisotropic linear and exponential-linear terms are excluded from the energy evaluator and optimizer. Their reported coefficients remain exactly zero; discovery operates on the other 20 terms. Isotropic linear terms remain because their reference stress is hydrostatic and cancels with pressure.

Every CV fold independently estimates weights/scales, generates supports, refits amplitudes and, for cardiac tissue, calibrates exponential parameters. Stress components from one deformation stay in the same fold. Copied Treloar EBT stresses are not counted twice. Final refits use the same loss policy as discovery.

LAR is the documented `method='lar'` algorithm and proposes supports; the physical coefficient refit enforces nonnegativity. OMP is explicitly the nonnegative variant. Numerically dependent libraries can admit nonunique parameterizations; rank and solver diagnostics are included in reports. Conditional AIC/BIC and selection CV are not claimed to be independent estimates of generalization error.

## Python API

```python
from dataclasses import replace
from mdisc import run_case
from Anisotropic.configuration import Configuration

configuration = Configuration()
configuration = replace(configuration, noise_levels=(0.0,))
results = run_case("cardiac", input_dir="Anisotropic/input",
                   output_dir="/tmp/cardiac-results", configuration=configuration)
```

Each result includes full coefficient vectors, original term IDs, nonlinear parameters, predictions, selected paths, CV membership, effective settings, mechanical diagnostics, input/source hashes and dependency versions. The API raises explicit errors on invalid input or failed solvers. It does not change the working directory, NumPy's global RNG, global warning filters or stdout redirection. Use separate processes when rendering concurrently because Matplotlib is not thread-safe.

## Validation

```bash
python -m unittest discover -s tests -v
python scripts/validate_results.py --quick
python scripts/validate_results.py              # full grids, all cases, all noise levels
# Run both acceptance gates:
make check
```

The validator runs each experiment twice in isolated folders and compares **all result, log and PDF bytes**, using deterministic seeds and native thread counts. It never overwrites the project's existing case outputs. Unit tests independently check energy derivatives, work conjugacy, objectivity, stress symmetry, reference-state conditions, nonlinear Jacobians/calibration, weighting and fold boundaries. [VALIDATION.md](docs/VALIDATION.md) records the completed validation.

The historical exact comparisons with `eaa8c20` remain available in Git at `ea0a3c6`; they established compatibility of the earlier refactor, not correctness of its mechanics. Current acceptance checks target mathematical contracts and reproducibility of the corrected formulation.

## Citation and license

See [CITATION.cff](CITATION.cff) and [LICENSE](LICENSE). Primary cardiac formulation/data references are linked in the mechanics documentation.
