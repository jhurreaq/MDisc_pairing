# MDisc_pairing

Code for the paper [Automated Constitutive Model Discovery by Pairing Sparse Regression Algorithms with Model Selection Criteria](https://arxiv.org/abs/2509.16040).

The project discovers constitutive models with LASSO, LARS, and OMP paired with cross-validation, AIC, and BIC. It includes synthetic isotropic benchmarks, Treloar/Kawabata isotropic experiments, and anisotropic cardiac tissue experiments.

## Installation and execution

Use Python 3.9 or 3.10 and the pinned scientific dependencies. From this repository:

```bash
python3 -m venv venvMDisc
source venvMDisc/bin/activate
python -m pip install -e .
```

Run a case from the centralized scripts folder:

```bash
python scripts/run_synthetic.py
python scripts/run_treloar.py
python scripts/run_cardiac.py
```

The same workflows are available through the package and installed command:

```bash
python -m mdisc synthetic
mdisc treloar
mdisc cardiac
mdisc all
```

`all` runs synthetic, Treloar, and cardiac experiments sequentially. Each case creates fresh settings and state. After installation, `mdisc` can be called from any working directory. The centralized scripts can also be called by their absolute paths without installing the package.

To use another root containing the same input/output folder layout:

```bash
mdisc cardiac --project-root /path/to/MDisc_pairing
```

The three original dated scripts remain as compatibility entry points; they call the same package workflows.

## Structure

```text
MDisc_pairing/
├── pyproject.toml
├── requirements.txt
├── mdisc/                         # Reusable functionality
│   ├── cli.py                     # Command-line interface
│   ├── runner.py                  # Case selection and path resolution
│   ├── configuration.py           # Original common plot settings/palettes
│   ├── isotropic.py               # Invariants, derivatives, MR terms, LaTeX names
│   ├── numerics.py                # Original safe-power/log variants
│   ├── preprocessing.py           # Scaling, stress weights, column normalization
│   ├── regression.py              # LASSO fitting, NNLS/Ridge refits, CV plot data
│   ├── paths.py                   # LARS support handling and positive OMP paths
│   └── criteria.py                # Original isotropic/weighted AIC and BIC
├── scripts/
│   ├── run_synthetic.py
│   ├── run_treloar.py
│   ├── run_cardiac.py
│   └── validate_equivalence.py
├── Isotropic/
│   ├── Benchmarks/
│   │   ├── configuration.py
│   │   ├── data.py                # Synthetic data generation
│   │   ├── models.py              # Isotropic library/design matrix/consistency
│   │   ├── discovery.py           # Case-specific regression/selection policies
│   │   ├── plotting.py            # Original prediction/heatmap/criteria templates
│   │   ├── workflow.py            # Original scenarios and reporting
│   │   ├── MDisc_synthetic_data_rel_noise_20251004_UT_PS_EBT_pos.py
│   │   └── outputs/
│   └── Treloar/                   # Same module separation for experimental data
│       ├── MDisc_TreloarKawabata_datasets_combined_20250827.py
│       ├── input/
│       └── outputs/
├── Anisotropic/                   # Cardiac data, models, plots, and workflow
│   ├── configuration.py
│   ├── data.py
│   ├── models.py
│   ├── discovery.py
│   ├── refinement.py              # Original Stage 2 nonlinear optimization
│   ├── plotting.py
│   ├── workflow.py
│   ├── HeartCANN_discovered_lasso_lars_omp_two_stages_20250827.py
│   ├── input/
│   └── outputs/
└── tests/
    └── test_contracts.py
```

Case modules accept an explicit `context` rather than reading mutable globals from a standalone script. Importing modules does not load datasets, seed random generators, create outputs, or execute an experiment. The runner retains the original process-wide random-number and stdout behavior while running a case. Run concurrent experiments in separate processes and separate roots.

## Preserved numerical behavior

This refactor starts from `eaa8c20` on `main`. It preserves the original scenario ordering, data loading and filtering, unit conversions, random seed choices, regression grids, support thresholds, CV splits, accumulation order, normalization, coefficient back-transforms, refits, and optimizer settings.

The workflows have different policies, so those remain explicit:

| Case | Original preprocessing and refit behavior |
| --- | --- |
| Synthetic | Original per-mode stress normalization; StandardScaler fitted on the first design matrix and applied to the final matrix; target remains in its original weighted units; original positive refits and Ridge alpha `1e-7`. |
| Treloar | Original data combination and fit-subset selection; StandardScaler applied to both X and y on the fitting subset; evaluation on full unweighted data; original NNLS fallbacks and Ridge alpha `1e-6`. |
| Cardiac | Original inverse-RMS stress weights; row multiplication by sqrt(weight); weighted column L2 normalization; coefficient division by column scales; original positive refits and Stage 2 L-BFGS-B refinement. |

Synthetic and Treloar retain their distinct OMP residual multiplication orders and LARS support policies. Cardiac retains the original C-based anisotropic invariants, tensor prediction, and Martonova reference parameters. The separate `dev_cofC` branch is not incorporated into this branch.

Keep the pinned dependency versions for reproduction, particularly scikit-learn `1.3.0`. The original solver settings and calls are preserved. Changing scientific dependencies, numerical libraries, fonts, or thread settings can change results or PDF serialization.

## Inputs and outputs

External datasets are kept in their original case directories and are not packaged or committed:

| Case | Required files | Source |
| --- | --- | --- |
| Cardiac | `Anisotropic/input/CANNsHEARTdata_shear05.xlsx` | [CANN heart tissue data](https://github.com/LivingMatterLab/CANN/blob/main/HEART/input/CANNsHEARTdata_shear05.xlsx) |
| Treloar/Kawabata | `Isotropic/Treloar/input/Kawabata.csv`, `TreloarDataEBT.csv`, `TreloarDataPS.csv`, `TreloarDataUT.csv` | [Zenodo dataset, DOI 10.5281/zenodo.14995273](https://doi.org/10.5281/zenodo.14995273) |
| Synthetic | No external inputs | Generated by the original framework |

Output filenames, scenario directories, plot sizes, fonts, colors, axes, legends, annotations, and reporting text retain the original templates:

- Synthetic: `Isotropic/Benchmarks/outputs/<scenario>/`, plus `outputs/log_clean.txt`.
- Treloar: `Isotropic/Treloar/outputs/Fit_Treloar_Only/` by default.
- Cardiac: `Anisotropic/outputs/RelNoise0pct/`, `RelNoise5pct/`, and `RelNoise10pct/`.

Treloar and cardiac print their logs to stdout, as the originals did. To save them:

```bash
python scripts/run_treloar.py > Isotropic/Treloar/log.txt 2>&1
python scripts/run_cardiac.py > Anisotropic/log.txt 2>&1
```

Running a case overwrites matching generated output files, as in the original scripts. Use `--project-root` with an independent folder layout when preserving an earlier run.

## Python API

```python
from mdisc import run_case

synthetic = run_case("synthetic")
initial_models = synthetic["initial_results"]
final_models = synthetic["refit_summaries"]

cardiac = run_case("cardiac")
noise_results = cardiac["noise_results"]
summary_rows = cardiac["summary"]
```

The isotropic cases return initial selection results and refit summaries. Cardiac returns the initial/refined results per noise level and the detailed summary rows. These contain the original coefficient arrays and metrics at full numerical precision; no new result files are written by the API.

## Verification

Fast numerical, state-isolation, CLI, and import checks:

```bash
python -m unittest discover -s tests -v
```

Complete original-versus-refactored comparisons, using the locally available datasets:

```bash
python scripts/validate_equivalence.py --reference eaa8c20
```

The validator copies original scripts from Git and copies inputs into a new temporary directory. It runs every original and refactored scenario, compares the complete file inventory and SHA-256 bytes of all PDFs and logs, compares stdout, and checks the exact bytes of numerical coefficient arrays and full-precision metrics. It retains both output sets and a JSON report. To select a case or retain results in a chosen new directory:

```bash
python scripts/validate_equivalence.py --cases treloar --work-dir /tmp/mdisc-treloar-validation
```

Normal runs contain changing timestamps, measured durations, and PDF creation dates. The original Treloar workflow also uses unseeded shuffled folds for LARS/OMP. For a strict byte comparison the validator controls those values identically in both versions; it does not alter production code or replace numerical outputs with rounded comparisons. Exact equivalence is checked in the same pinned environment with the same input data and initial RNG state.

See [docs/REFACTOR_VALIDATION.md](docs/REFACTOR_VALIDATION.md) for the recorded validation results.
