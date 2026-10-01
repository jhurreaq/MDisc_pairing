# Validation of the corrected formulation

Completed on 2026-10-01 with Python 3.9.6 and the scientific dependency versions pinned in `pyproject.toml`. The machine-readable [validation record](validation_report.json) contains the validated source digest, input hashes, effective configurations and output hashes.

## Independent contracts

**35 tests passed** with `python -m unittest discover -s tests -v`. They cover:

- Every enabled energy term's reference energy and pressure-corrected reference stress.
- Independent finite-difference energy gradients and nonlinear stress Jacobians.
- Frame indifference, orthotropic reflections and symmetric Cauchy stress.
- Nominal-stress work conjugacy along an incompressible rank-one path.
- Analytical neo-Hookean stress and the Ogden initial shear modulus.
- Tension-only fibre compression, explicit pressure reactions and invalid deformation/material inputs.
- Original cardiac IDs and permanent exclusion of the 12 specified terms.
- Recovery of a known nonlinear cardiac exponential term.
- Analytical weighted origin-preserving regression, checked fallback and invalid-weight rejection.
- Mode balancing, scaling back to physical coefficients and consistent conditional information criteria.
- Documented LAR invocation, distinct same-size supports, grouped measurements and fold-local discovery/scaling.
- CSV reference-row retention and copied-versus-independent stress components.
- Synthetic generation independent of display labels.
- Public API, centralized script invocation outside the checkout, import side effects and failure-state preservation.

## Full end-to-end reproducibility

`python scripts/validate_results.py` completed two isolated runs of **all default cases, noise levels and model-selection grids**. All **259 output files matched byte for byte**: 253 PDFs, three JSON reports and three logs. Neither run overwrote existing case outputs.

| Case | Scenarios | Selected method/criterion models | PDFs |
| --- | ---: | ---: | ---: |
| Synthetic | 12 | 108 | 192 |
| Treloar/Kawabata | 1 | 9 | 16 |
| Cardiac | 3 | 27 | 45 |
| Total | 16 | 144 | 253 |

All 144 selected models converged and passed the implemented sampled mechanics checks. Their reference stress norms were exactly zero in the validation environment. Every cardiac result retained a full 32-vector with all permanently excluded positions exactly zero. Experimental input hashes were checked against `data_manifest.json`.

The complete local artifacts are in `/private/tmp/mdisc-cleanup-validation-20261001/{first,second}`. They are temporary; the committed validation record preserves their hashes. Re-run the validator to regenerate artifacts in a fresh folder.

## Legacy cleanup comparison

The cleanup was separately compared with the corrected implementation at `cad0faf`. All 253 PDFs and three logs are byte-identical. All three result payloads are exactly equal after excluding the source-hash provenance field, which necessarily changes when unused code is deleted. Model supports, coefficients, nonlinear parameters, predictions, criteria, CV paths and metrics remain exactly the same.

## Installation

A wheel built without downloading dependencies. It contains neither the deleted legacy modules nor data-folder placeholders. It was installed into an isolated target and successfully ran a reduced synthetic discovery from outside the checkout. With no explicit output path, outputs went into the caller's workspace; nothing was written into the installed package. This used the existing pinned dependency environment, rather than a fresh dependency installation.

## Interpretation

These are acceptance checks for the **corrected formulation**, not comparisons with the former numerical results. The historical compatibility record is available in Git at `ea0a3c6`. PDF byte equality is established for repeated runs in the recorded environment; it is not a promise across operating systems, font libraries or numerical backends.

Sampled admissibility is not global strong-ellipticity or polyconvexity certification. Multistart nonlinear fitting does not certify a global optimum, and model-selection CV is not an independent post-selection test. See [MECHANICS.md](MECHANICS.md) for the assumptions and boundaries of each claim.
