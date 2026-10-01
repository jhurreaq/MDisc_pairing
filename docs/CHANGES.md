# Changes in the principled implementation

This replaces the compatibility implementation at `ea0a3c6` on `dev_MDisc_app`. Numerical changes are deliberate. The experimental data, three-case separation and prediction-figure templates are retained; `main` remains at `eaa8c20`.

| Priority | Issue addressed | Implementation |
| --- | --- | --- |
| P0 | Multiple inconsistent formulas for the same material | A typed energy registry and one analytical energy derivative provide feature matrices, final predictions and nonlinear Jacobians. |
| P0 | Cardiac anisotropic linear terms violate the required reference response | Permanently exclude original IDs 9, 10, 13, 14, 17, 18, 21, 22, 25, 26, 29, 30; preserve all 32 reported positions. |
| P0 | Cardiac surrogate omits fibre contributions and stretch factors | Use full tensor differentiation and push-forward for both biaxial and shear measurements. |
| P0 | Stress-measure, pressure and unit ambiguity | Explicit nominal/Cauchy observation contracts, native units, incompressible kinematics and boundary-condition pressure. |
| P0 | Tension-only invariant treated inconsistently | Differentiate its inactive-compression branch; keep the squared reference response continuously differentiable. |
| P0 | Constitutive stress intercept gives spurious reference stress | No stress intercept in any constitutive discovery/refit. Correct weighted centering in the optional generic affine helper. |
| P0 | Scaling, weights and support selection leak validation information | Repeat weighting, scaling, path generation, support selection and nonlinear calibration inside every training fold. |
| P0 | Duplicated or paired stresses counted as independent observations | Drop copied Treloar EBT components and group genuinely paired Kawabata/cardiac stresses. |
| P0 | Positive amplitudes treated as a stability certificate | Check energy, reference stress, symmetry and sampled isochoric rank-one curvature; reject failing candidates and report the absence of global certification. |
| P0 | Mechanical correctness has no independent acceptance gate | Independent derivative/conjugacy/objectivity/reference tests and a full deterministic two-run output validator; `make check` runs both. |
| P1 | Undocumented LARS mode and incompatible positivity flag | Use actual documented LAR support proposals with nonnegative physical refits; record rank degeneracy. |
| P1 | Signed OMP/refits allow inconsistent material amplitudes | Explicit nonnegative OMP and checked nonnegative calibration throughout. |
| P1 | Refits optimize a different loss than discovery | Share training weights, scales, dimensionless ridge policy and support tolerance. |
| P1 | Cardiac nonlinear shapes calibrated only after model selection | Calibrate within CV and before AIC/BIC scoring; use analytic Jacobians, deterministic multistart and recorded bounds. |
| P1 | Exponential clipping changes the constitutive equation | Use genuine exponentials and explicit parameter-domain bounds; invalid evaluations raise errors. |
| P1 | Different likelihoods and arbitrary complexity/variance rules | One conditional Gaussian working likelihood, weight determinant, variance parameter and local effective dimension. |
| P1 | Same-cardinality candidate supports silently discarded | Preserve different supports of the same size; cache only identical refits. |
| P1 | Tiny coefficients and nonlinear parameters counted inconsistently | One dimensionless amplitude tolerance with reduced-support recalibration. |
| P1 | CV graphics show a transformed or misleading quantity | Render actual dimensionless validation loss and fold standard deviation, with accurate criteria labels. |
| P1 | Training comparisons presented as external validation | Label selection CV and fitting metrics; report Kawabata as held out only when excluded from fitting. |
| P1 | CSV parsing drops the first Kawabata/reference row | Parse headers explicitly and validate columns and finite values. |
| P1 | Failed solvers or malformed input silently produce fabricated values | Checked solver fallback, explicit input/weight validation, propagated failures and useful error logs. |
| P2 | Three duplicated discovery engines and obsolete utilities | Delete case-specific discovery implementations; keep one shared engine and thin case workflows. |
| P2 | Mathematical meaning encoded in plot labels | Use term metadata, with labels reserved for display. |
| P2 | Inactive or misleading library configuration | Build libraries from the configured MR order and actual exponent tuples; reject invalid settings. |
| P2 | Plotting recomputes numerical fits or metrics | Rendering consumes already-computed model results and evaluation metrics. |
| P2 | Configuration stores runtime palettes/data | Immutable scientific settings and separate runtime rendering state. |
| P2 | Working-directory, stdout, warning and global RNG side effects | Explicit paths/log streams, local RNGs and narrowly scoped numerical warnings. |
| P2 | Hardcoded paths and case-specific invocation | Centralized CLI/scripts, configurable input/output roots and installed-wheel support. |
| P2 | Redundant dependency declarations | Package metadata is the dependency source; requirements delegates to it. |
| P2 | Missing data/result provenance | Input manifest, source and input hashes, effective configuration, library IDs and dependency versions in results. |
| P2 | Colliding or misleading plot filenames | Consistent method/criterion names; retain original prediction layouts and palettes. |
| P2 | Obsolete exact-output acceptance checks | Preserve historical checks in Git; replace active acceptance with mechanics tests and corrected-output reproducibility. |

The mathematical assumptions and remaining limits are described in [MECHANICS.md](MECHANICS.md). A successful sparse fit is not a uniqueness proof, a global nonlinear-optimality certificate or a global stability proof. The implementation reports these distinctions explicitly.
