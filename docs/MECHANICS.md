# Mechanical and statistical formulation

This project models **passive, incompressible hyperelasticity**. It does not identify a compressible volumetric response, active contraction, rate dependence or damage from these datasets.

## Energy, stress and pressure

The deformation gradient maps material coordinates into spatial coordinates. For reference-volume strain energy `W(F)` and `J = det(F) = 1`,

```text
P = dW/dF - p F^(-T)
sigma = (dW/dF) F^T - p I
P = sigma F^(-T)
```

Pressure is an incompressibility reaction determined by boundary conditions. It is never a fitted intercept. `EnergyLibrary.stress(..., pressure=p)` accepts an externally determined pressure. With a plane-stress boundary normal `n`, pressure is `n · [(dW/dF)F^T] · n`. Rotate this normal together with the specimen when checking frame indifference. Incompressible deformations and finite material parameters are checked explicitly.

For the diagonal isotropic experiments the third direction is traction-free. Thus `p = lambda3 * dW/dlambda3` and `P11 = dW/dlambda1 - (lambda3/lambda1) dW/dlambda3`. Plane-strain/pure-shear data have `lambda2=1`, with an in-plane reaction stress; imposing traction freedom in the thickness direction is compatible with that kinematics.

Cardiac biaxial experiments use material directions `(f,s,n)`, `F=diag(lambda_f, 1/(lambda_f*lambda_n), lambda_n)`, and the sheet direction is traction-free. Shear measurements do not determine pressure; the pressure convention affects normal reactions, not the fitted off-diagonal stresses.

The cardiac measurements are Cauchy stresses in kPa in the [Martonová et al. formulation](https://doi.org/10.1016/j.cma.2024.117078). The [upstream data inventory](https://github.com/LivingMatterLab/CANN/tree/main/HEART) identifies the workbook used here. Its generic `P` headers are not used to infer a different stress measure. Treloar/Kawabata CSVs are interpreted as nominal stresses in MPa, matching their project data contract.

## Energy libraries

Isotropic MR terms are `(I1-3)^i (I2-3)^j`, with `I1=tr(C)` and `I2=[tr(C)^2-tr(C^2)]/2`, where `C=F^T F`. Ogden terms are `sum(lambda_a^alpha)-3`. Their gradients use the singular-value decomposition of `F`; display labels never determine their mathematics. Amplitudes are nonnegative, including after refitting. All terms have zero reference energy and pressure-corrected zero reference stress.

Cardiac directions are fixed in the material frame. The fourth invariants are `I4a=a0·C·a0`; coupling invariants are `I8ab=a0·C·b0`. Fibre extension is tension-only: `q=max(I4a-1,0)`. Its chain rule includes the inactive-compression branch. For isotropic invariants `q=I1-3` or `I2-3`; coupling invariants use `q=I8ab`. The four original activations are `q`, `exp(w1*q)-1`, `q²` and `exp(w1*q²)-1`.

The original 32-term registry is retained. **One-based IDs 9, 10, 13, 14, 17, 18, 21, 22, 25, 26, 29 and 30 are permanently disabled**. These anisotropic linear/exponential-linear contributions do not provide the required smooth stress-free reference response. They are absent from every feature matrix, energy evaluation and optimizer; the full 32-vector contains zeros at their positions. Isotropic linear/exponential-linear terms remain admissible because reference stress is hydrostatic and cancels with pressure.

There is one analytic derivative implementation for Stage 1 and final predictions. Stage 1 fixes exponential shape parameters to one. Stage 2 changes those parameters, not the stress formula. In particular, biaxial fibre terms contain their pushed-forward stretch factors, and shear stresses include the fourth-invariant contributions omitted by the former surrogate.

Dataset shear names `ij` denote shear along `j` of the plane normal to `i`, hence `F_ji=gamma` in this workbook convention. This activates `I4i=1+gamma²`. The convention is consistent with the experiment description in [Martonová et al. (2025), section 2.3](https://link.springer.com/article/10.1007/s10237-025-02005-x); Cauchy component symmetry does not make swapped deformation gradients equivalent in an orthotropic material.

## Discovery, scaling and refitting

Every case minimizes an explicitly weighted squared-error objective. With mode RMS `s_m`, training count `n_m` and training size `N`, weights are proportional to `1/(n_m*s_m²)`, normalized to have training mean one. A configured RMS floor prevents division by zero. This balances mode-relative errors and mode counts; it is a chosen working loss, not a claim that experimental noise variances are known. Disable either balancing component through `WeightPolicy` when a different justified noise model is available.

Rows and targets are multiplied by `sqrt(w)`, consistently in discovery, refitting and scoring. Features use training-only weighted RMS scales and targets use a training-only weighted RMS stress scale. **Neither is centered.** Thus origin-preserving physical stress features remain origin-preserving after transformation. Dimensionless coefficients are converted back to the native stress units. The ridge penalty applies to dimensionless amplitudes, making its strength independent of choosing Pa, kPa or MPa.

LASSO uses positive coordinate descent. LAR uses the documented scikit-learn `method='lar', positive=False` algorithm to propose supports, then calibrates physical amplitudes nonnegatively. Its path is limited to numerical design rank; rank degeneracy and solver warnings are recorded. Nonnegative OMP explicitly selects positive residual correlations and performs bounded least-squares updates. Equal-cardinality supports are not discarded. LAR/OMP CV parameters are normalized path positions, allowing fold-local paths to differ in their term membership and length.

Selected amplitudes are calibrated using bounded least squares with no stress intercept. The optional generic affine regression helper uses correct weighted centering; constitutive workflows never enable its intercept. Nonlinear cardiac calibration uses deterministic multistart bounded least squares and an analytical stress Jacobian. The same amplitude regularization is retained. Shape bounds prevent overflow on the training deformation domain; exponentials are never clipped into a different constitutive law. Bounds, convergence and local effective dimension are reported. Multistart does **not** certify a global optimum.

Numerical-zero amplitudes are removed using one dimensionless coefficient threshold, followed by recalibration of the reduced support. Both discovery and final models use this rule.

## Validation and information criteria

A measurement group contains the stress components recorded at one deformation. Groups stay together in mode-stratified folds. Copied Treloar EBT `P22=P11` values are symmetry information rather than another measurement; they are not counted twice. Kawabata's independently measured components are both used and stay in one group. Its CSV reference row is preserved.

Each training fold independently estimates weights/scales, constructs the discovery path, selects its own support, calibrates amplitudes and, for cardiac tissue, nonlinear shape parameters. Validation targets never determine those steps. Only the final model is refitted on all fitting data.

CV compares dimensionless validation error using training weights and stress scales. The displayed curve is the actual loss with fold standard deviation; no min–max rescaling or clipping turns it into a different metric. This is **model-selection CV**, not an independent performance test after hyperparameter selection. Kawabata is reported as held out only when it was excluded from fitting. Clean cardiac comparisons are comparisons to the same experimental dataset, not external test data.

AIC/BIC use one conditional Gaussian working model, `Var(error_i)=sigma²/w_i`. The maximized likelihood includes `-sum(log(w_i))` and one estimated residual-variance parameter. Ridge/constrained/nonlinear fits use a local effective dimension; boundary and data-dependent selection make these **conditional approximate** criteria. Exact-fit variance uses a documented floating-point floor instead of an arbitrary likelihood reward or negative infinity. No claim of statistically exact post-selection inference is made.

## Mechanical acceptance and scope

Tests independently verify energy derivatives, reference conditions, frame indifference, orthotropic symmetry, angular momentum through Cauchy symmetry, nominal-stress work conjugacy and nonlinear Jacobians. Runtime checks reject nonfinite or nonphysical input instead of replacing it with zeros or fabricated stresses.

Candidate selection checks reference energy/stress, nonnegative amplitudes, nonnegative energy and symmetric stress. It also tests rank-one isochoric energy curvature at **every distinct fitting deformation and the reference state**, in six deterministic directions per state. Within CV, these are the training fold's states; held-out targets and deformations do not determine admissibility. Candidates that fail this sampled admissibility check cannot be selected; corresponding failed validation candidates receive infinite selection loss. The zero-response model remains a possible statistical null model and is explicitly identified as not strictly stable.

These checks do **not** prove global polyconvexity or strong ellipticity for every deformation and direction. Mixed MR terms and anisotropic coupling terms do not gain such a certificate merely from positive coefficients. Reports therefore contain `global_stability_certified: false`. A wider deployment domain requires an expanded stability study or a formally certified restricted library. This distinction is necessary for a principled implementation; matching old plots, positive initial shear modulus or successful fitting cannot supply that proof.
