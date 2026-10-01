# Reading the discovery plots

The selection PDFs contain already-computed results. Rendering does not change fits, normalization, standardization, criteria or selected models.

## Activation heatmaps

Each **row is a strain-energy term**, labeled with its original coefficient ID, such as `c12`. Each **column is a candidate on the discovery grid**, after fitting the proposed support to the full training dataset and performing the cardiac nonlinear refinement where applicable.

| Display | Meaning |
| --- | --- |
| Blue cell | This refitted model contains the term with a strictly positive amplitude. |
| White cell | This refitted model has zero amplitude for the term. |
| Vertical marker | Candidate selected by CV, AIC or BIC. The legend gives its parameter value. |
| Combined marker, e.g. AIC/BIC | These criteria selected the very same candidate. |

Shades do **not** encode amplitude, stress, error or stiffness. Discovery already removes numerical-zero amplitudes using its dimensionless tolerance; rendering applies no additional tolerance to coefficients in physical units. Changing stress units cannot change these activation cells.

As in the original heatmaps, only terms that become active somewhere on the path are displayed, ordered by their first activation. Original IDs are retained even when rows are omitted. Cardiac IDs **9, 10, 13, 14, 17, 18, 21, 22, 25, 26, 29, 30** remain permanently excluded and never appear as active rows. Other absent rows were zero across this path. The complete registry and coefficients remain in `results.json`.

These are **full-data refitted models**, not raw signed LAR coefficients or the models fitted inside individual CV folds. Cardiac exponential shape parameters can differ between columns. Two columns can have identical full-data activation but different CV errors because each fold independently discovers and refits its own model.

## Horizontal axes

Both plots follow the same discovery direction, starting at the sparse/zero end:

- **LASSO: Penalty α.** The dimensionless regularization parameter decreases from left to right. The criteria panels use a logarithmic axis; heatmap columns are equally spaced on the logarithmic candidate grid. Tick and selection labels display the actual parameter values, rounded for readability.
- **LAR and OMP: Path fraction.** The fraction runs from zero to one. For a raw path containing `L` models, fraction `r` selects position `round(r × (L−1))`, using NumPy's nearest-even rounding. Zero selects the initial zero model; one selects the final available path model. This fraction is neither a penalty nor a literal term count. Fold path lengths can differ, so CV compares the same fraction across each fold's own path.

The heatmap includes every evaluated candidate, including repeated full-data models. Its markers identify the exact columns selected; they are not shifted to neighboring columns to separate coincident choices.

## Criteria panels

The original **3 × 4.5 inch** three-panel layout and plasma colors are retained. Tick counts are reduced and labels use compact notation; the original font sizes are retained. The dashed line in each panel marks that panel's selected parameter, also printed above the panel.

| Panel | Quantity | Interpretation |
| --- | --- | --- |
| CV NMSE | Mean of the grouped validation folds' dimensionless weighted mean-square stress errors | Lower is better. Weights and stress scales come from each fold's training data. |
| AIC | Conditional Gaussian working-likelihood AIC | Lower is better; negative values are valid. |
| BIC | Conditional Gaussian working-likelihood BIC | Lower is better; negative values are valid. |

The CV band shows **± one sample standard deviation across fold losses**, not a confidence interval or standard error. Its lower edge is bounded at zero because squared error is nonnegative. The plotted mean remains unchanged, and values greater than one are shown without clipping. AIC/BIC use the existing likelihood, weight determinant and effective dimension, including the residual-variance parameter. Their values are not rescaled for display.

CV is used for model selection and is not an independent post-selection test of generalization. AIC and BIC are conditional approximations with the limitations documented in [MECHANICS.md](MECHANICS.md).

## Energy notation

Labels are generated from mathematical metadata rather than interpreting stored display strings. `c_i` is the physical energy amplitude. Ogden expressions show the complete principal-stretch sum `λ₁ᵃ + λ₂ᵃ + λ₃ᵃ − 3`.

In cardiac exponential labels, `b_i` denotes the shape parameter stored under the corresponding original term ID in the model's `nonlinear` dictionary. Fibre expressions use the tension-only positive part `⟨I₄a−1⟩₊`; their squares and exponential squares retain the reference-state exclusions. Coupling terms use squared `I₈ab` without a tension-only positive part. Stress is calculated from the energy, with pressure and the appropriate stress measure, rather than equating an energy label with a stress formula.

## Redrawing saved results

```bash
python scripts/replot_selection.py Anisotropic/output_mdisc_app/results.json
python scripts/replot_selection.py Isotropic/Treloar/output_mdisc_app/results.json
python scripts/replot_selection.py Isotropic/Benchmarks/output_mdisc_app/results.json
```

By default this replaces only the heatmap and criteria PDFs beside the report. Use `--output-dir /path/to/new/plots` to write into another folder. No fitting is performed; logs, predictions and numerical files remain intact. The report's existing source provenance describes the original numerical run, not the later rendering code.
