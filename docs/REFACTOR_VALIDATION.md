# Refactor validation

Validated on 1 October 2026 against the original scripts at `eaa8c207be800b1ff4ac4c1426a4ea5d36a8895a`.

Every scenario was run with the existing local datasets. The originals were read from Git, and both versions ran in isolated temporary case folders. Repository datasets and existing output files were not overwritten.

| Case | Scenarios | PDFs | Other generated files | Console log | Full-precision results |
| --- | --- | --- | --- | --- | --- |
| Synthetic | 4 ground-truth models × 3 noise levels | 192/192 byte-identical | `outputs/log_clean.txt` byte-identical | Byte-identical | Exact array bytes and metrics match |
| Treloar | Original `Fit_Treloar_Only` configuration | 16/16 byte-identical | None, matching the original | Byte-identical | Exact array bytes and metrics match |
| Cardiac | 3 noise levels, with original Stage 1/2 processing | 45/45 byte-identical | None, matching the original | Byte-identical | Exact array bytes and metrics match |

There were no missing, extra, or differing generated files. No PDF rendering tolerance, coefficient rounding, or approximate numerical comparisons were used in the complete comparisons.

The validator fixes the clock and PDF creation date and starts both versions from NumPy seed 42, Python hash seed 0, and one native numerical thread. The original synthetic and cardiac seed choices remain unchanged in production. The original Treloar workflow has unseeded shuffled folds, so matching initial RNG state is required for exact reproduction. Normal runtime timestamps and measured durations naturally differ between runs; their output templates are preserved.

Environment: Python 3.9.6, NumPy 1.25.2, SciPy 1.11.2, scikit-learn 1.3.0, pandas 2.2.3, matplotlib 3.7.2, openpyxl 3.1.5. The machine-readable [equivalence report](equivalence_report.json) records the versions, input hashes, all output hashes, and console/numerical snapshot hashes.

Additional checks passed:

- 14 unit/integration contract tests covering normalization, support/refit behavior, fallback solvers, state isolation, safe imports, CLI use outside the checkout, and working-directory restoration on failure.
- A wheel built successfully using temporary build tools (setuptools 75.8.2, wheel 0.45.1), without changing the project's scientific environment.
- The wheel installed into a temporary target. Its `mdisc` console entry point and all three workflows imported outside the checkout; the original MR4 library was generated successfully.
- Wheel contents included 37 Python modules and the input directory placeholders. External datasets, generated PDFs/CSVs, and virtual environments were excluded.
- Git whitespace checks passed; `main` remained at the reference commit.

Reproduce the complete checks with:

```bash
python -m unittest discover -s tests -v
python scripts/validate_equivalence.py --reference eaa8c20
```

The complete run used `/private/tmp/mdisc-validation-complete-20261001`; that local folder retains both output sets, stdout, stderr, and full numerical JSON snapshots. The committed report contains hashes rather than copies of datasets or generated plots.
