# MDisc_pairing
This repository contain the necessary codes to reproduce the results reported in the paper "Automated Constitutive Model Discovery by Pairing Sparse Regression Algorithms with Model Selection Criteria" ([https://arxiv.org/abs/2509.16040](https://arxiv.org/abs/2509.16040)).

### ⚠️ Note on `scikit-learn` Versions Used in the Paper

The simulations reported in the paper were performed using a version of `scikit-learn` that allowed:

```python
method="lars", positive=True
```

in the `lars_path` solver, e.g., scikit-learn==1.3.0 (see requirements.txt)

In **newer releases of scikit-learn (≥ 1.4)**, the API has changed:

* The option `method="lars"` has been **removed** and replaced by:

  ```python
  method="lar"
  ```
* The option `positive=True` has been **removed** from all LARS-based solvers.