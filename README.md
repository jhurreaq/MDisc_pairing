# MDisc_pairing
This repository contains the necessary code to reproduce the results reported in the paper ‘Automated Constitutive Model Discovery by Pairing Sparse Regression Algorithms with Model Selection Criteria’. ([https://arxiv.org/abs/2509.16040](https://arxiv.org/abs/2509.16040)).

## Project Structure

```
MDisc_pairing/
├── Anisotropic/                           # Anisotropic material simulations
│   ├── HeartCANN_discovered_lasso_lars_omp_two_stages_20250827.py
│   ├── input/                             # Input data for anisotropic models
│   │   └── CANNsHEARTdata_shear05.xlsx   # Heart tissue data
│   ├── outputs/                           # Simulation results
│   │   ├── RelNoise0pct/                 # Results with 0% noise
│   │   ├── RelNoise5pct/                 # Results with 5% noise
│   │   └── RelNoise10pct/                # Results with 10% noise
│   └── log.txt                           # Execution logs
│
├── Isotropic/                            # Isotropic material simulations
│   ├── Benchmarks/                       # Synthetic benchmark tests
│   │   ├── MDisc_synthetic_data_rel_noise_20251004_UT_PS_EBT_pos.py
│   │   └── outputs/                      # Results for different models & noise levels
│   │       ├── MR1O1_RelNoise0pct/
│   │       ├── MR2O2_RelNoise0pct/
│   │       ├── MR2_RelNoise0pct/
│   │       ├── O2_RelNoise0pct/
│   │       └── ... (5%, 10% noise variants)
│   │
│   └── Treloar/                          # Treloar & Kawabata experimental data
│       ├── MDisc_TreloarKawabata_datasets_combined_20250827.py
│       ├── input/                        # Experimental datasets
│       │   ├── Kawabata.csv
│       │   ├── TreloarDataEBT.csv       # Equal-biaxial tension
│       │   ├── TreloarDataPS.csv        # Pure shear
│       │   └── TreloarDataUT.csv        # Uniaxial tension
│       ├── outputs/                      # Discovery results
│       └── log.txt
│
├── requirements.txt                      # Python dependencies
├── CITATION.cff                         # Citation information
├── LICENSE                              # License file
└── README.md                            # This file
```

### Folder Descriptions

- **`Anisotropic/`**: Contains scripts and data for model discovery on anisotropic materials (i.e., heart tissue). The main script runs model discovery with multiple sparse regression algorithms and noise levels.

- **`Isotropic/`**: Contains two subdirectories:
  - **`Benchmarks/`**: Synthetic data experiments to validate the methodology with known ground truth models (Mooney-Rivlin, Ogden)
  - **`Treloar/`**: Model discovery using classical experimental data from Treloar

## External Data Requirements

Some simulations require external datasets that are not included in this repository. Please download the required data from the sources listed below and place them in the appropriate `input/` directories.

### Anisotropic Simulations

**Required file:** `CANNsHEARTdata_shear05.xlsx`  
**Location:** `Anisotropic/input/`  
**Source:** [CANN GitHub Repository](https://github.com/LivingMatterLab/CANN/blob/main/HEART/input/CANNsHEARTdata_shear05.xlsx)  

### Isotropic - Treloar Simulations

**Required files:**
- `Kawabata.csv`
- `TreloarDataEBT.csv`
- `TreloarDataPS.csv`
- `TreloarDataUT.csv`

**Location:** `Isotropic/Treloar/input/`  
**Source:** [Zenodo Dataset (DOI: 10.5281/zenodo.14995273)](https://doi.org/10.5281/zenodo.14995273)  

## Setup Instructions

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