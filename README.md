# Conformalized Percentile Interval: Finite Sample Validity and Improved Conditional Performance

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **CPI (Conformalized Percentile Interval)** for the ICML 2025 paper "Conformalized Percentile Interval: Finite Sample Validity and Improved Conditional Performance".

## Abstract

Prediction intervals are crucial for uncertainty quantification in regression. We propose CPI, a conformal method based on neural network conditional density estimation (NNCDE) via hazard function modeling. CPI constructs shortest-width prediction intervals with finite-sample marginal coverage guarantees while improving conditional coverage compared to existing methods (Residual, Rescaled, CQR). We evaluate CPI on 6 real benchmark datasets and demonstrate superior or competitive width across all problem settings while maintaining strict coverage guarantees.

## Key Features

- **Finite-sample coverage guarantee**: Marginal coverage is guaranteed at level 1-α regardless of model quality
- **Improved conditional coverage**: Empirically superior to Residual, Rescaled, and CQR baselines
- **Shortest-width intervals**: Optimal width selection via grid search over PIT quantiles
- **Hazard-based density estimation**: Efficient continuous density modeling using neural network conditional density estimation
- **Comprehensive evaluation**: Tested on 6 real datasets with PC1-based conditional grouping

## Installation

### From source (recommended)

```bash
git clone https://github.com/AprilZou852/conformal-prediction.git
cd conformal-prediction
pip install -e .
```

### Using conda

```bash
conda env create -f environment.yml
conda activate cpi-conformal
pip install -e .
```

### Using pip with requirements.txt

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Simulation Experiment

Run simulation on synthetic setups (0a and 0c1):

```bash
python scripts/run_simulation.py \
    --config configs/simulation.yaml \
    --setups 0a 0c1 \
    --output-dir results/simulation
```

### Real Data Experiment

After downloading datasets (see `data/README.md`):

```bash
python scripts/run_realdata.py \
    --config configs/realdata.yaml \
    --x-path data/Airfoil/X.csv \
    --y-path data/Airfoil/y.csv \
    --output-dir results/Airfoil
```

### Reproduce All Results

```bash
bash scripts/reproduce.sh
```

This will:
1. Run all simulation experiments
2. Run experiments on all real datasets (if downloaded)
3. Generate visualization plots

## Project Structure

```
conformal-prediction/
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── models/                   # Neural network models
│   │   ├── hazard_net.py         # Hazard CDF/PDF modeling
│   │   ├── baseline_nets.py      # Baselines (Simple, MeanVariance, Quantile NN)
│   │   └── residual.py           # Residual model (ridge/XGB/LGBM)
│   ├── conformal/                # Conformal methods
│   │   ├── hazard_inference.py   # PIT, z* grid search, quantile inversion
│   │   ├── cpi.py                # CPI calibration and inference
│   │   ├── dcp.py                # DCP (Dynamic Conformal Prediction)
│   │   └── baselines.py          # Residual, Rescaled, CQR methods
│   ├── data/                     # Data loading and generation
│   │   ├── dgp.py                # Synthetic data generation (setups 0a, 0c1)
│   │   ├── loader.py             # Real data loading
│   │   └── feature_selection.py  # Stability-based feature selection
│   └── utils/                    # Utilities
│       ├── helpers.py            # Seeding, RNG preservation
│       ├── preprocessing.py      # Standardization, transformation
│       ├── metrics.py            # Coverage, width, PC1 grouping
│       └── plotting.py           # Visualization helpers
├── configs/                      # Configuration files
│   ├── simulation.yaml           # Simulation hyperparameters
│   └── realdata.yaml             # Real data hyperparameters
├── scripts/                      # Executable scripts
│   ├── run_simulation.py         # Main simulation runner
│   ├── run_realdata.py           # Main real data runner
│   ├── plot_results.py           # Result visualization
│   └── reproduce.sh              # Full reproducibility script
├── data/                         # Benchmark datasets (see data/README.md)
│   ├── Airfoil/
│   ├── Computer/
│   ├── Abalone/
│   ├── Concrete/
│   ├── AutoMPG/
│   └── Crime/
├── results/                      # Output directory (auto-created)
├── figures/                      # Generated plots (auto-created)
├── requirements.txt              # Python dependencies
├── environment.yml               # Conda environment
├── setup.py                      # Package installer
├── LICENSE                       # MIT License
└── README.md                     # This file
```

## Reproducing Results

### 1. Simulation Results

Generates coverage/width plots for synthetic setups:

```bash
python scripts/run_simulation.py --config configs/simulation.yaml
```

Output: `results/simulation/simulation_results_*.csv`, `figures/plot_*.png`

### 2. Real Data Results

First, download datasets from UCI ML Repository (see `data/README.md`). Then:

```bash
for dataset in Airfoil Computer Abalone Concrete AutoMPG Crime; do
    python scripts/run_realdata.py \
        --config configs/realdata.yaml \
        --x-path data/${dataset}/X.csv \
        --y-path data/${dataset}/y.csv \
        --output-dir results/${dataset}
done
```

Output: `results/<dataset>/summary_realdata.csv`

### 3. Generate Plots

```bash
python scripts/plot_results.py --type simulation --input-dir results/simulation --output-dir figures
python scripts/plot_results.py --type conditional --input-dir results --output-dir figures
```

## Configuration

### Simulation Config (`configs/simulation.yaml`)

- `alpha`: Coverage level (default: 0.10)
- `n_train`: Training samples (default: 3000)
- `n_cal`: Calibration samples (default: 500)
- `n_test`: Test points for evaluation (default: 200)
- `n_rep`: Monte Carlo replicates (default: 3000)
- `hazard.hidden`: Hidden units for hazard network (default: 64)
- `baseline_config.hidden`: Hidden units for baselines (default: 32)

### Real Data Config (`configs/realdata.yaml`)

- `test_frac`: Test set fraction (default: 0.20)
- `cal_frac`: Calibration set fraction (default: 0.35)
- `hazard.ensemble_k`: Ensemble size (default: 15)
- `feature_selection.method`: FS method (default: "stability_lgbm")
- `pc1_groups`: Number of conditional groups (default: 4)

## Methods Evaluated

### Main Method

- **CPI (Conformalized Percentile Interval)**: Neural network conditional density estimation via hazard function modeling with optimal z* grid search

### Baselines

- **Residual**: Simple point prediction + residual quantile
- **Rescaled**: Mean-variance network with adaptive scaling
- **CQR**: Conditional quantile regression
- **DCP**: Dynamic conformal prediction (for simulation)

## Key Results

On 6 real datasets (Airfoil, Computer, Abalone, Concrete, AutoMPG, Crime):

- CPI achieves **1.3-2.1× narrower intervals** than Residual while maintaining coverage
- CPI improves **conditional coverage** (PC1-grouped) compared to CQR
- All methods satisfy **marginal coverage guarantees** by design

## Citation

```bibtex
@inproceedings{zou2025cpi,
  title={Conformalized Percentile Interval: Finite Sample Validity and Improved Conditional Performance},
  author={Zou, April},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

Thanks to the conformal prediction community and UCI ML Repository for datasets and foundational work on conformal inference.

## Questions?

For issues or questions, please open an issue on [GitHub](https://github.com/AprilZou852/conformal-prediction/issues).
