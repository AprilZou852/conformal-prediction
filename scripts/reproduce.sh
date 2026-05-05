#!/bin/bash
# Reproduce all paper results for "Conformalized Percentile Interval"
# Authors: AprilZou852
# Repository: conformal-prediction
# ICML 2025

set -e

echo "=========================================="
echo "CPI Paper Reproducibility Script"
echo "=========================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

# Create output directories
mkdir -p results/simulation
mkdir -p results/realdata
mkdir -p figures

# Step 1: Simulation experiments
echo ""
echo "=========================================="
echo "Step 1: Running Simulation Experiments"
echo "=========================================="
python scripts/run_simulation.py \
    --config configs/simulation.yaml \
    --setups 0a 0c1 \
    --output-dir results/simulation

# Step 2: Real data experiments
# Note: Requires downloading datasets from UCI ML Repository
# See data/README.md for download instructions

echo ""
echo "=========================================="
echo "Step 2: Real Data Experiments"
echo "=========================================="

DATASETS=("Airfoil" "Computer" "Abalone" "Concrete" "AutoMPG" "Crime")

for dataset in "${DATASETS[@]}"; do
    data_dir="data/${dataset}"

    if [ ! -f "${data_dir}/X.csv" ] || [ ! -f "${data_dir}/y.csv" ]; then
        echo "Skipping $dataset: data files not found at $data_dir"
        echo "Please download dataset and extract to $data_dir"
        continue
    fi

    echo ""
    echo "Processing dataset: $dataset"
    python scripts/run_realdata.py \
        --config configs/realdata.yaml \
        --x-path "${data_dir}/X.csv" \
        --y-path "${data_dir}/y.csv" \
        --output-dir "results/${dataset}"
done

# Step 3: Generate plots
echo ""
echo "=========================================="
echo "Step 3: Generating Plots"
echo "=========================================="

# Simulation plots
python scripts/plot_results.py \
    --type simulation \
    --input-dir results/simulation \
    --output-dir figures

# Conditional plots (real data)
python scripts/plot_results.py \
    --type conditional \
    --input-dir results \
    --output-dir figures

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - Simulation results: results/simulation/"
echo "  - Real data results: results/<dataset>/"
echo "  - Plots: figures/"
echo ""
echo "Key output files:"
echo "  - simulation_results_*.csv"
echo "  - summary_realdata.csv"
echo "  - plot_*.png"
