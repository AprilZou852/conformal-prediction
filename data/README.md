# Benchmark Datasets

This directory contains the real-world datasets used for evaluating CPI and baseline methods.

## Datasets

All datasets are sourced from the UCI Machine Learning Repository and are publicly available.

### Downloading Datasets

Create subdirectories for each dataset and download `X.csv` and `y.csv` files:

```bash
mkdir -p data/Airfoil
mkdir -p data/Computer
mkdir -p data/Concrete
mkdir -p data/AutoMPG
mkdir -p data/Crime
```

#### 1. Airfoil Self-Noise
- **Source**: UCI ML Repository (https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise)
- **Features**: 5 (frequency, angle of attack, chord length, free-stream velocity, suction side displacement)
- **Target**: Sound pressure level
- **Samples**: 1,503

```bash
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat -O data/Airfoil/airfoil.dat
```

#### 2. Computer Hardware
- **Source**: UCI ML Repository (https://archive.ics.uci.edu/ml/datasets/Computer+Hardware)
- **Features**: 6 (manufacturer, model name, MYCT, MMIN, MMAX, CACH)
- **Target**: Published relative performance
- **Samples**: 209

#### 3. Concrete Compressive Strength
- **Source**: UCI ML Repository (https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength)
- **Features**: 8 (cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate, age)
- **Target**: Compressive strength
- **Samples**: 1,030

#### 4. Auto MPG
- **Source**: UCI ML Repository (https://archive.ics.uci.edu/ml/datasets/Auto+MPG)
- **Features**: 7 (cylinders, displacement, horsepower, weight, acceleration, model year, origin)
- **Target**: Miles per gallon
- **Samples**: 398

#### 5. Communities and Crime
- **Source**: UCI ML Repository (https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime)
- **Features**: Selected socioeconomic and crime statistics
- **Target**: Per capita violent crime rate
- **Samples**: 1,994

## Data Format

Each dataset should have the following structure:

```
data/
└── DatasetName/
    ├── X.csv       # Features (n_samples × n_features)
    └── y.csv       # Target values (n_samples,)
```

### CSV Requirements

- **X.csv**: Tab or comma-separated, no header required, numeric values only
- **y.csv**: Single column, no header, numeric target values

## Preprocessing

The scripts handle basic preprocessing:
- Handling missing values
- Feature standardization (via `StandardScaler`)
- Train/Cal/Test splitting

No manual preprocessing is needed; datasets can be used directly after download.

## Usage

Once datasets are downloaded, run real data experiments:

```bash
python scripts/run_realdata.py \
    --config configs/realdata.yaml \
    --x-path data/Airfoil/X.csv \
    --y-path data/Airfoil/y.csv \
    --output-dir results/Airfoil
```

Or reproduce all results with:

```bash
bash scripts/reproduce.sh
```
