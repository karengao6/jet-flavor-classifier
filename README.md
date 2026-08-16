# Jet Flavor Classifier

Machine learning pipeline for classifying **b-, c-, and light-flavor particle jets** using open data from CERN's ATLAS experiment.

My project focuses on building a reproducible and scalable ML workflow—from raw CERN HDF5 data through preprocessing and model training to evaluation—while comparing traditional ML methods with deep learning approaches.

## Baseline Models

The baseline trains two multiclass classifiers using jet-level features:

* **Logistic Regression** — scaled linear baseline
* **XGBoost** — tree-based baseline

Each run produces evaluation metrics and confusion matrices for comparing model performance.

### Quick Start

Install dependencies:

```sh
uv sync
```

Run a small smoke test:

```sh
uv run python -m jet_flavor_classifier.training.baseline \
  --data data/raw/mc-flavtag-ttbar-small.h5 \
  --output-dir results/baseline \
  --max-samples 1000 \
  --seed 42
```

Run on the full dataset:

```sh
uv run python -m jet_flavor_classifier.training.baseline \
  --data data/jets.h5 \
  --output-dir results/baseline \
  --seed 42
```

## Dataset

The project uses the **CERN ATLAS Transforming Jet Flavour Open Data** dataset.

The models use reconstructed jet features as inputs. Truth-label information is used only to construct training targets and evaluate predictions, not as model features.

## Goals

My project aims to:

1. Establish reproducible traditional ML baselines.
2. Evaluate deep learning approaches for jet flavor classification.
3. Compare models using consistent evaluation metrics.
4. Build a maintainable ML codebase with automated testing, configuration, and experiment tracking.
