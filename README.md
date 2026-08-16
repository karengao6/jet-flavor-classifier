# jet-flavor-classifier

## Feature baselines

The baseline command trains a scaled logistic-regression classifier and an
XGBoost multiclass classifier on jet-level features, then writes metrics
and confusion matrices for each model.

Running a small smoke test:

```sh
uv sync
uv run python -m jet_flavor_classifier.training.baseline \
  --data data/raw/mc-flavtag-ttbar-small.h5 \
  --output-dir results/baseline \
  --max-samples 1000 \
  --seed 42
```

Full training run (omit `--max-samples`):

```sh
uv run python -m jet_flavor_classifier.training.baseline \
  --data data/jets.h5 \
  --output-dir results/baseline \
  --seed 42
```
