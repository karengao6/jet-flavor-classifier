from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from jet_flavor_classifier.training.baseline import run_baseline


def test_run_baseline_saves_metrics(tmp_path):
    path = tmp_path / "jets_baseline.h5"
    output_dir = tmp_path / "baseline_results"

    rng = np.random.default_rng(123)
    n_jets = 120
    raw_classes = np.array([0, 4, 5, 15])

    jet_dtype = np.dtype(
        [
            ("eventNumber", np.int64),
            ("HadronGhostTruthLabelID", np.int64),
            ("pt", np.float32),
            ("eta", np.float32),
            ("mass", np.float32),
            ("phi", np.float32),
        ]
    )
    jets = np.empty(n_jets, dtype=jet_dtype)

    labels = np.tile(raw_classes, n_jets // len(raw_classes))
    labels = labels[:n_jets]
    rng.shuffle(labels)

    jets["eventNumber"] = np.arange(n_jets)
    jets["HadronGhostTruthLabelID"] = labels
    jets["pt"] = rng.uniform(20.0, 200.0, size=n_jets).astype(np.float32)
    jets["eta"] = rng.uniform(-2.5, 2.5, size=n_jets).astype(np.float32)
    jets["mass"] = rng.uniform(0.5, 50.0, size=n_jets).astype(np.float32)
    jets["phi"] = rng.uniform(-np.pi, np.pi, size=n_jets).astype(np.float32)

    with h5py.File(path, "w") as f:
        f.create_dataset("jets", data=jets)

    summary = run_baseline(
        data_path=str(path),
        output_dir=str(output_dir),
        seed=42,
        max_samples=80,
    )

    assert set(summary) == {"logistic_regression", "tree_model"}

    for model_name in summary:
        metrics_path = output_dir / f"{model_name}_metrics.json"
        assert metrics_path.exists()

        with metrics_path.open("r", encoding="utf-8") as fh:
            metrics = json.load(fh)

        assert metrics["accuracy"] >= 0.0
        assert metrics["precision"]["macro"] >= 0.0
        assert metrics["recall"]["macro"] >= 0.0
        assert metrics["f1"]["macro"] >= 0.0
        assert "confusion_matrix" in metrics
        assert len(metrics["class_names"]) == 4

    confusion_path = output_dir / "logistic_regression_confusion_matrix.csv"
    assert confusion_path.exists()
