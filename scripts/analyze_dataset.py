from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


DATA_PATH = Path("data/raw/mc-flavtag-ttbar-small.h5")
OUTPUT_DIR = Path("results/eda")

TARGET = "HadronGhostTruthLabelID"

CLASS_NAMES = {
    0: "light",
    4: "charm",
    5: "bottom",
    15: "tau",
}


def plot_class_distribution(labels: np.ndarray) -> None:
    """Plot the distribution of truth labels."""

    values, counts = np.unique(labels, return_counts=True)

    names = [
        CLASS_NAMES.get(int(value), str(value))
        for value in values
    ]

    percentages = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(names, counts)

    ax.set_title("Jet flavor class distribution")
    ax.set_xlabel("Jet flavor")
    ax.set_ylabel("Number of jets")

    for bar, percentage in zip(bars, percentages):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{percentage:.2f}%",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "class_distribution.png",
        dpi=150,
    )
    plt.close(fig)


def plot_jet_kinematics(jets) -> None:
    """Plot basic reconstructed jet kinematics."""

    variables = {
        "pt": (
            jets["pt"],
            "Jet pT [GeV]",
            "jet_pt.png",
        ),
        "eta": (
            jets["eta"],
            "Jet eta",
            "jet_eta.png",
        ),
        "mass": (
            jets["mass"],
            "Jet mass [GeV]",
            "jet_mass.png",
        ),
    }

    for _, (values, xlabel, filename) in variables.items():
        values = np.asarray(values)

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.hist(
            values,
            bins=100,
            histtype="step",
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of jets")
        ax.set_title(f"Distribution of {xlabel}")

        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / filename,
            dpi=150,
        )
        plt.close(fig)

def plot_jet_pt_log_scale(jets) -> None:
    """Plot jet pT on a logarithmic x-axis."""

    pt = np.asarray(jets["pt"])

    positive_pt = pt[pt > 0]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        positive_pt,
        bins=np.logspace(
            np.log10(positive_pt.min()),
            np.log10(positive_pt.max()),
            100,
        ),
        histtype="step",
    )

    ax.set_xscale("log")

    ax.set_xlabel("Jet pT [GeV]")
    ax.set_ylabel("Number of jets")
    ax.set_title("Jet pT distribution")

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "jet_pt_log.png",
        dpi=150,
    )
    plt.close(fig)


def get_track_multiplicity(tracks) -> np.ndarray:
    """
    Count valid tracks for each jet.

    tracks.shape:
        (number_of_jets, 40)
    """

    valid = np.asarray(tracks["valid"])

    return valid.sum(axis=1)


def plot_track_multiplicity(tracks) -> None:
    """Plot the number of valid tracks per jet."""

    multiplicity = get_track_multiplicity(tracks)

    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.arange(
        multiplicity.max() + 2
    ) - 0.5

    ax.hist(
        multiplicity,
        bins=bins,
        histtype="step",
    )

    ax.set_xlabel("Number of valid tracks")
    ax.set_ylabel("Number of jets")
    ax.set_title("Track multiplicity per jet")

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "track_multiplicity.png",
        dpi=150,
    )
    plt.close(fig)


def plot_feature_distributions(jets) -> None:
    """Plot distributions of reconstructed model features."""

    features = {
        "pt": (
            jets["pt"],
            "Jet pT [GeV]",
        ),
        "eta": (
            jets["eta"],
            "Jet eta",
        ),
        "mass": (
            jets["mass"],
            "Jet mass [GeV]",
        ),
        "sin_phi": (
            np.sin(jets["phi"]),
            "sin(phi)",
        ),
        "cos_phi": (
            np.cos(jets["phi"]),
            "cos(phi)",
        ),
    }

    for name, (values, xlabel) in features.items():
        values = np.asarray(values)

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.hist(
            values,
            bins=100,
            histtype="step",
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of jets")
        ax.set_title(f"Feature distribution: {name}")

        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / f"feature_{name}.png",
            dpi=150,
        )
        plt.close(fig)


def plot_features_by_class(jets) -> None:
    """Compare reconstructed jet features between flavor classes."""

    labels = np.asarray(jets[TARGET])

    features = {
        "pt": (
            jets["pt"],
            "Jet pT [GeV]",
        ),
        "eta": (
            jets["eta"],
            "Jet eta",
        ),
        "mass": (
            jets["mass"],
            "Jet mass [GeV]",
        ),
    }

    for name, (values, xlabel) in features.items():
        fig, ax = plt.subplots(figsize=(8, 5))

        for label, class_name in CLASS_NAMES.items():
            class_values = np.asarray(
                values[labels == label]
            )

            ax.hist(
                class_values,
                bins=100,
                histtype="step",
                label=class_name,
                density=True,
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.set_title(f"{name} distribution by jet flavor")
        ax.legend()

        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / f"{name}_by_class.png",
            dpi=150,
        )
        plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading dataset...")

    with h5py.File(DATA_PATH, "r") as f:
        jets = f["jets"]
        tracks = f["tracks"]

        labels = np.asarray(jets[TARGET])

        print(f"Jets: {len(labels):,}")

        print("Creating class distribution plot...")
        plot_class_distribution(labels)

        print("Creating jet kinematic plots...")
        plot_jet_kinematics(jets)
        plot_jet_pt_log_scale(jets)

        print("Creating track multiplicity plot...")
        plot_track_multiplicity(tracks)

        print("Creating feature distributions...")
        plot_feature_distributions(jets)

        print("Creating flavor comparison plots...")
        plot_features_by_class(jets)

    print(f"Plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()