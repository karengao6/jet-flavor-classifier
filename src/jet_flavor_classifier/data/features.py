"""
Feature definitions for the CERN jet-flavor classifier.

RULE:
    Model inputs must be reconstructed detector-level quantities.

We DO NOT use:
    - truth labels
    - truth matching information
    - existing tagger outputs such as GN2 or DL1

Truth information is used only as the supervised-learning target
and for evaluation/analysis.
"""

from __future__ import annotations

import numpy as np


# ============================================================================
# RAW JET FIELDS
# ============================================================================

# These are actual fields in the HDF5 `jets` dataset.

RAW_JET_FEATURES = (
    "pt",
    "eta",
    "mass",
    "phi",
)


# ============================================================================
# MODEL JET FEATURES
# ============================================================================

# The model receives five values:
#
#     pt
#     eta
#     mass
#     sin(phi)
#     cos(phi)
#
# We transform phi because it is periodic.
#
# Without the transformation:
#
#     phi = -pi
#     phi = +pi
#
# look numerically very far apart even though they represent nearly
# the same physical direction.
#
# sin/cos provide a continuous representation.

JET_FEATURE_NAMES = (
    "pt",
    "eta",
    "mass",
    "sin_phi",
    "cos_phi",
)


def extract_jet_features(
    jets: np.ndarray,
) -> np.ndarray:
    """Extract reconstructed jet features."""

    pt = jets["pt"].astype(np.float32)
    eta = jets["eta"].astype(np.float32)
    mass = jets["mass"].astype(np.float32)
    phi = jets["phi"].astype(np.float32)

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    return np.column_stack(
        [
            pt,
            eta,
            mass,
            sin_phi,
            cos_phi,
        ]
    ).astype(np.float32)


# ============================================================================
# RAW TRACK FIELDS USED BY THE MODEL
# ============================================================================

# These are reconstructed track quantities.
#
# `valid` is deliberately NOT included here.
#
# It is a mask telling us whether a track slot contains a real track.

TRACK_FEATURES = (
    "pt",
    "ptfrac",
    "deta",
    "dphi",
    "d0",
    "z0SinTheta",
    "d0Uncertainty",
    "z0SinThetaUncertainty",
    "lifetimeSignedD0",
    "lifetimeSignedZ0SinTheta",
    "lifetimeSignedD0Significance",
    "lifetimeSignedZ0SinThetaSignificance",
)


# ============================================================================
# TARGET
# ============================================================================

# This is truth information.
#
# It is NOT a model input.
#
# We use it because supervised learning requires a target label.

TARGET = "HadronGhostTruthLabelID"