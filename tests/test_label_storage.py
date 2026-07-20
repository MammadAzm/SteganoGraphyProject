from __future__ import annotations

from pathlib import Path

import numpy as np

from src.labeling.storage import (
    inspect_label_file,
    save_label_file,
)


def test_label_file_round_trip(
    tmp_path: Path,
) -> None:
    output_path = (
        tmp_path / "sample_labels.npz"
    )

    score_map = np.linspace(
        0.0,
        1.0,
        25 * 25,
        dtype=np.float32,
    ).reshape(
        25,
        25,
    )

    feature_names = (
        "entropy",
        "variance",
        "edge_density",
        "lbp_non_uniform_ratio",
        "laplacian_variance",
    )

    group_names = (
        "intensity_complexity",
        "spatial_complexity",
        "pattern_complexity",
    )

    raw_maps = np.zeros(
        (5, 25, 25),
        dtype=np.float32,
    )

    normalized_maps = np.zeros_like(
        raw_maps
    )

    group_maps = np.zeros(
        (3, 25, 25),
        dtype=np.float32,
    )

    coordinates = np.zeros(
        (25, 25, 4),
        dtype=np.int32,
    )

    save_label_file(
        output_path,
        score_map=score_map,
        feature_names=feature_names,
        group_names=group_names,
        metadata={
            "label_fingerprint": "test-fingerprint",
        },
        compressed=True,
        raw_feature_maps=raw_maps,
        normalized_feature_maps=(
            normalized_maps
        ),
        group_score_maps=group_maps,
        window_coordinates=coordinates,
    )

    metadata, statistics = inspect_label_file(
        output_path,
        expected_score_shape=(
            25,
            25,
        ),
        expected_feature_names=(
            feature_names
        ),
        expected_label_fingerprint=(
            "test-fingerprint"
        ),
    )

    assert metadata[
        "label_fingerprint"
    ] == "test-fingerprint"

    assert statistics.score_min == 0.0
    assert statistics.score_max == 1.0
    assert statistics.score_count == 625