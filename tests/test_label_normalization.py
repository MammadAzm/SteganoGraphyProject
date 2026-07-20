from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from src.config.schema import (
    FeatureItemSettings,
)
from src.features.fusion import (
    FeatureGroupSettings,
    FeatureNormalizationBounds,
    GroupedFusionSettings,
)
from src.labeling.normalization import (
    NormalizationArtifact,
    fuse_feature_maps,
    normalize_feature_maps,
)


def test_feature_map_normalization_and_fusion() -> None:
    feature_names = (
        "entropy",
        "variance",
        "edge_density",
        "lbp_non_uniform_ratio",
        "laplacian_variance",
    )

    artifact = NormalizationArtifact(
        version=1,
        created_at_utc=datetime.now(
            timezone.utc
        ).isoformat(),
        fit_split="train",
        feature_names=feature_names,
        sample_count=4,
        image_count=1,
        percentile_low=0.0,
        percentile_high=100.0,
        epsilon=1e-8,
        raw_feature_fingerprint="test",
        bounds={
            feature_name: (
                FeatureNormalizationBounds(
                    feature_name=feature_name,
                    lower=0.0,
                    upper=1.0,
                )
            )
            for feature_name in feature_names
        },
    )

    feature_items = {
        feature_name: FeatureItemSettings(
            enabled=True,
            weight=1.0,
            direction="higher",
        )
        for feature_name in feature_names
    }

    raw_maps = np.asarray(
        [
            [[1.0, 0.0], [0.5, 0.5]],
            [[1.0, 0.0], [0.5, 0.5]],
            [[0.0, 1.0], [0.5, 0.5]],
            [[0.5, 0.5], [0.5, 0.5]],
            [[0.0, 1.0], [0.5, 0.5]],
        ],
        dtype=np.float32,
    )

    normalized = normalize_feature_maps(
        raw_maps,
        artifact,
        feature_items,
    )

    assert np.array_equal(
        normalized,
        raw_maps,
    )

    fusion = GroupedFusionSettings(
        method="grouped_weighted_mean",
        clip_min=0.0,
        clip_max=1.0,
        groups=(
            FeatureGroupSettings(
                name="intensity_complexity",
                weight=1.0,
                members={
                    "entropy": 0.5,
                    "variance": 0.5,
                },
            ),
            FeatureGroupSettings(
                name="spatial_complexity",
                weight=1.0,
                members={
                    "edge_density": 0.5,
                    "laplacian_variance": 0.5,
                },
            ),
            FeatureGroupSettings(
                name="pattern_complexity",
                weight=1.0,
                members={
                    "lbp_non_uniform_ratio": 1.0,
                },
            ),
        ),
    )

    group_maps, group_names, score_map = (
        fuse_feature_maps(
            normalized,
            feature_names,
            fusion,
        )
    )

    assert group_maps.shape == (
        3,
        2,
        2,
    )

    assert group_names == (
        "intensity_complexity",
        "spatial_complexity",
        "pattern_complexity",
    )

    assert score_map.shape == (
        2,
        2,
    )

    assert np.all(
        score_map >= 0.0
    )

    assert np.all(
        score_map <= 1.0
    )

    assert score_map[0, 0] == pytest.approx(
        0.5
    )