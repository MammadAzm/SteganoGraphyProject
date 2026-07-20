from __future__ import annotations

import pytest

from src.config.schema import FeatureItemSettings
from src.features.fusion import (
    FeatureGroupSettings,
    GroupedFusionSettings,
    fit_robust_normalization_bounds,
    fuse_normalized_features,
    normalize_feature_values,
)


def feature_items() -> dict[
    str,
    FeatureItemSettings,
]:
    return {
        "entropy": FeatureItemSettings(
            enabled=True,
            weight=1.0,
            direction="higher",
        ),
        "variance": FeatureItemSettings(
            enabled=True,
            weight=1.0,
            direction="higher",
        ),
        "edge_density": FeatureItemSettings(
            enabled=True,
            weight=1.0,
            direction="higher",
        ),
        "laplacian_variance": FeatureItemSettings(
            enabled=True,
            weight=1.0,
            direction="higher",
        ),
        "lbp_non_uniform_ratio": FeatureItemSettings(
            enabled=True,
            weight=1.0,
            direction="higher",
        ),
    }


def fusion_settings() -> GroupedFusionSettings:
    return GroupedFusionSettings(
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


def test_robust_normalization_clips_values() -> None:
    rows = [
        {
            "entropy": 0.0,
            "variance": 0.0,
        },
        {
            "entropy": 5.0,
            "variance": 0.1,
        },
        {
            "entropy": 10.0,
            "variance": 0.2,
        },
    ]

    bounds = fit_robust_normalization_bounds(
        rows,
        (
            "entropy",
            "variance",
        ),
        percentile_low=0.0,
        percentile_high=100.0,
        epsilon=1e-8,
    )

    items = {
        "entropy": FeatureItemSettings(
            enabled=True,
            weight=1.0,
            direction="higher",
        ),
        "variance": FeatureItemSettings(
            enabled=True,
            weight=1.0,
            direction="higher",
        ),
    }

    normalized = normalize_feature_values(
        {
            "entropy": 20.0,
            "variance": -1.0,
        },
        bounds,
        items,
    )

    assert normalized["entropy"] == pytest.approx(
        1.0
    )

    assert normalized["variance"] == pytest.approx(
        0.0
    )


def test_lower_direction_is_inverted() -> None:
    rows = [
        {"mse": 0.0},
        {"mse": 1.0},
    ]

    bounds = fit_robust_normalization_bounds(
        rows,
        ("mse",),
        percentile_low=0.0,
        percentile_high=100.0,
        epsilon=1e-8,
    )

    items = {
        "mse": FeatureItemSettings(
            enabled=True,
            weight=1.0,
            direction="lower",
        )
    }

    low_mse = normalize_feature_values(
        {"mse": 0.0},
        bounds,
        items,
    )

    high_mse = normalize_feature_values(
        {"mse": 1.0},
        bounds,
        items,
    )

    assert low_mse["mse"] == pytest.approx(
        1.0
    )

    assert high_mse["mse"] == pytest.approx(
        0.0
    )


def test_grouped_fusion_gives_equal_group_influence() -> None:
    settings = fusion_settings()

    normalized = {
        "entropy": 1.0,
        "variance": 1.0,
        "edge_density": 0.0,
        "laplacian_variance": 0.0,
        "lbp_non_uniform_ratio": 0.5,
    }

    group_scores, combined_score = (
        fuse_normalized_features(
            normalized,
            settings,
        )
    )

    assert group_scores[
        "intensity_complexity"
    ] == pytest.approx(
        1.0
    )

    assert group_scores[
        "spatial_complexity"
    ] == pytest.approx(
        0.0
    )

    assert group_scores[
        "pattern_complexity"
    ] == pytest.approx(
        0.5
    )

    assert combined_score == pytest.approx(
        0.5
    )


def test_fusion_settings_cover_all_enabled_features() -> None:
    settings = fusion_settings()

    settings.validate(
        feature_items()
    )