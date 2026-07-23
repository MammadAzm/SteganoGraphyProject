from __future__ import annotations

import numpy as np

from src.labeling.inspection import (
    detect_suspicious_flags,
)
from src.labeling.inspection_settings import (
    SuspiciousMapSettings,
)
from src.labeling.visualization import (
    build_pixel_coverage_map,
)


def suspicious_settings() -> SuspiciousMapSettings:
    return SuspiciousMapSettings(
        constant_map_std=1e-6,
        low_map_std=0.03,
        high_map_std=0.25,
        low_score_threshold=0.10,
        high_score_threshold=0.90,
        extreme_fraction=0.95,
        excessive_zero_fraction=0.20,
        excessive_one_fraction=0.20,
        group_disagreement_threshold=0.50,
        isolated_peak_difference=0.35,
        isolated_peak_fraction=0.01,
    )


def test_pixel_coverage_map_with_overlapping_windows() -> None:
    score_map = np.asarray(
        [
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    coordinates = np.asarray(
        [
            [
                [0, 0, 2, 2],
                [1, 0, 2, 2],
            ]
        ],
        dtype=np.int32,
    )

    coverage = build_pixel_coverage_map(
        score_map,
        coordinates,
        image_height=2,
        image_width=3,
    )

    expected = np.asarray(
        [
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
        ],
        dtype=np.float32,
    )

    assert np.allclose(
        coverage,
        expected,
    )


def test_constant_score_map_is_flagged() -> None:
    score_map = np.full(
        (25, 25),
        0.5,
        dtype=np.float32,
    )

    group_maps = np.full(
        (3, 25, 25),
        0.5,
        dtype=np.float32,
    )

    flags = detect_suspicious_flags(
        score_map,
        group_maps,
        suspicious_settings(),
        tolerance=1e-6,
    )

    assert (
        "CONSTANT_OR_NEAR_CONSTANT_MAP"
        in flags
    )


def test_excessive_zero_map_is_flagged() -> None:
    score_map = np.zeros(
        (25, 25),
        dtype=np.float32,
    )

    group_maps = np.zeros(
        (3, 25, 25),
        dtype=np.float32,
    )

    flags = detect_suspicious_flags(
        score_map,
        group_maps,
        suspicious_settings(),
        tolerance=1e-6,
    )

    assert (
        "EXCESSIVE_EXACT_ZEROS"
        in flags
    )

    assert (
        "ALMOST_ALL_SCORES_LOW"
        in flags
    )


def test_group_disagreement_is_flagged() -> None:
    score_map = np.full(
        (25, 25),
        0.5,
        dtype=np.float32,
    )

    group_maps = np.stack(
        [
            np.zeros(
                (25, 25),
                dtype=np.float32,
            ),
            np.full(
                (25, 25),
                0.5,
                dtype=np.float32,
            ),
            np.ones(
                (25, 25),
                dtype=np.float32,
            ),
        ]
    )

    flags = detect_suspicious_flags(
        score_map,
        group_maps,
        suspicious_settings(),
        tolerance=1e-6,
    )

    assert (
        "LARGE_GROUP_DISAGREEMENT"
        in flags
    )


def test_normal_nonconstant_map_is_not_constant() -> None:
    generator = np.random.default_rng(
        42
    )

    score_map = generator.uniform(
        0.2,
        0.8,
        size=(25, 25),
    ).astype(
        np.float32
    )

    group_maps = np.stack(
        [
            score_map,
            score_map,
            score_map,
        ]
    )

    flags = detect_suspicious_flags(
        score_map,
        group_maps,
        suspicious_settings(),
        tolerance=1e-6,
    )

    assert (
        "CONSTANT_OR_NEAR_CONSTANT_MAP"
        not in flags
    )