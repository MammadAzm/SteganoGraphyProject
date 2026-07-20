from __future__ import annotations

import math

import numpy as np
import pytest

from src.config.schema import FeatureItemSettings
from src.features.context import FeatureContext
from src.features.embedding import (
    deterministic_lsb_replacement,
)
from src.features.extractors import (
    edge_density,
    entropy,
    laplacian_variance,
    lbp_entropy,
    lbp_non_uniform_ratio,
    mse,
    ms_ssim,
    psnr,
    ssim,
    variance,
)
from src.features.pipeline import FeaturePipeline
from src.features.settings import (
    FeatureExtractionSettings,
)
from src.features.synthetic import (
    create_synthetic_patches,
)


@pytest.fixture
def settings() -> FeatureExtractionSettings:
    return FeatureExtractionSettings.defaults()


@pytest.fixture
def patches() -> dict[str, np.ndarray]:
    return create_synthetic_patches(
        width=64,
        height=64,
        seed=42,
    )


def create_context(
    patch: np.ndarray,
    settings: FeatureExtractionSettings,
) -> FeatureContext:
    return FeatureContext(
        patch=patch,
        settings=settings,
        seed=42,
    )


def test_lsb_replacement_is_deterministic(
    patches: dict[str, np.ndarray],
) -> None:
    gray = patches["random_noise"][:, :, 0]

    first = deterministic_lsb_replacement(
        gray,
        payload_rate=0.40,
        seed=123,
    )

    second = deterministic_lsb_replacement(
        gray,
        payload_rate=0.40,
        seed=123,
    )

    third = deterministic_lsb_replacement(
        gray,
        payload_rate=0.40,
        seed=456,
    )

    assert np.array_equal(
        first.stego,
        second.stego,
    )

    assert not np.array_equal(
        first.stego,
        third.stego,
    )

    assert first.selected_samples == round(
        0.40 * gray.size
    )

    assert (
        0
        <= first.changed_samples
        <= first.selected_samples
    )


def test_flat_patch_has_zero_cover_complexity(
    settings: FeatureExtractionSettings,
    patches: dict[str, np.ndarray],
) -> None:
    flat_context = create_context(
        patches["flat_gray"],
        settings,
    )

    assert entropy(
        flat_context
    ) == pytest.approx(
        0.0
    )

    assert variance(
        flat_context
    ) == pytest.approx(
        0.0
    )

    assert edge_density(
        flat_context
    ) == pytest.approx(
        0.0
    )

    assert laplacian_variance(
        flat_context
    ) == pytest.approx(
        0.0
    )


def test_lbp_border_is_excluded_for_flat_patch(
    settings: FeatureExtractionSettings,
    patches: dict[str, np.ndarray],
) -> None:
    flat_context = create_context(
        patches["flat_gray"],
        settings,
    )

    assert lbp_entropy(
        flat_context
    ) == pytest.approx(
        0.0,
        abs=1e-12,
    )

    assert lbp_non_uniform_ratio(
        flat_context
    ) == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_noise_is_more_complex_than_flat_patch(
    settings: FeatureExtractionSettings,
    patches: dict[str, np.ndarray],
) -> None:
    flat_context = create_context(
        patches["flat_gray"],
        settings,
    )

    noise_context = create_context(
        patches["random_noise"],
        settings,
    )

    assert entropy(
        noise_context
    ) > entropy(
        flat_context
    )

    assert variance(
        noise_context
    ) > variance(
        flat_context
    )

    assert lbp_entropy(
        noise_context
    ) > lbp_entropy(
        flat_context
    )

    assert lbp_non_uniform_ratio(
        noise_context
    ) > lbp_non_uniform_ratio(
        flat_context
    )

    assert laplacian_variance(
        noise_context
    ) > laplacian_variance(
        flat_context
    )


def test_edge_density_detects_clean_edge(
    settings: FeatureExtractionSettings,
    patches: dict[str, np.ndarray],
) -> None:
    flat_context = create_context(
        patches["flat_gray"],
        settings,
    )

    edge_context = create_context(
        patches["clean_vertical_edge"],
        settings,
    )

    assert edge_density(
        edge_context
    ) > edge_density(
        flat_context
    )


def test_distortion_metrics_share_one_stego_patch(
    settings: FeatureExtractionSettings,
    patches: dict[str, np.ndarray],
) -> None:
    patch_context = create_context(
        patches["random_noise"],
        settings,
    )

    mse_value = mse(
        patch_context
    )

    psnr_value = psnr(
        patch_context
    )

    assert 0.0 <= mse_value <= 1.0

    if mse_value > 0:
        expected_psnr = 10.0 * math.log10(
            (255.0 * 255.0) / mse_value
        )

        assert psnr_value == pytest.approx(
            min(
                expected_psnr,
                settings.stego.max_psnr,
            )
        )

    assert (
        0.0
        <= ssim(patch_context)
        <= 1.0
    )

    assert (
        0.0
        <= ms_ssim(patch_context)
        <= 1.0
    )


def test_feature_pipeline_returns_finite_values(
    settings: FeatureExtractionSettings,
    patches: dict[str, np.ndarray],
) -> None:
    feature_names = (
        "entropy",
        "variance",
        "edge_density",
        "lbp_non_uniform_ratio",
        "lbp_entropy",
        "laplacian_variance",
        "mse",
        "psnr",
        "ssim",
        "ms_ssim",
    )

    items = {
        name: FeatureItemSettings(
            enabled=True,
            weight=1.0,
            direction=(
                "lower"
                if name == "mse"
                else "higher"
            ),
        )
        for name in feature_names
    }

    pipeline = FeaturePipeline(
        settings,
        items,
        default_seed=42,
    )

    values = pipeline.extract(
        patches["random_noise"],
        include_disabled=True,
    )

    assert set(values) == set(
        feature_names
    )

    assert all(
        np.isfinite(value)
        for value in values.values()
    )