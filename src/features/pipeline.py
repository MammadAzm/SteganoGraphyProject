from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from src.config.loader import LoadedConfiguration
from src.config.schema import FeatureItemSettings
from src.features.context import FeatureContext
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
from src.features.settings import (
    FeatureExtractionSettings,
)


FeatureFunction = Callable[[FeatureContext], float]


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    function: FeatureFunction
    requires_stego: bool
    description: str
    expected_range: str


FEATURE_REGISTRY: dict[str, FeatureDefinition] = {
    "entropy": FeatureDefinition(
        name="entropy",
        function=entropy,
        requires_stego=False,
        description=(
            "Shannon entropy of grayscale intensity values."
        ),
        expected_range="[0, 8] bits",
    ),
    "variance": FeatureDefinition(
        name="variance",
        function=variance,
        requires_stego=False,
        description=(
            "Variance of grayscale intensities normalized to [0, 1]."
        ),
        expected_range="[0, 0.25]",
    ),
    "edge_density": FeatureDefinition(
        name="edge_density",
        function=edge_density,
        requires_stego=False,
        description=(
            "Fraction of pixels identified as Canny edges."
        ),
        expected_range="[0, 1]",
    ),
    "lbp_non_uniform_ratio": FeatureDefinition(
        name="lbp_non_uniform_ratio",
        function=lbp_non_uniform_ratio,
        requires_stego=False,
        description=(
            "Fraction of valid interior pixels assigned to the "
            "non-uniform LBP class."
        ),
        expected_range="[0, 1]",
    ),
    "lbp_entropy": FeatureDefinition(
        name="lbp_entropy",
        function=lbp_entropy,
        requires_stego=False,
        description=(
            "Normalized entropy of the valid-interior uniform-LBP histogram."
        ),
        expected_range="[0, 1]",
    ),
    "laplacian_variance": FeatureDefinition(
        name="laplacian_variance",
        function=laplacian_variance,
        requires_stego=False,
        description=(
            "Variance of the Laplacian response on normalized grayscale."
        ),
        expected_range="[0, +inf)",
    ),
    "mse": FeatureDefinition(
        name="mse",
        function=mse,
        requires_stego=True,
        description=(
            "Mean squared error between the cover and deterministic "
            "LSB-replacement stego patch."
        ),
        expected_range="[0, 1] for grayscale LSB replacement",
    ),
    "psnr": FeatureDefinition(
        name="psnr",
        function=psnr,
        requires_stego=True,
        description=(
            "Peak signal-to-noise ratio between cover and stego patch."
        ),
        expected_range="[0, configured maximum] dB",
    ),
    "ssim": FeatureDefinition(
        name="ssim",
        function=ssim,
        requires_stego=True,
        description=(
            "Single-scale structural similarity between cover and stego."
        ),
        expected_range="normally [0, 1]",
    ),
    "ms_ssim": FeatureDefinition(
        name="ms_ssim",
        function=ms_ssim,
        requires_stego=True,
        description=(
            "Multi-scale structural similarity using a Gaussian pyramid."
        ),
        expected_range="[0, 1]",
    ),
}


class FeaturePipeline:
    def __init__(
        self,
        extraction_settings: FeatureExtractionSettings,
        feature_items: Mapping[
            str,
            FeatureItemSettings,
        ],
        *,
        default_seed: int,
    ) -> None:
        self.extraction_settings = extraction_settings
        self.feature_items = dict(feature_items)
        self.default_seed = int(default_seed)

        unknown_features = (
            set(self.feature_items)
            - set(FEATURE_REGISTRY)
        )

        if unknown_features:
            raise ValueError(
                "Unknown configured features: "
                + ", ".join(
                    sorted(unknown_features)
                )
            )

    @classmethod
    def from_loaded_configuration(
        cls,
        loaded: LoadedConfiguration,
    ) -> "FeaturePipeline":
        return cls(
            extraction_settings=(
                FeatureExtractionSettings
                .from_loaded_configuration(loaded)
            ),
            feature_items=loaded.settings.features.items,
            default_seed=loaded.settings.project.random_seed,
        )

    def feature_names(
        self,
        *,
        include_disabled: bool = False,
    ) -> tuple[str, ...]:
        names: list[str] = []

        for name, item in self.feature_items.items():
            if include_disabled or item.enabled:
                names.append(name)

        return tuple(names)

    def extract(
        self,
        patch: np.ndarray,
        *,
        seed: int | None = None,
        include_disabled: bool = False,
    ) -> dict[str, float]:
        context = FeatureContext(
            patch=patch,
            settings=self.extraction_settings,
            seed=(
                self.default_seed
                if seed is None
                else int(seed)
            ),
        )

        output: dict[str, float] = {}

        for name in self.feature_names(
            include_disabled=include_disabled
        ):
            definition = FEATURE_REGISTRY[name]

            value = float(
                definition.function(context)
            )

            if not np.isfinite(value):
                raise ValueError(
                    f"Feature '{name}' produced a non-finite value."
                )

            output[name] = value

        return output

    def definitions(
        self,
        *,
        include_disabled: bool = False,
    ) -> dict[str, dict[str, object]]:
        output: dict[str, dict[str, object]] = {}

        for name in self.feature_names(
            include_disabled=include_disabled
        ):
            definition = FEATURE_REGISTRY[name]
            item = self.feature_items[name]

            output[name] = {
                "enabled": item.enabled,
                "weight": item.weight,
                "direction": item.direction,
                "requires_stego": definition.requires_stego,
                "description": definition.description,
                "expected_range": definition.expected_range,
            }

        return output