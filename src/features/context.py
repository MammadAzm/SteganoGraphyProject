from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import cv2
import numpy as np

from src.features.embedding import (
    EmbeddingResult,
    deterministic_lsb_replacement,
)
from src.features.settings import (
    FeatureExtractionSettings,
)


@dataclass
class FeatureContext:
    patch: np.ndarray
    settings: FeatureExtractionSettings
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.patch, np.ndarray):
            raise TypeError(
                "Patch must be a NumPy array."
            )

        if self.patch.dtype != np.uint8:
            raise TypeError(
                "Patch must use uint8 pixel values."
            )

        if self.patch.ndim not in {2, 3}:
            raise ValueError(
                "Patch must be grayscale or RGB."
            )

        if (
            self.patch.ndim == 3
            and self.patch.shape[2] != 3
        ):
            raise ValueError(
                "An RGB patch must have exactly three channels."
            )

        if self.patch.shape[0] < 8 or self.patch.shape[1] < 8:
            raise ValueError(
                "Patch dimensions must be at least 8x8."
            )

    @cached_property
    def gray(self) -> np.ndarray:
        if self.patch.ndim == 2:
            return self.patch.copy()

        return cv2.cvtColor(
            self.patch,
            cv2.COLOR_RGB2GRAY,
        )

    @cached_property
    def gray_float(self) -> np.ndarray:
        return self.gray.astype(np.float64) / 255.0

    @cached_property
    def embedding(self) -> EmbeddingResult:
        return deterministic_lsb_replacement(
            self.gray,
            payload_rate=self.settings.stego.payload_rate,
            seed=self.seed,
        )

    @cached_property
    def stego_gray(self) -> np.ndarray:
        return self.embedding.stego