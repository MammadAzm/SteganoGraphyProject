from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EmbeddingResult:
    stego: np.ndarray
    payload_rate: float
    selected_samples: int
    changed_samples: int
    changed_fraction: float


def deterministic_lsb_replacement(
    grayscale: np.ndarray,
    *,
    payload_rate: float,
    seed: int,
) -> EmbeddingResult:
    if grayscale.dtype != np.uint8:
        raise TypeError(
            "LSB replacement requires a uint8 grayscale image."
        )

    if grayscale.ndim != 2:
        raise ValueError(
            "LSB replacement requires a two-dimensional grayscale image."
        )

    if not 0 <= payload_rate <= 1:
        raise ValueError(
            "Payload rate must be between zero and one."
        )

    flat_cover = grayscale.reshape(-1)
    total_samples = flat_cover.size

    selected_samples = int(
        round(payload_rate * total_samples)
    )

    flat_stego = flat_cover.copy()

    if selected_samples == 0:
        return EmbeddingResult(
            stego=flat_stego.reshape(grayscale.shape),
            payload_rate=payload_rate,
            selected_samples=0,
            changed_samples=0,
            changed_fraction=0.0,
        )

    generator = np.random.default_rng(seed)

    selected_indices = generator.choice(
        total_samples,
        size=selected_samples,
        replace=False,
    )

    message_bits = generator.integers(
        low=0,
        high=2,
        size=selected_samples,
        dtype=np.uint8,
    )

    original_values = flat_stego[selected_indices].copy()

    flat_stego[selected_indices] = (
        flat_stego[selected_indices] & np.uint8(0xFE)
    ) | message_bits

    changed_samples = int(
        np.count_nonzero(
            original_values != flat_stego[selected_indices]
        )
    )

    return EmbeddingResult(
        stego=flat_stego.reshape(grayscale.shape),
        payload_rate=payload_rate,
        selected_samples=selected_samples,
        changed_samples=changed_samples,
        changed_fraction=(
            changed_samples / total_samples
        ),
    )