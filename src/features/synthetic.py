from __future__ import annotations

import numpy as np


def _to_rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(
        gray[:, :, None],
        repeats=3,
        axis=2,
    )


def create_synthetic_patches(
    *,
    width: int,
    height: int,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    generator = np.random.default_rng(seed)

    black = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    middle_gray = np.full(
        (height, width),
        fill_value=128,
        dtype=np.uint8,
    )

    horizontal_gradient = np.tile(
        np.linspace(
            0,
            255,
            width,
            dtype=np.uint8,
        ),
        (height, 1),
    )

    clean_edge = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    clean_edge[:, width // 2:] = 255

    y_indices, x_indices = np.indices(
        (height, width)
    )

    checkerboard = (
        (
            (x_indices // 8)
            + (y_indices // 8)
        )
        % 2
        * 255
    ).astype(np.uint8)

    vertical_stripes = (
        (x_indices // 4) % 2 * 255
    ).astype(np.uint8)

    random_noise = generator.integers(
        low=0,
        high=256,
        size=(height, width),
        dtype=np.uint8,
    )

    low_amplitude_noise = np.clip(
        128
        + generator.normal(
            loc=0.0,
            scale=8.0,
            size=(height, width),
        ),
        0,
        255,
    ).astype(np.uint8)

    return {
        "flat_black": _to_rgb(black),
        "flat_gray": _to_rgb(middle_gray),
        "horizontal_gradient": _to_rgb(
            horizontal_gradient
        ),
        "clean_vertical_edge": _to_rgb(
            clean_edge
        ),
        "checkerboard": _to_rgb(checkerboard),
        "vertical_stripes": _to_rgb(
            vertical_stripes
        ),
        "low_amplitude_noise": _to_rgb(
            low_amplitude_noise
        ),
        "random_noise": _to_rgb(random_noise),
    }