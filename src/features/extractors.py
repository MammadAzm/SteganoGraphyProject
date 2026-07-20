from __future__ import annotations

import math

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.metrics import structural_similarity

from src.features.context import FeatureContext


def entropy(context: FeatureContext) -> float:
    histogram = np.bincount(
        context.gray.reshape(-1),
        minlength=256,
    ).astype(np.float64)

    probabilities = histogram / histogram.sum()
    probabilities = probabilities[probabilities > 0]

    return float(
        -np.sum(
            probabilities * np.log2(probabilities)
        )
    )


def variance(context: FeatureContext) -> float:
    return float(
        np.var(
            context.gray_float,
            dtype=np.float64,
        )
    )


def edge_density(context: FeatureContext) -> float:
    settings = context.settings.canny

    edges = cv2.Canny(
        context.gray,
        threshold1=settings.low_threshold,
        threshold2=settings.high_threshold,
        apertureSize=settings.aperture_size,
        L2gradient=settings.l2_gradient,
    )

    return float(
        np.count_nonzero(edges) / edges.size
    )


def _valid_lbp_values(
    context: FeatureContext,
) -> np.ndarray:
    settings = context.settings.lbp

    lbp = local_binary_pattern(
        context.gray,
        P=settings.points,
        R=settings.radius,
        method=settings.method,
    )

    if not settings.exclude_border:
        return lbp.reshape(-1)

    border = settings.radius

    if (
        context.gray.shape[0] <= 2 * border
        or context.gray.shape[1] <= 2 * border
    ):
        raise ValueError(
            "Patch dimensions are too small after excluding the LBP border."
        )

    valid_lbp = lbp[
        border:-border,
        border:-border,
    ]

    return valid_lbp.reshape(-1)


def lbp_entropy(context: FeatureContext) -> float:
    settings = context.settings.lbp

    lbp_values = _valid_lbp_values(
        context
    ).astype(np.int64)

    number_of_bins = settings.points + 2

    histogram = np.bincount(
        lbp_values,
        minlength=number_of_bins,
    ).astype(np.float64)

    probabilities = histogram / histogram.sum()
    probabilities = probabilities[probabilities > 0]

    value = float(
        -np.sum(
            probabilities * np.log2(probabilities)
        )
    )

    if settings.normalize_entropy:
        maximum_entropy = math.log2(number_of_bins)

        if maximum_entropy > 0:
            value /= maximum_entropy

    return float(
        np.clip(
            value,
            0.0,
            1.0,
        )
    )


def lbp_non_uniform_ratio(
    context: FeatureContext,
) -> float:
    settings = context.settings.lbp

    lbp_values = _valid_lbp_values(
        context
    ).astype(np.int64)

    non_uniform_code = settings.points + 1

    non_uniform_count = np.count_nonzero(
        lbp_values == non_uniform_code
    )

    return float(
        non_uniform_count / lbp_values.size
    )


def laplacian_variance(
    context: FeatureContext,
) -> float:
    laplacian = cv2.Laplacian(
        context.gray_float.astype(np.float32),
        ddepth=cv2.CV_32F,
        ksize=context.settings.laplacian.kernel_size,
        borderType=cv2.BORDER_REFLECT101,
    )

    return float(
        np.var(
            laplacian,
            dtype=np.float64,
        )
    )


def mse(context: FeatureContext) -> float:
    difference = (
        context.gray.astype(np.float64)
        - context.stego_gray.astype(np.float64)
    )

    return float(
        np.mean(difference * difference)
    )


def psnr(context: FeatureContext) -> float:
    mse_value = mse(context)

    if mse_value == 0:
        return context.settings.stego.max_psnr

    value = 10.0 * math.log10(
        (255.0 * 255.0) / mse_value
    )

    return float(
        min(
            value,
            context.settings.stego.max_psnr,
        )
    )


def _ssim_window_size(
    image: np.ndarray,
    preferred: int = 7,
) -> int:
    minimum_dimension = min(image.shape[:2])

    window_size = min(
        preferred,
        minimum_dimension,
    )

    if window_size % 2 == 0:
        window_size -= 1

    if window_size < 3:
        raise ValueError(
            "SSIM requires image dimensions of at least 3x3."
        )

    return window_size


def ssim(context: FeatureContext) -> float:
    window_size = _ssim_window_size(
        context.gray,
    )

    return float(
        structural_similarity(
            context.gray,
            context.stego_gray,
            data_range=255,
            win_size=window_size,
        )
    )


def _ssim_components(
    first: np.ndarray,
    second: np.ndarray,
    *,
    window_size: int,
    sigma: float,
) -> tuple[float, float]:
    first = first.astype(np.float64)
    second = second.astype(np.float64)

    mu_first = cv2.GaussianBlur(
        first,
        (window_size, window_size),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT101,
    )

    mu_second = cv2.GaussianBlur(
        second,
        (window_size, window_size),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT101,
    )

    mu_first_squared = mu_first * mu_first
    mu_second_squared = mu_second * mu_second
    mu_product = mu_first * mu_second

    variance_first = cv2.GaussianBlur(
        first * first,
        (window_size, window_size),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT101,
    ) - mu_first_squared

    variance_second = cv2.GaussianBlur(
        second * second,
        (window_size, window_size),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT101,
    ) - mu_second_squared

    covariance = cv2.GaussianBlur(
        first * second,
        (window_size, window_size),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT101,
    ) - mu_product

    variance_first = np.maximum(
        variance_first,
        0.0,
    )

    variance_second = np.maximum(
        variance_second,
        0.0,
    )

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    luminance = (
        2.0 * mu_product + c1
    ) / (
        mu_first_squared
        + mu_second_squared
        + c1
    )

    contrast_structure = (
        2.0 * covariance + c2
    ) / (
        variance_first
        + variance_second
        + c2
    )

    border = window_size // 2

    if (
        first.shape[0] > 2 * border
        and first.shape[1] > 2 * border
    ):
        luminance = luminance[
            border:-border,
            border:-border,
        ]

        contrast_structure = contrast_structure[
            border:-border,
            border:-border,
        ]

    return (
        float(np.mean(luminance)),
        float(np.mean(contrast_structure)),
    )


def ms_ssim(context: FeatureContext) -> float:
    settings = context.settings.ms_ssim

    first = context.gray.copy()
    second = context.stego_gray.copy()

    weights = np.asarray(
        settings.weights,
        dtype=np.float64,
    )

    weights = weights / weights.sum()

    luminance_values: list[float] = []
    contrast_structure_values: list[float] = []

    for scale_index in range(settings.scales):
        minimum_dimension = min(first.shape[:2])

        window_size = min(
            settings.gaussian_window_size,
            minimum_dimension,
        )

        if window_size % 2 == 0:
            window_size -= 1

        window_size = max(
            window_size,
            3,
        )

        luminance_value, contrast_structure_value = (
            _ssim_components(
                first,
                second,
                window_size=window_size,
                sigma=settings.gaussian_sigma,
            )
        )

        luminance_values.append(
            max(
                luminance_value,
                1e-12,
            )
        )

        contrast_structure_values.append(
            max(
                contrast_structure_value,
                1e-12,
            )
        )

        if scale_index < settings.scales - 1:
            if min(first.shape[:2]) < 8:
                raise ValueError(
                    "The patch is too small for the configured "
                    "number of MS-SSIM scales."
                )

            first = cv2.pyrDown(first)
            second = cv2.pyrDown(second)

    result = 1.0

    for index in range(settings.scales - 1):
        result *= (
            contrast_structure_values[index]
            ** weights[index]
        )

    final_index = settings.scales - 1

    result *= (
        luminance_values[final_index]
        ** weights[final_index]
    )

    result *= (
        contrast_structure_values[final_index]
        ** weights[final_index]
    )

    return float(
        np.clip(
            result,
            0.0,
            1.0,
        )
    )