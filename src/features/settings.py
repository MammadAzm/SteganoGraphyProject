from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.config.loader import LoadedConfiguration
from src.config.schema import ConfigurationError


def _mapping(
    value: Any,
    field_name: str,
) -> Mapping[str, Any]:
    if value is None:
        return {}

    if not isinstance(value, Mapping):
        raise ConfigurationError(
            f"Configuration field '{field_name}' must be a mapping."
        )

    return value


def _integer(
    value: Any,
    field_name: str,
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
    ):
        raise ConfigurationError(
            f"Configuration field '{field_name}' must be an integer."
        )

    return value


def _float(
    value: Any,
    field_name: str,
) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ConfigurationError(
            f"Configuration field '{field_name}' must be numeric."
        ) from error


def _boolean(
    value: Any,
    field_name: str,
) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(
            f"Configuration field '{field_name}' must be boolean."
        )

    return value


@dataclass(frozen=True)
class CannySettings:
    low_threshold: int
    high_threshold: int
    aperture_size: int
    l2_gradient: bool


@dataclass(frozen=True)
class LBPSettings:
    points: int
    radius: int
    method: str
    normalize_entropy: bool
    exclude_border: bool


@dataclass(frozen=True)
class LaplacianSettings:
    kernel_size: int


@dataclass(frozen=True)
class StegoSimulationSettings:
    method: str
    payload_rate: float
    max_psnr: float


@dataclass(frozen=True)
class MSSSIMSettings:
    scales: int
    weights: tuple[float, ...]
    gaussian_window_size: int
    gaussian_sigma: float


@dataclass(frozen=True)
class FeatureExtractionSettings:
    grayscale_method: str
    canny: CannySettings
    lbp: LBPSettings
    laplacian: LaplacianSettings
    stego: StegoSimulationSettings
    ms_ssim: MSSSIMSettings

    @classmethod
    def defaults(cls) -> "FeatureExtractionSettings":
        return cls(
            grayscale_method="bt601",
            canny=CannySettings(
                low_threshold=100,
                high_threshold=200,
                aperture_size=3,
                l2_gradient=True,
            ),
            lbp=LBPSettings(
                points=8,
                radius=1,
                method="uniform",
                normalize_entropy=True,
                exclude_border=True,
            ),
            laplacian=LaplacianSettings(
                kernel_size=3,
            ),
            stego=StegoSimulationSettings(
                method="lsb_replacement",
                payload_rate=0.40,
                max_psnr=100.0,
            ),
            ms_ssim=MSSSIMSettings(
                scales=3,
                weights=(0.10, 0.30, 0.60),
                gaussian_window_size=11,
                gaussian_sigma=1.5,
            ),
        )

    @classmethod
    def from_loaded_configuration(
        cls,
        loaded: LoadedConfiguration,
    ) -> "FeatureExtractionSettings":
        defaults = cls.defaults()

        root = _mapping(
            loaded.raw.get("feature_extraction"),
            "feature_extraction",
        )

        grayscale = _mapping(
            root.get("grayscale"),
            "feature_extraction.grayscale",
        )

        canny = _mapping(
            root.get("canny"),
            "feature_extraction.canny",
        )

        lbp = _mapping(
            root.get("lbp"),
            "feature_extraction.lbp",
        )

        laplacian = _mapping(
            root.get("laplacian"),
            "feature_extraction.laplacian",
        )

        stego = _mapping(
            root.get("stego_simulation"),
            "feature_extraction.stego_simulation",
        )

        ms_ssim = _mapping(
            root.get("ms_ssim"),
            "feature_extraction.ms_ssim",
        )

        weights_value = ms_ssim.get(
            "weights",
            list(defaults.ms_ssim.weights),
        )

        if not isinstance(weights_value, list):
            raise ConfigurationError(
                "'feature_extraction.ms_ssim.weights' must be a list."
            )

        settings = cls(
            grayscale_method=str(
                grayscale.get(
                    "method",
                    defaults.grayscale_method,
                )
            ).lower(),
            canny=CannySettings(
                low_threshold=_integer(
                    canny.get(
                        "low_threshold",
                        defaults.canny.low_threshold,
                    ),
                    "feature_extraction.canny.low_threshold",
                ),
                high_threshold=_integer(
                    canny.get(
                        "high_threshold",
                        defaults.canny.high_threshold,
                    ),
                    "feature_extraction.canny.high_threshold",
                ),
                aperture_size=_integer(
                    canny.get(
                        "aperture_size",
                        defaults.canny.aperture_size,
                    ),
                    "feature_extraction.canny.aperture_size",
                ),
                l2_gradient=_boolean(
                    canny.get(
                        "l2_gradient",
                        defaults.canny.l2_gradient,
                    ),
                    "feature_extraction.canny.l2_gradient",
                ),
            ),
            lbp=LBPSettings(
                points=_integer(
                    lbp.get(
                        "points",
                        defaults.lbp.points,
                    ),
                    "feature_extraction.lbp.points",
                ),
                radius=_integer(
                    lbp.get(
                        "radius",
                        defaults.lbp.radius,
                    ),
                    "feature_extraction.lbp.radius",
                ),
                method=str(
                    lbp.get(
                        "method",
                        defaults.lbp.method,
                    )
                ).lower(),
                normalize_entropy=_boolean(
                    lbp.get(
                        "normalize_entropy",
                        defaults.lbp.normalize_entropy,
                    ),
                    "feature_extraction.lbp.normalize_entropy",
                ),
                exclude_border=_boolean(
                    lbp.get(
                        "exclude_border",
                        defaults.lbp.exclude_border,
                    ),
                    "feature_extraction.lbp.exclude_border",
                ),
            ),
            laplacian=LaplacianSettings(
                kernel_size=_integer(
                    laplacian.get(
                        "kernel_size",
                        defaults.laplacian.kernel_size,
                    ),
                    "feature_extraction.laplacian.kernel_size",
                )
            ),
            stego=StegoSimulationSettings(
                method=str(
                    stego.get(
                        "method",
                        defaults.stego.method,
                    )
                ).lower(),
                payload_rate=_float(
                    stego.get(
                        "payload_rate",
                        defaults.stego.payload_rate,
                    ),
                    (
                        "feature_extraction.stego_simulation."
                        "payload_rate"
                    ),
                ),
                max_psnr=_float(
                    stego.get(
                        "max_psnr",
                        defaults.stego.max_psnr,
                    ),
                    (
                        "feature_extraction.stego_simulation."
                        "max_psnr"
                    ),
                ),
            ),
            ms_ssim=MSSSIMSettings(
                scales=_integer(
                    ms_ssim.get(
                        "scales",
                        defaults.ms_ssim.scales,
                    ),
                    "feature_extraction.ms_ssim.scales",
                ),
                weights=tuple(
                    _float(
                        value,
                        "feature_extraction.ms_ssim.weights",
                    )
                    for value in weights_value
                ),
                gaussian_window_size=_integer(
                    ms_ssim.get(
                        "gaussian_window_size",
                        defaults.ms_ssim.gaussian_window_size,
                    ),
                    (
                        "feature_extraction.ms_ssim."
                        "gaussian_window_size"
                    ),
                ),
                gaussian_sigma=_float(
                    ms_ssim.get(
                        "gaussian_sigma",
                        defaults.ms_ssim.gaussian_sigma,
                    ),
                    (
                        "feature_extraction.ms_ssim."
                        "gaussian_sigma"
                    ),
                ),
            ),
        )

        settings.validate()
        return settings

    def validate(self) -> None:
        errors: list[str] = []

        if self.grayscale_method != "bt601":
            errors.append(
                "Only BT.601 grayscale conversion is currently supported."
            )

        if not 0 <= self.canny.low_threshold <= 255:
            errors.append(
                "Canny low threshold must be between 0 and 255."
            )

        if not 0 <= self.canny.high_threshold <= 255:
            errors.append(
                "Canny high threshold must be between 0 and 255."
            )

        if self.canny.low_threshold >= self.canny.high_threshold:
            errors.append(
                "Canny low threshold must be smaller than the high threshold."
            )

        if self.canny.aperture_size not in {3, 5, 7}:
            errors.append(
                "Canny aperture size must be 3, 5, or 7."
            )

        if self.lbp.points <= 0:
            errors.append(
                "LBP points must be greater than zero."
            )

        if self.lbp.radius <= 0:
            errors.append(
                "LBP radius must be greater than zero."
            )

        if self.lbp.method != "uniform":
            errors.append(
                "Only uniform LBP is currently supported."
            )

        if self.laplacian.kernel_size not in {1, 3, 5, 7}:
            errors.append(
                "Laplacian kernel size must be 1, 3, 5, or 7."
            )

        if self.stego.method != "lsb_replacement":
            errors.append(
                "Only deterministic LSB replacement is currently supported."
            )

        if not 0 <= self.stego.payload_rate <= 1:
            errors.append(
                "Stego payload rate must be between zero and one."
            )

        if self.stego.max_psnr <= 0:
            errors.append(
                "Maximum PSNR must be greater than zero."
            )

        if self.ms_ssim.scales <= 0:
            errors.append(
                "MS-SSIM scales must be greater than zero."
            )

        if len(self.ms_ssim.weights) != self.ms_ssim.scales:
            errors.append(
                "The number of MS-SSIM weights must equal the number "
                "of scales."
            )

        if any(weight <= 0 for weight in self.ms_ssim.weights):
            errors.append(
                "Every MS-SSIM weight must be greater than zero."
            )

        if (
            self.ms_ssim.gaussian_window_size < 3
            or self.ms_ssim.gaussian_window_size % 2 == 0
        ):
            errors.append(
                "MS-SSIM Gaussian window size must be an odd integer "
                "greater than or equal to three."
            )

        if self.ms_ssim.gaussian_sigma <= 0:
            errors.append(
                "MS-SSIM Gaussian sigma must be greater than zero."
            )

        if errors:
            formatted_errors = "\n".join(
                f"- {error}" for error in errors
            )

            raise ConfigurationError(
                "Invalid feature extraction configuration:\n"
                f"{formatted_errors}"
            )