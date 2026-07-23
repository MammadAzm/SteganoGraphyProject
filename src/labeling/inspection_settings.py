from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.config.loader import LoadedConfiguration
from src.config.schema import ConfigurationError
from src.utils.paths import resolve_path


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


def _boolean(
    value: Any,
    field_name: str,
) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(
            f"Configuration field '{field_name}' must be boolean."
        )

    return value


def _positive_integer(
    value: Any,
    field_name: str,
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ConfigurationError(
            f"Configuration field '{field_name}' must be a "
            "positive integer."
        )

    return value


def _non_negative_float(
    value: Any,
    field_name: str,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ConfigurationError(
            f"Configuration field '{field_name}' must be numeric."
        ) from error

    if parsed < 0:
        raise ConfigurationError(
            f"Configuration field '{field_name}' cannot be negative."
        )

    return parsed


@dataclass(frozen=True)
class DistributionInspectionSettings:
    comparison_sample_limit: int
    histogram_bins: int


@dataclass(frozen=True)
class SuspiciousMapSettings:
    constant_map_std: float
    low_map_std: float
    high_map_std: float

    low_score_threshold: float
    high_score_threshold: float
    extreme_fraction: float

    excessive_zero_fraction: float
    excessive_one_fraction: float

    group_disagreement_threshold: float

    isolated_peak_difference: float
    isolated_peak_fraction: float


@dataclass(frozen=True)
class VisualizationInspectionSettings:
    enabled: bool
    samples_per_category: int
    max_images_per_split: int
    top_windows: int
    bottom_windows: int
    overlay_alpha: float
    dpi: int
    random_seed: int


@dataclass(frozen=True)
class LabelInspectionSettings:
    output_root: Path
    expected_counts: dict[str, int]
    required_arrays: tuple[str, ...]
    preview_thresholds: tuple[float, ...]
    numerical_tolerance: float
    distribution: DistributionInspectionSettings
    suspicious: SuspiciousMapSettings
    visualization: VisualizationInspectionSettings
    overwrite: bool

    @classmethod
    def from_loaded_configuration(
        cls,
        loaded: LoadedConfiguration,
    ) -> "LabelInspectionSettings":
        root = _mapping(
            loaded.raw.get("label_inspection"),
            "label_inspection",
        )

        expected_counts_section = _mapping(
            root.get("expected_counts"),
            "label_inspection.expected_counts",
        )

        required_arrays = root.get(
            "required_arrays",
            [
                "score_map",
                "raw_feature_maps",
                "normalized_feature_maps",
                "group_score_maps",
                "window_coordinates",
                "feature_names",
                "group_names",
                "metadata_json",
            ],
        )

        if not isinstance(required_arrays, list):
            raise ConfigurationError(
                "'label_inspection.required_arrays' must be a list."
            )

        preview_thresholds = root.get(
            "preview_thresholds",
            [0.50, 0.55, 0.60, 0.65],
        )

        if not isinstance(preview_thresholds, list):
            raise ConfigurationError(
                "'label_inspection.preview_thresholds' must be a list."
            )

        distribution_section = _mapping(
            root.get("distribution"),
            "label_inspection.distribution",
        )

        suspicious_section = _mapping(
            root.get("suspicious"),
            "label_inspection.suspicious",
        )

        visualization_section = _mapping(
            root.get("visualization"),
            "label_inspection.visualization",
        )

        reports_section = _mapping(
            root.get("reports"),
            "label_inspection.reports",
        )

        settings = cls(
            output_root=resolve_path(
                str(
                    root.get(
                        "output_root",
                        "outputs/label_inspection",
                    )
                ),
                loaded.paths.project_root,
            ),
            expected_counts={
                split_name: _positive_integer(
                    expected_counts_section.get(
                        split_name,
                        default_count,
                    ),
                    (
                        "label_inspection.expected_counts."
                        f"{split_name}"
                    ),
                )
                for split_name, default_count in {
                    "train": 10000,
                    "validation": 3000,
                    "test": 2000,
                }.items()
            },
            required_arrays=tuple(
                str(name)
                for name in required_arrays
            ),
            preview_thresholds=tuple(
                float(value)
                for value in preview_thresholds
            ),
            numerical_tolerance=_non_negative_float(
                root.get(
                    "numerical_tolerance",
                    1e-6,
                ),
                "label_inspection.numerical_tolerance",
            ),
            distribution=DistributionInspectionSettings(
                comparison_sample_limit=_positive_integer(
                    distribution_section.get(
                        "comparison_sample_limit",
                        500000,
                    ),
                    (
                        "label_inspection.distribution."
                        "comparison_sample_limit"
                    ),
                ),
                histogram_bins=_positive_integer(
                    distribution_section.get(
                        "histogram_bins",
                        50,
                    ),
                    (
                        "label_inspection.distribution."
                        "histogram_bins"
                    ),
                ),
            ),
            suspicious=SuspiciousMapSettings(
                constant_map_std=_non_negative_float(
                    suspicious_section.get(
                        "constant_map_std",
                        1e-6,
                    ),
                    (
                        "label_inspection.suspicious."
                        "constant_map_std"
                    ),
                ),
                low_map_std=_non_negative_float(
                    suspicious_section.get(
                        "low_map_std",
                        0.03,
                    ),
                    (
                        "label_inspection.suspicious."
                        "low_map_std"
                    ),
                ),
                high_map_std=_non_negative_float(
                    suspicious_section.get(
                        "high_map_std",
                        0.25,
                    ),
                    (
                        "label_inspection.suspicious."
                        "high_map_std"
                    ),
                ),
                low_score_threshold=_non_negative_float(
                    suspicious_section.get(
                        "low_score_threshold",
                        0.10,
                    ),
                    (
                        "label_inspection.suspicious."
                        "low_score_threshold"
                    ),
                ),
                high_score_threshold=_non_negative_float(
                    suspicious_section.get(
                        "high_score_threshold",
                        0.90,
                    ),
                    (
                        "label_inspection.suspicious."
                        "high_score_threshold"
                    ),
                ),
                extreme_fraction=_non_negative_float(
                    suspicious_section.get(
                        "extreme_fraction",
                        0.95,
                    ),
                    (
                        "label_inspection.suspicious."
                        "extreme_fraction"
                    ),
                ),
                excessive_zero_fraction=_non_negative_float(
                    suspicious_section.get(
                        "excessive_zero_fraction",
                        0.20,
                    ),
                    (
                        "label_inspection.suspicious."
                        "excessive_zero_fraction"
                    ),
                ),
                excessive_one_fraction=_non_negative_float(
                    suspicious_section.get(
                        "excessive_one_fraction",
                        0.20,
                    ),
                    (
                        "label_inspection.suspicious."
                        "excessive_one_fraction"
                    ),
                ),
                group_disagreement_threshold=_non_negative_float(
                    suspicious_section.get(
                        "group_disagreement_threshold",
                        0.50,
                    ),
                    (
                        "label_inspection.suspicious."
                        "group_disagreement_threshold"
                    ),
                ),
                isolated_peak_difference=_non_negative_float(
                    suspicious_section.get(
                        "isolated_peak_difference",
                        0.35,
                    ),
                    (
                        "label_inspection.suspicious."
                        "isolated_peak_difference"
                    ),
                ),
                isolated_peak_fraction=_non_negative_float(
                    suspicious_section.get(
                        "isolated_peak_fraction",
                        0.01,
                    ),
                    (
                        "label_inspection.suspicious."
                        "isolated_peak_fraction"
                    ),
                ),
            ),
            visualization=VisualizationInspectionSettings(
                enabled=_boolean(
                    visualization_section.get(
                        "enabled",
                        True,
                    ),
                    (
                        "label_inspection.visualization."
                        "enabled"
                    ),
                ),
                samples_per_category=_positive_integer(
                    visualization_section.get(
                        "samples_per_category",
                        3,
                    ),
                    (
                        "label_inspection.visualization."
                        "samples_per_category"
                    ),
                ),
                max_images_per_split=_positive_integer(
                    visualization_section.get(
                        "max_images_per_split",
                        24,
                    ),
                    (
                        "label_inspection.visualization."
                        "max_images_per_split"
                    ),
                ),
                top_windows=_positive_integer(
                    visualization_section.get(
                        "top_windows",
                        5,
                    ),
                    (
                        "label_inspection.visualization."
                        "top_windows"
                    ),
                ),
                bottom_windows=_positive_integer(
                    visualization_section.get(
                        "bottom_windows",
                        5,
                    ),
                    (
                        "label_inspection.visualization."
                        "bottom_windows"
                    ),
                ),
                overlay_alpha=float(
                    visualization_section.get(
                        "overlay_alpha",
                        0.45,
                    )
                ),
                dpi=_positive_integer(
                    visualization_section.get(
                        "dpi",
                        160,
                    ),
                    "label_inspection.visualization.dpi",
                ),
                random_seed=int(
                    visualization_section.get(
                        "random_seed",
                        42,
                    )
                ),
            ),
            overwrite=_boolean(
                reports_section.get(
                    "overwrite",
                    True,
                ),
                "label_inspection.reports.overwrite",
            ),
        )

        settings.validate()
        return settings

    def validate(self) -> None:
        errors: list[str] = []

        if not self.required_arrays:
            errors.append(
                "At least one required label array must be configured."
            )

        if not self.preview_thresholds:
            errors.append(
                "At least one preview threshold must be configured."
            )

        if any(
            threshold < 0 or threshold > 1
            for threshold in self.preview_thresholds
        ):
            errors.append(
                "Preview thresholds must lie inside [0, 1]."
            )

        if (
            self.suspicious.low_score_threshold
            >= self.suspicious.high_score_threshold
        ):
            errors.append(
                "Low score threshold must be smaller than the high "
                "score threshold."
            )

        probability_values = [
            self.suspicious.extreme_fraction,
            self.suspicious.excessive_zero_fraction,
            self.suspicious.excessive_one_fraction,
            self.suspicious.isolated_peak_fraction,
            self.visualization.overlay_alpha,
        ]

        if any(
            value < 0 or value > 1
            for value in probability_values
        ):
            errors.append(
                "Configured fractions and overlay alpha must lie "
                "inside [0, 1]."
            )

        if errors:
            raise ConfigurationError(
                "Invalid label-inspection configuration:\n"
                + "\n".join(
                    f"- {error}"
                    for error in errors
                )
            )