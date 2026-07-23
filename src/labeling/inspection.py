from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import (
    ks_2samp,
    wasserstein_distance,
)

from src.config.loader import LoadedConfiguration
from src.data.sliding_window import SlidingWindowGrid
from src.features.fusion import GroupedFusionSettings
from src.labeling.generator import (
    list_split_images,
)
from src.labeling.inspection_settings import (
    LabelInspectionSettings,
    SuspiciousMapSettings,
)
from src.labeling.normalization import (
    load_normalization_artifact,
)
from src.labeling.settings import (
    LabelGenerationSettings,
)
from src.labeling.storage import (
    LABEL_FILE_VERSION,
)
from src.labeling.visualization import (
    plot_score_histograms,
    render_label_panel,
    select_visual_records,
)
from src.utils.hashing import (
    canonical_json_hash,
)


@dataclass(frozen=True)
class InspectionIssue:
    severity: str
    code: str
    message: str
    split: str | None = None
    label_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "split": self.split,
            "label_path": self.label_path,
        }


@dataclass
class RunningStatistics:
    count: int = 0
    total: float = 0.0
    squared_total: float = 0.0
    minimum: float = float("inf")
    maximum: float = float("-inf")

    def update(
        self,
        values: np.ndarray,
    ) -> None:
        values = np.asarray(
            values,
            dtype=np.float64,
        )

        self.count += int(
            values.size
        )

        self.total += float(
            np.sum(values)
        )

        self.squared_total += float(
            np.sum(values * values)
        )

        self.minimum = min(
            self.minimum,
            float(np.min(values)),
        )

        self.maximum = max(
            self.maximum,
            float(np.max(values)),
        )

    def to_dict(self) -> dict[str, float | int]:
        if self.count == 0:
            return {
                "count": 0,
                "minimum": 0.0,
                "maximum": 0.0,
                "mean": 0.0,
                "standard_deviation": 0.0,
            }

        mean = self.total / self.count

        variance = max(
            (
                self.squared_total
                / self.count
            )
            - mean * mean,
            0.0,
        )

        return {
            "count": self.count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": mean,
            "standard_deviation": (
                variance ** 0.5
            ),
        }


def detect_suspicious_flags(
    score_map: np.ndarray,
    group_score_maps: np.ndarray,
    settings: SuspiciousMapSettings,
    *,
    tolerance: float,
) -> list[str]:
    flags: list[str] = []

    score_standard_deviation = float(
        np.std(score_map)
    )

    if (
        score_standard_deviation
        <= settings.constant_map_std
    ):
        flags.append(
            "CONSTANT_OR_NEAR_CONSTANT_MAP"
        )

    elif (
        score_standard_deviation
        <= settings.low_map_std
    ):
        flags.append(
            "LOW_MAP_VARIATION"
        )

    if (
        score_standard_deviation
        >= settings.high_map_std
    ):
        flags.append(
            "HIGH_MAP_VARIATION"
        )

    low_fraction = float(
        np.mean(
            score_map
            <= settings.low_score_threshold
        )
    )

    high_fraction = float(
        np.mean(
            score_map
            >= settings.high_score_threshold
        )
    )

    if (
        low_fraction
        >= settings.extreme_fraction
    ):
        flags.append(
            "ALMOST_ALL_SCORES_LOW"
        )

    if (
        high_fraction
        >= settings.extreme_fraction
    ):
        flags.append(
            "ALMOST_ALL_SCORES_HIGH"
        )

    zero_fraction = float(
        np.mean(
            score_map <= tolerance
        )
    )

    one_fraction = float(
        np.mean(
            score_map >= 1.0 - tolerance
        )
    )

    if (
        zero_fraction
        >= settings.excessive_zero_fraction
    ):
        flags.append(
            "EXCESSIVE_EXACT_ZEROS"
        )

    if (
        one_fraction
        >= settings.excessive_one_fraction
    ):
        flags.append(
            "EXCESSIVE_EXACT_ONES"
        )

    group_means = np.mean(
        group_score_maps,
        axis=(1, 2),
    )

    if (
        float(
            np.max(group_means)
            - np.min(group_means)
        )
        >= settings.group_disagreement_threshold
    ):
        flags.append(
            "LARGE_GROUP_DISAGREEMENT"
        )

    local_median = median_filter(
        score_map,
        size=3,
        mode="nearest",
    )

    isolated_difference = (
        score_map - local_median
    )

    isolated_fraction = float(
        np.mean(
            isolated_difference
            >= settings.isolated_peak_difference
        )
    )

    if (
        isolated_fraction
        >= settings.isolated_peak_fraction
    ):
        flags.append(
            "ISOLATED_SCORE_PEAKS"
        )

    return flags


def _threshold_column(
    threshold: float,
) -> str:
    return (
        f"fraction_ge_"
        f"{threshold:.2f}"
    ).replace(
        ".",
        "_",
    )


class LabelMapInspector:
    def __init__(
        self,
        loaded: LoadedConfiguration,
    ) -> None:
        self.loaded = loaded
        self.project = loaded.settings
        self.paths = loaded.paths

        self.label_settings = (
            LabelGenerationSettings
            .from_loaded_configuration(
                loaded
            )
        )

        self.settings = (
            LabelInspectionSettings
            .from_loaded_configuration(
                loaded
            )
        )

        self.grid = (
            SlidingWindowGrid
            .from_configuration(
                self.project
            )
        )

        self.fusion = (
            GroupedFusionSettings
            .from_loaded_configuration(
                loaded
            )
        )

        self.feature_names = tuple(
            feature_name
            for feature_name, feature_item
            in self.project.features.items.items()
            if feature_item.enabled
        )

        self.group_names = tuple(
            group.name
            for group in self.fusion.groups
        )

        self.normalization_artifact = (
            load_normalization_artifact(
                self.label_settings
                .normalization
                .artifact_path
            )
        )

        self.normalization_fingerprint = (
            canonical_json_hash(
                self.normalization_artifact.to_dict()
            )
        )

        self.issues: list[
            InspectionIssue
        ] = []

        self.settings.output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

    def add_issue(
        self,
        severity: str,
        code: str,
        message: str,
        *,
        split: str | None = None,
        label_path: Path | None = None,
    ) -> None:
        self.issues.append(
            InspectionIssue(
                severity=severity,
                code=code,
                message=message,
                split=split,
                label_path=(
                    str(label_path.resolve())
                    if label_path is not None
                    else None
                ),
            )
        )

    def _validate_required_arrays(
        self,
        label_file: np.lib.npyio.NpzFile,
        *,
        split_name: str,
        label_path: Path,
    ) -> bool:
        available = set(
            label_file.files
        )

        missing = [
            array_name
            for array_name
            in self.settings.required_arrays
            if array_name not in available
        ]

        if missing:
            self.add_issue(
                "error",
                "MISSING_REQUIRED_ARRAYS",
                (
                    "Label file is missing required arrays: "
                    + ", ".join(missing)
                ),
                split=split_name,
                label_path=label_path,
            )

            return False

        return True

    def _inspect_label_file(
        self,
        *,
        split_name: str,
        label_path: Path,
        source_paths: set[str],
        seen_source_paths: set[str],
        split_score_buffer: np.ndarray,
        score_offset: int,
        group_statistics: dict[
            str,
            RunningStatistics,
        ],
        feature_clipping: dict[
            str,
            dict[str, int],
        ],
    ) -> tuple[
        dict[str, Any] | None,
        int,
    ]:
        tolerance = (
            self.settings.numerical_tolerance
        )

        try:
            with np.load(
                label_path,
                allow_pickle=False,
            ) as label_file:
                if not self._validate_required_arrays(
                    label_file,
                    split_name=split_name,
                    label_path=label_path,
                ):
                    return None, score_offset

                version = int(
                    label_file[
                        "label_file_version"
                    ]
                )

                if version != LABEL_FILE_VERSION:
                    self.add_issue(
                        "error",
                        "INVALID_LABEL_VERSION",
                        (
                            f"Expected label version "
                            f"{LABEL_FILE_VERSION}; found "
                            f"{version}."
                        ),
                        split=split_name,
                        label_path=label_path,
                    )

                    return None, score_offset

                score_map = label_file[
                    "score_map"
                ].astype(
                    np.float32
                )

                raw_feature_maps = label_file[
                    "raw_feature_maps"
                ].astype(
                    np.float32
                )

                normalized_feature_maps = (
                    label_file[
                        "normalized_feature_maps"
                    ].astype(
                        np.float32
                    )
                )

                group_score_maps = label_file[
                    "group_score_maps"
                ].astype(
                    np.float32
                )

                window_coordinates = label_file[
                    "window_coordinates"
                ].astype(
                    np.int32
                )

                feature_names = tuple(
                    str(value)
                    for value in label_file[
                        "feature_names"
                    ].tolist()
                )

                group_names = tuple(
                    str(value)
                    for value in label_file[
                        "group_names"
                    ].tolist()
                )

                metadata = json.loads(
                    str(
                        label_file[
                            "metadata_json"
                        ].item()
                    )
                )

        except Exception as error:
            self.add_issue(
                "error",
                "UNREADABLE_LABEL_FILE",
                (
                    f"Failed to read label file: "
                    f"{type(error).__name__}: {error}"
                ),
                split=split_name,
                label_path=label_path,
            )

            return None, score_offset

        expected_feature_shape = (
            len(self.feature_names),
            self.grid.grid_height,
            self.grid.grid_width,
        )

        expected_group_shape = (
            len(self.group_names),
            self.grid.grid_height,
            self.grid.grid_width,
        )

        expected_coordinate_shape = (
            self.grid.grid_height,
            self.grid.grid_width,
            4,
        )

        structural_checks = [
            (
                score_map.shape
                == self.grid.score_shape,
                "INVALID_SCORE_MAP_SHAPE",
                (
                    f"Expected score-map shape "
                    f"{self.grid.score_shape}; found "
                    f"{score_map.shape}."
                ),
            ),
            (
                raw_feature_maps.shape
                == expected_feature_shape,
                "INVALID_RAW_FEATURE_SHAPE",
                (
                    f"Expected raw-feature shape "
                    f"{expected_feature_shape}; found "
                    f"{raw_feature_maps.shape}."
                ),
            ),
            (
                normalized_feature_maps.shape
                == expected_feature_shape,
                "INVALID_NORMALIZED_FEATURE_SHAPE",
                (
                    f"Expected normalized-feature shape "
                    f"{expected_feature_shape}; found "
                    f"{normalized_feature_maps.shape}."
                ),
            ),
            (
                group_score_maps.shape
                == expected_group_shape,
                "INVALID_GROUP_MAP_SHAPE",
                (
                    f"Expected group-map shape "
                    f"{expected_group_shape}; found "
                    f"{group_score_maps.shape}."
                ),
            ),
            (
                window_coordinates.shape
                == expected_coordinate_shape,
                "INVALID_COORDINATE_SHAPE",
                (
                    f"Expected coordinate shape "
                    f"{expected_coordinate_shape}; found "
                    f"{window_coordinates.shape}."
                ),
            ),
            (
                feature_names
                == self.feature_names,
                "FEATURE_NAME_MISMATCH",
                (
                    f"Expected features "
                    f"{self.feature_names}; found "
                    f"{feature_names}."
                ),
            ),
            (
                group_names
                == self.group_names,
                "GROUP_NAME_MISMATCH",
                (
                    f"Expected groups "
                    f"{self.group_names}; found "
                    f"{group_names}."
                ),
            ),
        ]

        structure_valid = True

        for valid, code, message in structural_checks:
            if valid:
                continue

            structure_valid = False

            self.add_issue(
                "error",
                code,
                message,
                split=split_name,
                label_path=label_path,
            )

        if not structure_valid:
            return None, score_offset

        expected_coordinates = (
            self.grid.coordinate_array()
        )

        if not np.array_equal(
            window_coordinates,
            expected_coordinates,
        ):
            self.add_issue(
                "error",
                "WINDOW_COORDINATE_MISMATCH",
                (
                    "Saved window coordinates do not match the "
                    "configured sliding-window grid."
                ),
                split=split_name,
                label_path=label_path,
            )

        arrays_to_check = {
            "score_map": score_map,
            "raw_feature_maps": raw_feature_maps,
            "normalized_feature_maps": (
                normalized_feature_maps
            ),
            "group_score_maps": group_score_maps,
        }

        for array_name, array_value in (
            arrays_to_check.items()
        ):
            if not np.all(
                np.isfinite(array_value)
            ):
                self.add_issue(
                    "error",
                    "NON_FINITE_VALUES",
                    (
                        f"Array '{array_name}' contains "
                        "NaN or infinite values."
                    ),
                    split=split_name,
                    label_path=label_path,
                )

        bounded_arrays = {
            "score_map": score_map,
            "normalized_feature_maps": (
                normalized_feature_maps
            ),
            "group_score_maps": group_score_maps,
        }

        for array_name, array_value in (
            bounded_arrays.items()
        ):
            minimum = float(
                np.min(array_value)
            )

            maximum = float(
                np.max(array_value)
            )

            if minimum < -tolerance:
                self.add_issue(
                    "error",
                    "VALUE_BELOW_ZERO",
                    (
                        f"Array '{array_name}' contains value "
                        f"{minimum}, below zero."
                    ),
                    split=split_name,
                    label_path=label_path,
                )

            if maximum > 1.0 + tolerance:
                self.add_issue(
                    "error",
                    "VALUE_ABOVE_ONE",
                    (
                        f"Array '{array_name}' contains value "
                        f"{maximum}, above one."
                    ),
                    split=split_name,
                    label_path=label_path,
                )

        source_relative_path = str(
            metadata.get(
                "source_relative_path",
                "",
            )
        )

        if not source_relative_path:
            self.add_issue(
                "error",
                "MISSING_SOURCE_PATH",
                (
                    "Label metadata does not contain "
                    "'source_relative_path'."
                ),
                split=split_name,
                label_path=label_path,
            )

        elif (
            source_relative_path
            not in source_paths
        ):
            self.add_issue(
                "error",
                "ORPHAN_LABEL",
                (
                    f"No source image exists for "
                    f"'{source_relative_path}'."
                ),
                split=split_name,
                label_path=label_path,
            )

        if source_relative_path in seen_source_paths:
            self.add_issue(
                "error",
                "DUPLICATE_SOURCE_LABEL",
                (
                    f"More than one label maps to source image "
                    f"'{source_relative_path}'."
                ),
                split=split_name,
                label_path=label_path,
            )

        seen_source_paths.add(
            source_relative_path
        )

        if (
            metadata.get(
                "normalization_fingerprint"
            )
            != self.normalization_fingerprint
        ):
            self.add_issue(
                "error",
                "NORMALIZATION_FINGERPRINT_MISMATCH",
                (
                    "Label uses a different normalization "
                    "artifact fingerprint."
                ),
                split=split_name,
                label_path=label_path,
            )

        if (
            metadata.get("split")
            != split_name
        ):
            self.add_issue(
                "error",
                "SPLIT_METADATA_MISMATCH",
                (
                    f"Label metadata split is "
                    f"'{metadata.get('split')}', expected "
                    f"'{split_name}'."
                ),
                split=split_name,
                label_path=label_path,
            )

        score_values = score_map.reshape(-1)

        next_offset = (
            score_offset
            + score_values.size
        )

        split_score_buffer[
            score_offset:next_offset
        ] = score_values

        for feature_index, feature_name in (
            enumerate(self.feature_names)
        ):
            normalized_values = (
                normalized_feature_maps[
                    feature_index
                ]
            )

            clipping = feature_clipping[
                feature_name
            ]

            clipping["count"] += int(
                normalized_values.size
            )

            clipping["lower_clipped"] += int(
                np.count_nonzero(
                    normalized_values
                    <= tolerance
                )
            )

            clipping["upper_clipped"] += int(
                np.count_nonzero(
                    normalized_values
                    >= 1.0 - tolerance
                )
            )

        group_means: dict[str, float] = {}

        for group_index, group_name in (
            enumerate(self.group_names)
        ):
            group_map = group_score_maps[
                group_index
            ]

            group_statistics[
                group_name
            ].update(
                group_map
            )

            group_means[
                group_name
            ] = float(
                np.mean(group_map)
            )

        score_mean = float(
            np.mean(score_map)
        )

        score_std = float(
            np.std(score_map)
        )

        flags = detect_suspicious_flags(
            score_map,
            group_score_maps,
            self.settings.suspicious,
            tolerance=tolerance,
        )

        record: dict[str, Any] = {
            "split": split_name,
            "source_relative_path": (
                source_relative_path
            ),
            "label_path": str(
                label_path.resolve()
            ),
            "label_fingerprint": str(
                metadata.get(
                    "label_fingerprint",
                    "",
                )
            ),
            "normalization_fingerprint": str(
                metadata.get(
                    "normalization_fingerprint",
                    "",
                )
            ),
            "score_min": float(
                np.min(score_map)
            ),
            "score_q01": float(
                np.quantile(
                    score_map,
                    0.01,
                )
            ),
            "score_q25": float(
                np.quantile(
                    score_map,
                    0.25,
                )
            ),
            "score_median": float(
                np.quantile(
                    score_map,
                    0.50,
                )
            ),
            "score_mean": score_mean,
            "score_q75": float(
                np.quantile(
                    score_map,
                    0.75,
                )
            ),
            "score_q99": float(
                np.quantile(
                    score_map,
                    0.99,
                )
            ),
            "score_max": float(
                np.max(score_map)
            ),
            "score_standard_deviation": (
                score_std
            ),
            "zero_fraction": float(
                np.mean(
                    score_map <= tolerance
                )
            ),
            "one_fraction": float(
                np.mean(
                    score_map
                    >= 1.0 - tolerance
                )
            ),
            "low_score_fraction": float(
                np.mean(
                    score_map
                    <= self.settings
                    .suspicious
                    .low_score_threshold
                )
            ),
            "high_score_fraction": float(
                np.mean(
                    score_map
                    >= self.settings
                    .suspicious
                    .high_score_threshold
                )
            ),
            "suspicious_flags": "|".join(
                flags
            ),
            "suspicious_flag_count": len(
                flags
            ),
        }

        for threshold in (
            self.settings.preview_thresholds
        ):
            record[
                _threshold_column(
                    threshold
                )
            ] = float(
                np.mean(
                    score_map >= threshold
                )
            )

        for group_name, group_mean in (
            group_means.items()
        ):
            record[
                f"group_mean_{group_name}"
            ] = group_mean

        return record, next_offset

    def _inspect_split(
        self,
        split_name: str,
    ) -> dict[str, Any]:
        source_directory = (
            self.paths.dataset_split(
                split_name
            )
        )

        label_directory = (
            self.label_settings.output_root
            / split_name
        )

        source_images = list_split_images(
            source_directory,
            recursive=self.project.dataset.recursive,
            extensions=(
                self.project.dataset
                .allowed_extensions
            ),
        )

        source_paths = {
            image_path.relative_to(
                source_directory
            ).as_posix()
            for image_path in source_images
        }

        label_paths = sorted(
            label_directory.glob(
                f"*{self.project.labels.filename_suffix}"
            ),
            key=lambda path: path.name.lower(),
        )

        expected_count = (
            self.settings.expected_counts[
                split_name
            ]
        )

        if len(source_images) != expected_count:
            self.add_issue(
                "error",
                "SOURCE_IMAGE_COUNT_MISMATCH",
                (
                    f"Expected {expected_count} source images; "
                    f"found {len(source_images)}."
                ),
                split=split_name,
            )

        if len(label_paths) != expected_count:
            self.add_issue(
                "error",
                "LABEL_FILE_COUNT_MISMATCH",
                (
                    f"Expected {expected_count} label files; "
                    f"found {len(label_paths)}."
                ),
                split=split_name,
            )

        score_buffer = np.empty(
            (
                len(label_paths)
                * self.grid.windows_per_image
            ),
            dtype=np.float32,
        )

        group_statistics = {
            group_name: RunningStatistics()
            for group_name in self.group_names
        }

        feature_clipping = {
            feature_name: {
                "count": 0,
                "lower_clipped": 0,
                "upper_clipped": 0,
            }
            for feature_name in self.feature_names
        }

        records: list[
            dict[str, Any]
        ] = []

        seen_source_paths: set[str] = set()

        score_offset = 0

        for label_index, label_path in enumerate(
            label_paths,
            start=1,
        ):
            record, score_offset = (
                self._inspect_label_file(
                    split_name=split_name,
                    label_path=label_path,
                    source_paths=source_paths,
                    seen_source_paths=(
                        seen_source_paths
                    ),
                    split_score_buffer=(
                        score_buffer
                    ),
                    score_offset=score_offset,
                    group_statistics=(
                        group_statistics
                    ),
                    feature_clipping=(
                        feature_clipping
                    ),
                )
            )

            if record is not None:
                records.append(
                    record
                )

            if (
                label_index % 100 == 0
                or label_index
                == len(label_paths)
            ):
                print(
                    f"[{split_name}] "
                    f"inspected {label_index}/"
                    f"{len(label_paths)} labels"
                )

        missing_source_labels = (
            source_paths
            - seen_source_paths
        )

        for missing_path in sorted(
            missing_source_labels
        ):
            self.add_issue(
                "error",
                "MISSING_LABEL_FOR_SOURCE",
                (
                    f"No label file was found for source "
                    f"'{missing_path}'."
                ),
                split=split_name,
            )

        valid_scores = score_buffer[
            :score_offset
        ]

        if valid_scores.size == 0:
            self.add_issue(
                "error",
                "NO_VALID_LABEL_SCORES",
                (
                    f"No valid scores were loaded for "
                    f"split '{split_name}'."
                ),
                split=split_name,
            )

            valid_scores = np.asarray(
                [0.0],
                dtype=np.float32,
            )

        label_fingerprints = {
            str(
                record[
                    "label_fingerprint"
                ]
            )
            for record in records
            if record[
                "label_fingerprint"
            ]
        }

        if len(label_fingerprints) != 1:
            self.add_issue(
                "error",
                "INCONSISTENT_LABEL_FINGERPRINTS",
                (
                    f"Found {len(label_fingerprints)} distinct "
                    f"label fingerprints in split "
                    f"'{split_name}'."
                ),
                split=split_name,
            )

        split_statistics: dict[str, Any] = {
            "split": split_name,
            "source_image_count": len(
                source_images
            ),
            "label_file_count": len(
                label_paths
            ),
            "valid_label_count": len(
                records
            ),
            "total_score_count": int(
                valid_scores.size
            ),
            "score_min": float(
                np.min(valid_scores)
            ),
            "score_q01": float(
                np.quantile(
                    valid_scores,
                    0.01,
                )
            ),
            "score_q25": float(
                np.quantile(
                    valid_scores,
                    0.25,
                )
            ),
            "score_median": float(
                np.quantile(
                    valid_scores,
                    0.50,
                )
            ),
            "score_mean": float(
                np.mean(valid_scores)
            ),
            "score_q75": float(
                np.quantile(
                    valid_scores,
                    0.75,
                )
            ),
            "score_q99": float(
                np.quantile(
                    valid_scores,
                    0.99,
                )
            ),
            "score_max": float(
                np.max(valid_scores)
            ),
            "score_standard_deviation": float(
                np.std(valid_scores)
            ),
            "zero_fraction": float(
                np.mean(
                    valid_scores
                    <= self.settings
                    .numerical_tolerance
                )
            ),
            "one_fraction": float(
                np.mean(
                    valid_scores
                    >= 1.0
                    - self.settings
                    .numerical_tolerance
                )
            ),
            "suspicious_image_count": sum(
                int(
                    record[
                        "suspicious_flag_count"
                    ]
                    > 0
                )
                for record in records
            ),
            "label_fingerprint": (
                next(
                    iter(label_fingerprints)
                )
                if len(label_fingerprints) == 1
                else ""
            ),
        }

        for threshold in (
            self.settings.preview_thresholds
        ):
            split_statistics[
                _threshold_column(
                    threshold
                )
            ] = float(
                np.mean(
                    valid_scores >= threshold
                )
            )

        clipping_rows: list[
            dict[str, Any]
        ] = []

        for feature_name, clipping in (
            feature_clipping.items()
        ):
            count = clipping["count"]

            clipping_rows.append(
                {
                    "split": split_name,
                    "feature": feature_name,
                    "count": count,
                    "lower_clipped_count": (
                        clipping[
                            "lower_clipped"
                        ]
                    ),
                    "upper_clipped_count": (
                        clipping[
                            "upper_clipped"
                        ]
                    ),
                    "lower_clipped_fraction": (
                        clipping[
                            "lower_clipped"
                        ]
                        / count
                        if count > 0
                        else 0.0
                    ),
                    "upper_clipped_fraction": (
                        clipping[
                            "upper_clipped"
                        ]
                        / count
                        if count > 0
                        else 0.0
                    ),
                }
            )

        group_rows = [
            {
                "split": split_name,
                "group": group_name,
                **statistics.to_dict(),
            }
            for group_name, statistics
            in group_statistics.items()
        ]

        return {
            "split": split_name,
            "records": records,
            "scores": valid_scores,
            "statistics": split_statistics,
            "clipping_rows": clipping_rows,
            "group_rows": group_rows,
        }

    def _distribution_sample(
        self,
        values: np.ndarray,
        *,
        seed: int,
    ) -> np.ndarray:
        sample_limit = (
            self.settings.distribution
            .comparison_sample_limit
        )

        if values.size <= sample_limit:
            return values

        generator = np.random.default_rng(
            seed
        )

        selected_indices = generator.choice(
            values.size,
            size=sample_limit,
            replace=False,
        )

        return values[
            selected_indices
        ]

    def _distribution_comparisons(
        self,
        split_results: Mapping[
            str,
            Mapping[str, Any],
        ],
    ) -> list[dict[str, Any]]:
        split_names = list(
            split_results
        )

        output: list[
            dict[str, Any]
        ] = []

        for first_index in range(
            len(split_names)
        ):
            for second_index in range(
                first_index + 1,
                len(split_names),
            ):
                first_name = split_names[
                    first_index
                ]

                second_name = split_names[
                    second_index
                ]

                first_values = (
                    self._distribution_sample(
                        split_results[
                            first_name
                        ]["scores"],
                        seed=1000 + first_index,
                    )
                )

                second_values = (
                    self._distribution_sample(
                        split_results[
                            second_name
                        ]["scores"],
                        seed=2000 + second_index,
                    )
                )

                ks_result = ks_2samp(
                    first_values,
                    second_values,
                )

                output.append(
                    {
                        "split_a": first_name,
                        "split_b": second_name,
                        "sample_count_a": int(
                            first_values.size
                        ),
                        "sample_count_b": int(
                            second_values.size
                        ),
                        "mean_difference": float(
                            np.mean(first_values)
                            - np.mean(
                                second_values
                            )
                        ),
                        "wasserstein_distance": float(
                            wasserstein_distance(
                                first_values,
                                second_values,
                            )
                        ),
                        "ks_statistic": float(
                            ks_result.statistic
                        ),
                        "ks_pvalue": float(
                            ks_result.pvalue
                        ),
                    }
                )

        return output

    def _write_csv(
        self,
        path: Path,
        rows: Sequence[
            Mapping[str, Any]
        ],
    ) -> None:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not rows:
            path.write_text(
                "",
                encoding="utf-8",
            )

            return

        fieldnames: list[str] = []

        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(
                        key
                    )

        temporary_path = (
            path.with_suffix(
                path.suffix + ".tmp"
            )
        )

        with temporary_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for row in rows:
                writer.writerow(
                    {
                        field_name: row.get(
                            field_name,
                            "",
                        )
                        for field_name
                        in fieldnames
                    }
                )

        temporary_path.replace(
            path
        )

    def _write_json(
        self,
        path: Path,
        payload: Mapping[str, Any],
    ) -> None:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = (
            path.with_suffix(
                path.suffix + ".tmp"
            )
        )

        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as output_file:
            json.dump(
                payload,
                output_file,
                indent=2,
                ensure_ascii=False,
            )

        temporary_path.replace(
            path
        )

    def run(
        self,
        *,
        splits: Sequence[str] = (
            "train",
            "validation",
            "test",
        ),
        generate_galleries: bool = True,
    ) -> dict[str, Any]:
        split_results: dict[
            str,
            dict[str, Any],
        ] = {}

        for split_name in splits:
            split_results[
                split_name
            ] = self._inspect_split(
                split_name
            )

        global_label_fingerprints = {
            result[
                "statistics"
            ]["label_fingerprint"]
            for result in split_results.values()
            if result[
                "statistics"
            ]["label_fingerprint"]
        }

        if (
            len(global_label_fingerprints)
            != 1
        ):
            self.add_issue(
                "error",
                "CROSS_SPLIT_FINGERPRINT_MISMATCH",
                (
                    "Train, validation, and test labels do not "
                    "share one label fingerprint."
                ),
            )

        image_rows = [
            record
            for result in split_results.values()
            for record in result[
                "records"
            ]
        ]

        split_statistics_rows = [
            result[
                "statistics"
            ]
            for result in split_results.values()
        ]

        clipping_rows = [
            row
            for result in split_results.values()
            for row in result[
                "clipping_rows"
            ]
        ]

        group_rows = [
            row
            for result in split_results.values()
            for row in result[
                "group_rows"
            ]
        ]

        suspicious_rows = [
            record
            for record in image_rows
            if int(
                record[
                    "suspicious_flag_count"
                ]
            ) > 0
        ]

        distribution_rows = (
            self._distribution_comparisons(
                split_results
            )
        )

        self._write_csv(
            self.settings.output_root
            / "image_statistics.csv",
            image_rows,
        )

        self._write_csv(
            self.settings.output_root
            / "split_statistics.csv",
            split_statistics_rows,
        )

        self._write_csv(
            self.settings.output_root
            / "feature_clipping_statistics.csv",
            clipping_rows,
        )

        self._write_csv(
            self.settings.output_root
            / "group_statistics.csv",
            group_rows,
        )

        self._write_csv(
            self.settings.output_root
            / "distribution_comparison.csv",
            distribution_rows,
        )

        self._write_csv(
            self.settings.output_root
            / "suspicious_images.csv",
            suspicious_rows,
        )

        issue_counts = {
            "error": sum(
                issue.severity == "error"
                for issue in self.issues
            ),
            "warning": sum(
                issue.severity == "warning"
                for issue in self.issues
            ),
        }

        integrity_report = {
            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "critical_error_count": (
                issue_counts["error"]
            ),
            "warning_count": (
                issue_counts["warning"]
            ),
            "normalization_artifact": str(
                self.label_settings
                .normalization
                .artifact_path
            ),
            "normalization_fit_split": (
                self.normalization_artifact
                .fit_split
            ),
            "normalization_fingerprint": (
                self.normalization_fingerprint
            ),
            "label_fingerprints": sorted(
                global_label_fingerprints
            ),
            "expected_score_shape": list(
                self.grid.score_shape
            ),
            "expected_feature_names": list(
                self.feature_names
            ),
            "expected_group_names": list(
                self.group_names
            ),
            "issues": [
                issue.to_dict()
                for issue in self.issues
            ],
        }

        self._write_json(
            self.settings.output_root
            / "integrity_report.json",
            integrity_report,
        )

        plot_score_histograms(
            {
                split_name: result[
                    "scores"
                ]
                for split_name, result
                in split_results.items()
            },
            self.settings.output_root
            / "score_histograms.png",
            bins=(
                self.settings
                .distribution
                .histogram_bins
            ),
            dpi=(
                self.settings
                .visualization
                .dpi
            ),
        )

        visualized_records: list[
            dict[str, Any]
        ] = []

        if (
            generate_galleries
            and self.settings
            .visualization
            .enabled
        ):
            for split_name, result in (
                split_results.items()
            ):
                selected_records = (
                    select_visual_records(
                        result["records"],
                        samples_per_category=(
                            self.settings
                            .visualization
                            .samples_per_category
                        ),
                        maximum_records=(
                            self.settings
                            .visualization
                            .max_images_per_split
                        ),
                        random_seed=(
                            self.settings
                            .visualization
                            .random_seed
                        ),
                    )
                )

                source_directory = (
                    self.paths.dataset_split(
                        split_name
                    )
                )

                gallery_directory = (
                    self.settings.output_root
                    / "galleries"
                    / split_name
                )

                for gallery_index, record in (
                    enumerate(
                        selected_records,
                        start=1,
                    )
                ):
                    source_path = (
                        source_directory
                        / str(
                            record[
                                "source_relative_path"
                            ]
                        )
                    )

                    label_path = Path(
                        str(
                            record[
                                "label_path"
                            ]
                        )
                    )

                    output_path = (
                        gallery_directory
                        / (
                            f"{gallery_index:03d}_"
                            f"{source_path.stem}.png"
                        )
                    )

                    render_label_panel(
                        source_image_path=(
                            source_path
                        ),
                        label_path=(
                            label_path
                        ),
                        output_path=(
                            output_path
                        ),
                        overlay_alpha=(
                            self.settings
                            .visualization
                            .overlay_alpha
                        ),
                        top_windows=(
                            self.settings
                            .visualization
                            .top_windows
                        ),
                        bottom_windows=(
                            self.settings
                            .visualization
                            .bottom_windows
                        ),
                        dpi=(
                            self.settings
                            .visualization
                            .dpi
                        ),
                    )

                    visualized_record = dict(
                        record
                    )

                    visualized_record[
                        "gallery_path"
                    ] = str(
                        output_path.resolve()
                    )

                    visualized_records.append(
                        visualized_record
                    )

                print(
                    f"[{split_name}] generated "
                    f"{len(selected_records)} gallery panels"
                )

        self._write_csv(
            self.settings.output_root
            / "visualized_images.csv",
            visualized_records,
        )

        summary = {
            "stage": 4,
            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "automated_integrity_passed": (
                issue_counts["error"] == 0
            ),
            "critical_error_count": (
                issue_counts["error"]
            ),
            "warning_count": (
                issue_counts["warning"]
            ),
            "suspicious_image_count": len(
                suspicious_rows
            ),
            "visualized_image_count": len(
                visualized_records
            ),
            "normalization_fingerprint": (
                self.normalization_fingerprint
            ),
            "label_fingerprint": (
                next(
                    iter(
                        global_label_fingerprints
                    )
                )
                if len(
                    global_label_fingerprints
                ) == 1
                else ""
            ),
            "splits": {
                split_name: result[
                    "statistics"
                ]
                for split_name, result
                in split_results.items()
            },
            "reports": {
                "integrity_report": str(
                    (
                        self.settings.output_root
                        / "integrity_report.json"
                    ).resolve()
                ),
                "image_statistics": str(
                    (
                        self.settings.output_root
                        / "image_statistics.csv"
                    ).resolve()
                ),
                "split_statistics": str(
                    (
                        self.settings.output_root
                        / "split_statistics.csv"
                    ).resolve()
                ),
                "suspicious_images": str(
                    (
                        self.settings.output_root
                        / "suspicious_images.csv"
                    ).resolve()
                ),
                "histograms": str(
                    (
                        self.settings.output_root
                        / "score_histograms.png"
                    ).resolve()
                ),
                "galleries": str(
                    (
                        self.settings.output_root
                        / "galleries"
                    ).resolve()
                ),
            },
        }

        self._write_json(
            self.settings.output_root
            / "inspection_summary.json",
            summary,
        )

        pending_approval = {
            "stage": 4,
            "approved": False,
            "status": (
                "awaiting_manual_visual_review"
                if issue_counts["error"] == 0
                else "blocked_by_integrity_errors"
            ),
            "critical_error_count": (
                issue_counts["error"]
            ),
            "normalization_fingerprint": (
                self.normalization_fingerprint
            ),
            "label_fingerprint": (
                summary[
                    "label_fingerprint"
                ]
            ),
            "reviewed_visual_examples": 0,
            "reviewer": "",
            "review_notes": [],
        }

        self._write_json(
            self.settings.output_root
            / "label_approval.json",
            pending_approval,
        )

        return summary