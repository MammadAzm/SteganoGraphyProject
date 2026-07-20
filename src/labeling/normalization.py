from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.config.schema import FeatureItemSettings
from src.features.fusion import (
    FeatureNormalizationBounds,
    GroupedFusionSettings,
)


NORMALIZATION_ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class NormalizationArtifact:
    version: int
    created_at_utc: str
    fit_split: str
    feature_names: tuple[str, ...]
    sample_count: int
    image_count: int
    percentile_low: float
    percentile_high: float
    epsilon: float
    raw_feature_fingerprint: str
    bounds: dict[str, FeatureNormalizationBounds]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at_utc": self.created_at_utc,
            "fit_split": self.fit_split,
            "feature_names": list(self.feature_names),
            "sample_count": self.sample_count,
            "image_count": self.image_count,
            "percentile_low": self.percentile_low,
            "percentile_high": self.percentile_high,
            "epsilon": self.epsilon,
            "raw_feature_fingerprint": (
                self.raw_feature_fingerprint
            ),
            "bounds": {
                feature_name: asdict(feature_bounds)
                for feature_name, feature_bounds
                in self.bounds.items()
            },
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "NormalizationArtifact":
        bounds_payload = payload.get(
            "bounds",
            {},
        )

        bounds = {
            str(feature_name): FeatureNormalizationBounds(
                feature_name=str(
                    feature_data["feature_name"]
                ),
                lower=float(
                    feature_data["lower"]
                ),
                upper=float(
                    feature_data["upper"]
                ),
            )
            for feature_name, feature_data
            in bounds_payload.items()
        }

        artifact = cls(
            version=int(payload["version"]),
            created_at_utc=str(
                payload["created_at_utc"]
            ),
            fit_split=str(payload["fit_split"]),
            feature_names=tuple(
                str(name)
                for name in payload["feature_names"]
            ),
            sample_count=int(
                payload["sample_count"]
            ),
            image_count=int(
                payload["image_count"]
            ),
            percentile_low=float(
                payload["percentile_low"]
            ),
            percentile_high=float(
                payload["percentile_high"]
            ),
            epsilon=float(
                payload["epsilon"]
            ),
            raw_feature_fingerprint=str(
                payload["raw_feature_fingerprint"]
            ),
            bounds=bounds,
        )

        artifact.validate()
        return artifact

    def validate(self) -> None:
        errors: list[str] = []

        if self.version != NORMALIZATION_ARTIFACT_VERSION:
            errors.append(
                f"Unsupported normalization artifact version "
                f"{self.version}."
            )

        if self.fit_split != "train":
            errors.append(
                "Normalization artifact was not fitted on training data."
            )

        if tuple(self.bounds) != self.feature_names:
            errors.append(
                "Normalization feature order does not match its bounds."
            )

        if self.sample_count <= 0:
            errors.append(
                "Normalization sample count must be positive."
            )

        if self.image_count <= 0:
            errors.append(
                "Normalization image count must be positive."
            )

        for feature_name, bounds in self.bounds.items():
            if not np.isfinite(bounds.lower):
                errors.append(
                    f"Feature '{feature_name}' has a non-finite "
                    "lower bound."
                )

            if not np.isfinite(bounds.upper):
                errors.append(
                    f"Feature '{feature_name}' has a non-finite "
                    "upper bound."
                )

            if bounds.upper <= bounds.lower:
                errors.append(
                    f"Feature '{feature_name}' has invalid bounds."
                )

        if errors:
            raise ValueError(
                "Invalid normalization artifact:\n"
                + "\n".join(
                    f"- {error}"
                    for error in errors
                )
            )


def save_normalization_artifact(
    artifact: NormalizationArtifact,
    path: Path,
) -> None:
    artifact.validate()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            artifact.to_dict(),
            output_file,
            indent=2,
            ensure_ascii=False,
        )

    temporary_path.replace(path)


def load_normalization_artifact(
    path: Path,
) -> NormalizationArtifact:
    if not path.is_file():
        raise FileNotFoundError(
            f"Normalization artifact not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        payload = json.load(input_file)

    return NormalizationArtifact.from_dict(
        payload
    )


def fit_exact_normalization(
    feature_matrix: np.memmap,
    feature_names: Sequence[str],
    *,
    image_count: int,
    percentile_low: float,
    percentile_high: float,
    epsilon: float,
    raw_feature_fingerprint: str,
) -> NormalizationArtifact:
    if feature_matrix.ndim != 2:
        raise ValueError(
            "Feature matrix must be two-dimensional."
        )

    if feature_matrix.shape[1] != len(feature_names):
        raise ValueError(
            "Feature matrix column count does not match feature names."
        )

    bounds: dict[str, FeatureNormalizationBounds] = {}

    for feature_index, feature_name in enumerate(
        feature_names
    ):
        values = np.asarray(
            feature_matrix[:, feature_index],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Feature '{feature_name}' contains non-finite values."
            )

        lower, upper = np.percentile(
            values,
            [
                percentile_low,
                percentile_high,
            ],
        )

        lower = float(lower)
        upper = float(upper)

        if upper - lower < epsilon:
            upper = lower + epsilon

        bounds[feature_name] = (
            FeatureNormalizationBounds(
                feature_name=feature_name,
                lower=lower,
                upper=upper,
            )
        )

    artifact = NormalizationArtifact(
        version=NORMALIZATION_ARTIFACT_VERSION,
        created_at_utc=datetime.now(
            timezone.utc
        ).isoformat(),
        fit_split="train",
        feature_names=tuple(feature_names),
        sample_count=int(
            feature_matrix.shape[0]
        ),
        image_count=int(image_count),
        percentile_low=float(percentile_low),
        percentile_high=float(percentile_high),
        epsilon=float(epsilon),
        raw_feature_fingerprint=raw_feature_fingerprint,
        bounds=bounds,
    )

    artifact.validate()
    return artifact


def normalize_feature_maps(
    raw_feature_maps: np.ndarray,
    artifact: NormalizationArtifact,
    feature_items: Mapping[
        str,
        FeatureItemSettings,
    ],
) -> np.ndarray:
    expected_shape = (
        len(artifact.feature_names),
        *raw_feature_maps.shape[1:],
    )

    if raw_feature_maps.shape != expected_shape:
        raise ValueError(
            f"Expected raw feature shape {expected_shape}; "
            f"found {raw_feature_maps.shape}."
        )

    normalized = np.empty_like(
        raw_feature_maps,
        dtype=np.float32,
    )

    for feature_index, feature_name in enumerate(
        artifact.feature_names
    ):
        bounds = artifact.bounds[
            feature_name
        ]

        feature_map = raw_feature_maps[
            feature_index
        ].astype(
            np.float64,
            copy=False,
        )

        output = (
            feature_map - bounds.lower
        ) / (
            bounds.upper - bounds.lower
        )

        output = np.clip(
            output,
            0.0,
            1.0,
        )

        direction = feature_items[
            feature_name
        ].direction

        if direction == "lower":
            output = 1.0 - output
        elif direction != "higher":
            raise ValueError(
                f"Unsupported feature direction "
                f"'{direction}' for '{feature_name}'."
            )

        normalized[
            feature_index
        ] = output.astype(
            np.float32
        )

    return normalized


def fuse_feature_maps(
    normalized_feature_maps: np.ndarray,
    feature_names: Sequence[str],
    fusion: GroupedFusionSettings,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    feature_indices = {
        feature_name: index
        for index, feature_name
        in enumerate(feature_names)
    }

    spatial_shape = normalized_feature_maps.shape[1:]

    group_maps = np.empty(
        (
            len(fusion.groups),
            *spatial_shape,
        ),
        dtype=np.float32,
    )

    group_names: list[str] = []
    total_group_weight = 0.0

    combined = np.zeros(
        spatial_shape,
        dtype=np.float64,
    )

    for group_index, group in enumerate(
        fusion.groups
    ):
        group_value = np.zeros(
            spatial_shape,
            dtype=np.float64,
        )

        total_member_weight = 0.0

        for feature_name, member_weight in (
            group.members.items()
        ):
            if feature_name not in feature_indices:
                raise KeyError(
                    f"Feature '{feature_name}' is missing from "
                    "the normalized maps."
                )

            group_value += (
                normalized_feature_maps[
                    feature_indices[feature_name]
                ]
                * member_weight
            )

            total_member_weight += member_weight

        group_value /= total_member_weight

        group_value = np.clip(
            group_value,
            fusion.clip_min,
            fusion.clip_max,
        )

        group_maps[group_index] = (
            group_value.astype(np.float32)
        )

        group_names.append(group.name)

        combined += (
            group_value * group.weight
        )

        total_group_weight += group.weight

    combined /= total_group_weight

    combined = np.clip(
        combined,
        fusion.clip_min,
        fusion.clip_max,
    ).astype(np.float32)

    return (
        group_maps,
        tuple(group_names),
        combined,
    )