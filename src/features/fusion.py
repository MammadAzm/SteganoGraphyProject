from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from src.config.loader import LoadedConfiguration
from src.config.schema import (
    ConfigurationError,
    FeatureItemSettings,
)


@dataclass(frozen=True)
class FeatureNormalizationBounds:
    feature_name: str
    lower: float
    upper: float

    @property
    def span(self) -> float:
        return self.upper - self.lower


@dataclass(frozen=True)
class FeatureGroupSettings:
    name: str
    weight: float
    members: dict[str, float]


@dataclass(frozen=True)
class GroupedFusionSettings:
    method: str
    clip_min: float
    clip_max: float
    groups: tuple[FeatureGroupSettings, ...]

    @classmethod
    def from_loaded_configuration(
        cls,
        loaded: LoadedConfiguration,
    ) -> "GroupedFusionSettings":
        features_section = loaded.raw.get(
            "features",
            {},
        )

        if not isinstance(features_section, Mapping):
            raise ConfigurationError(
                "'features' must be a mapping."
            )

        fusion_section = features_section.get(
            "score_fusion",
            {},
        )

        if not isinstance(fusion_section, Mapping):
            raise ConfigurationError(
                "'features.score_fusion' must be a mapping."
            )

        method = str(
            fusion_section.get(
                "method",
                "grouped_weighted_mean",
            )
        ).lower()

        if method != "grouped_weighted_mean":
            raise ConfigurationError(
                "Stage 2 requires score-fusion method "
                "'grouped_weighted_mean'."
            )

        clip_min = float(
            fusion_section.get(
                "clip_min",
                0.0,
            )
        )

        clip_max = float(
            fusion_section.get(
                "clip_max",
                1.0,
            )
        )

        if clip_min >= clip_max:
            raise ConfigurationError(
                "Feature-fusion clip_min must be smaller than clip_max."
            )

        groups_value = fusion_section.get(
            "groups",
        )

        if not isinstance(groups_value, Mapping):
            raise ConfigurationError(
                "'features.score_fusion.groups' must be a mapping."
            )

        groups: list[FeatureGroupSettings] = []

        for group_name, group_value in groups_value.items():
            if not isinstance(group_value, Mapping):
                raise ConfigurationError(
                    f"Feature group '{group_name}' must be a mapping."
                )

            group_weight = float(
                group_value.get(
                    "weight",
                    1.0,
                )
            )

            if group_weight <= 0:
                raise ConfigurationError(
                    f"Feature group '{group_name}' must have a "
                    "positive weight."
                )

            members_value = group_value.get(
                "members",
            )

            if not isinstance(members_value, Mapping):
                raise ConfigurationError(
                    f"Feature group '{group_name}.members' must be "
                    "a mapping."
                )

            members: dict[str, float] = {}

            for feature_name, member_weight in members_value.items():
                parsed_weight = float(
                    member_weight
                )

                if parsed_weight <= 0:
                    raise ConfigurationError(
                        f"Feature '{feature_name}' in group "
                        f"'{group_name}' must have a positive weight."
                    )

                members[str(feature_name)] = parsed_weight

            if not members:
                raise ConfigurationError(
                    f"Feature group '{group_name}' cannot be empty."
                )

            groups.append(
                FeatureGroupSettings(
                    name=str(group_name),
                    weight=group_weight,
                    members=members,
                )
            )

        settings = cls(
            method=method,
            clip_min=clip_min,
            clip_max=clip_max,
            groups=tuple(groups),
        )

        settings.validate(
            loaded.settings.features.items
        )

        return settings

    def validate(
        self,
        feature_items: Mapping[
            str,
            FeatureItemSettings,
        ],
    ) -> None:
        enabled_features = {
            name
            for name, item in feature_items.items()
            if item.enabled
        }

        grouped_features: list[str] = []

        for group in self.groups:
            grouped_features.extend(
                group.members.keys()
            )

        grouped_feature_set = set(
            grouped_features
        )

        unknown = (
            grouped_feature_set
            - set(feature_items)
        )

        if unknown:
            raise ConfigurationError(
                "Feature groups reference unknown features: "
                + ", ".join(sorted(unknown))
            )

        disabled_group_members = {
            feature_name
            for feature_name in grouped_feature_set
            if not feature_items[feature_name].enabled
        }

        if disabled_group_members:
            raise ConfigurationError(
                "Feature groups contain disabled features: "
                + ", ".join(
                    sorted(disabled_group_members)
                )
            )

        missing_enabled = (
            enabled_features
            - grouped_feature_set
        )

        if missing_enabled:
            raise ConfigurationError(
                "Enabled features missing from score-fusion groups: "
                + ", ".join(
                    sorted(missing_enabled)
                )
            )

        duplicates = {
            feature_name
            for feature_name in grouped_features
            if grouped_features.count(feature_name) > 1
        }

        if duplicates:
            raise ConfigurationError(
                "Features cannot belong to more than one fusion group: "
                + ", ".join(sorted(duplicates))
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
            "groups": [
                asdict(group)
                for group in self.groups
            ],
        }


def fit_robust_normalization_bounds(
    rows: Sequence[Mapping[str, float]],
    feature_names: Sequence[str],
    *,
    percentile_low: float,
    percentile_high: float,
    epsilon: float,
) -> dict[str, FeatureNormalizationBounds]:
    if not rows:
        raise ValueError(
            "Cannot fit feature normalization without samples."
        )

    if not 0 <= percentile_low < percentile_high <= 100:
        raise ValueError(
            "Normalization percentiles must satisfy "
            "0 <= low < high <= 100."
        )

    if epsilon <= 0:
        raise ValueError(
            "Normalization epsilon must be greater than zero."
        )

    bounds: dict[str, FeatureNormalizationBounds] = {}

    for feature_name in feature_names:
        values = np.asarray(
            [
                float(row[feature_name])
                for row in rows
            ],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Feature '{feature_name}' contains non-finite values."
            )

        lower = float(
            np.percentile(
                values,
                percentile_low,
            )
        )

        upper = float(
            np.percentile(
                values,
                percentile_high,
            )
        )

        if upper - lower < epsilon:
            upper = lower + epsilon

        bounds[feature_name] = (
            FeatureNormalizationBounds(
                feature_name=feature_name,
                lower=lower,
                upper=upper,
            )
        )

    return bounds


def normalize_feature_values(
    values: Mapping[str, float],
    bounds: Mapping[
        str,
        FeatureNormalizationBounds,
    ],
    feature_items: Mapping[
        str,
        FeatureItemSettings,
    ],
) -> dict[str, float]:
    output: dict[str, float] = {}

    for feature_name, normalization in bounds.items():
        raw_value = float(
            values[feature_name]
        )

        normalized = (
            raw_value - normalization.lower
        ) / normalization.span

        normalized = float(
            np.clip(
                normalized,
                0.0,
                1.0,
            )
        )

        direction = feature_items[
            feature_name
        ].direction

        if direction == "lower":
            normalized = 1.0 - normalized
        elif direction != "higher":
            raise ValueError(
                f"Unsupported direction '{direction}' "
                f"for feature '{feature_name}'."
            )

        output[feature_name] = normalized

    return output


def fuse_normalized_features(
    normalized_values: Mapping[str, float],
    fusion_settings: GroupedFusionSettings,
) -> tuple[dict[str, float], float]:
    group_scores: dict[str, float] = {}

    weighted_group_sum = 0.0
    total_group_weight = 0.0

    for group in fusion_settings.groups:
        member_sum = 0.0
        total_member_weight = 0.0

        for feature_name, member_weight in group.members.items():
            if feature_name not in normalized_values:
                raise KeyError(
                    f"Normalized feature '{feature_name}' is missing."
                )

            member_sum += (
                normalized_values[feature_name]
                * member_weight
            )

            total_member_weight += member_weight

        group_score = (
            member_sum / total_member_weight
        )

        group_score = float(
            np.clip(
                group_score,
                fusion_settings.clip_min,
                fusion_settings.clip_max,
            )
        )

        group_scores[group.name] = group_score

        weighted_group_sum += (
            group_score * group.weight
        )

        total_group_weight += group.weight

    if total_group_weight <= 0:
        raise ValueError(
            "Total feature-group weight must be positive."
        )

    combined_score = (
        weighted_group_sum / total_group_weight
    )

    combined_score = float(
        np.clip(
            combined_score,
            fusion_settings.clip_min,
            fusion_settings.clip_max,
        )
    )

    return group_scores, combined_score