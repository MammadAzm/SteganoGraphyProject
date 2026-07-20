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


def _non_negative_integer(
    value: Any,
    field_name: str,
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        raise ConfigurationError(
            f"Configuration field '{field_name}' must be a "
            "non-negative integer."
        )

    return value


@dataclass(frozen=True)
class LabelNormalizationSettings:
    fit_split: str
    fit_mode: str
    artifact_path: Path
    feature_matrix_path: Path
    raw_cache_dir: Path
    cache_training_features: bool
    reuse_existing_artifact: bool
    delete_feature_matrix_after_fit: bool
    delete_raw_cache_after_success: bool


@dataclass(frozen=True)
class LabelStorageSettings:
    compressed: bool
    save_raw_feature_maps: bool
    save_normalized_feature_maps: bool
    save_group_score_maps: bool
    save_window_coordinates: bool
    save_metadata: bool


@dataclass(frozen=True)
class LabelExecutionSettings:
    max_workers: int
    progress_interval: int
    resume: bool
    overwrite_existing: bool


@dataclass(frozen=True)
class LabelIntegritySettings:
    verify_after_write: bool
    fail_on_non_finite: bool
    score_tolerance: float


@dataclass(frozen=True)
class LabelReportSettings:
    manifest_filename: str
    summary_filename: str
    global_summary_path: Path


@dataclass(frozen=True)
class LabelGenerationSettings:
    output_root: Path
    normalization: LabelNormalizationSettings
    storage: LabelStorageSettings
    execution: LabelExecutionSettings
    integrity: LabelIntegritySettings
    reports: LabelReportSettings

    @classmethod
    def from_loaded_configuration(
        cls,
        loaded: LoadedConfiguration,
    ) -> "LabelGenerationSettings":
        root = _mapping(
            loaded.raw.get("label_generation"),
            "label_generation",
        )

        normalization = _mapping(
            root.get("normalization"),
            "label_generation.normalization",
        )

        storage = _mapping(
            root.get("storage"),
            "label_generation.storage",
        )

        execution = _mapping(
            root.get("execution"),
            "label_generation.execution",
        )

        integrity = _mapping(
            root.get("integrity"),
            "label_generation.integrity",
        )

        reports = _mapping(
            root.get("reports"),
            "label_generation.reports",
        )

        project_root = loaded.paths.project_root

        settings = cls(
            output_root=resolve_path(
                str(
                    root.get(
                        "output_root",
                        loaded.settings.labels.root_dir,
                    )
                ),
                project_root,
            ),
            normalization=LabelNormalizationSettings(
                fit_split=str(
                    normalization.get(
                        "fit_split",
                        "train",
                    )
                ).lower(),
                fit_mode=str(
                    normalization.get(
                        "fit_mode",
                        "exact",
                    )
                ).lower(),
                artifact_path=resolve_path(
                    str(
                        normalization.get(
                            "artifact_path",
                            (
                                "outputs/label_generation/"
                                "feature_normalization.json"
                            ),
                        )
                    ),
                    project_root,
                ),
                feature_matrix_path=resolve_path(
                    str(
                        normalization.get(
                            "feature_matrix_path",
                            (
                                "outputs/label_generation/"
                                "train_feature_matrix.float32.dat"
                            ),
                        )
                    ),
                    project_root,
                ),
                raw_cache_dir=resolve_path(
                    str(
                        normalization.get(
                            "raw_cache_dir",
                            (
                                "outputs/label_generation/"
                                "raw_train_features"
                            ),
                        )
                    ),
                    project_root,
                ),
                cache_training_features=_boolean(
                    normalization.get(
                        "cache_training_features",
                        True,
                    ),
                    (
                        "label_generation.normalization."
                        "cache_training_features"
                    ),
                ),
                reuse_existing_artifact=_boolean(
                    normalization.get(
                        "reuse_existing_artifact",
                        True,
                    ),
                    (
                        "label_generation.normalization."
                        "reuse_existing_artifact"
                    ),
                ),
                delete_feature_matrix_after_fit=_boolean(
                    normalization.get(
                        "delete_feature_matrix_after_fit",
                        True,
                    ),
                    (
                        "label_generation.normalization."
                        "delete_feature_matrix_after_fit"
                    ),
                ),
                delete_raw_cache_after_success=_boolean(
                    normalization.get(
                        "delete_raw_cache_after_success",
                        False,
                    ),
                    (
                        "label_generation.normalization."
                        "delete_raw_cache_after_success"
                    ),
                ),
            ),
            storage=LabelStorageSettings(
                compressed=_boolean(
                    storage.get(
                        "compressed",
                        True,
                    ),
                    "label_generation.storage.compressed",
                ),
                save_raw_feature_maps=_boolean(
                    storage.get(
                        "save_raw_feature_maps",
                        True,
                    ),
                    (
                        "label_generation.storage."
                        "save_raw_feature_maps"
                    ),
                ),
                save_normalized_feature_maps=_boolean(
                    storage.get(
                        "save_normalized_feature_maps",
                        True,
                    ),
                    (
                        "label_generation.storage."
                        "save_normalized_feature_maps"
                    ),
                ),
                save_group_score_maps=_boolean(
                    storage.get(
                        "save_group_score_maps",
                        True,
                    ),
                    (
                        "label_generation.storage."
                        "save_group_score_maps"
                    ),
                ),
                save_window_coordinates=_boolean(
                    storage.get(
                        "save_window_coordinates",
                        True,
                    ),
                    (
                        "label_generation.storage."
                        "save_window_coordinates"
                    ),
                ),
                save_metadata=_boolean(
                    storage.get(
                        "save_metadata",
                        True,
                    ),
                    (
                        "label_generation.storage."
                        "save_metadata"
                    ),
                ),
            ),
            execution=LabelExecutionSettings(
                max_workers=_non_negative_integer(
                    execution.get(
                        "max_workers",
                        loaded.settings.runtime.num_workers,
                    ),
                    "label_generation.execution.max_workers",
                ),
                progress_interval=_non_negative_integer(
                    execution.get(
                        "progress_interval",
                        50,
                    ),
                    (
                        "label_generation.execution."
                        "progress_interval"
                    ),
                ),
                resume=_boolean(
                    execution.get(
                        "resume",
                        True,
                    ),
                    "label_generation.execution.resume",
                ),
                overwrite_existing=_boolean(
                    execution.get(
                        "overwrite_existing",
                        loaded.settings.runtime.overwrite_existing,
                    ),
                    (
                        "label_generation.execution."
                        "overwrite_existing"
                    ),
                ),
            ),
            integrity=LabelIntegritySettings(
                verify_after_write=_boolean(
                    integrity.get(
                        "verify_after_write",
                        True,
                    ),
                    (
                        "label_generation.integrity."
                        "verify_after_write"
                    ),
                ),
                fail_on_non_finite=_boolean(
                    integrity.get(
                        "fail_on_non_finite",
                        True,
                    ),
                    (
                        "label_generation.integrity."
                        "fail_on_non_finite"
                    ),
                ),
                score_tolerance=float(
                    integrity.get(
                        "score_tolerance",
                        1e-6,
                    )
                ),
            ),
            reports=LabelReportSettings(
                manifest_filename=str(
                    reports.get(
                        "manifest_filename",
                        "manifest.csv",
                    )
                ),
                summary_filename=str(
                    reports.get(
                        "summary_filename",
                        "summary.json",
                    )
                ),
                global_summary_path=resolve_path(
                    str(
                        reports.get(
                            "global_summary_path",
                            (
                                "outputs/label_generation/"
                                "label_generation_summary.json"
                            ),
                        )
                    ),
                    project_root,
                ),
            ),
        )

        settings.validate()
        return settings

    def validate(self) -> None:
        errors: list[str] = []

        if self.normalization.fit_split != "train":
            errors.append(
                "Normalization must be fitted using the training split."
            )

        if self.normalization.fit_mode != "exact":
            errors.append(
                "Stage 3 currently supports only exact normalization fitting."
            )

        if self.integrity.score_tolerance < 0:
            errors.append(
                "Score tolerance cannot be negative."
            )

        if (
            not self.storage.save_raw_feature_maps
            and not self.storage.save_normalized_feature_maps
            and not self.storage.save_group_score_maps
        ):
            # The score map itself is always saved.
            pass

        if errors:
            raise ConfigurationError(
                "Invalid label-generation configuration:\n"
                + "\n".join(
                    f"- {error}"
                    for error in errors
                )
            )