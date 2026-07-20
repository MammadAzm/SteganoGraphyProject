from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class ConfigurationError(ValueError):
    """Raised when the project configuration is missing or invalid."""


def _require_mapping(
    value: Any,
    field_name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(
            f"Configuration field '{field_name}' must be a mapping."
        )

    return value


def _require_key(
    mapping: Mapping[str, Any],
    key: str,
    parent_name: str,
) -> Any:
    if key not in mapping:
        raise ConfigurationError(
            f"Missing required configuration field: "
            f"'{parent_name}.{key}'."
        )

    return mapping[key]


def _positive_integer(
    value: Any,
    field_name: str,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigurationError(
            f"Configuration field '{field_name}' must be a positive integer."
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
class ProjectSettings:
    name: str
    root_dir: str
    random_seed: int

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> "ProjectSettings":
        return cls(
            name=str(_require_key(data, "name", "project")),
            root_dir=str(_require_key(data, "root_dir", "project")),
            random_seed=int(
                _require_key(data, "random_seed", "project")
            ),
        )


@dataclass(frozen=True)
class ImageSizeSettings:
    width: int
    height: int

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> "ImageSizeSettings":
        return cls(
            width=_positive_integer(
                _require_key(data, "width", "dataset.image_size"),
                "dataset.image_size.width",
            ),
            height=_positive_integer(
                _require_key(data, "height", "dataset.image_size"),
                "dataset.image_size.height",
            ),
        )


@dataclass(frozen=True)
class DatasetSettings:
    root_dir: str
    splits: dict[str, str]
    image_size: ImageSizeSettings
    channels: int
    color_mode: str
    recursive: bool
    allowed_extensions: tuple[str, ...]

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> "DatasetSettings":
        split_mapping = _require_mapping(
            _require_key(data, "splits", "dataset"),
            "dataset.splits",
        )

        required_splits = ("train", "validation", "test")
        splits: dict[str, str] = {}

        for split_name in required_splits:
            splits[split_name] = str(
                _require_key(
                    split_mapping,
                    split_name,
                    "dataset.splits",
                )
            )

        image_size_mapping = _require_mapping(
            _require_key(data, "image_size", "dataset"),
            "dataset.image_size",
        )

        extensions_value = _require_key(
            data,
            "allowed_extensions",
            "dataset",
        )

        if not isinstance(extensions_value, list) or not extensions_value:
            raise ConfigurationError(
                "'dataset.allowed_extensions' must be a non-empty list."
            )

        normalized_extensions: list[str] = []

        for extension in extensions_value:
            extension_string = str(extension).strip().lower()

            if not extension_string.startswith("."):
                extension_string = f".{extension_string}"

            normalized_extensions.append(extension_string)

        return cls(
            root_dir=str(_require_key(data, "root_dir", "dataset")),
            splits=splits,
            image_size=ImageSizeSettings.from_mapping(
                image_size_mapping
            ),
            channels=_positive_integer(
                _require_key(data, "channels", "dataset"),
                "dataset.channels",
            ),
            color_mode=str(
                _require_key(data, "color_mode", "dataset")
            ).lower(),
            recursive=bool(
                _require_key(data, "recursive", "dataset")
            ),
            allowed_extensions=tuple(normalized_extensions),
        )


@dataclass(frozen=True)
class StrideSettings:
    x: int
    y: int

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> "StrideSettings":
        return cls(
            x=_positive_integer(
                _require_key(data, "x", "sliding_window.stride"),
                "sliding_window.stride.x",
            ),
            y=_positive_integer(
                _require_key(data, "y", "sliding_window.stride"),
                "sliding_window.stride.y",
            ),
        )


@dataclass(frozen=True)
class ExpectedGridSettings:
    width: int
    height: int

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> "ExpectedGridSettings":
        return cls(
            width=_positive_integer(
                _require_key(
                    data,
                    "width",
                    "sliding_window.expected_grid",
                ),
                "sliding_window.expected_grid.width",
            ),
            height=_positive_integer(
                _require_key(
                    data,
                    "height",
                    "sliding_window.expected_grid",
                ),
                "sliding_window.expected_grid.height",
            ),
        )


@dataclass(frozen=True)
class SlidingWindowSettings:
    width: int
    height: int
    stride: StrideSettings
    expected_grid: ExpectedGridSettings

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> "SlidingWindowSettings":
        stride_mapping = _require_mapping(
            _require_key(data, "stride", "sliding_window"),
            "sliding_window.stride",
        )

        expected_grid_mapping = _require_mapping(
            _require_key(data, "expected_grid", "sliding_window"),
            "sliding_window.expected_grid",
        )

        return cls(
            width=_positive_integer(
                _require_key(data, "width", "sliding_window"),
                "sliding_window.width",
            ),
            height=_positive_integer(
                _require_key(data, "height", "sliding_window"),
                "sliding_window.height",
            ),
            stride=StrideSettings.from_mapping(stride_mapping),
            expected_grid=ExpectedGridSettings.from_mapping(
                expected_grid_mapping
            ),
        )


@dataclass(frozen=True)
class FeatureNormalizationSettings:
    method: str
    percentile_low: float
    percentile_high: float
    epsilon: float

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> "FeatureNormalizationSettings":
        return cls(
            method=str(
                _require_key(data, "method", "features.normalization")
            ).lower(),
            percentile_low=float(
                _require_key(
                    data,
                    "percentile_low",
                    "features.normalization",
                )
            ),
            percentile_high=float(
                _require_key(
                    data,
                    "percentile_high",
                    "features.normalization",
                )
            ),
            epsilon=float(
                _require_key(
                    data,
                    "epsilon",
                    "features.normalization",
                )
            ),
        )


@dataclass(frozen=True)
class ScoreFusionSettings:
    method: str
    clip_min: float
    clip_max: float

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> "ScoreFusionSettings":
        return cls(
            method=str(
                _require_key(data, "method", "features.score_fusion")
            ).lower(),
            clip_min=float(
                _require_key(
                    data,
                    "clip_min",
                    "features.score_fusion",
                )
            ),
            clip_max=float(
                _require_key(
                    data,
                    "clip_max",
                    "features.score_fusion",
                )
            ),
        )


@dataclass(frozen=True)
class FeatureItemSettings:
    enabled: bool
    weight: float
    direction: str

    @classmethod
    def from_mapping(
        cls,
        feature_name: str,
        data: Mapping[str, Any],
    ) -> "FeatureItemSettings":
        return cls(
            enabled=bool(
                _require_key(
                    data,
                    "enabled",
                    f"features.items.{feature_name}",
                )
            ),
            weight=_non_negative_float(
                _require_key(
                    data,
                    "weight",
                    f"features.items.{feature_name}",
                ),
                f"features.items.{feature_name}.weight",
            ),
            direction=str(
                _require_key(
                    data,
                    "direction",
                    f"features.items.{feature_name}",
                )
            ).lower(),
        )


@dataclass(frozen=True)
class FeatureSettings:
    normalization: FeatureNormalizationSettings
    score_fusion: ScoreFusionSettings
    items: dict[str, FeatureItemSettings]

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> "FeatureSettings":
        normalization_mapping = _require_mapping(
            _require_key(data, "normalization", "features"),
            "features.normalization",
        )

        score_fusion_mapping = _require_mapping(
            _require_key(data, "score_fusion", "features"),
            "features.score_fusion",
        )

        items_mapping = _require_mapping(
            _require_key(data, "items", "features"),
            "features.items",
        )

        if not items_mapping:
            raise ConfigurationError(
                "'features.items' must contain at least one feature."
            )

        items: dict[str, FeatureItemSettings] = {}

        for feature_name, feature_data in items_mapping.items():
            feature_mapping = _require_mapping(
                feature_data,
                f"features.items.{feature_name}",
            )

            items[str(feature_name)] = FeatureItemSettings.from_mapping(
                str(feature_name),
                feature_mapping,
            )

        return cls(
            normalization=FeatureNormalizationSettings.from_mapping(
                normalization_mapping
            ),
            score_fusion=ScoreFusionSettings.from_mapping(
                score_fusion_mapping
            ),
            items=items,
        )

    @property
    def enabled_items(self) -> dict[str, FeatureItemSettings]:
        return {
            name: item
            for name, item in self.items.items()
            if item.enabled
        }


@dataclass(frozen=True)
class LabelSettings:
    root_dir: str
    file_format: str
    dtype: str
    include_feature_maps: bool
    include_window_coordinates: bool
    include_metadata: bool
    filename_suffix: str

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> "LabelSettings":
        return cls(
            root_dir=str(_require_key(data, "root_dir", "labels")),
            file_format=str(
                _require_key(data, "file_format", "labels")
            ).lower(),
            dtype=str(_require_key(data, "dtype", "labels")).lower(),
            include_feature_maps=bool(
                _require_key(
                    data,
                    "include_feature_maps",
                    "labels",
                )
            ),
            include_window_coordinates=bool(
                _require_key(
                    data,
                    "include_window_coordinates",
                    "labels",
                )
            ),
            include_metadata=bool(
                _require_key(data, "include_metadata", "labels")
            ),
            filename_suffix=str(
                _require_key(data, "filename_suffix", "labels")
            ),
        )


@dataclass(frozen=True)
class ModelSettings:
    architecture: str
    input_channels: int
    output_activation: str
    output_layout: str

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> "ModelSettings":
        return cls(
            architecture=str(
                _require_key(data, "architecture", "model")
            ),
            input_channels=_positive_integer(
                _require_key(data, "input_channels", "model"),
                "model.input_channels",
            ),
            output_activation=str(
                _require_key(data, "output_activation", "model")
            ).lower(),
            output_layout=str(
                _require_key(data, "output_layout", "model")
            ).lower(),
        )


@dataclass(frozen=True)
class EarlyStoppingSettings:
    enabled: bool
    monitor: str
    patience: int
    restore_best_weights: bool

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> "EarlyStoppingSettings":
        return cls(
            enabled=bool(
                _require_key(
                    data,
                    "enabled",
                    "training.early_stopping",
                )
            ),
            monitor=str(
                _require_key(
                    data,
                    "monitor",
                    "training.early_stopping",
                )
            ),
            patience=_positive_integer(
                _require_key(
                    data,
                    "patience",
                    "training.early_stopping",
                ),
                "training.early_stopping.patience",
            ),
            restore_best_weights=bool(
                _require_key(
                    data,
                    "restore_best_weights",
                    "training.early_stopping",
                )
            ),
        )


@dataclass(frozen=True)
class CheckpointSettings:
    enabled: bool
    monitor: str
    save_best_only: bool

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> "CheckpointSettings":
        return cls(
            enabled=bool(
                _require_key(
                    data,
                    "enabled",
                    "training.checkpoint",
                )
            ),
            monitor=str(
                _require_key(
                    data,
                    "monitor",
                    "training.checkpoint",
                )
            ),
            save_best_only=bool(
                _require_key(
                    data,
                    "save_best_only",
                    "training.checkpoint",
                )
            ),
        )


@dataclass(frozen=True)
class TrainingSettings:
    batch_size: int
    epochs: int
    learning_rate: float
    loss: str
    early_stopping: EarlyStoppingSettings
    checkpoint: CheckpointSettings

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> "TrainingSettings":
        early_stopping_mapping = _require_mapping(
            _require_key(data, "early_stopping", "training"),
            "training.early_stopping",
        )

        checkpoint_mapping = _require_mapping(
            _require_key(data, "checkpoint", "training"),
            "training.checkpoint",
        )

        learning_rate = float(
            _require_key(data, "learning_rate", "training")
        )

        if learning_rate <= 0:
            raise ConfigurationError(
                "'training.learning_rate' must be greater than zero."
            )

        return cls(
            batch_size=_positive_integer(
                _require_key(data, "batch_size", "training"),
                "training.batch_size",
            ),
            epochs=_positive_integer(
                _require_key(data, "epochs", "training"),
                "training.epochs",
            ),
            learning_rate=learning_rate,
            loss=str(_require_key(data, "loss", "training")).lower(),
            early_stopping=EarlyStoppingSettings.from_mapping(
                early_stopping_mapping
            ),
            checkpoint=CheckpointSettings.from_mapping(
                checkpoint_mapping
            ),
        )


@dataclass(frozen=True)
class ThresholdConstraintSettings:
    minimum_secure_precision: float
    minimum_secure_recall: float

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> "ThresholdConstraintSettings":
        return cls(
            minimum_secure_precision=float(
                _require_key(
                    data,
                    "minimum_secure_precision",
                    "threshold_selection.constraints",
                )
            ),
            minimum_secure_recall=float(
                _require_key(
                    data,
                    "minimum_secure_recall",
                    "threshold_selection.constraints",
                )
            ),
        )


@dataclass(frozen=True)
class ThresholdSelectionSettings:
    start: float
    end: float
    step: float
    reference_threshold: float
    selection_metric: str
    constraints: ThresholdConstraintSettings

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> "ThresholdSelectionSettings":
        constraints_mapping = _require_mapping(
            _require_key(
                data,
                "constraints",
                "threshold_selection",
            ),
            "threshold_selection.constraints",
        )

        return cls(
            start=float(
                _require_key(data, "start", "threshold_selection")
            ),
            end=float(
                _require_key(data, "end", "threshold_selection")
            ),
            step=float(
                _require_key(data, "step", "threshold_selection")
            ),
            reference_threshold=float(
                _require_key(
                    data,
                    "reference_threshold",
                    "threshold_selection",
                )
            ),
            selection_metric=str(
                _require_key(
                    data,
                    "selection_metric",
                    "threshold_selection",
                )
            ).lower(),
            constraints=ThresholdConstraintSettings.from_mapping(
                constraints_mapping
            ),
        )

    def candidates(self) -> tuple[float, ...]:
        count = round((self.end - self.start) / self.step)

        return tuple(
            round(self.start + index * self.step, 10)
            for index in range(count + 1)
        )


@dataclass(frozen=True)
class OutputSettings:
    root_dir: str
    directories: dict[str, str]

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> "OutputSettings":
        directories_mapping = _require_mapping(
            _require_key(data, "directories", "outputs"),
            "outputs.directories",
        )

        required_directories = (
            "feature_statistics",
            "models",
            "training_reports",
            "validation_reports",
            "test_reports",
            "heatmaps",
        )

        directories: dict[str, str] = {}

        for directory_name in required_directories:
            directories[directory_name] = str(
                _require_key(
                    directories_mapping,
                    directory_name,
                    "outputs.directories",
                )
            )

        return cls(
            root_dir=str(_require_key(data, "root_dir", "outputs")),
            directories=directories,
        )


@dataclass(frozen=True)
class RuntimeSettings:
    num_workers: int
    overwrite_existing: bool
    fail_fast: bool
    deterministic: bool

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> "RuntimeSettings":
        num_workers = int(
            _require_key(data, "num_workers", "runtime")
        )

        if num_workers < 0:
            raise ConfigurationError(
                "'runtime.num_workers' cannot be negative."
            )

        return cls(
            num_workers=num_workers,
            overwrite_existing=bool(
                _require_key(
                    data,
                    "overwrite_existing",
                    "runtime",
                )
            ),
            fail_fast=bool(
                _require_key(data, "fail_fast", "runtime")
            ),
            deterministic=bool(
                _require_key(data, "deterministic", "runtime")
            ),
        )


@dataclass(frozen=True)
class ProjectConfiguration:
    project: ProjectSettings
    dataset: DatasetSettings
    sliding_window: SlidingWindowSettings
    features: FeatureSettings
    labels: LabelSettings
    model: ModelSettings
    training: TrainingSettings
    threshold_selection: ThresholdSelectionSettings
    outputs: OutputSettings
    runtime: RuntimeSettings

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> "ProjectConfiguration":
        required_sections = (
            "project",
            "dataset",
            "sliding_window",
            "features",
            "labels",
            "model",
            "training",
            "threshold_selection",
            "outputs",
            "runtime",
        )

        sections: dict[str, Mapping[str, Any]] = {}

        for section_name in required_sections:
            sections[section_name] = _require_mapping(
                _require_key(data, section_name, "configuration"),
                section_name,
            )

        configuration = cls(
            project=ProjectSettings.from_mapping(
                sections["project"]
            ),
            dataset=DatasetSettings.from_mapping(
                sections["dataset"]
            ),
            sliding_window=SlidingWindowSettings.from_mapping(
                sections["sliding_window"]
            ),
            features=FeatureSettings.from_mapping(
                sections["features"]
            ),
            labels=LabelSettings.from_mapping(
                sections["labels"]
            ),
            model=ModelSettings.from_mapping(
                sections["model"]
            ),
            training=TrainingSettings.from_mapping(
                sections["training"]
            ),
            threshold_selection=(
                ThresholdSelectionSettings.from_mapping(
                    sections["threshold_selection"]
                )
            ),
            outputs=OutputSettings.from_mapping(
                sections["outputs"]
            ),
            runtime=RuntimeSettings.from_mapping(
                sections["runtime"]
            ),
        )

        configuration.validate()
        return configuration

    @property
    def grid_width(self) -> int:
        return (
            (
                self.dataset.image_size.width
                - self.sliding_window.width
            )
            // self.sliding_window.stride.x
        ) + 1

    @property
    def grid_height(self) -> int:
        return (
            (
                self.dataset.image_size.height
                - self.sliding_window.height
            )
            // self.sliding_window.stride.y
        ) + 1

    @property
    def windows_per_image(self) -> int:
        return self.grid_width * self.grid_height

    @property
    def output_shape(self) -> tuple[int, int]:
        return self.grid_height, self.grid_width

    def validate(self) -> None:
        errors: list[str] = []

        image_width = self.dataset.image_size.width
        image_height = self.dataset.image_size.height

        window_width = self.sliding_window.width
        window_height = self.sliding_window.height

        stride_x = self.sliding_window.stride.x
        stride_y = self.sliding_window.stride.y

        if window_width > image_width:
            errors.append(
                "Sliding-window width cannot exceed image width."
            )

        if window_height > image_height:
            errors.append(
                "Sliding-window height cannot exceed image height."
            )

        if (image_width - window_width) % stride_x != 0:
            errors.append(
                "The horizontal stride does not exactly cover the image. "
                "The final window would not align with the image boundary."
            )

        if (image_height - window_height) % stride_y != 0:
            errors.append(
                "The vertical stride does not exactly cover the image. "
                "The final window would not align with the image boundary."
            )

        expected_width = self.sliding_window.expected_grid.width
        expected_height = self.sliding_window.expected_grid.height

        if self.grid_width != expected_width:
            errors.append(
                "Configured grid width does not match the calculated grid "
                f"width. Expected {expected_width}, calculated "
                f"{self.grid_width}."
            )

        if self.grid_height != expected_height:
            errors.append(
                "Configured grid height does not match the calculated grid "
                f"height. Expected {expected_height}, calculated "
                f"{self.grid_height}."
            )

        if self.dataset.channels != self.model.input_channels:
            errors.append(
                "'dataset.channels' must match 'model.input_channels'."
            )

        if self.dataset.color_mode not in {"rgb", "grayscale"}:
            errors.append(
                "'dataset.color_mode' must be 'rgb' or 'grayscale'."
            )

        valid_normalization_methods = {
            "robust_minmax",
            "minmax",
            "zscore",
            "percentile_rank",
        }

        if (
            self.features.normalization.method
            not in valid_normalization_methods
        ):
            errors.append(
                "Unsupported feature-normalization method: "
                f"{self.features.normalization.method}."
            )

        percentile_low = (
            self.features.normalization.percentile_low
        )
        percentile_high = (
            self.features.normalization.percentile_high
        )

        if not 0 <= percentile_low < percentile_high <= 100:
            errors.append(
                "Feature-normalization percentiles must satisfy "
                "0 <= percentile_low < percentile_high <= 100."
            )

        if self.features.normalization.epsilon <= 0:
            errors.append(
                "Feature-normalization epsilon must be greater than zero."
            )

        if (
            self.features.score_fusion.clip_min
            >= self.features.score_fusion.clip_max
        ):
            errors.append(
                "Feature score clip_min must be smaller than clip_max."
            )

        valid_directions = {"higher", "lower"}

        for feature_name, feature_item in self.features.items.items():
            if feature_item.direction not in valid_directions:
                errors.append(
                    f"Feature '{feature_name}' has invalid direction "
                    f"'{feature_item.direction}'."
                )

            if feature_item.enabled and feature_item.weight <= 0:
                errors.append(
                    f"Enabled feature '{feature_name}' must have a "
                    "weight greater than zero."
                )

        if not self.features.enabled_items:
            errors.append(
                "At least one feature must be enabled."
            )

        if self.labels.file_format not in {
            "npz",
            "hdf5",
            "zarr",
        }:
            errors.append(
                "Unsupported label file format. Use npz, hdf5, or zarr."
            )

        if self.labels.dtype not in {
            "float16",
            "float32",
            "float64",
        }:
            errors.append(
                "Unsupported label dtype."
            )

        if self.model.output_activation != "sigmoid":
            errors.append(
                "The initial scoring model must use sigmoid output "
                "activation so scores remain between zero and one."
            )

        threshold = self.threshold_selection

        if not 0 <= threshold.start <= 1:
            errors.append(
                "Threshold start must be between zero and one."
            )

        if not 0 <= threshold.end <= 1:
            errors.append(
                "Threshold end must be between zero and one."
            )

        if threshold.start > threshold.end:
            errors.append(
                "Threshold start cannot exceed threshold end."
            )

        if threshold.step <= 0:
            errors.append(
                "Threshold step must be greater than zero."
            )

        if not 0 <= threshold.reference_threshold <= 1:
            errors.append(
                "Reference threshold must be between zero and one."
            )

        if threshold.selection_metric not in {
            "f1",
            "precision",
            "recall",
            "accuracy",
            "balanced_accuracy",
        }:
            errors.append(
                "Unsupported threshold-selection metric."
            )

        for constraint_name, constraint_value in {
            "minimum_secure_precision": (
                threshold.constraints.minimum_secure_precision
            ),
            "minimum_secure_recall": (
                threshold.constraints.minimum_secure_recall
            ),
        }.items():
            if not 0 <= constraint_value <= 1:
                errors.append(
                    f"Threshold constraint '{constraint_name}' must be "
                    "between zero and one."
                )

        expected_thresholds = tuple(
            round(0.50 + index * 0.01, 2)
            for index in range(17)
        )

        actual_thresholds = tuple(
            round(value, 2)
            for value in threshold.candidates()
        )

        if actual_thresholds != expected_thresholds:
            errors.append(
                "The initial threshold scan must contain exactly "
                "0.50, 0.51, ..., 0.66."
            )

        if errors:
            formatted_errors = "\n".join(
                f"- {error}" for error in errors
            )

            raise ConfigurationError(
                "Invalid project configuration:\n"
                f"{formatted_errors}"
            )