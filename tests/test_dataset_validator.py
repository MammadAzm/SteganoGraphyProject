from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from PIL import Image

from src.config.loader import load_configuration
from src.data.dataset_validator import DatasetValidator


def create_rgb_image(
    path: Path,
    *,
    size: tuple[int, int] = (256, 256),
    color: tuple[int, int, int] = (20, 40, 60),
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = Image.new(
        "RGB",
        size,
        color=color,
    )

    image.save(path)


def create_configuration(
    temporary_directory: Path,
    *,
    expected_train: int = 1,
    expected_validation: int = 1,
    expected_test: int = 1,
) -> Path:
    project_root = temporary_directory / "project"
    configuration_directory = (
        project_root / "configs"
    )

    dataset_root = (
        temporary_directory / "dataset"
    )

    configuration_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    configuration = {
        "project": {
            "name": "test_project",
            "root_dir": "..",
            "random_seed": 42,
        },
        "dataset": {
            "root_dir": str(dataset_root),
            "splits": {
                "train": "train",
                "validation": "validation",
                "test": "test",
            },
            "image_size": {
                "width": 256,
                "height": 256,
            },
            "channels": 3,
            "color_mode": "rgb",
            "recursive": True,
            "allowed_extensions": [
                ".png",
                ".jpg",
                ".jpeg",
            ],
        },
        "sliding_window": {
            "width": 64,
            "height": 64,
            "stride": {
                "x": 8,
                "y": 8,
            },
            "expected_grid": {
                "width": 25,
                "height": 25,
            },
        },
        "features": {
            "normalization": {
                "method": "robust_minmax",
                "percentile_low": 1.0,
                "percentile_high": 99.0,
                "epsilon": 1e-8,
            },
            "score_fusion": {
                "method": "weighted_mean",
                "clip_min": 0.0,
                "clip_max": 1.0,
            },
            "items": {
                "entropy": {
                    "enabled": True,
                    "weight": 1.0,
                    "direction": "higher",
                }
            },
        },
        "labels": {
            "root_dir": "data/labels",
            "file_format": "npz",
            "dtype": "float32",
            "include_feature_maps": True,
            "include_window_coordinates": True,
            "include_metadata": True,
            "filename_suffix": "_labels.npz",
        },
        "model": {
            "architecture": "full_image_cnn_v1",
            "input_channels": 3,
            "output_activation": "sigmoid",
            "output_layout": "channels_last",
        },
        "training": {
            "batch_size": 4,
            "epochs": 2,
            "learning_rate": 0.001,
            "loss": "huber",
            "early_stopping": {
                "enabled": True,
                "monitor": "val_loss",
                "patience": 2,
                "restore_best_weights": True,
            },
            "checkpoint": {
                "enabled": True,
                "monitor": "val_loss",
                "save_best_only": True,
            },
        },
        "threshold_selection": {
            "start": 0.50,
            "end": 0.66,
            "step": 0.01,
            "reference_threshold": 0.50,
            "selection_metric": "f1",
            "constraints": {
                "minimum_secure_precision": 0.0,
                "minimum_secure_recall": 0.0,
            },
        },
        "outputs": {
            "root_dir": "outputs",
            "directories": {
                "feature_statistics": (
                    "feature_statistics"
                ),
                "models": "models",
                "training_reports": (
                    "training_reports"
                ),
                "validation_reports": (
                    "validation_reports"
                ),
                "test_reports": "test_reports",
                "heatmaps": "heatmaps",
            },
        },
        "runtime": {
            "num_workers": 1,
            "overwrite_existing": False,
            "fail_fast": True,
            "deterministic": True,
        },
        "dataset_validation": {
            "report_dir": "outputs/dataset_reports",
            "expected_counts": {
                "train": expected_train,
                "validation": expected_validation,
                "test": expected_test,
            },
            "checks": {
                "dimensions": True,
                "color_mode": True,
                "channels": True,
                "full_decode": True,
            },
            "duplicates": {
                "check_filenames": True,
                "check_image_ids": True,
                "check_content": True,
                "content_hash": "pixels",
                "within_split_severity": "warning",
                "cross_split_severity": "error",
            },
            "unsupported_files": {
                "severity": "ignore",
            },
            "reports": {
                "save_manifest": True,
                "save_issues": True,
                "save_summary": True,
                "overwrite": True,
            },
            "execution": {
                "max_workers": 1,
                "progress_interval": 0,
            },
            "fail_on_error": True,
        },
    }

    configuration_path = (
        configuration_directory / "test.yaml"
    )

    with configuration_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        yaml.safe_dump(
            configuration,
            output_file,
            sort_keys=False,
        )

    return configuration_path


def prepare_split_directories(
    temporary_directory: Path,
) -> Path:
    dataset_root = temporary_directory / "dataset"

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        (
            dataset_root / split_name
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    return dataset_root


def test_valid_dataset_passes(
    tmp_path: Path,
) -> None:
    dataset_root = prepare_split_directories(
        tmp_path
    )

    create_rgb_image(
        dataset_root / "train" / "train.png",
        color=(10, 20, 30),
    )

    create_rgb_image(
        (
            dataset_root
            / "validation"
            / "validation.png"
        ),
        color=(40, 50, 60),
    )

    create_rgb_image(
        dataset_root / "test" / "test.png",
        color=(70, 80, 90),
    )

    configuration_path = create_configuration(
        tmp_path
    )

    loaded = load_configuration(
        configuration_path
    )

    result = DatasetValidator(loaded).run()

    assert result.summary.valid is True
    assert result.summary.error_count == 0
    assert result.summary.total_files == 3
    assert result.summary.readable_images == 3
    assert (
        result.summary
        .sliding_window["windows_per_image"]
        == 625
    )


def test_wrong_image_dimensions_fail(
    tmp_path: Path,
) -> None:
    dataset_root = prepare_split_directories(
        tmp_path
    )

    create_rgb_image(
        dataset_root / "train" / "train.png",
        size=(128, 128),
        color=(10, 20, 30),
    )

    create_rgb_image(
        (
            dataset_root
            / "validation"
            / "validation.png"
        ),
        color=(40, 50, 60),
    )

    create_rgb_image(
        dataset_root / "test" / "test.png",
        color=(70, 80, 90),
    )

    configuration_path = create_configuration(
        tmp_path
    )

    loaded = load_configuration(
        configuration_path
    )

    result = DatasetValidator(loaded).run()

    assert result.summary.valid is False

    issue_codes = {
        issue.code
        for issue in result.issues
    }

    assert (
        "INVALID_IMAGE_DIMENSIONS"
        in issue_codes
    )


def test_cross_split_duplicate_content_fails(
    tmp_path: Path,
) -> None:
    dataset_root = prepare_split_directories(
        tmp_path
    )

    duplicate_color = (100, 110, 120)

    create_rgb_image(
        dataset_root / "train" / "image_a.png",
        color=duplicate_color,
    )

    create_rgb_image(
        (
            dataset_root
            / "validation"
            / "image_b.png"
        ),
        color=duplicate_color,
    )

    create_rgb_image(
        dataset_root / "test" / "image_c.png",
        color=(1, 2, 3),
    )

    configuration_path = create_configuration(
        tmp_path
    )

    loaded = load_configuration(
        configuration_path
    )

    result = DatasetValidator(loaded).run()

    assert result.summary.valid is False

    issue_codes = {
        issue.code
        for issue in result.issues
    }

    assert (
        "DUPLICATE_CONTENT_ACROSS_SPLITS"
        in issue_codes
    )


def test_unreadable_image_fails(
    tmp_path: Path,
) -> None:
    dataset_root = prepare_split_directories(
        tmp_path
    )

    invalid_image = (
        dataset_root / "train" / "broken.jpg"
    )

    invalid_image.write_bytes(
        b"this is not a valid image"
    )

    create_rgb_image(
        (
            dataset_root
            / "validation"
            / "validation.png"
        ),
        color=(40, 50, 60),
    )

    create_rgb_image(
        dataset_root / "test" / "test.png",
        color=(70, 80, 90),
    )

    configuration_path = create_configuration(
        tmp_path
    )

    loaded = load_configuration(
        configuration_path
    )

    result = DatasetValidator(loaded).run()

    assert result.summary.valid is False
    assert result.summary.unreadable_images == 1

    issue_codes = {
        issue.code
        for issue in result.issues
    }

    assert "UNREADABLE_IMAGE" in issue_codes


def test_count_mismatch_fails(
    tmp_path: Path,
) -> None:
    dataset_root = prepare_split_directories(
        tmp_path
    )

    create_rgb_image(
        dataset_root / "train" / "train.png",
        color=(10, 20, 30),
    )

    create_rgb_image(
        (
            dataset_root
            / "validation"
            / "validation.png"
        ),
        color=(40, 50, 60),
    )

    create_rgb_image(
        dataset_root / "test" / "test.png",
        color=(70, 80, 90),
    )

    configuration_path = create_configuration(
        tmp_path,
        expected_train=2,
    )

    loaded = load_configuration(
        configuration_path
    )

    result = DatasetValidator(loaded).run()

    assert result.summary.valid is False

    issue_codes = {
        issue.code
        for issue in result.issues
    }

    assert "IMAGE_COUNT_MISMATCH" in issue_codes