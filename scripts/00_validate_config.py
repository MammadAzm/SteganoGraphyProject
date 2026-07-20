from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config.loader import load_configuration
from src.config.schema import ConfigurationError
from src.utils.reproducibility import set_global_seed


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the project configuration and display the "
            "resolved project settings."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "default.yaml",
        help="Path to the YAML configuration file.",
    )

    parser.add_argument(
        "--check-dataset-directories",
        action="store_true",
        help="Verify that train, validation, and test directories exist.",
    )

    parser.add_argument(
        "--create-generated-directories",
        action="store_true",
        help="Create label and output directories.",
    )

    return parser.parse_args()


def print_summary(loaded_configuration) -> None:
    settings = loaded_configuration.settings
    paths = loaded_configuration.paths

    enabled_features = settings.features.enabled_items

    print("=" * 80)
    print("PROJECT CONFIGURATION")
    print("=" * 80)

    print(f"Configuration file : {loaded_configuration.source_path}")
    print(f"Project name       : {settings.project.name}")
    print(f"Project root       : {paths.project_root}")
    print(f"Random seed        : {settings.project.random_seed}")

    print()
    print("DATASET")
    print("-" * 80)
    print(f"Dataset root       : {paths.dataset_root}")

    for split_name, split_path in paths.dataset_splits.items():
        print(f"{split_name:<18} : {split_path}")

    print(
        f"Image dimensions   : "
        f"{settings.dataset.image_size.width} × "
        f"{settings.dataset.image_size.height}"
    )
    print(f"Channels           : {settings.dataset.channels}")
    print(f"Color mode         : {settings.dataset.color_mode}")

    print()
    print("SLIDING WINDOW")
    print("-" * 80)
    print(
        f"Window dimensions  : "
        f"{settings.sliding_window.width} × "
        f"{settings.sliding_window.height}"
    )
    print(
        f"Stride             : "
        f"x={settings.sliding_window.stride.x}, "
        f"y={settings.sliding_window.stride.y}"
    )
    print(
        f"Output grid        : "
        f"{settings.grid_width} × {settings.grid_height}"
    )
    print(
        f"Windows per image  : "
        f"{settings.windows_per_image}"
    )

    print()
    print("FEATURES")
    print("-" * 80)
    print(
        f"Normalization      : "
        f"{settings.features.normalization.method}"
    )
    print(
        f"Score fusion       : "
        f"{settings.features.score_fusion.method}"
    )

    for feature_name, feature_settings in enabled_features.items():
        print(
            f"{feature_name:<20} "
            f"weight={feature_settings.weight:<8} "
            f"direction={feature_settings.direction}"
        )

    print()
    print("LABELS")
    print("-" * 80)
    print(f"Labels root        : {paths.labels_root}")
    print(f"Format             : {settings.labels.file_format}")
    print(f"Data type          : {settings.labels.dtype}")

    print()
    print("MODEL")
    print("-" * 80)
    print(f"Architecture       : {settings.model.architecture}")
    print(
        f"Input shape        : "
        f"{settings.dataset.image_size.height} × "
        f"{settings.dataset.image_size.width} × "
        f"{settings.model.input_channels}"
    )
    print(
        f"Output shape       : "
        f"{settings.grid_height} × "
        f"{settings.grid_width}"
    )
    print(
        f"Output activation  : "
        f"{settings.model.output_activation}"
    )

    print()
    print("THRESHOLD SCAN")
    print("-" * 80)

    thresholds = settings.threshold_selection.candidates()

    print(
        "Thresholds         : "
        + ", ".join(f"{value:.2f}" for value in thresholds)
    )
    print(
        f"Selection metric   : "
        f"{settings.threshold_selection.selection_metric}"
    )

    print()
    print("OUTPUT DIRECTORIES")
    print("-" * 80)

    for output_name, output_path in paths.output_directories.items():
        print(f"{output_name:<20}: {output_path}")

    print()
    print("REPRODUCIBILITY")
    print("-" * 80)

    seed_status = set_global_seed(
        settings.project.random_seed,
        deterministic=settings.runtime.deterministic,
    )

    print(f"Python seed set    : {seed_status['python']}")
    print(f"NumPy seed set     : {seed_status['numpy']}")
    print(f"TensorFlow present : {seed_status['tensorflow']}")
    print(f"Deterministic mode : {seed_status['deterministic']}")

    print("=" * 80)
    print("Configuration validation completed successfully.")
    print("=" * 80)


def main() -> None:
    arguments = parse_arguments()

    try:
        loaded_configuration = load_configuration(
            arguments.config,
            create_generated_directories=(
                arguments.create_generated_directories
            ),
            validate_dataset_directories=(
                arguments.check_dataset_directories
            ),
        )

        print_summary(loaded_configuration)

    except (
        ConfigurationError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as error:
        print(
            f"Configuration validation failed:\n{error}",
            file=sys.stderr,
        )

        raise SystemExit(1) from error


if __name__ == "__main__":
    main()