from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config.loader import load_configuration
from src.config.schema import ConfigurationError
from src.data.dataset_validator import DatasetValidator
from src.utils.reproducibility import set_global_seed


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the resized train, validation, and test "
            "image datasets."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "default.yaml",
        help="Path to the project YAML configuration file.",
    )

    return parser.parse_args()


def print_result(result) -> None:
    summary = result.summary

    print()
    print("=" * 80)
    print("DATASET VALIDATION RESULT")
    print("=" * 80)

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        statistics = summary.split_statistics[
            split_name
        ]

        print(
            f"{split_name.upper():<12} "
            f"files={statistics['files']:<7} "
            f"readable={statistics['readable']:<7} "
            f"unreadable={statistics['unreadable']:<7} "
            f"errors={statistics['errors']:<7} "
            f"warnings={statistics['warnings']}"
        )

    print("-" * 80)
    print(f"Total files       : {summary.total_files}")
    print(
        f"Readable images   : "
        f"{summary.readable_images}"
    )
    print(
        f"Unreadable images : "
        f"{summary.unreadable_images}"
    )
    print(f"Errors            : {summary.error_count}")
    print(
        f"Warnings          : "
        f"{summary.warning_count}"
    )
    print(f"Information       : {summary.info_count}")

    print()
    print("Sliding-window contract")
    print("-" * 80)

    window = summary.sliding_window

    print(
        f"Image             : "
        f"{window['image_width']}x"
        f"{window['image_height']}"
    )
    print(
        f"Window            : "
        f"{window['window_width']}x"
        f"{window['window_height']}"
    )
    print(
        f"Stride            : "
        f"x={window['stride_x']}, "
        f"y={window['stride_y']}"
    )
    print(
        f"Output grid       : "
        f"{window['grid_width']}x"
        f"{window['grid_height']}"
    )
    print(
        f"Windows per image : "
        f"{window['windows_per_image']}"
    )

    if summary.issue_codes:
        print()
        print("Issue counts")
        print("-" * 80)

        for issue_code, count in (
            summary.issue_codes.items()
        ):
            print(f"{issue_code:<45} {count}")

    if summary.report_paths:
        print()
        print("Reports")
        print("-" * 80)

        for report_name, report_path in (
            summary.report_paths.items()
        ):
            print(
                f"{report_name:<12}: {report_path}"
            )

    print("=" * 80)

    if summary.valid:
        print("RESULT: DATASET IS VALID")
    else:
        print("RESULT: DATASET VALIDATION FAILED")

    print("=" * 80)


def main() -> None:
    arguments = parse_arguments()

    try:
        loaded_configuration = load_configuration(
            arguments.config,
            create_generated_directories=True,
            validate_dataset_directories=False,
        )

        set_global_seed(
            loaded_configuration
            .settings
            .project
            .random_seed,
            deterministic=(
                loaded_configuration
                .settings
                .runtime
                .deterministic
            ),
        )

        validator = DatasetValidator(
            loaded_configuration
        )

        result = validator.run()

        print_result(result)

        if (
            not result.summary.valid
            and validator
            .validation_settings
            .fail_on_error
        ):
            raise SystemExit(1)

    except (
        ConfigurationError,
        FileNotFoundError,
        PermissionError,
        OSError,
        ValueError,
    ) as error:
        print(
            f"Dataset validation could not complete:\n{error}",
            file=sys.stderr,
        )

        raise SystemExit(1) from error


if __name__ == "__main__":
    main()