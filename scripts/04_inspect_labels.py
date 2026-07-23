from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.config.loader import (
    load_configuration,
)
from src.labeling.inspection import (
    LabelMapInspector,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect generated continuous sliding-window label maps."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "default.yaml"
        ),
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        choices=(
            "train",
            "validation",
            "test",
        ),
        default=[
            "train",
            "validation",
            "test",
        ],
    )

    parser.add_argument(
        "--no-galleries",
        action="store_true",
        help=(
            "Run integrity and statistical checks without "
            "generating visual panels."
        ),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    loaded = load_configuration(
        arguments.config,
        create_generated_directories=True,
        validate_dataset_directories=True,
    )

    inspector = LabelMapInspector(
        loaded
    )

    print("=" * 80)
    print("STAGE 4 — LABEL MAP INSPECTION")
    print("=" * 80)
    print(
        f"Score shape: "
        f"{inspector.grid.score_shape}"
    )
    print(
        f"Enabled features: "
        f"{', '.join(inspector.feature_names)}"
    )
    print(
        f"Feature groups: "
        f"{', '.join(inspector.group_names)}"
    )
    print("=" * 80)

    summary = inspector.run(
        splits=arguments.splits,
        generate_galleries=(
            not arguments.no_galleries
        ),
    )

    print()
    print("=" * 80)
    print("LABEL INSPECTION COMPLETED")
    print("=" * 80)
    print(
        f"Automated integrity passed: "
        f"{summary['automated_integrity_passed']}"
    )
    print(
        f"Critical errors: "
        f"{summary['critical_error_count']}"
    )
    print(
        f"Suspicious images: "
        f"{summary['suspicious_image_count']}"
    )
    print(
        f"Visualized images: "
        f"{summary['visualized_image_count']}"
    )
    print(
        f"Reports: "
        f"{inspector.settings.output_root}"
    )
    print("=" * 80)

    if summary[
        "critical_error_count"
    ] > 0:
        raise SystemExit(1)

    print(
        "\nAutomated checks passed. Review the generated "
        "galleries before running 04_approve_labels.py."
    )


if __name__ == "__main__":
    main()