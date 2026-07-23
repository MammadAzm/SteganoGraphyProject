from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.config.loader import (
    load_configuration,
)
from src.labeling.inspection_settings import (
    LabelInspectionSettings,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Approve Stage 4 labels after manual visual inspection."
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
        "--reviewer",
        required=True,
        help="Name or identifier of the reviewer.",
    )

    parser.add_argument(
        "--reviewed-count",
        type=int,
        required=True,
        help=(
            "Number of generated visual panels that were "
            "manually reviewed."
        ),
    )

    parser.add_argument(
        "--note",
        action="append",
        default=[],
        help=(
            "Review note. This argument may be supplied "
            "multiple times."
        ),
    )

    return parser.parse_args()


def read_json(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required report not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        return json.load(
            input_file
        )


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
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


def main() -> None:
    arguments = parse_arguments()

    if arguments.reviewed_count <= 0:
        raise ValueError(
            "reviewed-count must be greater than zero."
        )

    loaded = load_configuration(
        arguments.config
    )

    settings = (
        LabelInspectionSettings
        .from_loaded_configuration(
            loaded
        )
    )

    integrity_report = read_json(
        settings.output_root
        / "integrity_report.json"
    )

    inspection_summary = read_json(
        settings.output_root
        / "inspection_summary.json"
    )

    critical_errors = int(
        integrity_report.get(
            "critical_error_count",
            -1,
        )
    )

    if critical_errors != 0:
        raise RuntimeError(
            "Labels cannot be approved because the integrity "
            f"report contains {critical_errors} critical errors."
        )

    visualized_count = int(
        inspection_summary.get(
            "visualized_image_count",
            0,
        )
    )

    if visualized_count <= 0:
        raise RuntimeError(
            "No gallery images were generated. Run Stage 4 "
            "inspection with gallery generation enabled."
        )

    if (
        arguments.reviewed_count
        > visualized_count
    ):
        raise ValueError(
            f"reviewed-count is {arguments.reviewed_count}, "
            f"but only {visualized_count} gallery panels exist."
        )

    approval = {
        "stage": 4,
        "approved": True,
        "status": "approved",
        "approved_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "reviewer": arguments.reviewer,
        "reviewed_visual_examples": (
            arguments.reviewed_count
        ),
        "review_notes": list(
            arguments.note
        ),
        "critical_error_count": 0,
        "warning_count": int(
            integrity_report.get(
                "warning_count",
                0,
            )
        ),
        "suspicious_image_count": int(
            inspection_summary.get(
                "suspicious_image_count",
                0,
            )
        ),
        "normalization_fingerprint": (
            inspection_summary.get(
                "normalization_fingerprint",
                "",
            )
        ),
        "label_fingerprint": (
            inspection_summary.get(
                "label_fingerprint",
                "",
            )
        ),
        "split_counts": {
            split_name: int(
                split_data[
                    "valid_label_count"
                ]
            )
            for split_name, split_data
            in inspection_summary[
                "splits"
            ].items()
        },
        "score_shape": [25, 25],
        "inspection_summary": str(
            (
                settings.output_root
                / "inspection_summary.json"
            ).resolve()
        ),
        "integrity_report": str(
            (
                settings.output_root
                / "integrity_report.json"
            ).resolve()
        ),
    }

    approval_path = (
        settings.output_root
        / "label_approval.json"
    )

    write_json(
        approval_path,
        approval,
    )

    print("=" * 80)
    print("STAGE 4 LABELS APPROVED")
    print("=" * 80)
    print(
        f"Reviewer: "
        f"{arguments.reviewer}"
    )
    print(
        f"Reviewed gallery panels: "
        f"{arguments.reviewed_count}"
    )
    print(
        f"Approval record: "
        f"{approval_path}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()