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


from src.config.loader import load_configuration
from src.labeling.generator import (
    SlidingWindowLabelGenerator,
)
from src.utils.reproducibility import (
    set_global_seed,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate continuous sliding-window label maps for "
            "train, validation, and test images."
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
        "--phase",
        choices=(
            "all",
            "fit-normalization",
            "generate",
        ),
        default="all",
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
        "--force-normalization",
        action="store_true",
        help=(
            "Ignore an existing normalization artifact and "
            "fit new training bounds."
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

    set_global_seed(
        loaded.settings.project.random_seed,
        deterministic=(
            loaded.settings.runtime.deterministic
        ),
    )

    generator = SlidingWindowLabelGenerator(
        loaded
    )

    print("=" * 80)
    print("STAGE 3 — SLIDING-WINDOW LABEL GENERATION")
    print("=" * 80)
    print(
        f"Enabled features: "
        f"{', '.join(generator.feature_names)}"
    )
    print(
        f"Grid: "
        f"{generator.grid.grid_height}x"
        f"{generator.grid.grid_width}"
    )
    print(
        f"Windows per image: "
        f"{generator.grid.windows_per_image}"
    )
    print(
        "Normalization fit split: train"
    )
    print("=" * 80)

    if arguments.phase == "fit-normalization":
        artifact = generator.fit_normalization(
            force=arguments.force_normalization
        )

        print(
            json.dumps(
                artifact.to_dict(),
                indent=2,
            )
        )

        return

    if arguments.phase == "generate":
        artifact = generator.fit_normalization(
            force=False
        )

        summaries = {
            split_name: generator.generate_split(
                split_name,
                artifact,
            )
            for split_name in arguments.splits
        }

        print(
            json.dumps(
                summaries,
                indent=2,
            )
        )

        return

    summary = generator.run(
        splits=arguments.splits,
        force_normalization=(
            arguments.force_normalization
        ),
    )

    print("=" * 80)
    print("LABEL GENERATION COMPLETED")
    print("=" * 80)

    for split_name, split_summary in (
        summary["splits"].items()
    ):
        print(
            f"{split_name:<12} "
            f"images={split_summary['image_count']:<7} "
            f"generated={split_summary['generated_count']:<7} "
            f"reused={split_summary['reused_count']:<7} "
            f"mean={split_summary['score_mean']:.6f} "
            f"std={split_summary['score_standard_deviation']:.6f}"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()