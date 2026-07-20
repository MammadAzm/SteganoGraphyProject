from __future__ import annotations

import argparse
from pathlib import Path


DIRECTORIES = [
    "configs",
    "data",
    "data/labels",
    "data/labels/train",
    "data/labels/validation",
    "data/labels/test",
    "legacy",
    "outputs",
    "outputs/feature_statistics",
    "outputs/heatmaps",
    "outputs/models",
    "outputs/test_reports",
    "outputs/training_reports",
    "outputs/validation_reports",
    "scripts",
    "src",
    "src/config",
    "src/data",
    "src/evaluation",
    "src/features",
    "src/labeling",
    "src/models",
    "src/training",
    "src/utils",
    "src/validation",
    "tests",
]

PYTHON_PACKAGES = [
    "src",
    "src/config",
    "src/data",
    "src/evaluation",
    "src/features",
    "src/labeling",
    "src/models",
    "src/training",
    "src/utils",
    "src/validation",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the modular project directory structure."
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help=(
            "Project root directory. By default, the parent directory "
            "of the scripts folder is used."
        ),
    )

    return parser.parse_args()


def create_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def create_empty_file(path: Path) -> None:
    if not path.exists():
        path.touch()


def initialize_project(project_root: Path) -> None:
    project_root = project_root.expanduser().resolve()

    print("=" * 72)
    print("Initializing steganography patch-scoring project")
    print(f"Project root: {project_root}")
    print("=" * 72)

    for relative_directory in DIRECTORIES:
        directory = project_root / relative_directory
        create_directory(directory)
        print(f"[directory] {directory}")

    for relative_package in PYTHON_PACKAGES:
        init_file = project_root / relative_package / "__init__.py"
        create_empty_file(init_file)
        print(f"[package]   {init_file}")

    for output_directory in [
        "data/labels/train",
        "data/labels/validation",
        "data/labels/test",
        "outputs/feature_statistics",
        "outputs/heatmaps",
        "outputs/models",
        "outputs/test_reports",
        "outputs/training_reports",
        "outputs/validation_reports",
        "tests",
        "legacy",
    ]:
        gitkeep_file = project_root / output_directory / ".gitkeep"
        create_empty_file(gitkeep_file)

    print("=" * 72)
    print("Project structure created successfully.")
    print("Existing files were not overwritten.")
    print("=" * 72)


def main() -> None:
    arguments = parse_arguments()
    initialize_project(arguments.root)


if __name__ == "__main__":
    main()