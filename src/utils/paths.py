from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from src.config.schema import (
    ConfigurationError,
    ProjectConfiguration,
)


WINDOWS_ABSOLUTE_PATH_PATTERN = re.compile(
    r"^[A-Za-z]:[\\/]"
)


def _expand_path_value(value: str) -> str:
    expanded_environment = os.path.expandvars(value)
    expanded_user = os.path.expanduser(expanded_environment)

    if "${" in expanded_user:
        raise ConfigurationError(
            f"Path contains an unresolved environment variable: {value}"
        )

    return expanded_user


def resolve_path(
    value: str,
    base_directory: Path,
) -> Path:
    expanded_value = _expand_path_value(value)

    if (
        WINDOWS_ABSOLUTE_PATH_PATTERN.match(expanded_value)
        and os.name != "nt"
    ):
        raise ConfigurationError(
            "A Windows absolute path was provided while running on a "
            f"non-Windows operating system: {expanded_value}. "
            "Update the path in the YAML configuration."
        )

    candidate = Path(expanded_value)

    if not candidate.is_absolute():
        candidate = base_directory / candidate

    return candidate.resolve()


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    dataset_root: Path
    dataset_splits: dict[str, Path]
    labels_root: Path
    label_splits: dict[str, Path]
    outputs_root: Path
    output_directories: dict[str, Path]

    @classmethod
    def from_configuration(
        cls,
        configuration: ProjectConfiguration,
        configuration_path: Path,
    ) -> "ProjectPaths":
        configuration_directory = configuration_path.parent

        project_root = resolve_path(
            configuration.project.root_dir,
            configuration_directory,
        )

        dataset_root = resolve_path(
            configuration.dataset.root_dir,
            project_root,
        )

        dataset_splits = {
            split_name: (
                dataset_root / relative_directory
            ).resolve()
            for split_name, relative_directory
            in configuration.dataset.splits.items()
        }

        labels_root = resolve_path(
            configuration.labels.root_dir,
            project_root,
        )

        label_splits = {
            split_name: (labels_root / split_name).resolve()
            for split_name in configuration.dataset.splits
        }

        outputs_root = resolve_path(
            configuration.outputs.root_dir,
            project_root,
        )

        output_directories = {
            output_name: (
                outputs_root / relative_directory
            ).resolve()
            for output_name, relative_directory
            in configuration.outputs.directories.items()
        }

        return cls(
            project_root=project_root,
            dataset_root=dataset_root,
            dataset_splits=dataset_splits,
            labels_root=labels_root,
            label_splits=label_splits,
            outputs_root=outputs_root,
            output_directories=output_directories,
        )

    def create_generated_directories(self) -> None:
        self.labels_root.mkdir(parents=True, exist_ok=True)
        self.outputs_root.mkdir(parents=True, exist_ok=True)

        for split_directory in self.label_splits.values():
            split_directory.mkdir(parents=True, exist_ok=True)

        for output_directory in self.output_directories.values():
            output_directory.mkdir(parents=True, exist_ok=True)

    def validate_dataset_directories(self) -> None:
        missing_directories = [
            path
            for path in self.dataset_splits.values()
            if not path.is_dir()
        ]

        if missing_directories:
            formatted = "\n".join(
                f"- {path}" for path in missing_directories
            )

            raise FileNotFoundError(
                "The following dataset directories do not exist:\n"
                f"{formatted}"
            )

    def dataset_split(self, split_name: str) -> Path:
        if split_name not in self.dataset_splits:
            raise KeyError(
                f"Unknown dataset split: {split_name}"
            )

        return self.dataset_splits[split_name]

    def label_split(self, split_name: str) -> Path:
        if split_name not in self.label_splits:
            raise KeyError(
                f"Unknown label split: {split_name}"
            )

        return self.label_splits[split_name]

    def output_directory(self, name: str) -> Path:
        if name not in self.output_directories:
            raise KeyError(
                f"Unknown output directory: {name}"
            )

        return self.output_directories[name]