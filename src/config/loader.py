from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.config.schema import (
    ConfigurationError,
    ProjectConfiguration,
)
from src.utils.paths import ProjectPaths


@dataclass(frozen=True)
class LoadedConfiguration:
    source_path: Path
    settings: ProjectConfiguration
    paths: ProjectPaths
    raw: Mapping[str, Any]


def _expand_environment_variables(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)

    if isinstance(value, list):
        return [
            _expand_environment_variables(item)
            for item in value
        ]

    if isinstance(value, dict):
        return {
            key: _expand_environment_variables(item)
            for key, item in value.items()
        }

    return value


def load_configuration(
    configuration_path: str | Path,
    *,
    create_generated_directories: bool = False,
    validate_dataset_directories: bool = False,
) -> LoadedConfiguration:
    source_path = Path(configuration_path).expanduser().resolve()

    if not source_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {source_path}"
        )

    try:
        with source_path.open(
            "r",
            encoding="utf-8",
        ) as configuration_file:
            raw_configuration = yaml.safe_load(configuration_file)

    except yaml.YAMLError as error:
        raise ConfigurationError(
            f"Invalid YAML syntax in {source_path}: {error}"
        ) from error

    if raw_configuration is None:
        raise ConfigurationError(
            f"Configuration file is empty: {source_path}"
        )

    if not isinstance(raw_configuration, dict):
        raise ConfigurationError(
            "The root YAML configuration value must be a mapping."
        )

    expanded_configuration = _expand_environment_variables(
        raw_configuration
    )

    settings = ProjectConfiguration.from_mapping(
        expanded_configuration
    )

    paths = ProjectPaths.from_configuration(
        configuration=settings,
        configuration_path=source_path,
    )

    if create_generated_directories:
        paths.create_generated_directories()

    if validate_dataset_directories:
        paths.validate_dataset_directories()

    return LoadedConfiguration(
        source_path=source_path,
        settings=settings,
        paths=paths,
        raw=expanded_configuration,
    )