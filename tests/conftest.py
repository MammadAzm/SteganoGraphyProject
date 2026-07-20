from __future__ import annotations

from pathlib import Path

import pytest

from src.config.loader import (
    load_configuration,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def loaded_configuration():
    return load_configuration(
        PROJECT_ROOT
        / "configs"
        / "default.yaml"
    )