from __future__ import annotations

import numpy as np

from src.config.schema import (
    DatasetSettings,
    ExpectedGridSettings,
    ImageSizeSettings,
    ProjectConfiguration,
    ProjectSettings,
    SlidingWindowSettings,
    StrideSettings,
)


def test_sliding_window_grid_from_real_configuration(
    loaded_configuration,
) -> None:
    from src.data.sliding_window import (
        SlidingWindowGrid,
    )

    grid = SlidingWindowGrid.from_configuration(
        loaded_configuration.settings
    )

    assert grid.grid_width == 25
    assert grid.grid_height == 25
    assert grid.windows_per_image == 625

    assert grid.x_positions[0] == 0
    assert grid.x_positions[-1] == 192

    assert grid.y_positions[0] == 0
    assert grid.y_positions[-1] == 192

    coordinates = grid.coordinate_array()

    assert coordinates.shape == (
        25,
        25,
        4,
    )

    assert tuple(
        coordinates[0, 0]
    ) == (
        0,
        0,
        64,
        64,
    )

    assert tuple(
        coordinates[-1, -1]
    ) == (
        192,
        192,
        64,
        64,
    )

    image = np.zeros(
        (256, 256, 3),
        dtype=np.uint8,
    )

    grid.validate_image(image)

    final_location = list(
        grid.locations()
    )[-1]

    patch = grid.extract(
        image,
        final_location,
    )

    assert patch.shape == (
        64,
        64,
        3,
    )