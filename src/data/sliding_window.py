from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from src.config.schema import ProjectConfiguration


@dataclass(frozen=True)
class WindowLocation:
    row: int
    column: int
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class SlidingWindowGrid:
    image_width: int
    image_height: int
    window_width: int
    window_height: int
    stride_x: int
    stride_y: int
    x_positions: tuple[int, ...]
    y_positions: tuple[int, ...]

    @classmethod
    def from_configuration(
        cls,
        configuration: ProjectConfiguration,
    ) -> "SlidingWindowGrid":
        image_width = configuration.dataset.image_size.width
        image_height = configuration.dataset.image_size.height

        window_width = configuration.sliding_window.width
        window_height = configuration.sliding_window.height

        stride_x = configuration.sliding_window.stride.x
        stride_y = configuration.sliding_window.stride.y

        x_positions = tuple(
            range(
                0,
                image_width - window_width + 1,
                stride_x,
            )
        )

        y_positions = tuple(
            range(
                0,
                image_height - window_height + 1,
                stride_y,
            )
        )

        grid = cls(
            image_width=image_width,
            image_height=image_height,
            window_width=window_width,
            window_height=window_height,
            stride_x=stride_x,
            stride_y=stride_y,
            x_positions=x_positions,
            y_positions=y_positions,
        )

        grid.validate(configuration)
        return grid

    @property
    def grid_width(self) -> int:
        return len(self.x_positions)

    @property
    def grid_height(self) -> int:
        return len(self.y_positions)

    @property
    def windows_per_image(self) -> int:
        return self.grid_width * self.grid_height

    @property
    def score_shape(self) -> tuple[int, int]:
        return self.grid_height, self.grid_width

    def validate(
        self,
        configuration: ProjectConfiguration,
    ) -> None:
        expected_width = (
            configuration.sliding_window.expected_grid.width
        )

        expected_height = (
            configuration.sliding_window.expected_grid.height
        )

        errors: list[str] = []

        if self.grid_width != expected_width:
            errors.append(
                f"Calculated grid width is {self.grid_width}; "
                f"expected {expected_width}."
            )

        if self.grid_height != expected_height:
            errors.append(
                f"Calculated grid height is {self.grid_height}; "
                f"expected {expected_height}."
            )

        if self.x_positions[-1] + self.window_width != self.image_width:
            errors.append(
                "The final horizontal window does not align with "
                "the right image boundary."
            )

        if self.y_positions[-1] + self.window_height != self.image_height:
            errors.append(
                "The final vertical window does not align with "
                "the bottom image boundary."
            )

        if errors:
            raise ValueError(
                "Invalid sliding-window grid:\n"
                + "\n".join(
                    f"- {error}"
                    for error in errors
                )
            )

    def validate_image(
        self,
        image: np.ndarray,
    ) -> None:
        if image.ndim not in {2, 3}:
            raise ValueError(
                "Image must be grayscale or multi-channel."
            )

        height, width = image.shape[:2]

        if width != self.image_width or height != self.image_height:
            raise ValueError(
                f"Expected image dimensions "
                f"{self.image_width}x{self.image_height}; "
                f"found {width}x{height}."
            )

    def locations(self) -> Iterator[WindowLocation]:
        for row, y in enumerate(self.y_positions):
            for column, x in enumerate(self.x_positions):
                yield WindowLocation(
                    row=row,
                    column=column,
                    x=x,
                    y=y,
                    width=self.window_width,
                    height=self.window_height,
                )

    def extract(
        self,
        image: np.ndarray,
        location: WindowLocation,
    ) -> np.ndarray:
        return image[
            location.y:location.y + location.height,
            location.x:location.x + location.width,
        ]

    def coordinate_array(self) -> np.ndarray:
        coordinates = np.empty(
            (
                self.grid_height,
                self.grid_width,
                4,
            ),
            dtype=np.int32,
        )

        for location in self.locations():
            coordinates[
                location.row,
                location.column,
            ] = (
                location.x,
                location.y,
                location.width,
                location.height,
            )

        return coordinates