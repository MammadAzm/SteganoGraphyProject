from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def build_pixel_coverage_map(
    score_map: np.ndarray,
    window_coordinates: np.ndarray,
    *,
    image_height: int,
    image_width: int,
) -> np.ndarray:
    if score_map.shape != window_coordinates.shape[:2]:
        raise ValueError(
            "Score-map shape must match the first two coordinate "
            "dimensions."
        )

    score_sum = np.zeros(
        (image_height, image_width),
        dtype=np.float64,
    )

    coverage_count = np.zeros(
        (image_height, image_width),
        dtype=np.float64,
    )

    for row in range(score_map.shape[0]):
        for column in range(score_map.shape[1]):
            x, y, width, height = (
                int(value)
                for value in window_coordinates[
                    row,
                    column,
                ]
            )

            score_sum[
                y:y + height,
                x:x + width,
            ] += float(
                score_map[row, column]
            )

            coverage_count[
                y:y + height,
                x:x + width,
            ] += 1.0

    if np.any(coverage_count == 0):
        raise ValueError(
            "Some image pixels are not covered by any sliding window."
        )

    return (
        score_sum / coverage_count
    ).astype(np.float32)


def _window_montage(
    image: Image.Image,
    score_map: np.ndarray,
    window_coordinates: np.ndarray,
    flat_indices: Sequence[int],
    *,
    tile_size: int = 112,
) -> Image.Image:
    label_height = 20

    montage = Image.new(
        "RGB",
        (
            tile_size * len(flat_indices),
            tile_size + label_height,
        ),
        color=(255, 255, 255),
    )

    draw = ImageDraw.Draw(montage)

    for montage_index, flat_index in enumerate(
        flat_indices
    ):
        row, column = np.unravel_index(
            int(flat_index),
            score_map.shape,
        )

        x, y, width, height = (
            int(value)
            for value in window_coordinates[
                row,
                column,
            ]
        )

        patch = image.crop(
            (
                x,
                y,
                x + width,
                y + height,
            )
        )

        patch = patch.resize(
            (tile_size, tile_size),
            Image.Resampling.LANCZOS,
        )

        left = montage_index * tile_size

        montage.paste(
            patch,
            (left, 0),
        )

        score = float(
            score_map[row, column]
        )

        draw.text(
            (
                left + 4,
                tile_size + 2,
            ),
            f"{score:.3f}",
            fill=(0, 0, 0),
        )

    return montage


def render_label_panel(
    *,
    source_image_path: Path,
    label_path: Path,
    output_path: Path,
    overlay_alpha: float,
    top_windows: int,
    bottom_windows: int,
    dpi: int,
) -> None:
    with Image.open(
        source_image_path
    ) as image_file:
        image = image_file.convert("RGB")

    image_array = np.asarray(
        image,
        dtype=np.uint8,
    )

    with np.load(
        label_path,
        allow_pickle=False,
    ) as label_file:
        score_map = label_file[
            "score_map"
        ].astype(
            np.float32
        )

        group_score_maps = label_file[
            "group_score_maps"
        ].astype(
            np.float32
        )

        group_names = tuple(
            str(value)
            for value in label_file[
                "group_names"
            ].tolist()
        )

        window_coordinates = label_file[
            "window_coordinates"
        ].astype(
            np.int32
        )

    coverage_map = build_pixel_coverage_map(
        score_map,
        window_coordinates,
        image_height=image_array.shape[0],
        image_width=image_array.shape[1],
    )

    flat_scores = score_map.reshape(-1)

    top_count = min(
        top_windows,
        flat_scores.size,
    )

    bottom_count = min(
        bottom_windows,
        flat_scores.size,
    )

    top_indices = np.argsort(
        flat_scores
    )[-top_count:][::-1]

    bottom_indices = np.argsort(
        flat_scores
    )[:bottom_count]

    top_montage = _window_montage(
        image,
        score_map,
        window_coordinates,
        top_indices,
    )

    bottom_montage = _window_montage(
        image,
        score_map,
        window_coordinates,
        bottom_indices,
    )

    figure, axes = plt.subplots(
        2,
        4,
        figsize=(20, 10),
    )

    axes[0, 0].imshow(
        image_array
    )

    axes[0, 0].set_title(
        "Original image"
    )

    score_artist = axes[0, 1].imshow(
        score_map,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )

    axes[0, 1].set_title(
        (
            "25×25 score map\n"
            f"mean={np.mean(score_map):.3f}, "
            f"std={np.std(score_map):.3f}"
        )
    )

    figure.colorbar(
        score_artist,
        ax=axes[0, 1],
        fraction=0.046,
    )

    axes[0, 2].imshow(
        image_array
    )

    overlay_artist = axes[0, 2].imshow(
        coverage_map,
        vmin=0.0,
        vmax=1.0,
        alpha=overlay_alpha,
    )

    axes[0, 2].set_title(
        "Pixel-aligned score overlay"
    )

    figure.colorbar(
        overlay_artist,
        ax=axes[0, 2],
        fraction=0.046,
    )

    axes[0, 3].imshow(
        top_montage
    )

    axes[0, 3].set_title(
        "Highest-scoring windows"
    )

    for group_index in range(3):
        group_artist = axes[
            1,
            group_index,
        ].imshow(
            group_score_maps[
                group_index
            ],
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )

        axes[
            1,
            group_index,
        ].set_title(
            group_names[group_index]
        )

        figure.colorbar(
            group_artist,
            ax=axes[1, group_index],
            fraction=0.046,
        )

    axes[1, 3].imshow(
        bottom_montage
    )

    axes[1, 3].set_title(
        "Lowest-scoring windows"
    )

    for axis in axes.reshape(-1):
        axis.axis("off")

    figure.suptitle(
        source_image_path.name,
        fontsize=14,
    )

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_score_histograms(
    split_scores: Mapping[str, np.ndarray],
    output_path: Path,
    *,
    bins: int,
    dpi: int,
) -> None:
    figure, axis = plt.subplots(
        figsize=(10, 6),
    )

    histogram_edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    for split_name, values in (
        split_scores.items()
    ):
        axis.hist(
            values,
            bins=histogram_edges,
            density=True,
            histtype="step",
            linewidth=2,
            label=split_name,
        )

    axis.set_title(
        "Sliding-window score distributions"
    )

    axis.set_xlabel(
        "Combined suitability score"
    )

    axis.set_ylabel(
        "Density"
    )

    axis.set_xlim(
        0.0,
        1.0,
    )

    axis.grid(
        alpha=0.25
    )

    axis.legend()

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(figure)


def select_visual_records(
    records: Sequence[Mapping[str, Any]],
    *,
    samples_per_category: int,
    maximum_records: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    if not records:
        return []

    records_list = [
        dict(record)
        for record in records
    ]

    selected: list[dict[str, Any]] = []
    selected_paths: set[str] = set()

    def add_candidates(
        candidates: Sequence[Mapping[str, Any]],
    ) -> None:
        for candidate in candidates:
            key = str(
                candidate["label_path"]
            )

            if key in selected_paths:
                continue

            selected.append(
                dict(candidate)
            )

            selected_paths.add(
                key
            )

            if len(selected) >= maximum_records:
                return

    by_mean = sorted(
        records_list,
        key=lambda record: float(
            record["score_mean"]
        ),
    )

    by_std = sorted(
        records_list,
        key=lambda record: float(
            record["score_standard_deviation"]
        ),
    )

    by_zero = sorted(
        records_list,
        key=lambda record: float(
            record["zero_fraction"]
        ),
        reverse=True,
    )

    by_one = sorted(
        records_list,
        key=lambda record: float(
            record["one_fraction"]
        ),
        reverse=True,
    )

    add_candidates(
        by_mean[
            :samples_per_category
        ]
    )

    add_candidates(
        by_mean[
            -samples_per_category:
        ][::-1]
    )

    median_mean = float(
        np.median(
            [
                float(record["score_mean"])
                for record in records_list
            ]
        )
    )

    median_candidates = sorted(
        records_list,
        key=lambda record: abs(
            float(record["score_mean"])
            - median_mean
        ),
    )

    add_candidates(
        median_candidates[
            :samples_per_category
        ]
    )

    add_candidates(
        by_std[
            :samples_per_category
        ]
    )

    add_candidates(
        by_std[
            -samples_per_category:
        ][::-1]
    )

    add_candidates(
        by_zero[
            :samples_per_category
        ]
    )

    add_candidates(
        by_one[
            :samples_per_category
        ]
    )

    if len(selected) < maximum_records:
        remaining = [
            record
            for record in records_list
            if str(
                record["label_path"]
            ) not in selected_paths
        ]

        generator = np.random.default_rng(
            random_seed
        )

        generator.shuffle(
            remaining
        )

        add_candidates(
            remaining
        )

    return selected[
        :maximum_records
    ]