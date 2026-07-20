from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
from scipy.stats import rankdata


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.config.loader import load_configuration
from src.features.fusion import (
    GroupedFusionSettings,
    fit_robust_normalization_bounds,
    fuse_normalized_features,
    normalize_feature_values,
)
from src.features.pipeline import FeaturePipeline
from src.features.synthetic import (
    create_synthetic_patches,
)
from src.utils.reproducibility import set_global_seed


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Review corrected patch features, feature correlations, "
            "normalization, and grouped score fusion."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "default.yaml",
    )

    parser.add_argument(
        "--split",
        choices=(
            "train",
            "validation",
            "test",
        ),
        default=None,
    )

    parser.add_argument(
        "--sample-images",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--patches-per-image",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--enabled-only",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def stable_seed(
    base_seed: int,
    *parts: object,
) -> int:
    message = "|".join(
        [
            str(base_seed),
            *[
                str(part)
                for part in parts
            ],
        ]
    )

    digest = hashlib.blake2b(
        message.encode("utf-8"),
        digest_size=8,
    ).digest()

    return int.from_bytes(
        digest,
        byteorder="little",
        signed=False,
    ) % (2**32)


def list_images(
    directory: Path,
    *,
    recursive: bool,
    extensions: Sequence[str],
) -> list[Path]:
    if recursive:
        candidates = directory.rglob("*")
    else:
        candidates = directory.iterdir()

    extension_set = {
        extension.lower()
        for extension in extensions
    }

    paths = [
        path
        for path in candidates
        if (
            path.is_file()
            and path.suffix.lower() in extension_set
        )
    ]

    return sorted(
        paths,
        key=lambda path: str(path).lower(),
    )


def sliding_positions(
    image_width: int,
    image_height: int,
    window_width: int,
    window_height: int,
    stride_x: int,
    stride_y: int,
) -> list[tuple[int, int]]:
    return [
        (x, y)
        for y in range(
            0,
            image_height - window_height + 1,
            stride_y,
        )
        for x in range(
            0,
            image_width - window_width + 1,
            stride_x,
        )
    ]


def ensure_writable(
    path: Path,
    *,
    overwrite: bool,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output exists and overwrite is disabled: {path}"
        )


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
    *,
    overwrite: bool,
) -> None:
    ensure_writable(
        path,
        overwrite=overwrite,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(fieldnames),
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field_name: row.get(
                        field_name,
                        "",
                    )
                    for field_name in fieldnames
                }
            )


def calculate_statistics(
    rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    for feature_name in feature_names:
        values = np.asarray(
            [
                float(row[feature_name])
                for row in rows
            ],
            dtype=np.float64,
        )

        finite_values = values[
            np.isfinite(values)
        ]

        if finite_values.size == 0:
            raise ValueError(
                f"No finite values found for '{feature_name}'."
            )

        output.append(
            {
                "feature": feature_name,
                "count": int(values.size),
                "finite_count": int(
                    finite_values.size
                ),
                "minimum": float(
                    np.min(finite_values)
                ),
                "q01": float(
                    np.quantile(
                        finite_values,
                        0.01,
                    )
                ),
                "q25": float(
                    np.quantile(
                        finite_values,
                        0.25,
                    )
                ),
                "median": float(
                    np.quantile(
                        finite_values,
                        0.50,
                    )
                ),
                "mean": float(
                    np.mean(finite_values)
                ),
                "q75": float(
                    np.quantile(
                        finite_values,
                        0.75,
                    )
                ),
                "q99": float(
                    np.quantile(
                        finite_values,
                        0.99,
                    )
                ),
                "maximum": float(
                    np.max(finite_values)
                ),
                "standard_deviation": float(
                    np.std(finite_values)
                ),
                "unique_values": int(
                    np.unique(finite_values).size
                ),
            }
        )

    return output


def build_value_matrix(
    rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
) -> np.ndarray:
    return np.asarray(
        [
            [
                float(row[feature_name])
                for feature_name in feature_names
            ]
            for row in rows
        ],
        dtype=np.float64,
    )


def correlation_matrix(
    rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    *,
    method: str,
) -> np.ndarray:
    matrix = build_value_matrix(
        rows,
        feature_names,
    )

    if matrix.shape[1] == 1:
        return np.asarray(
            [[1.0]],
            dtype=np.float64,
        )

    if method == "pearson":
        prepared_matrix = matrix
    elif method == "spearman":
        prepared_matrix = np.column_stack(
            [
                rankdata(
                    matrix[:, column_index],
                    method="average",
                )
                for column_index in range(
                    matrix.shape[1]
                )
            ]
        )
    else:
        raise ValueError(
            f"Unsupported correlation method: {method}"
        )

    return np.atleast_2d(
        np.corrcoef(
            prepared_matrix,
            rowvar=False,
        )
    )


def correlation_rows(
    matrix: np.ndarray,
    feature_names: Sequence[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    for row_index, feature_name in enumerate(
        feature_names
    ):
        row: dict[str, Any] = {
            "feature": feature_name,
        }

        for column_index, column_name in enumerate(
            feature_names
        ):
            value = matrix[
                row_index,
                column_index,
            ]

            row[column_name] = (
                float(value)
                if np.isfinite(value)
                else ""
            )

        output.append(row)

    return output


def high_correlation_pairs(
    matrix: np.ndarray,
    feature_names: Sequence[str],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    for first_index in range(
        len(feature_names)
    ):
        for second_index in range(
            first_index + 1,
            len(feature_names),
        ):
            value = matrix[
                first_index,
                second_index,
            ]

            if (
                np.isfinite(value)
                and abs(value) >= threshold
            ):
                output.append(
                    {
                        "feature_a": feature_names[
                            first_index
                        ],
                        "feature_b": feature_names[
                            second_index
                        ],
                        "correlation": float(value),
                        "absolute_correlation": float(
                            abs(value)
                        ),
                    }
                )

    return sorted(
        output,
        key=lambda item: item[
            "absolute_correlation"
        ],
        reverse=True,
    )


def apply_normalization_and_fusion(
    rows: Sequence[Mapping[str, Any]],
    *,
    enabled_features: Sequence[str],
    bounds,
    feature_items,
    fusion_settings,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    for input_row in rows:
        normalized = normalize_feature_values(
            input_row,
            bounds,
            feature_items,
        )

        group_scores, combined_score = (
            fuse_normalized_features(
                normalized,
                fusion_settings,
            )
        )

        row = dict(input_row)

        for feature_name in enabled_features:
            row[
                f"normalized_{feature_name}"
            ] = normalized[feature_name]

        for group_name, group_score in (
            group_scores.items()
        ):
            row[
                f"group_{group_name}"
            ] = group_score

        row["combined_score"] = combined_score

        output.append(row)

    return output


def main() -> None:
    arguments = parse_arguments()

    loaded = load_configuration(
        arguments.config,
        create_generated_directories=True,
        validate_dataset_directories=True,
    )

    settings = loaded.settings

    review = loaded.raw.get(
        "feature_review",
        {},
    )

    if not isinstance(review, Mapping):
        raise ValueError(
            "'feature_review' must be a mapping."
        )

    split_name = (
        arguments.split
        or str(
            review.get(
                "split",
                "train",
            )
        )
    )

    sample_images = (
        arguments.sample_images
        if arguments.sample_images is not None
        else int(
            review.get(
                "sample_images",
                100,
            )
        )
    )

    patches_per_image = (
        arguments.patches_per_image
        if arguments.patches_per_image is not None
        else int(
            review.get(
                "patches_per_image",
                10,
            )
        )
    )

    include_disabled = (
        not arguments.enabled_only
        and bool(
            review.get(
                "include_disabled_features",
                True,
            )
        )
    )

    overwrite = (
        arguments.overwrite
        or bool(
            review.get(
                "overwrite",
                False,
            )
        )
    )

    high_correlation_threshold = float(
        review.get(
            "high_correlation_threshold",
            0.85,
        )
    )

    if sample_images <= 0:
        raise ValueError(
            "sample-images must be greater than zero."
        )

    if patches_per_image <= 0:
        raise ValueError(
            "patches-per-image must be greater than zero."
        )

    if not 0 <= high_correlation_threshold <= 1:
        raise ValueError(
            "high_correlation_threshold must be between zero and one."
        )

    set_global_seed(
        settings.project.random_seed,
        deterministic=settings.runtime.deterministic,
    )

    pipeline = FeaturePipeline.from_loaded_configuration(
        loaded
    )

    fusion_settings = (
        GroupedFusionSettings
        .from_loaded_configuration(
            loaded
        )
    )

    reviewed_feature_names = pipeline.feature_names(
        include_disabled=include_disabled
    )

    enabled_feature_names = pipeline.feature_names(
        include_disabled=False
    )

    group_names = tuple(
        group.name
        for group in fusion_settings.groups
    )

    output_directory = loaded.paths.output_directory(
        "feature_statistics"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_directory = loaded.paths.dataset_split(
        split_name
    )

    image_paths = list_images(
        split_directory,
        recursive=settings.dataset.recursive,
        extensions=settings.dataset.allowed_extensions,
    )

    if not image_paths:
        raise RuntimeError(
            f"No images found in {split_directory}"
        )

    generator = np.random.default_rng(
        settings.project.random_seed
    )

    number_to_sample = min(
        sample_images,
        len(image_paths),
    )

    selected_indices = generator.choice(
        len(image_paths),
        size=number_to_sample,
        replace=False,
    )

    selected_images = [
        image_paths[int(index)]
        for index in sorted(
            selected_indices
        )
    ]

    positions = sliding_positions(
        image_width=settings.dataset.image_size.width,
        image_height=settings.dataset.image_size.height,
        window_width=settings.sliding_window.width,
        window_height=settings.sliding_window.height,
        stride_x=settings.sliding_window.stride.x,
        stride_y=settings.sliding_window.stride.y,
    )

    real_rows: list[dict[str, Any]] = []

    for image_index, image_path in enumerate(
        selected_images,
        start=1,
    ):
        with Image.open(image_path) as image:
            image_array = np.asarray(
                image.convert("RGB"),
                dtype=np.uint8,
            )

        number_of_positions = min(
            patches_per_image,
            len(positions),
        )

        local_generator = np.random.default_rng(
            stable_seed(
                settings.project.random_seed,
                split_name,
                image_path.name,
            )
        )

        selected_position_indices = (
            local_generator.choice(
                len(positions),
                size=number_of_positions,
                replace=False,
            )
        )

        for position_index in sorted(
            selected_position_indices
        ):
            x, y = positions[
                int(position_index)
            ]

            patch = image_array[
                y:y + settings.sliding_window.height,
                x:x + settings.sliding_window.width,
            ]

            feature_values = pipeline.extract(
                patch,
                seed=stable_seed(
                    settings.project.random_seed,
                    split_name,
                    image_path.name,
                    x,
                    y,
                ),
                include_disabled=include_disabled,
            )

            real_rows.append(
                {
                    "split": split_name,
                    "image": image_path.name,
                    "x": x,
                    "y": y,
                    **feature_values,
                }
            )

        print(
            f"Reviewed {image_index}/"
            f"{len(selected_images)} images"
        )

    normalization = settings.features.normalization

    bounds = fit_robust_normalization_bounds(
        real_rows,
        enabled_feature_names,
        percentile_low=normalization.percentile_low,
        percentile_high=normalization.percentile_high,
        epsilon=normalization.epsilon,
    )

    scored_real_rows = apply_normalization_and_fusion(
        real_rows,
        enabled_features=enabled_feature_names,
        bounds=bounds,
        feature_items=settings.features.items,
        fusion_settings=fusion_settings,
    )

    synthetic_patches = create_synthetic_patches(
        width=settings.sliding_window.width,
        height=settings.sliding_window.height,
        seed=settings.project.random_seed,
    )

    synthetic_rows: list[dict[str, Any]] = []

    for patch_name, patch in synthetic_patches.items():
        feature_values = pipeline.extract(
            patch,
            seed=stable_seed(
                settings.project.random_seed,
                "synthetic",
                patch_name,
            ),
            include_disabled=include_disabled,
        )

        synthetic_rows.append(
            {
                "patch_name": patch_name,
                **feature_values,
            }
        )

    scored_synthetic_rows = (
        apply_normalization_and_fusion(
            synthetic_rows,
            enabled_features=enabled_feature_names,
            bounds=bounds,
            feature_items=settings.features.items,
            fusion_settings=fusion_settings,
        )
    )

    statistics_rows = calculate_statistics(
        real_rows,
        reviewed_feature_names,
    )

    pearson_matrix = correlation_matrix(
        real_rows,
        reviewed_feature_names,
        method="pearson",
    )

    spearman_matrix = correlation_matrix(
        real_rows,
        reviewed_feature_names,
        method="spearman",
    )

    enabled_spearman_matrix = correlation_matrix(
        real_rows,
        enabled_feature_names,
        method="spearman",
    )

    strong_enabled_correlations = (
        high_correlation_pairs(
            enabled_spearman_matrix,
            enabled_feature_names,
            threshold=high_correlation_threshold,
        )
    )

    synthetic_path = (
        output_directory
        / "synthetic_feature_review.csv"
    )

    sampled_features_path = (
        output_directory
        / "sampled_patch_features.csv"
    )

    sampled_scores_path = (
        output_directory
        / "sampled_patch_scores.csv"
    )

    statistics_path = (
        output_directory
        / "feature_statistics.csv"
    )

    pearson_path = (
        output_directory
        / "feature_correlations_pearson.csv"
    )

    spearman_path = (
        output_directory
        / "feature_correlations_spearman.csv"
    )

    normalization_path = (
        output_directory
        / "feature_normalization_preview.json"
    )

    definitions_path = (
        output_directory
        / "feature_definitions.json"
    )

    summary_path = (
        output_directory
        / "feature_review_summary.json"
    )

    score_columns = [
        *[
            f"normalized_{feature_name}"
            for feature_name in enabled_feature_names
        ],
        *[
            f"group_{group_name}"
            for group_name in group_names
        ],
        "combined_score",
    ]

    write_csv(
        synthetic_path,
        scored_synthetic_rows,
        [
            "patch_name",
            *reviewed_feature_names,
            *score_columns,
        ],
        overwrite=overwrite,
    )

    write_csv(
        sampled_features_path,
        real_rows,
        [
            "split",
            "image",
            "x",
            "y",
            *reviewed_feature_names,
        ],
        overwrite=overwrite,
    )

    write_csv(
        sampled_scores_path,
        scored_real_rows,
        [
            "split",
            "image",
            "x",
            "y",
            *reviewed_feature_names,
            *score_columns,
        ],
        overwrite=overwrite,
    )

    write_csv(
        statistics_path,
        statistics_rows,
        [
            "feature",
            "count",
            "finite_count",
            "minimum",
            "q01",
            "q25",
            "median",
            "mean",
            "q75",
            "q99",
            "maximum",
            "standard_deviation",
            "unique_values",
        ],
        overwrite=overwrite,
    )

    write_csv(
        pearson_path,
        correlation_rows(
            pearson_matrix,
            reviewed_feature_names,
        ),
        [
            "feature",
            *reviewed_feature_names,
        ],
        overwrite=overwrite,
    )

    write_csv(
        spearman_path,
        correlation_rows(
            spearman_matrix,
            reviewed_feature_names,
        ),
        [
            "feature",
            *reviewed_feature_names,
        ],
        overwrite=overwrite,
    )

    ensure_writable(
        normalization_path,
        overwrite=overwrite,
    )

    with normalization_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            {
                feature_name: asdict(
                    feature_bounds
                )
                for feature_name, feature_bounds
                in bounds.items()
            },
            output_file,
            indent=2,
        )

    ensure_writable(
        definitions_path,
        overwrite=overwrite,
    )

    with definitions_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            {
                "features": pipeline.definitions(
                    include_disabled=include_disabled
                ),
                "fusion": fusion_settings.to_dict(),
            },
            output_file,
            indent=2,
        )

    ensure_writable(
        summary_path,
        overwrite=overwrite,
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            {
                "split": split_name,
                "sample_images": len(
                    selected_images
                ),
                "patches_per_image": (
                    patches_per_image
                ),
                "sampled_patches": len(real_rows),
                "reviewed_features": list(
                    reviewed_feature_names
                ),
                "enabled_features": list(
                    enabled_feature_names
                ),
                "include_disabled": include_disabled,
                "normalization": {
                    "method": normalization.method,
                    "percentile_low": (
                        normalization.percentile_low
                    ),
                    "percentile_high": (
                        normalization.percentile_high
                    ),
                    "epsilon": normalization.epsilon,
                },
                "fusion": fusion_settings.to_dict(),
                "high_correlation_threshold": (
                    high_correlation_threshold
                ),
                "strong_enabled_spearman_correlations": (
                    strong_enabled_correlations
                ),
                "outputs": {
                    "synthetic": str(
                        synthetic_path
                    ),
                    "sampled_features": str(
                        sampled_features_path
                    ),
                    "sampled_scores": str(
                        sampled_scores_path
                    ),
                    "statistics": str(
                        statistics_path
                    ),
                    "pearson_correlations": str(
                        pearson_path
                    ),
                    "spearman_correlations": str(
                        spearman_path
                    ),
                    "normalization_preview": str(
                        normalization_path
                    ),
                    "definitions": str(
                        definitions_path
                    ),
                },
            },
            output_file,
            indent=2,
        )

    print("=" * 80)
    print("FEATURE REVIEW COMPLETED")
    print("=" * 80)
    print(
        "Enabled features: "
        + ", ".join(enabled_feature_names)
    )
    print(
        "Reviewed features: "
        + ", ".join(reviewed_feature_names)
    )
    print(
        f"Sampled images: {len(selected_images)}"
    )
    print(
        f"Sampled patches: {len(real_rows)}"
    )
    print(
        f"Strong enabled Spearman pairs: "
        f"{len(strong_enabled_correlations)}"
    )
    print(
        f"Reports: {output_directory}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()