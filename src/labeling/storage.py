from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


LABEL_FILE_VERSION = 1


@dataclass(frozen=True)
class LabelFileStatistics:
    score_min: float
    score_max: float
    score_mean: float
    score_standard_deviation: float
    score_sum: float
    score_squared_sum: float
    score_count: int


def calculate_label_statistics(
    score_map: np.ndarray,
) -> LabelFileStatistics:
    score_values = score_map.astype(
        np.float64,
        copy=False,
    )

    return LabelFileStatistics(
        score_min=float(
            np.min(score_values)
        ),
        score_max=float(
            np.max(score_values)
        ),
        score_mean=float(
            np.mean(score_values)
        ),
        score_standard_deviation=float(
            np.std(score_values)
        ),
        score_sum=float(
            np.sum(score_values)
        ),
        score_squared_sum=float(
            np.sum(score_values * score_values)
        ),
        score_count=int(
            score_values.size
        ),
    )


def save_label_file(
    path: Path,
    *,
    score_map: np.ndarray,
    feature_names: tuple[str, ...],
    group_names: tuple[str, ...],
    metadata: Mapping[str, Any],
    compressed: bool,
    raw_feature_maps: np.ndarray | None = None,
    normalized_feature_maps: np.ndarray | None = None,
    group_score_maps: np.ndarray | None = None,
    window_coordinates: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    arrays: dict[str, np.ndarray] = {
        "label_file_version": np.asarray(
            LABEL_FILE_VERSION,
            dtype=np.int32,
        ),
        "score_map": score_map.astype(
            np.float32,
            copy=False,
        ),
        "feature_names": np.asarray(
            feature_names,
            dtype=np.str_,
        ),
        "group_names": np.asarray(
            group_names,
            dtype=np.str_,
        ),
        "metadata_json": np.asarray(
            json.dumps(
                dict(metadata),
                sort_keys=True,
                ensure_ascii=False,
            ),
            dtype=np.str_,
        ),
    }

    if raw_feature_maps is not None:
        arrays["raw_feature_maps"] = (
            raw_feature_maps.astype(
                np.float32,
                copy=False,
            )
        )

    if normalized_feature_maps is not None:
        arrays["normalized_feature_maps"] = (
            normalized_feature_maps.astype(
                np.float32,
                copy=False,
            )
        )

    if group_score_maps is not None:
        arrays["group_score_maps"] = (
            group_score_maps.astype(
                np.float32,
                copy=False,
            )
        )

    if window_coordinates is not None:
        arrays["window_coordinates"] = (
            window_coordinates.astype(
                np.int32,
                copy=False,
            )
        )

    with temporary_path.open("wb") as output_file:
        if compressed:
            np.savez_compressed(
                output_file,
                **arrays,
            )
        else:
            np.savez(
                output_file,
                **arrays,
            )

    temporary_path.replace(path)


def inspect_label_file(
    path: Path,
    *,
    expected_score_shape: tuple[int, int],
    expected_feature_names: tuple[str, ...],
    expected_label_fingerprint: str | None = None,
    score_tolerance: float = 1e-6,
) -> tuple[dict[str, Any], LabelFileStatistics]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Label file not found: {path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as label_file:
        version = int(
            label_file["label_file_version"]
        )

        if version != LABEL_FILE_VERSION:
            raise ValueError(
                f"Unsupported label file version: {version}"
            )

        score_map = label_file["score_map"]

        if score_map.shape != expected_score_shape:
            raise ValueError(
                f"Expected score-map shape "
                f"{expected_score_shape}; found "
                f"{score_map.shape}."
            )

        if not np.all(np.isfinite(score_map)):
            raise ValueError(
                "Score map contains non-finite values."
            )

        if float(np.min(score_map)) < -score_tolerance:
            raise ValueError(
                "Score map contains values below zero."
            )

        if float(np.max(score_map)) > 1.0 + score_tolerance:
            raise ValueError(
                "Score map contains values above one."
            )

        feature_names = tuple(
            str(value)
            for value in label_file[
                "feature_names"
            ].tolist()
        )

        if feature_names != expected_feature_names:
            raise ValueError(
                "Label feature names do not match the current "
                "configuration."
            )

        metadata = json.loads(
            str(
                label_file[
                    "metadata_json"
                ].item()
            )
        )

        if expected_label_fingerprint is not None:
            actual_fingerprint = metadata.get(
                "label_fingerprint"
            )

            if actual_fingerprint != expected_label_fingerprint:
                raise ValueError(
                    "Label file was produced by a different "
                    "configuration or normalization artifact."
                )

        statistics = calculate_label_statistics(
            score_map
        )

    return metadata, statistics