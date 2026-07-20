from __future__ import annotations

import csv
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

from src.config.loader import LoadedConfiguration
from src.data.sliding_window import SlidingWindowGrid
from src.features.fusion import GroupedFusionSettings
from src.features.pipeline import FeaturePipeline
from src.labeling.normalization import (
    NormalizationArtifact,
    fit_exact_normalization,
    fuse_feature_maps,
    load_normalization_artifact,
    normalize_feature_maps,
    save_normalization_artifact,
)
from src.labeling.settings import (
    LabelGenerationSettings,
)
from src.labeling.storage import (
    LabelFileStatistics,
    inspect_label_file,
    save_label_file,
)
from src.utils.hashing import (
    canonical_json_hash,
    image_key_from_relative_path,
    stable_uint32_seed,
)


def list_split_images(
    split_directory: Path,
    *,
    recursive: bool,
    extensions: Sequence[str],
) -> list[Path]:
    candidates: Iterable[Path]

    if recursive:
        candidates = split_directory.rglob("*")
    else:
        candidates = split_directory.iterdir()

    allowed_extensions = {
        extension.lower()
        for extension in extensions
    }

    images = [
        path
        for path in candidates
        if (
            path.is_file()
            and path.suffix.lower()
            in allowed_extensions
        )
    ]

    return sorted(
        images,
        key=lambda path: (
            path.relative_to(split_directory)
            .as_posix()
            .lower()
        ),
    )


class SlidingWindowLabelGenerator:
    def __init__(
        self,
        loaded: LoadedConfiguration,
    ) -> None:
        self.loaded = loaded
        self.project = loaded.settings
        self.paths = loaded.paths

        self.settings = (
            LabelGenerationSettings
            .from_loaded_configuration(
                loaded
            )
        )

        self.pipeline = (
            FeaturePipeline
            .from_loaded_configuration(
                loaded
            )
        )

        self.fusion = (
            GroupedFusionSettings
            .from_loaded_configuration(
                loaded
            )
        )

        self.grid = SlidingWindowGrid.from_configuration(
            self.project
        )

        self.feature_names = self.pipeline.feature_names(
            include_disabled=False
        )

        self.raw_feature_fingerprint = (
            self._build_raw_feature_fingerprint()
        )

        self.settings.output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.settings.normalization.artifact_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.settings.normalization.raw_cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _build_raw_feature_fingerprint(self) -> str:
        enabled_items = {
            name: {
                "weight": item.weight,
                "direction": item.direction,
            }
            for name, item
            in self.project.features.items.items()
            if item.enabled
        }

        payload = {
            "image_size": {
                "width": (
                    self.project.dataset.image_size.width
                ),
                "height": (
                    self.project.dataset.image_size.height
                ),
                "channels": (
                    self.project.dataset.channels
                ),
                "color_mode": (
                    self.project.dataset.color_mode
                ),
            },
            "sliding_window": {
                "width": self.grid.window_width,
                "height": self.grid.window_height,
                "stride_x": self.grid.stride_x,
                "stride_y": self.grid.stride_y,
            },
            "features": enabled_items,
            "feature_extraction": self.loaded.raw.get(
                "feature_extraction",
                {},
            ),
        }

        return canonical_json_hash(payload)

    def _build_label_fingerprint(
        self,
        artifact: NormalizationArtifact,
    ) -> str:
        payload = {
            "raw_feature_fingerprint": (
                self.raw_feature_fingerprint
            ),
            "normalization": artifact.to_dict(),
            "fusion": self.fusion.to_dict(),
            "label_file_version": 1,
        }

        return canonical_json_hash(payload)

    def _load_image(
        self,
        path: Path,
    ) -> np.ndarray:
        with Image.open(path) as image:
            if self.project.dataset.color_mode == "rgb":
                image = image.convert("RGB")
            else:
                image = image.convert("L")

            image_array = np.asarray(
                image,
                dtype=np.uint8,
            )

        self.grid.validate_image(
            image_array
        )

        return image_array

    def _image_source_state(
        self,
        image_path: Path,
    ) -> dict[str, int]:
        stat = image_path.stat()

        return {
            "size_bytes": int(
                stat.st_size
            ),
            "modified_time_ns": int(
                stat.st_mtime_ns
            ),
        }

    def _cache_path(
        self,
        relative_path: Path,
    ) -> Path:
        image_key = image_key_from_relative_path(
            relative_path
        )

        return (
            self.settings.normalization.raw_cache_dir
            / f"{image_key}.npz"
        )

    def _label_path(
        self,
        split_name: str,
        relative_path: Path,
    ) -> Path:
        image_key = image_key_from_relative_path(
            relative_path
        )

        return (
            self.settings.output_root
            / split_name
            / f"{image_key}{self.project.labels.filename_suffix}"
        )

    def _extract_raw_feature_maps(
        self,
        *,
        split_name: str,
        split_directory: Path,
        image_path: Path,
    ) -> tuple[
        Path,
        str,
        np.ndarray,
        dict[str, int],
    ]:
        relative_path = image_path.relative_to(
            split_directory
        )

        source_state = self._image_source_state(
            image_path
        )

        image_array = self._load_image(
            image_path
        )

        raw_maps = np.empty(
            (
                len(self.feature_names),
                self.grid.grid_height,
                self.grid.grid_width,
            ),
            dtype=np.float32,
        )

        for location in self.grid.locations():
            patch = self.grid.extract(
                image_array,
                location,
            )

            seed = stable_uint32_seed(
                self.project.project.random_seed,
                split_name,
                relative_path.as_posix(),
                location.row,
                location.column,
                location.x,
                location.y,
            )

            feature_values = self.pipeline.extract(
                patch,
                seed=seed,
                include_disabled=False,
            )

            for feature_index, feature_name in enumerate(
                self.feature_names
            ):
                raw_maps[
                    feature_index,
                    location.row,
                    location.column,
                ] = feature_values[
                    feature_name
                ]

        if (
            self.settings.integrity.fail_on_non_finite
            and not np.all(np.isfinite(raw_maps))
        ):
            raise ValueError(
                f"Non-finite feature value generated for "
                f"{image_path}."
            )

        return (
            relative_path,
            image_key_from_relative_path(
                relative_path
            ),
            raw_maps,
            source_state,
        )

    def _save_raw_cache(
        self,
        *,
        cache_path: Path,
        relative_path: Path,
        raw_maps: np.ndarray,
        source_state: dict[str, int],
    ) -> None:
        metadata = {
            "relative_path": relative_path.as_posix(),
            "source_state": source_state,
            "raw_feature_fingerprint": (
                self.raw_feature_fingerprint
            ),
            "feature_names": list(
                self.feature_names
            ),
            "shape": list(
                raw_maps.shape
            ),
        }

        cache_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = cache_path.with_suffix(
            cache_path.suffix + ".tmp"
        )

        with temporary_path.open("wb") as output_file:
            np.savez_compressed(
                output_file,
                raw_feature_maps=raw_maps,
                metadata_json=np.asarray(
                    json.dumps(
                        metadata,
                        sort_keys=True,
                    ),
                    dtype=np.str_,
                ),
            )

        temporary_path.replace(
            cache_path
        )

    def _load_raw_cache(
        self,
        *,
        cache_path: Path,
        relative_path: Path,
        source_state: dict[str, int],
    ) -> np.ndarray | None:
        if not cache_path.is_file():
            return None

        try:
            with np.load(
                cache_path,
                allow_pickle=False,
            ) as cache_file:
                raw_maps = cache_file[
                    "raw_feature_maps"
                ]

                metadata = json.loads(
                    str(
                        cache_file[
                            "metadata_json"
                        ].item()
                    )
                )

            if metadata.get(
                "relative_path"
            ) != relative_path.as_posix():
                return None

            if metadata.get(
                "source_state"
            ) != source_state:
                return None

            if metadata.get(
                "raw_feature_fingerprint"
            ) != self.raw_feature_fingerprint:
                return None

            if tuple(
                metadata.get(
                    "feature_names",
                    [],
                )
            ) != self.feature_names:
                return None

            expected_shape = (
                len(self.feature_names),
                self.grid.grid_height,
                self.grid.grid_width,
            )

            if raw_maps.shape != expected_shape:
                return None

            if not np.all(
                np.isfinite(raw_maps)
            ):
                return None

            return raw_maps.astype(
                np.float32,
                copy=False,
            )

        except (
            OSError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
        ):
            return None

    def _get_training_raw_features(
        self,
        *,
        split_directory: Path,
        image_path: Path,
    ) -> tuple[
        Path,
        str,
        np.ndarray,
        dict[str, int],
    ]:
        relative_path = image_path.relative_to(
            split_directory
        )

        source_state = self._image_source_state(
            image_path
        )

        cache_path = self._cache_path(
            relative_path
        )

        if (
            self.settings.normalization
            .cache_training_features
        ):
            cached_maps = self._load_raw_cache(
                cache_path=cache_path,
                relative_path=relative_path,
                source_state=source_state,
            )

            if cached_maps is not None:
                return (
                    relative_path,
                    image_key_from_relative_path(
                        relative_path
                    ),
                    cached_maps,
                    source_state,
                )

        result = self._extract_raw_feature_maps(
            split_name="train",
            split_directory=split_directory,
            image_path=image_path,
        )

        if (
            self.settings.normalization
            .cache_training_features
        ):
            self._save_raw_cache(
                cache_path=cache_path,
                relative_path=result[0],
                raw_maps=result[2],
                source_state=result[3],
            )

        return result

    def _max_workers(self) -> int | None:
        workers = (
            self.settings.execution.max_workers
        )

        return None if workers == 0 else workers

    def fit_normalization(
        self,
        *,
        force: bool = False,
    ) -> NormalizationArtifact:
        artifact_path = (
            self.settings.normalization.artifact_path
        )

        if (
            artifact_path.is_file()
            and self.settings.normalization
            .reuse_existing_artifact
            and not force
        ):
            artifact = load_normalization_artifact(
                artifact_path
            )

            if (
                artifact.raw_feature_fingerprint
                == self.raw_feature_fingerprint
                and artifact.feature_names
                == self.feature_names
            ):
                print(
                    "Using existing compatible normalization "
                    f"artifact: {artifact_path}"
                )

                return artifact

            print(
                "Existing normalization artifact is incompatible "
                "and will be regenerated."
            )

        split_directory = self.paths.dataset_split(
            "train"
        )

        image_paths = list_split_images(
            split_directory,
            recursive=self.project.dataset.recursive,
            extensions=(
                self.project.dataset.allowed_extensions
            ),
        )

        if not image_paths:
            raise RuntimeError(
                "No training images found."
            )

        total_rows = (
            len(image_paths)
            * self.grid.windows_per_image
        )

        feature_count = len(
            self.feature_names
        )

        matrix_path = (
            self.settings.normalization.feature_matrix_path
        )

        matrix_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if matrix_path.exists():
            matrix_path.unlink()

        feature_matrix = np.memmap(
            matrix_path,
            mode="w+",
            dtype=np.float32,
            shape=(
                total_rows,
                feature_count,
            ),
        )

        progress_interval = (
            self.settings.execution.progress_interval
        )

        row_offset = 0

        with ThreadPoolExecutor(
            max_workers=self._max_workers()
        ) as executor:
            results = executor.map(
                lambda path: (
                    self._get_training_raw_features(
                        split_directory=split_directory,
                        image_path=path,
                    )
                ),
                image_paths,
            )

            for image_index, result in enumerate(
                results,
                start=1,
            ):
                raw_maps = result[2]

                patch_matrix = (
                    raw_maps
                    .reshape(
                        feature_count,
                        -1,
                    )
                    .T
                )

                next_offset = (
                    row_offset
                    + patch_matrix.shape[0]
                )

                feature_matrix[
                    row_offset:next_offset
                ] = patch_matrix

                row_offset = next_offset

                if (
                    progress_interval > 0
                    and (
                        image_index
                        % progress_interval
                        == 0
                        or image_index
                        == len(image_paths)
                    )
                ):
                    print(
                        "[normalization] "
                        f"{image_index}/{len(image_paths)} "
                        "training images processed"
                    )

        if row_offset != total_rows:
            raise RuntimeError(
                f"Expected {total_rows} feature rows; "
                f"generated {row_offset}."
            )

        feature_matrix.flush()

        normalization = (
            self.project.features.normalization
        )

        artifact = fit_exact_normalization(
            feature_matrix,
            self.feature_names,
            image_count=len(image_paths),
            percentile_low=(
                normalization.percentile_low
            ),
            percentile_high=(
                normalization.percentile_high
            ),
            epsilon=normalization.epsilon,
            raw_feature_fingerprint=(
                self.raw_feature_fingerprint
            ),
        )

        save_normalization_artifact(
            artifact,
            artifact_path,
        )

        del feature_matrix

        if (
            self.settings.normalization
            .delete_feature_matrix_after_fit
            and matrix_path.exists()
        ):
            matrix_path.unlink()

        print(
            "Normalization artifact saved to: "
            f"{artifact_path}"
        )

        return artifact

    def _build_label_metadata(
        self,
        *,
        split_name: str,
        relative_path: Path,
        source_state: dict[str, int],
        artifact: NormalizationArtifact,
        label_fingerprint: str,
    ) -> dict[str, Any]:
        return {
            "split": split_name,
            "source_relative_path": (
                relative_path.as_posix()
            ),
            "source_state": source_state,
            "image_width": self.grid.image_width,
            "image_height": self.grid.image_height,
            "window_width": self.grid.window_width,
            "window_height": self.grid.window_height,
            "stride_x": self.grid.stride_x,
            "stride_y": self.grid.stride_y,
            "grid_width": self.grid.grid_width,
            "grid_height": self.grid.grid_height,
            "feature_names": list(
                self.feature_names
            ),
            "normalization_artifact": str(
                self.settings.normalization
                .artifact_path
            ),
            "normalization_fingerprint": (
                canonical_json_hash(
                    artifact.to_dict()
                )
            ),
            "raw_feature_fingerprint": (
                self.raw_feature_fingerprint
            ),
            "label_fingerprint": (
                label_fingerprint
            ),
            "fusion": self.fusion.to_dict(),
        }

    def _generate_one_label(
        self,
        *,
        split_name: str,
        split_directory: Path,
        image_path: Path,
        artifact: NormalizationArtifact,
        label_fingerprint: str,
    ) -> dict[str, Any]:
        relative_path = image_path.relative_to(
            split_directory
        )

        image_key = image_key_from_relative_path(
            relative_path
        )

        output_path = self._label_path(
            split_name,
            relative_path,
        )

        if output_path.exists():
            if self.settings.execution.overwrite_existing:
                output_path.unlink()

            elif self.settings.execution.resume:
                try:
                    _, statistics = inspect_label_file(
                        output_path,
                        expected_score_shape=(
                            self.grid.score_shape
                        ),
                        expected_feature_names=(
                            self.feature_names
                        ),
                        expected_label_fingerprint=(
                            label_fingerprint
                        ),
                        score_tolerance=(
                            self.settings.integrity
                            .score_tolerance
                        ),
                    )

                    return self._manifest_row(
                        split_name=split_name,
                        relative_path=relative_path,
                        image_key=image_key,
                        label_path=output_path,
                        status="reused",
                        statistics=statistics,
                    )

                except (
                    OSError,
                    ValueError,
                    KeyError,
                ):
                    output_path.unlink()

            else:
                raise FileExistsError(
                    f"Label already exists: {output_path}"
                )

        source_state = self._image_source_state(
            image_path
        )

        if split_name == "train":
            result = self._get_training_raw_features(
                split_directory=split_directory,
                image_path=image_path,
            )

        else:
            result = self._extract_raw_feature_maps(
                split_name=split_name,
                split_directory=split_directory,
                image_path=image_path,
            )

        raw_maps = result[2]

        normalized_maps = normalize_feature_maps(
            raw_maps,
            artifact,
            self.project.features.items,
        )

        (
            group_maps,
            group_names,
            score_map,
        ) = fuse_feature_maps(
            normalized_maps,
            self.feature_names,
            self.fusion,
        )

        if (
            self.settings.integrity.fail_on_non_finite
            and (
                not np.all(
                    np.isfinite(normalized_maps)
                )
                or not np.all(
                    np.isfinite(group_maps)
                )
                or not np.all(
                    np.isfinite(score_map)
                )
            )
        ):
            raise ValueError(
                f"Non-finite label value generated for "
                f"{image_path}."
            )

        metadata = self._build_label_metadata(
            split_name=split_name,
            relative_path=relative_path,
            source_state=source_state,
            artifact=artifact,
            label_fingerprint=label_fingerprint,
        )

        save_label_file(
            output_path,
            score_map=score_map,
            feature_names=self.feature_names,
            group_names=group_names,
            metadata=metadata,
            compressed=(
                self.settings.storage.compressed
            ),
            raw_feature_maps=(
                raw_maps
                if (
                    self.settings.storage
                    .save_raw_feature_maps
                )
                else None
            ),
            normalized_feature_maps=(
                normalized_maps
                if (
                    self.settings.storage
                    .save_normalized_feature_maps
                )
                else None
            ),
            group_score_maps=(
                group_maps
                if (
                    self.settings.storage
                    .save_group_score_maps
                )
                else None
            ),
            window_coordinates=(
                self.grid.coordinate_array()
                if (
                    self.settings.storage
                    .save_window_coordinates
                )
                else None
            ),
        )

        if (
            self.settings.integrity
            .verify_after_write
        ):
            _, statistics = inspect_label_file(
                output_path,
                expected_score_shape=(
                    self.grid.score_shape
                ),
                expected_feature_names=(
                    self.feature_names
                ),
                expected_label_fingerprint=(
                    label_fingerprint
                ),
                score_tolerance=(
                    self.settings.integrity
                    .score_tolerance
                ),
            )

        else:
            score_values = score_map.astype(
                np.float64
            )

            statistics = LabelFileStatistics(
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
                    np.sum(
                        score_values
                        * score_values
                    )
                ),
                score_count=int(
                    score_values.size
                ),
            )

        return self._manifest_row(
            split_name=split_name,
            relative_path=relative_path,
            image_key=image_key,
            label_path=output_path,
            status="generated",
            statistics=statistics,
        )

    def _manifest_row(
        self,
        *,
        split_name: str,
        relative_path: Path,
        image_key: str,
        label_path: Path,
        status: str,
        statistics: LabelFileStatistics,
    ) -> dict[str, Any]:
        return {
            "split": split_name,
            "image_key": image_key,
            "source_relative_path": (
                relative_path.as_posix()
            ),
            "label_path": str(
                label_path.resolve()
            ),
            "status": status,
            "grid_height": (
                self.grid.grid_height
            ),
            "grid_width": (
                self.grid.grid_width
            ),
            "score_min": (
                statistics.score_min
            ),
            "score_max": (
                statistics.score_max
            ),
            "score_mean": (
                statistics.score_mean
            ),
            "score_standard_deviation": (
                statistics
                .score_standard_deviation
            ),
            "score_sum": (
                statistics.score_sum
            ),
            "score_squared_sum": (
                statistics.score_squared_sum
            ),
            "score_count": (
                statistics.score_count
            ),
        }

    def _write_split_reports(
        self,
        *,
        split_name: str,
        rows: list[dict[str, Any]],
        label_fingerprint: str,
    ) -> dict[str, Any]:
        split_output_directory = (
            self.settings.output_root
            / split_name
        )

        split_output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        manifest_path = (
            split_output_directory
            / self.settings.reports
            .manifest_filename
        )

        summary_path = (
            split_output_directory
            / self.settings.reports
            .summary_filename
        )

        manifest_fields = [
            "split",
            "image_key",
            "source_relative_path",
            "label_path",
            "status",
            "grid_height",
            "grid_width",
            "score_min",
            "score_max",
            "score_mean",
            "score_standard_deviation",
            "score_sum",
            "score_squared_sum",
            "score_count",
        ]

        temporary_manifest = (
            manifest_path.with_suffix(
                manifest_path.suffix + ".tmp"
            )
        )

        with temporary_manifest.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=manifest_fields,
            )

            writer.writeheader()
            writer.writerows(rows)

        temporary_manifest.replace(
            manifest_path
        )

        total_count = sum(
            int(row["score_count"])
            for row in rows
        )

        total_sum = sum(
            float(row["score_sum"])
            for row in rows
        )

        total_squared_sum = sum(
            float(
                row["score_squared_sum"]
            )
            for row in rows
        )

        global_mean = (
            total_sum / total_count
        )

        global_variance = max(
            (
                total_squared_sum
                / total_count
            )
            - global_mean * global_mean,
            0.0,
        )

        summary = {
            "split": split_name,
            "image_count": len(rows),
            "generated_count": sum(
                row["status"] == "generated"
                for row in rows
            ),
            "reused_count": sum(
                row["status"] == "reused"
                for row in rows
            ),
            "windows_per_image": (
                self.grid.windows_per_image
            ),
            "total_window_scores": (
                total_count
            ),
            "score_min": min(
                float(row["score_min"])
                for row in rows
            ),
            "score_max": max(
                float(row["score_max"])
                for row in rows
            ),
            "score_mean": global_mean,
            "score_standard_deviation": (
                global_variance ** 0.5
            ),
            "feature_names": list(
                self.feature_names
            ),
            "label_fingerprint": (
                label_fingerprint
            ),
            "manifest_path": str(
                manifest_path.resolve()
            ),
        }

        temporary_summary = (
            summary_path.with_suffix(
                summary_path.suffix + ".tmp"
            )
        )

        with temporary_summary.open(
            "w",
            encoding="utf-8",
        ) as output_file:
            json.dump(
                summary,
                output_file,
                indent=2,
            )

        temporary_summary.replace(
            summary_path
        )

        return summary

    def generate_split(
        self,
        split_name: str,
        artifact: NormalizationArtifact,
    ) -> dict[str, Any]:
        if split_name not in {
            "train",
            "validation",
            "test",
        }:
            raise ValueError(
                f"Unsupported dataset split: {split_name}"
            )

        split_directory = self.paths.dataset_split(
            split_name
        )

        image_paths = list_split_images(
            split_directory,
            recursive=self.project.dataset.recursive,
            extensions=(
                self.project.dataset.allowed_extensions
            ),
        )

        if not image_paths:
            raise RuntimeError(
                f"No images found for split '{split_name}'."
            )

        label_fingerprint = (
            self._build_label_fingerprint(
                artifact
            )
        )

        rows: list[dict[str, Any]] = []

        progress_interval = (
            self.settings.execution
            .progress_interval
        )

        with ThreadPoolExecutor(
            max_workers=self._max_workers()
        ) as executor:
            results = executor.map(
                lambda path: self._generate_one_label(
                    split_name=split_name,
                    split_directory=split_directory,
                    image_path=path,
                    artifact=artifact,
                    label_fingerprint=(
                        label_fingerprint
                    ),
                ),
                image_paths,
            )

            for image_index, row in enumerate(
                results,
                start=1,
            ):
                rows.append(row)

                if (
                    progress_interval > 0
                    and (
                        image_index
                        % progress_interval
                        == 0
                        or image_index
                        == len(image_paths)
                    )
                ):
                    print(
                        f"[{split_name}] "
                        f"{image_index}/{len(image_paths)} "
                        "label files processed"
                    )

        return self._write_split_reports(
            split_name=split_name,
            rows=rows,
            label_fingerprint=label_fingerprint,
        )

    def run(
        self,
        *,
        splits: Sequence[str] = (
            "train",
            "validation",
            "test",
        ),
        force_normalization: bool = False,
    ) -> dict[str, Any]:
        artifact = self.fit_normalization(
            force=force_normalization
        )

        split_summaries: dict[str, Any] = {}

        for split_name in splits:
            split_summaries[split_name] = (
                self.generate_split(
                    split_name,
                    artifact,
                )
            )

        global_summary = {
            "normalization_artifact": (
                str(
                    self.settings.normalization
                    .artifact_path
                )
            ),
            "normalization": (
                artifact.to_dict()
            ),
            "feature_names": list(
                self.feature_names
            ),
            "fusion": self.fusion.to_dict(),
            "grid": {
                "image_width": (
                    self.grid.image_width
                ),
                "image_height": (
                    self.grid.image_height
                ),
                "window_width": (
                    self.grid.window_width
                ),
                "window_height": (
                    self.grid.window_height
                ),
                "stride_x": (
                    self.grid.stride_x
                ),
                "stride_y": (
                    self.grid.stride_y
                ),
                "grid_width": (
                    self.grid.grid_width
                ),
                "grid_height": (
                    self.grid.grid_height
                ),
                "windows_per_image": (
                    self.grid.windows_per_image
                ),
            },
            "splits": split_summaries,
        }

        global_summary_path = (
            self.settings.reports
            .global_summary_path
        )

        global_summary_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = (
            global_summary_path.with_suffix(
                global_summary_path.suffix
                + ".tmp"
            )
        )

        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as output_file:
            json.dump(
                global_summary,
                output_file,
                indent=2,
            )

        temporary_path.replace(
            global_summary_path
        )

        if (
            self.settings.normalization
            .delete_raw_cache_after_success
            and self.settings.normalization
            .raw_cache_dir.exists()
        ):
            shutil.rmtree(
                self.settings.normalization
                .raw_cache_dir
            )

        return global_summary