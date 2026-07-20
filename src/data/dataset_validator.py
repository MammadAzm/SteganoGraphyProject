from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, UnidentifiedImageError

from src.config.loader import LoadedConfiguration
from src.config.schema import ConfigurationError


VALID_SEVERITIES = {
    "ignore",
    "info",
    "warning",
    "error",
}

VALID_CONTENT_HASH_METHODS = {
    "file",
    "pixels",
}


def _require_mapping(
    value: Any,
    field_name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(
            f"Configuration field '{field_name}' must be a mapping."
        )

    return value


def _optional_mapping(
    value: Any,
    field_name: str,
) -> Mapping[str, Any]:
    if value is None:
        return {}

    return _require_mapping(value, field_name)


def _parse_boolean(
    value: Any,
    field_name: str,
) -> bool:
    if isinstance(value, bool):
        return value

    raise ConfigurationError(
        f"Configuration field '{field_name}' must be boolean."
    )


def _parse_optional_positive_integer(
    value: Any,
    field_name: str,
) -> int | None:
    if value is None:
        return None

    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ConfigurationError(
            f"Configuration field '{field_name}' must be null "
            "or a positive integer."
        )

    return value


def _parse_non_negative_integer(
    value: Any,
    field_name: str,
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        raise ConfigurationError(
            f"Configuration field '{field_name}' must be a "
            "non-negative integer."
        )

    return value


def _parse_severity(
    value: Any,
    field_name: str,
) -> str:
    severity = str(value).strip().lower()

    if severity not in VALID_SEVERITIES:
        allowed = ", ".join(sorted(VALID_SEVERITIES))

        raise ConfigurationError(
            f"Configuration field '{field_name}' has invalid "
            f"severity '{severity}'. Allowed values: {allowed}."
        )

    return severity


@dataclass(frozen=True)
class ValidationCheckSettings:
    dimensions: bool = True
    color_mode: bool = True
    channels: bool = True
    full_decode: bool = True

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> "ValidationCheckSettings":
        return cls(
            dimensions=_parse_boolean(
                data.get("dimensions", True),
                "dataset_validation.checks.dimensions",
            ),
            color_mode=_parse_boolean(
                data.get("color_mode", True),
                "dataset_validation.checks.color_mode",
            ),
            channels=_parse_boolean(
                data.get("channels", True),
                "dataset_validation.checks.channels",
            ),
            full_decode=_parse_boolean(
                data.get("full_decode", True),
                "dataset_validation.checks.full_decode",
            ),
        )


@dataclass(frozen=True)
class DuplicateCheckSettings:
    check_filenames: bool = True
    check_image_ids: bool = True
    check_content: bool = True
    content_hash: str = "pixels"
    within_split_severity: str = "warning"
    cross_split_severity: str = "error"

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> "DuplicateCheckSettings":
        content_hash = str(
            data.get("content_hash", "pixels")
        ).strip().lower()

        if content_hash not in VALID_CONTENT_HASH_METHODS:
            allowed = ", ".join(
                sorted(VALID_CONTENT_HASH_METHODS)
            )

            raise ConfigurationError(
                "'dataset_validation.duplicates.content_hash' "
                f"must be one of: {allowed}."
            )

        return cls(
            check_filenames=_parse_boolean(
                data.get("check_filenames", True),
                (
                    "dataset_validation.duplicates."
                    "check_filenames"
                ),
            ),
            check_image_ids=_parse_boolean(
                data.get("check_image_ids", True),
                (
                    "dataset_validation.duplicates."
                    "check_image_ids"
                ),
            ),
            check_content=_parse_boolean(
                data.get("check_content", True),
                (
                    "dataset_validation.duplicates."
                    "check_content"
                ),
            ),
            content_hash=content_hash,
            within_split_severity=_parse_severity(
                data.get(
                    "within_split_severity",
                    "warning",
                ),
                (
                    "dataset_validation.duplicates."
                    "within_split_severity"
                ),
            ),
            cross_split_severity=_parse_severity(
                data.get(
                    "cross_split_severity",
                    "error",
                ),
                (
                    "dataset_validation.duplicates."
                    "cross_split_severity"
                ),
            ),
        )


@dataclass(frozen=True)
class UnsupportedFileSettings:
    severity: str = "ignore"

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> "UnsupportedFileSettings":
        return cls(
            severity=_parse_severity(
                data.get("severity", "ignore"),
                (
                    "dataset_validation.unsupported_files."
                    "severity"
                ),
            )
        )


@dataclass(frozen=True)
class ReportSettings:
    save_manifest: bool = True
    save_issues: bool = True
    save_summary: bool = True
    overwrite: bool = True

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> "ReportSettings":
        return cls(
            save_manifest=_parse_boolean(
                data.get("save_manifest", True),
                (
                    "dataset_validation.reports."
                    "save_manifest"
                ),
            ),
            save_issues=_parse_boolean(
                data.get("save_issues", True),
                (
                    "dataset_validation.reports."
                    "save_issues"
                ),
            ),
            save_summary=_parse_boolean(
                data.get("save_summary", True),
                (
                    "dataset_validation.reports."
                    "save_summary"
                ),
            ),
            overwrite=_parse_boolean(
                data.get("overwrite", True),
                (
                    "dataset_validation.reports."
                    "overwrite"
                ),
            ),
        )


@dataclass(frozen=True)
class ValidationExecutionSettings:
    max_workers: int | None = None
    progress_interval: int = 500

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> "ValidationExecutionSettings":
        max_workers_value = data.get("max_workers")

        if max_workers_value is not None:
            max_workers_value = _parse_non_negative_integer(
                max_workers_value,
                (
                    "dataset_validation.execution."
                    "max_workers"
                ),
            )

            if max_workers_value == 0:
                max_workers_value = None

        progress_interval = _parse_non_negative_integer(
            data.get("progress_interval", 500),
            (
                "dataset_validation.execution."
                "progress_interval"
            ),
        )

        return cls(
            max_workers=max_workers_value,
            progress_interval=progress_interval,
        )


@dataclass(frozen=True)
class DatasetValidationSettings:
    report_dir: str
    expected_counts: dict[str, int | None]
    checks: ValidationCheckSettings
    duplicates: DuplicateCheckSettings
    unsupported_files: UnsupportedFileSettings
    reports: ReportSettings
    execution: ValidationExecutionSettings
    fail_on_error: bool

    @classmethod
    def from_loaded_configuration(
        cls,
        loaded_configuration: LoadedConfiguration,
    ) -> "DatasetValidationSettings":
        raw_section = loaded_configuration.raw.get(
            "dataset_validation",
            {},
        )

        section = _optional_mapping(
            raw_section,
            "dataset_validation",
        )

        expected_counts_mapping = _optional_mapping(
            section.get("expected_counts"),
            "dataset_validation.expected_counts",
        )

        expected_counts: dict[str, int | None] = {}

        for split_name in (
            "train",
            "validation",
            "test",
        ):
            expected_counts[split_name] = (
                _parse_optional_positive_integer(
                    expected_counts_mapping.get(split_name),
                    (
                        "dataset_validation.expected_counts."
                        f"{split_name}"
                    ),
                )
            )

        checks = ValidationCheckSettings.from_mapping(
            _optional_mapping(
                section.get("checks"),
                "dataset_validation.checks",
            )
        )

        duplicates = DuplicateCheckSettings.from_mapping(
            _optional_mapping(
                section.get("duplicates"),
                "dataset_validation.duplicates",
            )
        )

        unsupported_files = (
            UnsupportedFileSettings.from_mapping(
                _optional_mapping(
                    section.get("unsupported_files"),
                    (
                        "dataset_validation."
                        "unsupported_files"
                    ),
                )
            )
        )

        reports = ReportSettings.from_mapping(
            _optional_mapping(
                section.get("reports"),
                "dataset_validation.reports",
            )
        )

        execution = ValidationExecutionSettings.from_mapping(
            _optional_mapping(
                section.get("execution"),
                "dataset_validation.execution",
            )
        )

        return cls(
            report_dir=str(
                section.get(
                    "report_dir",
                    "outputs/dataset_reports",
                )
            ),
            expected_counts=expected_counts,
            checks=checks,
            duplicates=duplicates,
            unsupported_files=unsupported_files,
            reports=reports,
            execution=execution,
            fail_on_error=_parse_boolean(
                section.get("fail_on_error", True),
                "dataset_validation.fail_on_error",
            ),
        )


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    split: str | None = None
    path: str | None = None

    def to_row(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "code": self.code,
            "split": self.split or "",
            "path": self.path or "",
            "message": self.message,
        }


@dataclass
class ImageInspection:
    split: str
    absolute_path: str
    relative_path: str
    filename: str
    image_id: str
    extension: str
    file_size_bytes: int

    readable: bool = False
    width: int | None = None
    height: int | None = None
    mode: str | None = None
    channels: int | None = None
    content_hash: str | None = None
    error_message: str | None = None

    issue_codes: list[str] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "absolute_path": self.absolute_path,
            "relative_path": self.relative_path,
            "filename": self.filename,
            "image_id": self.image_id,
            "extension": self.extension,
            "file_size_bytes": self.file_size_bytes,
            "readable": self.readable,
            "width": self.width if self.width is not None else "",
            "height": (
                self.height if self.height is not None else ""
            ),
            "mode": self.mode or "",
            "channels": (
                self.channels
                if self.channels is not None
                else ""
            ),
            "content_hash": self.content_hash or "",
            "issue_count": len(self.issue_codes),
            "issue_codes": "|".join(
                sorted(set(self.issue_codes))
            ),
            "error_message": self.error_message or "",
        }


@dataclass(frozen=True)
class ValidationSummary:
    valid: bool
    total_files: int
    readable_images: int
    unreadable_images: int
    total_issues: int
    error_count: int
    warning_count: int
    info_count: int
    split_statistics: dict[str, dict[str, int]]
    sliding_window: dict[str, int]
    issue_codes: dict[str, int]
    report_paths: dict[str, str]


@dataclass
class DatasetValidationResult:
    inspections: list[ImageInspection]
    issues: list[ValidationIssue]
    summary: ValidationSummary


class DatasetValidator:
    def __init__(
        self,
        loaded_configuration: LoadedConfiguration,
    ) -> None:
        self.loaded = loaded_configuration
        self.project_settings = loaded_configuration.settings
        self.paths = loaded_configuration.paths

        self.validation_settings = (
            DatasetValidationSettings.from_loaded_configuration(
                loaded_configuration
            )
        )

        self.issues: list[ValidationIssue] = []
        self.inspections: list[ImageInspection] = []

        self._issues_by_path: dict[str, list[str]] = (
            defaultdict(list)
        )

    @property
    def expected_mode(self) -> str:
        color_mode = self.project_settings.dataset.color_mode

        if color_mode == "rgb":
            return "RGB"

        if color_mode == "grayscale":
            return "L"

        raise ConfigurationError(
            f"Unsupported dataset color mode: {color_mode}"
        )

    @property
    def report_directory(self) -> Path:
        configured_path = Path(
            self.validation_settings.report_dir
        ).expanduser()

        if configured_path.is_absolute():
            return configured_path.resolve()

        return (
            self.paths.project_root / configured_path
        ).resolve()

    def add_issue(
        self,
        severity: str,
        code: str,
        message: str,
        *,
        split: str | None = None,
        path: Path | str | None = None,
    ) -> None:
        normalized_severity = severity.strip().lower()

        if normalized_severity == "ignore":
            return

        path_string = (
            str(Path(path).resolve())
            if path is not None
            else None
        )

        issue = ValidationIssue(
            severity=normalized_severity,
            code=code,
            message=message,
            split=split,
            path=path_string,
        )

        self.issues.append(issue)

        if path_string is not None:
            self._issues_by_path[path_string].append(code)

    def _list_split_files(
        self,
        split_name: str,
        split_directory: Path,
    ) -> tuple[list[Path], list[Path]]:
        if not split_directory.is_dir():
            self.add_issue(
                "error",
                "MISSING_SPLIT_DIRECTORY",
                (
                    f"Dataset split directory does not exist: "
                    f"{split_directory}"
                ),
                split=split_name,
                path=split_directory,
            )

            return [], []

        recursive = self.project_settings.dataset.recursive

        if recursive:
            candidates = [
                path
                for path in split_directory.rglob("*")
                if path.is_file()
            ]
        else:
            candidates = [
                path
                for path in split_directory.iterdir()
                if path.is_file()
            ]

        candidates.sort(
            key=lambda path: str(path).lower()
        )

        allowed_extensions = set(
            self.project_settings.dataset.allowed_extensions
        )

        image_files: list[Path] = []
        unsupported_files: list[Path] = []

        for candidate in candidates:
            if candidate.suffix.lower() in allowed_extensions:
                image_files.append(candidate)
            else:
                unsupported_files.append(candidate)

        unsupported_severity = (
            self.validation_settings
            .unsupported_files
            .severity
        )

        for unsupported_file in unsupported_files:
            self.add_issue(
                unsupported_severity,
                "UNSUPPORTED_FILE",
                (
                    "File extension is not included in "
                    "'dataset.allowed_extensions'."
                ),
                split=split_name,
                path=unsupported_file,
            )

        return image_files, unsupported_files

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()

        with path.open("rb") as input_file:
            while True:
                block = input_file.read(1024 * 1024)

                if not block:
                    break

                digest.update(block)

        return digest.hexdigest()

    def _pixel_sha256(
        self,
        image: Image.Image,
    ) -> str:
        canonical_image = image.convert(
            self.expected_mode
        )

        digest = hashlib.sha256()

        digest.update(
            canonical_image.mode.encode("utf-8")
        )

        digest.update(
            (
                f"{canonical_image.width}x"
                f"{canonical_image.height}"
            ).encode("utf-8")
        )

        digest.update(canonical_image.tobytes())

        return digest.hexdigest()

    def _inspect_image(
        self,
        split_name: str,
        split_directory: Path,
        image_path: Path,
    ) -> ImageInspection:
        inspection = ImageInspection(
            split=split_name,
            absolute_path=str(image_path.resolve()),
            relative_path=str(
                image_path.relative_to(split_directory)
            ),
            filename=image_path.name,
            image_id=image_path.stem,
            extension=image_path.suffix.lower(),
            file_size_bytes=image_path.stat().st_size,
        )

        try:
            with Image.open(image_path) as image:
                if (
                    self.validation_settings
                    .checks
                    .full_decode
                ):
                    image.load()

                inspection.width = image.width
                inspection.height = image.height
                inspection.mode = image.mode
                inspection.channels = len(image.getbands())
                inspection.readable = True

                if (
                    self.validation_settings
                    .duplicates
                    .check_content
                ):
                    hash_method = (
                        self.validation_settings
                        .duplicates
                        .content_hash
                    )

                    if hash_method == "file":
                        inspection.content_hash = (
                            self._file_sha256(image_path)
                        )
                    elif hash_method == "pixels":
                        inspection.content_hash = (
                            self._pixel_sha256(image)
                        )
                    else:
                        raise RuntimeError(
                            "Unexpected content-hash method: "
                            f"{hash_method}"
                        )

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            RuntimeError,
        ) as error:
            inspection.readable = False
            inspection.error_message = (
                f"{type(error).__name__}: {error}"
            )

        return inspection

    def _validate_inspection(
        self,
        inspection: ImageInspection,
    ) -> None:
        path = Path(inspection.absolute_path)
        split = inspection.split

        if not inspection.readable:
            self.add_issue(
                "error",
                "UNREADABLE_IMAGE",
                (
                    "Image could not be opened and decoded. "
                    f"Reason: {inspection.error_message}"
                ),
                split=split,
                path=path,
            )
            return

        expected_width = (
            self.project_settings.dataset.image_size.width
        )
        expected_height = (
            self.project_settings.dataset.image_size.height
        )

        if (
            self.validation_settings.checks.dimensions
            and (
                inspection.width != expected_width
                or inspection.height != expected_height
            )
        ):
            self.add_issue(
                "error",
                "INVALID_IMAGE_DIMENSIONS",
                (
                    f"Expected {expected_width}x{expected_height}, "
                    f"found {inspection.width}x"
                    f"{inspection.height}."
                ),
                split=split,
                path=path,
            )

        if (
            self.validation_settings.checks.color_mode
            and inspection.mode != self.expected_mode
        ):
            self.add_issue(
                "error",
                "INVALID_COLOR_MODE",
                (
                    f"Expected PIL mode '{self.expected_mode}', "
                    f"found '{inspection.mode}'."
                ),
                split=split,
                path=path,
            )

        expected_channels = (
            self.project_settings.dataset.channels
        )

        if (
            self.validation_settings.checks.channels
            and inspection.channels != expected_channels
        ):
            self.add_issue(
                "error",
                "INVALID_CHANNEL_COUNT",
                (
                    f"Expected {expected_channels} channels, "
                    f"found {inspection.channels}."
                ),
                split=split,
                path=path,
            )

    def _resolve_max_workers(self) -> int | None:
        configured_workers = (
            self.validation_settings.execution.max_workers
        )

        if configured_workers is not None:
            return configured_workers

        runtime_workers = (
            self.project_settings.runtime.num_workers
        )

        if runtime_workers == 0:
            return None

        return runtime_workers

    def _inspect_split(
        self,
        split_name: str,
        split_directory: Path,
        image_files: Sequence[Path],
    ) -> list[ImageInspection]:
        total = len(image_files)

        if total == 0:
            self.add_issue(
                "error",
                "EMPTY_DATASET_SPLIT",
                "No supported image files were found.",
                split=split_name,
                path=split_directory,
            )

            return []

        progress_interval = (
            self.validation_settings
            .execution
            .progress_interval
        )

        max_workers = self._resolve_max_workers()

        def inspect(path: Path) -> ImageInspection:
            return self._inspect_image(
                split_name,
                split_directory,
                path,
            )

        inspections: list[ImageInspection] = []

        with ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:
            for index, inspection in enumerate(
                executor.map(inspect, image_files),
                start=1,
            ):
                inspections.append(inspection)
                self._validate_inspection(inspection)

                if (
                    progress_interval > 0
                    and (
                        index % progress_interval == 0
                        or index == total
                    )
                ):
                    print(
                        f"[{split_name}] "
                        f"inspected {index}/{total} images"
                    )

        return inspections

    def _check_expected_count(
        self,
        split_name: str,
        split_directory: Path,
        actual_count: int,
    ) -> None:
        expected_count = (
            self.validation_settings
            .expected_counts
            .get(split_name)
        )

        if expected_count is None:
            return

        if actual_count != expected_count:
            self.add_issue(
                "error",
                "IMAGE_COUNT_MISMATCH",
                (
                    f"Expected {expected_count} supported images, "
                    f"found {actual_count}."
                ),
                split=split_name,
                path=split_directory,
            )

    def _duplicate_severity(
        self,
        records: Sequence[ImageInspection],
    ) -> tuple[str, bool]:
        split_names = {
            record.split
            for record in records
        }

        cross_split = len(split_names) > 1

        if cross_split:
            severity = (
                self.validation_settings
                .duplicates
                .cross_split_severity
            )
        else:
            severity = (
                self.validation_settings
                .duplicates
                .within_split_severity
            )

        return severity, cross_split

    def _report_duplicate_group(
        self,
        records: Sequence[ImageInspection],
        *,
        group_type: str,
        group_value: str,
    ) -> None:
        severity, cross_split = (
            self._duplicate_severity(records)
        )

        location_label = (
            "across dataset splits"
            if cross_split
            else "within one dataset split"
        )

        code = (
            f"DUPLICATE_{group_type.upper()}_"
            f"{'ACROSS_SPLITS' if cross_split else 'WITHIN_SPLIT'}"
        )

        paths = [
            record.absolute_path
            for record in records
        ]

        message = (
            f"Duplicate {group_type.replace('_', ' ')} "
            f"'{group_value}' detected {location_label}. "
            f"Matching files: {paths}"
        )

        for record in records:
            self.add_issue(
                severity,
                code,
                message,
                split=record.split,
                path=record.absolute_path,
            )

    def _find_duplicates(
        self,
        records: Sequence[ImageInspection],
        *,
        group_type: str,
        key_getter,
    ) -> None:
        groups: dict[str, list[ImageInspection]] = (
            defaultdict(list)
        )

        for record in records:
            key = key_getter(record)

            if key is None or key == "":
                continue

            groups[str(key)].append(record)

        for group_value, group_records in groups.items():
            if len(group_records) <= 1:
                continue

            self._report_duplicate_group(
                group_records,
                group_type=group_type,
                group_value=group_value,
            )

    def _check_duplicates(self) -> None:
        readable_records = [
            inspection
            for inspection in self.inspections
            if inspection.readable
        ]

        duplicate_settings = (
            self.validation_settings.duplicates
        )

        if duplicate_settings.check_filenames:
            self._find_duplicates(
                readable_records,
                group_type="filename",
                key_getter=lambda record: (
                    record.filename.casefold()
                ),
            )

        if duplicate_settings.check_image_ids:
            self._find_duplicates(
                readable_records,
                group_type="image_id",
                key_getter=lambda record: (
                    record.image_id.casefold()
                ),
            )

        if duplicate_settings.check_content:
            self._find_duplicates(
                readable_records,
                group_type="content",
                key_getter=lambda record: (
                    record.content_hash
                ),
            )

    def _attach_issue_codes(self) -> None:
        for inspection in self.inspections:
            path = str(
                Path(inspection.absolute_path).resolve()
            )

            inspection.issue_codes = list(
                self._issues_by_path.get(path, [])
            )

    def _validate_sliding_window_contract(self) -> None:
        settings = self.project_settings

        calculated_grid_width = settings.grid_width
        calculated_grid_height = settings.grid_height

        expected_grid_width = (
            settings.sliding_window.expected_grid.width
        )
        expected_grid_height = (
            settings.sliding_window.expected_grid.height
        )

        if calculated_grid_width != expected_grid_width:
            self.add_issue(
                "error",
                "INVALID_WINDOW_GRID_WIDTH",
                (
                    f"Expected grid width "
                    f"{expected_grid_width}, calculated "
                    f"{calculated_grid_width}."
                ),
            )

        if calculated_grid_height != expected_grid_height:
            self.add_issue(
                "error",
                "INVALID_WINDOW_GRID_HEIGHT",
                (
                    f"Expected grid height "
                    f"{expected_grid_height}, calculated "
                    f"{calculated_grid_height}."
                ),
            )

        expected_windows = (
            expected_grid_width * expected_grid_height
        )

        if settings.windows_per_image != expected_windows:
            self.add_issue(
                "error",
                "INVALID_WINDOWS_PER_IMAGE",
                (
                    f"Expected {expected_windows} windows per "
                    f"image, calculated "
                    f"{settings.windows_per_image}."
                ),
            )

    def _prepare_report_path(
        self,
        filename: str,
    ) -> Path:
        report_directory = self.report_directory
        report_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        report_path = report_directory / filename

        if (
            report_path.exists()
            and not self.validation_settings.reports.overwrite
        ):
            raise FileExistsError(
                f"Report already exists and overwrite is disabled: "
                f"{report_path}"
            )

        return report_path

    def _write_manifest(
        self,
    ) -> Path:
        report_path = self._prepare_report_path(
            "dataset_manifest.csv"
        )

        fieldnames = [
            "split",
            "absolute_path",
            "relative_path",
            "filename",
            "image_id",
            "extension",
            "file_size_bytes",
            "readable",
            "width",
            "height",
            "mode",
            "channels",
            "content_hash",
            "issue_count",
            "issue_codes",
            "error_message",
        ]

        with report_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for inspection in self.inspections:
                writer.writerow(inspection.to_row())

        return report_path

    def _write_issues(
        self,
    ) -> Path:
        report_path = self._prepare_report_path(
            "dataset_validation_issues.csv"
        )

        fieldnames = [
            "severity",
            "code",
            "split",
            "path",
            "message",
        ]

        with report_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for issue in self.issues:
                writer.writerow(issue.to_row())

        return report_path

    def _build_split_statistics(
        self,
    ) -> dict[str, dict[str, int]]:
        statistics: dict[str, dict[str, int]] = {}

        for split_name in (
            "train",
            "validation",
            "test",
        ):
            split_records = [
                record
                for record in self.inspections
                if record.split == split_name
            ]

            split_issues = [
                issue
                for issue in self.issues
                if issue.split == split_name
            ]

            statistics[split_name] = {
                "files": len(split_records),
                "readable": sum(
                    record.readable
                    for record in split_records
                ),
                "unreadable": sum(
                    not record.readable
                    for record in split_records
                ),
                "errors": sum(
                    issue.severity == "error"
                    for issue in split_issues
                ),
                "warnings": sum(
                    issue.severity == "warning"
                    for issue in split_issues
                ),
                "info": sum(
                    issue.severity == "info"
                    for issue in split_issues
                ),
            }

        return statistics

    def _build_summary(
        self,
        report_paths: dict[str, str],
    ) -> ValidationSummary:
        severity_counts = Counter(
            issue.severity
            for issue in self.issues
        )

        issue_code_counts = Counter(
            issue.code
            for issue in self.issues
        )

        error_count = severity_counts.get("error", 0)
        warning_count = severity_counts.get(
            "warning",
            0,
        )
        info_count = severity_counts.get("info", 0)

        readable_images = sum(
            inspection.readable
            for inspection in self.inspections
        )

        total_files = len(self.inspections)

        settings = self.project_settings

        return ValidationSummary(
            valid=error_count == 0,
            total_files=total_files,
            readable_images=readable_images,
            unreadable_images=(
                total_files - readable_images
            ),
            total_issues=len(self.issues),
            error_count=error_count,
            warning_count=warning_count,
            info_count=info_count,
            split_statistics=self._build_split_statistics(),
            sliding_window={
                "image_width": (
                    settings.dataset.image_size.width
                ),
                "image_height": (
                    settings.dataset.image_size.height
                ),
                "window_width": (
                    settings.sliding_window.width
                ),
                "window_height": (
                    settings.sliding_window.height
                ),
                "stride_x": (
                    settings.sliding_window.stride.x
                ),
                "stride_y": (
                    settings.sliding_window.stride.y
                ),
                "grid_width": settings.grid_width,
                "grid_height": settings.grid_height,
                "windows_per_image": (
                    settings.windows_per_image
                ),
            },
            issue_codes=dict(
                sorted(issue_code_counts.items())
            ),
            report_paths=report_paths,
        )

    def _write_summary(
        self,
        summary: ValidationSummary,
    ) -> Path:
        report_path = self._prepare_report_path(
            "dataset_validation_summary.json"
        )

        payload = asdict(summary)

        payload["configuration"] = {
            "source": str(self.loaded.source_path),
            "dataset_root": str(
                self.paths.dataset_root
            ),
            "dataset_splits": {
                split_name: str(split_path)
                for split_name, split_path
                in self.paths.dataset_splits.items()
            },
            "expected_counts": (
                self.validation_settings.expected_counts
            ),
            "expected_mode": self.expected_mode,
            "expected_channels": (
                self.project_settings.dataset.channels
            ),
            "allowed_extensions": list(
                self.project_settings
                .dataset
                .allowed_extensions
            ),
            "duplicate_content_hash": (
                self.validation_settings
                .duplicates
                .content_hash
            ),
        }

        with report_path.open(
            "w",
            encoding="utf-8",
        ) as output_file:
            json.dump(
                payload,
                output_file,
                indent=2,
                ensure_ascii=False,
            )

        return report_path

    def run(self) -> DatasetValidationResult:
        print("=" * 80)
        print("DATASET VALIDATION")
        print("=" * 80)
        print(f"Dataset root: {self.paths.dataset_root}")
        print(
            f"Expected image size: "
            f"{self.project_settings.dataset.image_size.width}"
            f"x"
            f"{self.project_settings.dataset.image_size.height}"
        )
        print(f"Expected mode: {self.expected_mode}")
        print(
            f"Expected channels: "
            f"{self.project_settings.dataset.channels}"
        )
        print(
            f"Window grid: "
            f"{self.project_settings.grid_width}x"
            f"{self.project_settings.grid_height}"
        )
        print(
            f"Windows per image: "
            f"{self.project_settings.windows_per_image}"
        )
        print("=" * 80)

        self._validate_sliding_window_contract()

        for split_name in (
            "train",
            "validation",
            "test",
        ):
            split_directory = (
                self.paths.dataset_splits[split_name]
            )

            image_files, _ = self._list_split_files(
                split_name,
                split_directory,
            )

            self._check_expected_count(
                split_name,
                split_directory,
                len(image_files),
            )

            split_inspections = self._inspect_split(
                split_name,
                split_directory,
                image_files,
            )

            self.inspections.extend(
                split_inspections
            )

        self._check_duplicates()
        self._attach_issue_codes()

        report_paths: dict[str, str] = {}

        if (
            self.validation_settings
            .reports
            .save_manifest
        ):
            manifest_path = self._write_manifest()
            report_paths["manifest"] = str(
                manifest_path
            )

        if (
            self.validation_settings
            .reports
            .save_issues
        ):
            issues_path = self._write_issues()
            report_paths["issues"] = str(
                issues_path
            )

        preliminary_summary = self._build_summary(
            report_paths
        )

        if (
            self.validation_settings
            .reports
            .save_summary
        ):
            summary_path = self._prepare_report_path(
                "dataset_validation_summary.json"
            )

            report_paths["summary"] = str(
                summary_path
            )

        final_summary = self._build_summary(
            report_paths
        )

        if (
            self.validation_settings
            .reports
            .save_summary
        ):
            self._write_summary(final_summary)

        return DatasetValidationResult(
            inspections=self.inspections,
            issues=self.issues,
            summary=final_summary,
        )