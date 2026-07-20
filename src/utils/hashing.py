from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_uint32_seed(
    *parts: object,
) -> int:
    message = "|".join(
        str(part)
        for part in parts
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


def canonical_json_hash(
    value: Any,
) -> str:
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


def file_sha256(
    path: Path,
    *,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as input_file:
        while True:
            block = input_file.read(chunk_size)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def image_key_from_relative_path(
    relative_path: Path,
) -> str:
    normalized = relative_path.as_posix()

    digest = hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()[:16]

    safe_stem = "".join(
        character
        if character.isalnum() or character in {"-", "_"}
        else "_"
        for character in relative_path.stem
    )

    return f"{safe_stem}__{digest}"