"""Small, dependency-free helpers for MiniMax-H3 generation provenance."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

PROVENANCE_SCHEMA_VERSION = 1


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Hash a regular file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_identity(
    model_path: str | os.PathLike[str],
    model_subfolder: str = "FL2VA",
    *,
    label: str | None = None,
) -> dict[str, Any]:
    """Return a cheap checkpoint identity without hashing multi-GB weight contents.

    The layout digest covers every relative filename and byte size. The small
    model index is content-hashed separately. Operators can provide a stronger
    human-managed identity through ``MINIMAX_H3_CHECKPOINT_ID``.
    """
    root = (Path(model_path) / model_subfolder).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"MiniMax-H3 checkpoint directory not found: {root}")

    model_index = root / "model_index.json"
    if not model_index.is_file():
        raise FileNotFoundError(f"MiniMax-H3 model index not found: {model_index}")

    layout_digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for candidate in sorted(path for path in root.rglob("*") if path.is_file()):
        relative = candidate.relative_to(root).as_posix()
        size = candidate.stat().st_size
        layout_digest.update(relative.encode("utf-8"))
        layout_digest.update(b"\0")
        layout_digest.update(str(size).encode("ascii"))
        layout_digest.update(b"\n")
        file_count += 1
        total_bytes += size

    identity: dict[str, Any] = {
        "root": str(root),
        "model_index_sha256": sha256_file(model_index),
        "layout_sha256": layout_digest.hexdigest(),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }
    if label:
        identity["label"] = label
    return identity


def video_provenance_path(video_path: str | os.PathLike[str]) -> Path:
    """Return the ``.mp4.json`` sidecar path for a normalized video."""
    path = Path(video_path)
    return path.with_name(f"{path.name}.json")


def write_video_provenance(video_path: str | os.PathLike[str], payload: dict[str, Any]) -> Path:
    """Atomically persist a provenance sidecar next to a completed video."""
    destination = video_provenance_path(video_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_video_provenance(video_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and minimally validate a MiniMax-H3 video sidecar."""
    path = video_provenance_path(video_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"MiniMax-H3 provenance must be a JSON object: {path}")
    if payload.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported MiniMax-H3 provenance schema: {path}")
    if payload.get("provider") != "minimax-h3":
        raise ValueError(f"Unexpected MiniMax-H3 provenance provider: {path}")
    return payload


def check_video_provenance(
    video_path: str | os.PathLike[str],
    *,
    condition_image: str | os.PathLike[str],
    mode: str,
    duration: str | int | float,
    seed: int,
    launcher_mode: str | None = None,
) -> tuple[str, str]:
    """Compare an existing sidecar with the generation request.

    Returns ``("match", detail)``, ``("missing", detail)``,
    ``("invalid", detail)``, or ``("mismatch", detail)``. A missing sidecar is
    kept distinct so old, already-reviewed batches can remain resumable.
    """
    sidecar = video_provenance_path(video_path)
    if not sidecar.is_file():
        return "missing", f"legacy video has no provenance sidecar: {sidecar}"
    try:
        payload = load_video_provenance(video_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return "invalid", str(error)

    request = payload.get("request")
    if not isinstance(request, dict):
        return "invalid", f"MiniMax-H3 provenance request is missing: {sidecar}"

    try:
        expected_duration = int(float(duration))
        condition_sha256 = sha256_file(condition_image)
    except (OSError, TypeError, ValueError) as error:
        return "invalid", f"cannot build expected MiniMax-H3 provenance: {error}"

    expected: dict[str, Any] = {
        "mode": str(mode),
        "duration_seconds": expected_duration,
        "seed": int(seed),
        "condition_sha256": condition_sha256,
    }
    if launcher_mode is not None:
        expected["launcher_mode"] = launcher_mode

    mismatches = {
        key: {"actual": request.get(key), "expected": value}
        for key, value in expected.items()
        if request.get(key) != value
    }
    if mismatches:
        return "mismatch", json.dumps(mismatches, sort_keys=True)
    return "match", str(sidecar)


__all__ = [
    "PROVENANCE_SCHEMA_VERSION",
    "check_video_provenance",
    "checkpoint_identity",
    "load_video_provenance",
    "sha256_file",
    "video_provenance_path",
    "write_video_provenance",
]
