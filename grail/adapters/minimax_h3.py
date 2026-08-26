"""MiniMax-H3 image-to-video adapter for a resident Sol-Engine service.

Connection and filesystem details come from a service descriptor so this
adapter remains independent of a particular cluster or host layout.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from grail.adapters.minimax_h3_provenance import (
    PROVENANCE_SCHEMA_VERSION,
    sha256_file,
    write_video_provenance,
)

DEFAULT_FPS = 24
_SSH_HOST = re.compile(r"(?:(?:[A-Za-z0-9_][A-Za-z0-9_.-]*)@)?[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


class MiniMaxH3Error(RuntimeError):
    """Base class for provider failures with an explicit retry contract."""

    retryable = False


class MiniMaxH3PermanentError(MiniMaxH3Error):
    """A deterministic request, configuration, checkpoint, or resource failure."""


class MiniMaxH3TransientError(MiniMaxH3Error):
    """A transport or temporary service failure that can be retried."""

    retryable = True


_PERMANENT_ERROR_MARKERS = (
    "out of memory",
    "cuda error",
    "checkpoint",
    "does not support",
    "invalid ",
    "must be",
    "not found",
    "outside configured",
    "permission denied",
    "requires ",
    "service mode",
    "unsupported",
)
_TRANSIENT_ERROR_MARKERS = (
    "broken pipe",
    "connection aborted",
    "connection refused",
    "connection reset",
    "network is unreachable",
    "remote end closed",
    "temporarily unavailable",
    "timed out",
    "timeout",
)


def _classify_error(
    error: BaseException,
    *,
    default_retryable: bool = True,
) -> MiniMaxH3Error:
    if isinstance(error, MiniMaxH3Error):
        return error

    message = str(error) or error.__class__.__name__
    lowered = message.lower()
    if isinstance(error, (ValueError, FileNotFoundError, PermissionError)) or any(
        marker in lowered for marker in _PERMANENT_ERROR_MARKERS
    ):
        return MiniMaxH3PermanentError(message)
    if isinstance(
        error,
        (ConnectionError, subprocess.TimeoutExpired, TimeoutError, urllib.error.URLError),
    ) or any(marker in lowered for marker in _TRANSIENT_ERROR_MARKERS):
        return MiniMaxH3TransientError(message)
    error_type = MiniMaxH3TransientError if default_retryable else MiniMaxH3PermanentError
    return error_type(message)


_REMOTE_HTTP_CLIENT = r"""
import json
import sys
import urllib.error
import urllib.request

url = sys.argv[1]
payload = json.load(sys.stdin) if sys.argv[2] == "POST" else None
request = urllib.request.Request(
    url,
    data=None if payload is None else json.dumps(payload).encode("utf-8"),
    method=sys.argv[2],
    headers={"Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(request, timeout=int(sys.argv[3])) as response:
        print(json.dumps(json.load(response)))
except urllib.error.HTTPError as error:
    sys.stderr.write(error.read().decode("utf-8", errors="replace"))
    raise
""".strip()


def _run(argv, *, timeout, input_text=None):
    result = subprocess.run(
        argv,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no command output"
        raise RuntimeError(f"Command failed ({result.returncode}): {detail}")
    return result


def _ssh(host, argv, *, timeout, input_text=None):
    command = shlex.join([str(value) for value in argv])
    return _run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, command],
        timeout=timeout,
        input_text=input_text,
    )


def _request_json_ssh(host, api_url, method, path, *, payload=None, timeout=1800):
    try:
        result = _ssh(
            host,
            [
                "python3",
                "-c",
                _REMOTE_HTTP_CLIENT,
                f"{api_url.rstrip('/')}{path}",
                method,
                timeout,
            ],
            timeout=timeout + 30,
            input_text=json.dumps(payload or {}),
        )
    except Exception as error:
        raise _classify_error(error) from error
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise MiniMaxH3TransientError(f"Invalid MiniMax-H3 response: {result.stdout!r}") from error


def _request_json_direct(api_url, method, path, *, payload=None, timeout=1800):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        retryable = error.code in {408, 425, 429} or error.code >= 500
        classified = _classify_error(
            RuntimeError(f"MiniMax-H3 HTTP {error.code}: {detail}"),
            default_retryable=retryable,
        )
        raise classified from error
    except (TimeoutError, urllib.error.URLError) as error:
        raise MiniMaxH3TransientError(f"MiniMax-H3 HTTP request failed: {error}") from error


def _load_service_config(path):
    path = Path(path)
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise MiniMaxH3PermanentError(f"MiniMax-H3 service file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise MiniMaxH3PermanentError(f"Invalid MiniMax-H3 service file: {path}") from error

    if not isinstance(config, dict):
        raise MiniMaxH3PermanentError(f"MiniMax-H3 service file must contain an object: {path}")

    required = ("transport", "api_url", "host_root", "container_root")
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise MiniMaxH3PermanentError(
            f"MiniMax-H3 service file is missing {', '.join(missing)}: {path}"
        )
    if config["transport"] not in {"ssh", "shared-http"}:
        raise MiniMaxH3PermanentError(f"Unsupported MiniMax-H3 service transport in {path}")
    for root_key in ("host_root", "container_root"):
        root = Path(config[root_key])
        if not root.is_absolute() or ".." in root.parts:
            raise MiniMaxH3PermanentError(
                f"MiniMax-H3 service {root_key} must be an absolute path without '..': {path}"
            )
    if config["transport"] == "ssh":
        ssh_host = config.get("ssh_host")
        if not ssh_host:
            raise MiniMaxH3PermanentError(
                f"MiniMax-H3 SSH service file is missing ssh_host: {path}"
            )
        if not isinstance(ssh_host, str) or _SSH_HOST.fullmatch(ssh_host) is None:
            raise MiniMaxH3PermanentError(
                f"MiniMax-H3 service file has an invalid ssh_host: {path}"
            )
    return config


def _service_output_to_host_path(service_path, host_root, container_root):
    service_path = Path(service_path)
    host_root = Path(host_root)
    container_root = Path(container_root)
    for name, candidate in (
        ("service output", service_path),
        ("host root", host_root),
        ("container root", container_root),
    ):
        if not candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"MiniMax-H3 {name} must be an absolute path without '..'")
    try:
        relative = service_path.relative_to(container_root)
    except ValueError:
        try:
            service_path.relative_to(host_root)
        except ValueError as error:
            raise ValueError(
                f"Service output is outside configured roots: {service_path}"
            ) from error
        return service_path
    return host_root / relative


def _cleanup_service_artifacts(transport, ssh_host, paths):
    """Best-effort cleanup after the normalized result is safely persisted."""
    try:
        if transport == "ssh":
            _ssh(ssh_host, ["rm", "-f", *paths], timeout=30)
        else:
            for path in paths:
                Path(path).unlink(missing_ok=True)
    except Exception as error:
        print(f"MiniMax-H3 output cleanup warning: {error}")


def _service_provenance(service_config, health):
    """Keep reproducibility fields while excluding endpoints and staging roots."""
    provenance = {
        "backend": health.get("backend") or service_config.get("backend"),
        "mode": service_config.get("mode"),
        "profile": health.get("profile") or service_config.get("profile"),
        "hardware_target": service_config.get("hardware"),
        "runtime_hardware": health.get("hardware"),
        "num_gpus": service_config.get("num_gpus"),
        "deployment": health.get("deployment") or service_config.get("deployment"),
        "quantization": health.get("quantization", service_config.get("quantization")),
        "quantization_ignored_layers": service_config.get("quantization_ignored_layers"),
        "vae_resident": health.get("vae_resident", service_config.get("vae_resident")),
        "rng_contract": health.get("rng_contract") or service_config.get("rng_contract"),
        "model_path": service_config.get("model_path"),
        "model_subfolder": service_config.get("model_subfolder"),
        "model_revision": health.get("model_revision") or service_config.get("model_revision"),
        "checkpoint": health.get("checkpoint") or service_config.get("checkpoint"),
        "hostname": health.get("hostname") or service_config.get("hostname"),
        "load_time_s": health.get("load_time_s"),
    }
    return {key: value for key, value in provenance.items() if value is not None}


def _target_frame_count(duration, fps=DEFAULT_FPS):
    seconds = float(duration)
    if seconds not in {5.0, 10.0}:
        raise ValueError("GRAIL MiniMax-H3 duration must be 5 or 10 seconds")
    return round(seconds * fps) + 1


def _validate_video(path, width, height, frame_count, fps=DEFAULT_FPS, *, ffprobe=None):
    ffprobe = ffprobe or os.getenv("FFPROBE_BIN") or shutil.which("ffprobe")
    if not ffprobe:
        raise FileNotFoundError("ffprobe is required to validate MiniMax-H3 output")
    result = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,width,height,r_frame_rate,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        timeout=60,
    )
    streams = json.loads(result.stdout).get("streams", [])
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(videos) != 1 or audio:
        raise RuntimeError(
            "Normalized MiniMax-H3 output must contain one video stream and no audio"
        )
    video = videos[0]
    expected = {
        "width": width,
        "height": height,
        "r_frame_rate": f"{fps}/1",
        "nb_frames": str(frame_count),
    }
    mismatches = {
        key: (video.get(key), value) for key, value in expected.items() if video.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Normalized MiniMax-H3 output failed validation: {mismatches}")


def _normalize_video(raw_path, output_path, width, height, duration, *, ffmpeg=None):
    ffmpeg = ffmpeg or os.getenv("FFMPEG_BIN") or shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg is required to normalize MiniMax-H3 output")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.{uuid.uuid4().hex}.mp4")
    target_frames = _target_frame_count(duration)
    try:
        _run(
            [
                ffmpeg,
                "-y",
                "-v",
                "error",
                "-i",
                str(raw_path),
                "-map",
                "0:v:0",
                "-an",
                "-frames:v",
                str(target_frames),
                "-vf",
                f"scale={width}:{height}:flags=lanczos",
                "-r",
                str(DEFAULT_FPS),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(temporary_path),
            ],
            timeout=600,
        )
        _run(
            [ffmpeg, "-v", "error", "-i", str(temporary_path), "-f", "null", "-"],
            timeout=600,
        )
        _validate_video(temporary_path, width, height, target_frames)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return str(output_path)


def generate_video(
    image_path,
    prompt,
    output_dir,
    base_name,
    *,
    mode="quality",
    duration="5",
    seed=42,
    image_tail_path=None,
    service_file=None,
    timeout=1800,
    launcher_mode=None,
):
    """Generate one image-conditioned MiniMax-H3 video through Sol-Engine.

    The service currently supports a starting image but not an explicit ending
    image. Returns the normalized local MP4 path and writes an adjacent
    ``.mp4.json`` provenance sidecar. Provider failures raise
    :class:`MiniMaxH3Error` with an explicit ``retryable`` contract.
    """
    if image_tail_path is not None:
        raise MiniMaxH3PermanentError("MiniMax-H3 Sol-Engine does not support image_tail_path")

    service_file = service_file or os.getenv("MINIMAX_H3_SERVICE_FILE")
    if not service_file:
        raise MiniMaxH3PermanentError(
            "MiniMax-H3 requires --minimax_h3_service_file or MINIMAX_H3_SERVICE_FILE"
        )
    service_config = _load_service_config(service_file)
    transport = service_config["transport"]
    api_url = service_config["api_url"]
    host_root = service_config["host_root"]
    container_root = service_config["container_root"]
    ssh_host = service_config.get("ssh_host")

    host_root = Path(host_root)
    container_root = Path(container_root)
    try:
        timeout = int(timeout)
    except (TypeError, ValueError) as error:
        raise MiniMaxH3PermanentError("MiniMax-H3 timeout must be an integer") from error
    staged_paths = []

    try:
        image_path = Path(image_path).resolve()
        if not image_path.is_file():
            raise MiniMaxH3PermanentError(f"Input image not found: {image_path}")
        if mode not in {"dense", "quality", "fast"}:
            raise MiniMaxH3PermanentError("MiniMax-H3 mode must be 'dense', 'quality', or 'fast'")
        target_frames = _target_frame_count(duration)

        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except OSError as error:
            raise MiniMaxH3PermanentError(
                f"MiniMax-H3 condition image is unreadable: {image_path}"
            ) from error
        condition_sha256 = sha256_file(image_path)

        if service_config.get("mode") not in {None, mode}:
            raise MiniMaxH3PermanentError(
                "MiniMax-H3 service mode does not match request: "
                f"{service_config.get('mode')} != {mode}"
            )

        if transport == "ssh":

            def request_json(method, path, **kwargs):
                return _request_json_ssh(ssh_host, api_url, method, path, **kwargs)

        else:

            def request_json(method, path, **kwargs):
                return _request_json_direct(api_url, method, path, **kwargs)

        health = request_json("GET", "/health", timeout=10)
        if health.get("backend") != "sol-engine" or mode not in health.get("modes", []):
            raise MiniMaxH3PermanentError(
                f"MiniMax-H3 Sol-Engine contract does not match the request: {health}"
            )

        token = uuid.uuid4().hex
        safe_name = "".join(c if c.isalnum() or c in "-_" else "-" for c in base_name)
        remote_input_dir = host_root / "grail-inputs"
        remote_image = remote_input_dir / f"{safe_name}-{token}{image_path.suffix.lower()}"
        remote_partial = remote_image.with_name(f".{remote_image.name}.partial")
        staged_paths = [remote_partial, remote_image]
        if transport == "ssh":
            _ssh(ssh_host, ["mkdir", "-p", remote_input_dir], timeout=30)
            _run(
                ["scp", "-q", str(image_path), f"{ssh_host}:{remote_partial}"],
                timeout=300,
            )
            _ssh(ssh_host, ["mv", remote_partial, remote_image], timeout=30)
        else:
            remote_input_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, remote_partial)
            os.replace(remote_partial, remote_image)

        container_image = container_root / remote_image.relative_to(host_root)
        job_id = f"grail-{safe_name}-seed{int(seed)}-{token[:8]}"
        result = request_json(
            "POST",
            "/generate",
            payload={
                "prompt": prompt,
                "mode": mode,
                "seconds": int(float(duration)),
                "seed": int(seed),
                "job_id": job_id,
                "image_path": str(container_image),
            },
            timeout=timeout,
        )
        if result.get("status") != "completed" or not result.get("file_path"):
            error = RuntimeError(f"MiniMax-H3 generation failed: {result}")
            raise _classify_error(error, default_retryable=False)

        remote_output = _service_output_to_host_path(result["file_path"], host_root, container_root)
        output_path = Path(output_dir) / f"{base_name}.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="grail-minimax-h3-") as temporary_dir:
            raw_path = Path(temporary_dir) / "raw.mp4"
            if transport == "ssh":
                _run(
                    ["scp", "-q", f"{ssh_host}:{remote_output}", str(raw_path)],
                    timeout=300,
                )
            else:
                if not remote_output.is_file():
                    raise MiniMaxH3TransientError(
                        f"MiniMax-H3 shared output is not yet available: {remote_output}"
                    )
                shutil.copy2(remote_output, raw_path)
            normalized = _normalize_video(
                raw_path,
                output_path,
                width,
                height,
                duration,
            )
        output_path = Path(normalized)
        provenance = {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "provider": "minimax-h3",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "request": {
                "condition_image": str(image_path),
                "condition_sha256": condition_sha256,
                "prompt": prompt,
                "mode": mode,
                "launcher_mode": launcher_mode,
                "duration_seconds": int(float(duration)),
                "seed": int(seed),
            },
            "output": {
                "path": output_path.name,
                "width": width,
                "height": height,
                "fps": DEFAULT_FPS,
                "frame_count": target_frames,
                "codec": "h264",
                "pixel_format": "yuv420p",
                "size_bytes": output_path.stat().st_size,
            },
            "service": _service_provenance(service_config, health),
            "result": {
                key: result[key]
                for key in ("inference_time_s", "generation_wall_s", "metrics")
                if key in result
            },
        }
        sidecar = write_video_provenance(output_path, provenance)
        _cleanup_service_artifacts(
            transport,
            ssh_host,
            (remote_output, remote_output.with_suffix(".json")),
        )
        print(
            f"MiniMax-H3: {mode} {duration}s seed={seed} -> {normalized} "
            f"({result.get('inference_time_s', 'unknown')}s inference; provenance={sidecar})"
        )
        return normalized
    except MiniMaxH3Error as error:
        disposition = "retryable" if error.retryable else "permanent"
        print(f"MiniMax-H3 {disposition} error: {error}")
        raise
    except Exception as error:
        classified = _classify_error(error)
        disposition = "retryable" if classified.retryable else "permanent"
        print(f"MiniMax-H3 {disposition} error: {classified}")
        raise classified from error
    finally:
        if staged_paths:
            if transport == "ssh":
                try:
                    _ssh(ssh_host, ["rm", "-f", *staged_paths], timeout=30)
                except Exception as cleanup_error:
                    print(f"MiniMax-H3 staging cleanup warning: {cleanup_error}")
            else:
                for path in staged_paths:
                    try:
                        Path(path).unlink(missing_ok=True)
                    except OSError as cleanup_error:
                        print(f"MiniMax-H3 staging cleanup warning: {cleanup_error}")
