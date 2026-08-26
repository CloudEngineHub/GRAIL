"""Own or borrow a resident MiniMax-H3 Sol-Engine service process.

The lifecycle is intentionally scheduler-neutral. Local Docker and cluster
containers launch the same GRAIL-owned foreground service backed by the pinned,
unmodified Sana submodule; resource allocation belongs to the outer launcher.
"""

from __future__ import annotations

import atexit
import os
import signal
import socket
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from grail.adapters.minimax_h3 import (
    _load_service_config,
    _request_json_direct,
    _request_json_ssh,
)

_OWNED_LAUNCHER_MODES = {
    "dense": "dense",
    "quality": "quality",
    "fast": "fast",
    "cache-fast": "fast",
    "dual": "dense",
    "dual-fast": "fast",
    "single": "dense",
    "single-gb-parity": "quality",
    "single-gb-parity-vae-resident": "quality",
    "single-cache-fast": "fast",
    "single-cache-fast-vae-resident": "fast",
    "single-cache-fast-fp8-resident": "fast",
    "gb-parity": "quality",
}
_DEFAULT_LAUNCHER_BY_REQUEST_MODE = {
    "dense": "single",
    "quality": "single-gb-parity-vae-resident",
    "fast": "single-cache-fast-vae-resident",
}


def resolve_minimax_h3_launcher_mode(request_mode: str, launcher_mode: str | None = None) -> str:
    """Resolve the launcher identity used for provenance and process startup."""
    try:
        default_launcher_mode = _DEFAULT_LAUNCHER_BY_REQUEST_MODE[request_mode]
    except KeyError as error:
        choices = ", ".join(sorted(_DEFAULT_LAUNCHER_BY_REQUEST_MODE))
        raise ValueError(f"MiniMax-H3 request mode must be one of: {choices}") from error
    return launcher_mode or default_launcher_mode


class MiniMaxH3ServiceProcess:
    """Borrow or lazily start one MiniMax-H3 service for a pipeline process."""

    def __init__(
        self,
        *,
        request_mode: str,
        launcher_mode: str | None = None,
        service_file: str | os.PathLike[str] | None = None,
        startup_timeout: int = 1800,
        repo_root: str | os.PathLike[str] | None = None,
        service_root: str | os.PathLike[str] | None = None,
        launcher: str | os.PathLike[str] | None = None,
    ) -> None:
        self.request_mode = request_mode
        self.launcher_mode = resolve_minimax_h3_launcher_mode(request_mode, launcher_mode)
        self.startup_timeout = int(startup_timeout)
        if self.startup_timeout <= 0:
            raise ValueError("MiniMax-H3 startup timeout must be positive")

        configured_service = service_file or os.getenv("MINIMAX_H3_SERVICE_FILE")
        self.preferred_service_file = (
            Path(configured_service).expanduser() if configured_service else None
        )
        self.repo_root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
        self.sol_engine_root = (
            Path(os.getenv("MINIMAX_H3_SOL_ENGINE_ROOT", self.repo_root / "imports/Sana"))
            .expanduser()
            .resolve()
        )
        configured_launcher = launcher or os.getenv("MINIMAX_H3_SERVICE_LAUNCHER")
        self.launcher = (
            Path(configured_launcher or self.repo_root / "grail/adapters/minimax_h3_service.sh")
            .expanduser()
            .resolve()
        )
        self.configured_service_root = (
            Path(service_root).expanduser().resolve() if service_root else None
        )

        self.process: subprocess.Popen[str] | None = None
        self.owned_service_file: Path | None = None
        self.service_root: Path | None = None
        self.log_path: Path | None = None
        self._log_handle: Any | None = None
        self._failure: BaseException | None = None
        self._closed = False
        self._scope_token = uuid.uuid4().hex[:8]
        atexit.register(self.close)

    def _validate_owned_mode(self) -> None:
        """Validate launcher settings only when this session must own a service."""
        try:
            launcher_request_mode = _OWNED_LAUNCHER_MODES[self.launcher_mode]
        except KeyError as error:
            choices = ", ".join(sorted(_OWNED_LAUNCHER_MODES))
            raise ValueError(
                f"MiniMax-H3 service launcher mode must be one of: {choices}"
            ) from error
        if self.request_mode != launcher_request_mode:
            raise ValueError(
                "MiniMax-H3 request mode does not match service launcher mode: "
                f"{self.request_mode} != {launcher_request_mode}"
            )

    def _health(self, service_file: Path) -> dict[str, Any] | None:
        try:
            config = _load_service_config(service_file)
            if config.get("mode") not in {None, self.request_mode}:
                return None
            if config["transport"] == "ssh":
                health = _request_json_ssh(
                    config["ssh_host"],
                    config["api_url"],
                    "GET",
                    "/health",
                    timeout=10,
                )
            else:
                health = _request_json_direct(config["api_url"], "GET", "/health", timeout=10)
        except Exception:
            return None
        if health.get("backend") != "sol-engine":
            return None
        if self.request_mode not in health.get("modes", []):
            return None
        return health

    def _default_service_root(self) -> Path:
        if self.configured_service_root is not None:
            return self.configured_service_root
        configured_base = os.getenv("MINIMAX_H3_AUTOSTART_ROOT")
        base = (
            Path(configured_base).expanduser().resolve()
            if configured_base
            else Path(tempfile.gettempdir()) / f"grail-minimax-h3-{os.getuid()}"
        )
        hostname = "".join(
            character if character.isalnum() or character in "_.-" else "-"
            for character in socket.gethostname()
        )
        identity = f"{hostname}-pid-{os.getpid()}-{self._scope_token}"
        return base / identity

    def _log_tail(self) -> str:
        if self.log_path is None or not self.log_path.is_file():
            return "service log is unavailable"
        try:
            content = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            return f"failed to read service log: {error}"
        return content[-8000:].strip() or "service log is empty"

    def _start_owned(self) -> Path:
        self._validate_owned_mode()
        if not self.launcher.is_file():
            raise FileNotFoundError(f"MiniMax-H3 service launcher not found: {self.launcher}")

        self.service_root = self._default_service_root()
        self.service_root.mkdir(parents=True, exist_ok=True)
        self.owned_service_file = self.service_root / "service.json"
        self.log_path = self.service_root / "service.log"
        self.owned_service_file.unlink(missing_ok=True)

        # The HTTP server binds port 0 before model loading and records the
        # assigned port in its descriptor. Keep the SGLang rendezvous port in a
        # separate deterministic range scoped by this process.
        master_port = 30000 + (os.getpid() % 20000)

        environment = os.environ.copy()
        environment.update(
            {
                "MINIMAX_H3_SOL_ENGINE_ROOT": str(self.sol_engine_root),
                "MINIMAX_H3_SERVICE_ROOT": str(self.service_root),
                "MINIMAX_H3_SERVICE_FILE": str(self.owned_service_file),
                "H3_MASTER_PORT": str(master_port),
            }
        )
        command = ["bash", str(self.launcher), self.launcher_mode, "0"]
        self._log_handle = self.log_path.open("a", encoding="utf-8")
        print(
            "MiniMax-H3: starting a resident service "
            f"(mode={self.launcher_mode}, log={self.log_path})"
        )
        self.process = subprocess.Popen(
            command,
            cwd=self.repo_root,
            env=environment,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )

        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                code = self.process.returncode
                self._finish_log()
                raise RuntimeError(
                    f"MiniMax-H3 service exited during startup ({code}):\n" f"{self._log_tail()}"
                )
            if self.owned_service_file.is_file():
                health = self._health(self.owned_service_file)
                if health is not None:
                    print(
                        "MiniMax-H3: resident service is ready "
                        f"on {health.get('hostname', 'localhost')}"
                    )
                    return self.owned_service_file
            time.sleep(2)

        self._stop_owned()
        raise TimeoutError(
            f"MiniMax-H3 service was not ready within {self.startup_timeout}s:\n"
            f"{self._log_tail()}"
        )

    def _finish_log(self) -> None:
        if self._log_handle is not None:
            self._log_handle.flush()
            self._log_handle.close()
            self._log_handle = None

    def _archive_descriptor(self) -> None:
        if self.owned_service_file is None or not self.owned_service_file.exists():
            return
        stopped_file = self.owned_service_file.with_name("service.stopped.json")
        os.replace(self.owned_service_file, stopped_file)

    def _stop_owned(self) -> None:
        process = self.process
        try:
            if process is None:
                return
            self._archive_descriptor()
            if process.poll() is None:
                print("MiniMax-H3: stopping the resident service")
                process.terminate()
                try:
                    process.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait(timeout=30)
            # The launcher is a process-group leader. Ensure no engine worker
            # survives even if the HTTP parent exited before reaping children.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            else:
                try:
                    time.sleep(1)
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    pass
        finally:
            self.process = None
            self._finish_log()
            self._archive_descriptor()

    def _start_or_remember_failure(self) -> Path:
        try:
            return self._start_owned()
        except BaseException as error:
            self._stop_owned()
            self._failure = error
            raise

    def ensure_running(self) -> str:
        """Return a healthy descriptor, starting/restarting an owned service if needed."""
        if self._closed:
            raise RuntimeError("MiniMax-H3 service process is already closed")
        if self._failure is not None:
            raise RuntimeError(
                "MiniMax-H3 service previously failed; refusing an implicit restart"
            ) from self._failure

        if self.owned_service_file is not None:
            health = self._health(self.owned_service_file)
            if health is not None:
                return str(self.owned_service_file)
            self._stop_owned()
            self._failure = RuntimeError(
                "MiniMax-H3 service became unhealthy; see " f"{self.log_path or 'the service log'}"
            )
            raise self._failure

        if self.preferred_service_file is not None:
            health = self._health(self.preferred_service_file)
            if health is not None:
                print(
                    "MiniMax-H3: using the existing healthy service descriptor "
                    f"{self.preferred_service_file}"
                )
                return str(self.preferred_service_file)
            print("MiniMax-H3: configured service is unavailable; starting a local process")
            self.preferred_service_file = None

        return str(self._start_or_remember_failure())

    def close(self) -> None:
        """Stop only a service owned by this session; borrowed services remain running."""
        if self._closed:
            return
        self._closed = True
        self._stop_owned()
        atexit.unregister(self.close)


__all__ = ["MiniMaxH3ServiceProcess", "resolve_minimax_h3_launcher_mode"]
