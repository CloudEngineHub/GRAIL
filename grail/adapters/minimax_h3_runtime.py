"""GRAIL-owned runtime profiles for the pinned MiniMax-H3 Sol Engine.

Sana remains an unmodified vendor submodule. This shim adds only the deployment
profiles GRAIL needs for its resident service: dynamic A100 topology,
Diffusers-reference execution, and BF16 FirstBlockCache without Sol attention.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
from pathlib import Path
from typing import Any

UPSTREAM_MODEL_MODULE = "sglang.multimodal_gen.runtime.models.dits.minimax_h3"
SUPPORTED_A100_GPU_COUNTS = {1, 2, 4}


def _locked_env(name: str, value: str) -> None:
    current = os.environ.get(name)
    if current is not None and current != value:
        raise RuntimeError(
            f"{name} is locked by the selected MiniMax-H3 profile: "
            f"expected {value!r}, got {current!r}"
        )
    os.environ[name] = value


def _runtime_values(
    hardware: Any,
    profile: Any,
    num_gpus: int,
    revision: str,
) -> dict[str, str]:
    return {
        "H3_HARDWARE": hardware.name,
        "H3_NUM_GPUS": str(num_gpus),
        "H3_ULYSSES_DEGREE": str(num_gpus),
        "H3_MODEL_REVISION": revision,
        "H3_OFFLOAD": "0",
        "H3_TORCH_COMPILE": "0",
        "H3_REORDER": "0",
        "H3_POLICY_NAME": profile.name,
        "H3_SOL_ATTN": "1" if profile.sol_attention else "0",
        "H3_SOL_TAU": str(profile.tau),
        "H3_SOL_THRESH_TYPE": profile.threshold_type,
        "H3_SOL_DENSE_STEPS": str(profile.dense_steps),
        "H3_SOL_DENSE_LAYERS": str(profile.dense_layers),
        "H3_SOL_SINK_MODE": "prefix",
        "H3_SOL_DENSITY_MODE": "warmup",
        "H3_SOL_CORRECTNESS_GATE": "1",
        "H3_FIRSTBLOCKCACHE": "1" if profile.cache == "firstblock" else "0",
        "H3_EASYCACHE": "1" if profile.cache == "easycache" else "0",
        "H3_CACHE_THRESHOLD": str(profile.cache_threshold),
        "H3_EASYCACHE_THRESHOLD": str(profile.cache_threshold),
        "H3_EASYCACHE_RETAIN_STEPS": str(profile.easycache_retain_steps),
        "H3_EASYCACHE_COOLDOWN_STEPS": str(profile.easycache_cooldown_steps),
        "H3_EASYCACHE_MAX_HITS": str(profile.easycache_max_hits),
        "H3_EXPECTED_SOL_BACKEND": hardware.sol_backend,
        "SOL_ATTN_STRICT": "1",
    }


def _verify_upstream_model(expected_sha256: str) -> Path:
    spec = importlib.util.find_spec(UPSTREAM_MODEL_MODULE)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"Pinned SGLang module is unavailable: {UPSTREAM_MODEL_MODULE}")
    path = Path(spec.origin)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(
            f"Unexpected SGLang MiniMax-H3 source: {actual}; expected {expected_sha256}"
        )
    return path


def configure_a100_runtime() -> tuple[Any, Any]:
    """Configure GRAIL's A100 service topology without modifying Sana."""
    from models.minimax_h3.A100.profiles import (
        HARDWARE,
        PINNED_MODEL_REVISION,
        PROFILES,
        RuntimeProfile,
    )

    profiles = dict(PROFILES)
    profiles["gb_parity"] = RuntimeProfile(
        name="gb_parity",
        sol_attention=False,
        tau=1.0,
        threshold_type="exact",
        dense_steps=10,
        dense_layers=2,
        cache="none",
        cache_threshold=0.08,
    )
    profiles["cache_only"] = RuntimeProfile(
        name="cache_only",
        sol_attention=False,
        tau=1.0,
        threshold_type="exact",
        dense_steps=10,
        dense_layers=2,
        cache="firstblock",
        cache_threshold=0.08,
    )

    profile_name = os.environ.get("H3_SOL_PROFILE", "dense").strip().lower()
    try:
        profile = profiles[profile_name]
    except KeyError as error:
        choices = ", ".join(sorted(profiles))
        raise ValueError(f"H3_SOL_PROFILE must be one of: {choices}") from error

    num_gpus = int(os.environ.get("H3_NUM_GPUS", "4"))
    if num_gpus not in SUPPORTED_A100_GPU_COUNTS:
        raise ValueError("H3_NUM_GPUS must be 1, 2, or 4 for A100")

    for name, value in _runtime_values(
        HARDWARE,
        profile,
        num_gpus,
        PINNED_MODEL_REVISION,
    ).items():
        _locked_env(name, value)
    return HARDWARE, profile


def _register_a100_runtime() -> tuple[Any, Any]:
    from models.minimax_h3.A100.profiles import PINNED_SGLANG_MODEL_SHA256

    hardware, profile = configure_a100_runtime()
    _verify_upstream_model(PINNED_SGLANG_MODEL_SHA256)
    if profile.name == "dense":
        return hardware, profile

    from models.minimax_h3.A100.model import MiniMaxH3DiTModel
    from sglang.multimodal_gen.runtime.models.registry import ModelRegistry

    ModelRegistry.register_model("MiniMaxH3DiTModel", MiniMaxH3DiTModel)
    ModelRegistry.register_model("MiniMaxH3Transformer3DModel", MiniMaxH3DiTModel)
    return hardware, profile


def register_runtime(hardware_target: str) -> tuple[Any, Any]:
    """Register a service runtime while treating Sana as read-only."""
    target = hardware_target.strip().upper()
    if target == "A100":
        return _register_a100_runtime()
    if target == "H100":
        registration = importlib.import_module("models.minimax_h3.H100.registration")
        return registration.register_runtime()
    raise RuntimeError("The resident MiniMax-H3 service supports A100 or H100")


__all__ = ["configure_a100_runtime", "register_runtime"]
