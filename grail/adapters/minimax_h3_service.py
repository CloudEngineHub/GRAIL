"""Portable resident MiniMax-H3 FL2VA service backed by Sol-Engine/SGLang."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import socket
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from grail.adapters.minimax_h3_provenance import checkpoint_identity

HARDWARE_TARGET = os.environ.get("MINIMAX_H3_HARDWARE", "A100").strip().upper()
if not re.fullmatch(r"[A-Z][A-Z0-9_]*", HARDWARE_TARGET):
    raise RuntimeError(f"Invalid MiniMax-H3 hardware target: {HARDWARE_TARGET!r}")
if HARDWARE_TARGET not in {"A100", "H100"}:
    raise RuntimeError("The resident MiniMax-H3 service supports A100 or H100")
SERVICE_NUM_GPUS = int(os.environ.get("H3_NUM_GPUS", "4"))
if SERVICE_NUM_GPUS not in {1, 2, 4}:
    raise RuntimeError("H3_NUM_GPUS must be 1, 2, or 4")

REQUESTED_PROFILE = os.environ.get("H3_SOL_PROFILE", "dense").strip().lower()
QUANTIZATION = os.environ.get("H3_QUANTIZATION", "").strip().lower() or None
QUANTIZATION_IGNORED_LAYERS = [
    value.strip()
    for value in os.environ.get("H3_QUANTIZATION_IGNORED_LAYERS", "").split(",")
    if value.strip()
]
DIT_RESIDENT = os.environ.get("H3_DIT_RESIDENT", "0") == "1"
VAE_RESIDENT = os.environ.get("H3_VAE_RESIDENT", "0") == "1"
if QUANTIZATION not in {None, "fp8"}:
    raise RuntimeError("H3_QUANTIZATION must be empty or fp8")
if QUANTIZATION is not None and SERVICE_NUM_GPUS != 1:
    raise RuntimeError("MiniMax-H3 quantization is currently validated on one GPU only")
if DIT_RESIDENT and QUANTIZATION != "fp8":
    raise RuntimeError("A resident one-GPU DiT requires fp8 quantization")
if VAE_RESIDENT and SERVICE_NUM_GPUS != 1:
    raise RuntimeError("H3_VAE_RESIDENT is supported on one GPU only")

if SERVICE_NUM_GPUS == 1 and REQUESTED_PROFILE not in {
    "dense",
    "gb_parity",
    "cache_only",
}:
    raise RuntimeError("The single-GPU deployment supports dense, gb_parity, or cache_only")

# Registration must happen at module scope so SGLang's spawned workers install
# the same process-local model class as the parent process. The GRAIL shim keeps
# topology/profile extensions out of the pristine Sana submodule.
from grail.adapters.minimax_h3_runtime import register_runtime  # noqa: E402

HARDWARE, PROFILE = register_runtime(HARDWARE_TARGET)


RNG_CONTRACT = os.environ.get("H3_RNG_CONTRACT", "diffusers_shared").strip().lower()
if RNG_CONTRACT not in {"diffusers_shared", "sglang_legacy"}:
    raise RuntimeError("H3_RNG_CONTRACT must be diffusers_shared or sglang_legacy")


def _install_diffusers_shared_rng_contract() -> None:
    """Use the released Diffusers FL2VA random-draw order.

    The GB300 Diffusers runtime uses one CPU generator for condition, target
    video, and target audio noise, in that order. The pinned SGLang runtime
    otherwise re-seeds the modalities independently and draws condition noise
    with a different shape. This override uses all three draws in the same
    semantic locations. PyTorch CPU normal samples remain architecture-native,
    so different CPU architectures are not expected to be bit-identical.
    ``sglang_legacy`` remains available as a rollback.
    """
    if RNG_CONTRACT != "diffusers_shared":
        return

    import torch

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
        MINIMAX_H3_DENOISE_STATE_EXTRA_KEY,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
        minimax_h3_patchify_video_latent,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages import (
        denoising as denoising_module,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.latent_preparation import (
        MiniMaxH3LatentPreparationStage,
    )

    condition_rows_key = "diffusers_condition_noise_rows"
    original_apply_condition_noise = denoising_module._apply_condition_noise_aug

    def tensor_sha256(value: torch.Tensor) -> str:
        array = value.detach().contiguous().cpu().numpy()
        return hashlib.sha256(array.tobytes()).hexdigest()

    def prepare_with_shared_rng(self, batch, plan) -> None:
        if MINIMAX_H3_DENOISE_STATE_EXTRA_KEY in batch.extra:
            return
        if str(plan.task) != "fl2va":
            raise ValueError(
                "The diffusers_shared RNG contract supports FL2VA only, " f"got {plan.task!r}"
            )

        shape = plan.shape
        if str(shape["geometry"]) != "resolved_v2":
            raise ValueError("diffusers_shared RNG requires resolved_v2 geometry")
        latent_h = int(shape["height"]) // 16
        latent_w = int(shape["width"]) // 16
        latent_t = int(shape["video_latent_t"])
        audio_t = int(shape["audio_latent_t"])
        seed = 42 if plan.seed is None else int(plan.seed)

        generator = torch.Generator(device="cpu").manual_seed(seed)
        condition_count = sum(
            str(material.material_chain) == "image.target_canvas" for material in plan.materials
        )
        if condition_count not in {1, 2}:
            raise ValueError(
                "FL2VA diffusers_shared RNG expected one or two keyframes, "
                f"got {condition_count}"
            )
        condition_rows = [
            minimax_h3_patchify_video_latent(
                torch.randn(
                    1,
                    24,
                    1,
                    latent_h,
                    latent_w,
                    generator=generator,
                    dtype=torch.float32,
                    device="cpu",
                ),
                patch_size=[1, 2, 2],
            ).to(torch.float32)
            for _ in range(condition_count)
        ]
        condition_rows = (
            condition_rows[0] if condition_count == 1 else torch.cat(condition_rows, dim=0)
        ).contiguous()

        video_tensor = torch.randn(
            1,
            24,
            latent_t,
            latent_h,
            latent_w,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )
        video_rows = minimax_h3_patchify_video_latent(video_tensor, patch_size=[1, 2, 2]).to(
            torch.float32
        )
        audio_rows = torch.randn(
            audio_t * 2,
            32,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )

        batch.extra[MINIMAX_H3_DENOISE_STATE_EXTRA_KEY] = {
            "initial_video_rows": video_rows,
            "initial_audio_rows": audio_rows,
            condition_rows_key: condition_rows,
            "latent_t": latent_t,
            "latent_h": latent_h,
            "latent_w": latent_w,
            "audio_t": audio_t,
        }
        print(
            "[minimax-h3-rng-contract] "
            + json.dumps(
                {
                    "contract": RNG_CONTRACT,
                    "seed": seed,
                    "condition_count": condition_count,
                    "condition_rows_shape": list(condition_rows.shape),
                    "condition_rows_sha256": tensor_sha256(condition_rows),
                    "video_raw_sha256": tensor_sha256(video_tensor),
                    "video_rows_sha256": tensor_sha256(video_rows),
                    "audio_rows_sha256": tensor_sha256(audio_rows),
                    "video_rows_shape": list(video_rows.shape),
                    "audio_rows_shape": list(audio_rows.shape),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def apply_shared_condition_noise(
        ctx,
        *,
        sampling,
        imgvid_noise_aug: float,
        audio_noise_aug: float,
    ) -> None:
        """Mix the shared generator's condition draw into FL2VA keyframes."""

        condition_rows = ctx.state.get(condition_rows_key)
        if condition_rows is None:
            return original_apply_condition_noise(
                ctx,
                sampling=sampling,
                imgvid_noise_aug=imgvid_noise_aug,
                audio_noise_aug=audio_noise_aug,
            )
        if ctx.is_ref2va or ctx.audio_ref_rows is not None:
            raise ValueError("diffusers_shared condition noise supports FL2VA only")
        if ctx.cond_rows is None:
            raise ValueError("diffusers_shared requires clean FL2VA condition rows")
        if list(condition_rows.shape) != list(ctx.cond_rows.shape):
            raise ValueError(
                "diffusers_shared condition noise shape "
                f"{list(condition_rows.shape)} does not match clean condition "
                f"shape {list(ctx.cond_rows.shape)}"
            )

        noise_aug = float(imgvid_noise_aug)
        if not 0.0 <= noise_aug <= 1.0:
            raise ValueError(
                f"imgvid condition noise augmentation must be in [0, 1], got {noise_aug}"
            )
        if noise_aug == 1.0:
            return

        clean_rows = ctx.cond_rows.to(torch.float32)
        noise_rows = condition_rows.to(
            device=clean_rows.device,
            dtype=torch.float32,
        )
        timestep = torch.tensor(
            noise_aug,
            dtype=torch.float32,
            device=clean_rows.device,
        )
        ctx.cond_rows = (timestep * clean_rows + (1.0 - timestep) * noise_rows).contiguous()
        print(
            "[minimax-h3-rng-contract] "
            + json.dumps(
                {
                    "contract": RNG_CONTRACT,
                    "condition_noise_aug": noise_aug,
                    "condition_augmented_rows_sha256": tensor_sha256(ctx.cond_rows),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    MiniMaxH3LatentPreparationStage._prepare_denoise_state_from_plan = prepare_with_shared_rng
    denoising_module._apply_condition_noise_aug = apply_shared_condition_noise
    print(
        "[minimax-h3-rng-contract] installed diffusers_shared RNG contract",
        flush=True,
    )


_install_diffusers_shared_rng_contract()

from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (  # noqa: E402
    DiffGenerator,
)

FPS = 24
ALLOWED_SECONDS = {5, 10}
PROFILE_TO_MODE = {
    "dense": "dense",
    "gb_parity": "quality",
    "quality": "quality",
    "cache_only": "fast",
    "fullopt_exact": "fast",
}
SAFE_JOB_ID = re.compile(r"[^A-Za-z0-9_.-]+")


def _deployment_name() -> str:
    if SERVICE_NUM_GPUS == 1:
        deployment = "layerwise-offload"
        if QUANTIZATION == "fp8":
            deployment += "-fp8-marlin"
        if DIT_RESIDENT:
            deployment += "-resident-dit"
        if VAE_RESIDENT:
            deployment += "-resident-vae"
        return deployment
    if SERVICE_NUM_GPUS == 2:
        return "fsdp-ulysses-component-offload"
    return "fsdp-ulysses"


def _runtime_info() -> dict[str, Any]:
    import torch
    import triton

    expected_torch = os.environ.get("H3_EXPECTED_TORCH", "2.11.0+cu130")
    expected_triton = os.environ.get("H3_EXPECTED_TRITON", "3.6.0")
    if torch.__version__ != expected_torch:
        raise RuntimeError(f"Expected torch {expected_torch}, found {torch.__version__}")
    if triton.__version__ != expected_triton:
        raise RuntimeError(f"Expected Triton {expected_triton}, found {triton.__version__}")
    if torch.cuda.device_count() != SERVICE_NUM_GPUS:
        raise RuntimeError(
            f"MiniMax-H3 service requires exactly {SERVICE_NUM_GPUS} visible GPU(s), "
            f"found {torch.cuda.device_count()}"
        )

    devices = []
    for index in range(SERVICE_NUM_GPUS):
        capability = tuple(torch.cuda.get_device_capability(index))
        if capability != HARDWARE.capability:
            raise RuntimeError(
                f"GPU {index} has capability {capability}; expected {HARDWARE.capability}"
            )
        devices.append(torch.cuda.get_device_name(index))

    backend = None
    if PROFILE.sol_attention:
        from sol_attn import get_sol_attn_backend

        backend = get_sol_attn_backend(0)
        if backend != HARDWARE.sol_backend:
            raise RuntimeError(f"Expected Sol-Attn backend {HARDWARE.sol_backend}, got {backend}")
    return {
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "devices": devices,
        "sol_attn_backend": backend,
    }


def _result_record(result: Any) -> dict[str, Any]:
    metrics = dict(result.metrics or {})
    inference_time = metrics.get("total_duration_s")
    if inference_time is None and "total_duration_ms" in metrics:
        inference_time = float(metrics["total_duration_ms"]) / 1000.0
    if inference_time is None:
        inference_time = float(result.generation_time)
    return {
        "inference_time_s": round(float(inference_time), 3),
        "generation_wall_s": float(result.generation_time),
        "file_path": str(result.output_file_path),
        "metrics": metrics,
    }


class SolEngineH3Service:
    def __init__(
        self,
        *,
        model_path: Path,
        output_dir: Path,
        shared_root: Path,
        service_file: Path,
        advertise_host: str,
        port: int,
    ) -> None:
        self.model_path = model_path.resolve()
        self.output_dir = output_dir.resolve()
        self.shared_root = shared_root.resolve()
        self.service_file = service_file.resolve()
        self.model_subfolder = os.environ.get("H3_MODEL_SUBFOLDER", "FL2VA")
        self.model_revision = os.environ["H3_MODEL_REVISION"]
        self.checkpoint = checkpoint_identity(
            self.model_path,
            self.model_subfolder,
            label=os.environ.get("MINIMAX_H3_CHECKPOINT_ID"),
        )
        self.mode = PROFILE_TO_MODE.get(PROFILE.name)
        if self.mode is None:
            raise RuntimeError(
                "Service profile must be dense, gb_parity, quality, "
                "cache_only, or fullopt_exact, "
                f"got {PROFILE.name!r}"
            )
        if SERVICE_NUM_GPUS == 1 and PROFILE.name not in {
            "dense",
            "gb_parity",
            "cache_only",
        }:
            raise RuntimeError(
                "The single-GPU deployment supports dense, gb_parity, or " "cache_only"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.service_file.parent.mkdir(parents=True, exist_ok=True)
        self.epoch_file = self.service_file.parent / "request_epoch.txt"
        os.environ["H3_REQUEST_EPOCH_FILE"] = str(self.epoch_file)
        os.environ["H3_SOL_EVENT_LOG"] = str(
            self.service_file.parent / "sol_events_rank{rank}.jsonl"
        )
        self.lock = threading.Lock()
        self.busy = False
        self.requests = 0
        self.started_at = time.time()
        self.runtime = _runtime_info()

        load_start = time.perf_counter()
        deployment = {
            "num_gpus": SERVICE_NUM_GPUS,
            "tp_size": 1,
            "ulysses_degree": SERVICE_NUM_GPUS,
            "enable_cfg_parallel": False,
            "performance_mode": "speed",
            "use_fsdp_inference": True,
            "layerwise_offload_components": [],
        }
        if SERVICE_NUM_GPUS == 1:
            offload_components = ["text_encoder"]
            if not VAE_RESIDENT:
                offload_components.append("vae")
            if not DIT_RESIDENT:
                offload_components.insert(0, "dit")
            deployment.update(
                {
                    "performance_mode": "memory",
                    "use_fsdp_inference": False,
                    "layerwise_offload_components": offload_components,
                    "dit_offload_prefetch_size": int(
                        os.environ.get("H3_DIT_OFFLOAD_PREFETCH_SIZE", "1")
                    ),
                    "dit_layerwise_resident_layers": int(
                        os.environ.get("H3_DIT_RESIDENT_LAYERS", "0")
                    ),
                    "pin_cpu_memory": False,
                }
            )
            if DIT_RESIDENT:
                deployment["dit_cpu_offload"] = False
        elif SERVICE_NUM_GPUS == 2:
            # The sharded DiT fits on two A100s, but keeping both conditioning
            # and VAE stages resident leaves too little activation headroom.
            deployment["layerwise_offload_components"] = ["text_encoder", "vae"]

        if QUANTIZATION is not None:
            deployment["quantization"] = QUANTIZATION
            deployment["quantization_ignored_layers"] = QUANTIZATION_IGNORED_LAYERS

        self.generator = DiffGenerator.from_pretrained(
            local_mode=True,
            model_path=str(self.model_path),
            model_subfolder=self.model_subfolder,
            model_variant="fl2va",
            revision=self.model_revision,
            enable_torch_compile=False,
            regional_compile=False,
            server_warmup=False,
            master_port=int(os.environ.get("H3_MASTER_PORT", "30005")),
            **deployment,
        )
        self.load_time_s = round(time.perf_counter() - load_start, 3)
        self._write_service_file(advertise_host, port)

    def _write_service_file(self, advertise_host: str, port: int) -> None:
        payload = {
            "schema_version": 1,
            "backend": "sol-engine",
            "transport": "shared-http",
            "api_url": f"http://{advertise_host}:{port}",
            "host_root": str(self.shared_root),
            "container_root": str(self.shared_root),
            "mode": self.mode,
            "profile": PROFILE.name,
            "hardware": HARDWARE.name,
            "num_gpus": SERVICE_NUM_GPUS,
            "deployment": _deployment_name(),
            "quantization": QUANTIZATION,
            "quantization_ignored_layers": QUANTIZATION_IGNORED_LAYERS,
            "vae_resident": VAE_RESIDENT,
            "rng_contract": RNG_CONTRACT,
            "hostname": socket.gethostname(),
            "model_path": str(self.model_path),
            "model_subfolder": self.model_subfolder,
            "model_revision": self.model_revision,
            "checkpoint": self.checkpoint,
            "started_at": time.time(),
        }
        temporary = self.service_file.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.service_file)

    def _within_shared_root(self, path: str) -> Path:
        resolved = Path(path).resolve(strict=True)
        try:
            resolved.relative_to(self.shared_root)
        except ValueError as error:
            raise ValueError(f"path is outside shared root: {resolved}") from error
        return resolved

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "backend": "sol-engine",
            "model": "MiniMaxAI/MiniMax-H3",
            "partition": "FL2VA",
            "modes": [self.mode],
            "profile": PROFILE.name,
            "hardware": f"{SERVICE_NUM_GPUS}x {self.runtime['devices'][0]}",
            "deployment": _deployment_name(),
            "quantization": QUANTIZATION,
            "vae_resident": VAE_RESIDENT,
            "rng_contract": RNG_CONTRACT,
            "model_revision": self.model_revision,
            "checkpoint": self.checkpoint,
            "busy": self.busy,
            "requests": self.requests,
            "load_time_s": self.load_time_s,
            "uptime_s": round(time.time() - self.started_at),
            "hostname": socket.gethostname(),
        }

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = str(payload.get("prompt") or "").strip()
        mode = str(payload.get("mode") or self.mode).strip().lower()
        seconds = int(payload.get("seconds") or 5)
        seed = int(payload.get("seed") or 0)
        image_path = payload.get("image_path")
        if not prompt:
            raise ValueError("prompt is required")
        if mode != self.mode:
            raise ValueError(f"service mode is {self.mode}, got {mode}")
        if seconds not in ALLOWED_SECONDS:
            raise ValueError("seconds must be 5 or 10")
        if not image_path:
            raise ValueError("image_path is required for FL2VA")
        image = self._within_shared_root(str(image_path))

        raw_job_id = str(payload.get("job_id") or f"job-{int(time.time())}")
        job_id = SAFE_JOB_ID.sub("-", raw_job_id).strip(".-") or "job"
        output_name = f"{job_id}.mp4"
        request_metadata = self.output_dir / f"{job_id}.json"
        sampling_params = {
            "prompt": prompt,
            "task": "fl2va",
            "conditions": [
                {
                    "type": "image",
                    "uri": str(image),
                    "role": "keyframe",
                    "frame_index": 0,
                }
            ],
            "target": {
                "short_edge": 768,
                "aspect_ratio": "16:9",
                "duration_seconds": float(seconds),
            },
            "num_outputs_per_prompt": 1,
            "num_inference_steps": 50,
            "flow_shift": 12.0,
            "audio_flow_shift": 3.0,
            "seed": seed,
            "output_path": str(self.output_dir),
            "output_file_name": output_name,
            "save_output": True,
            "return_file_paths_only": True,
        }

        with self.lock:
            self.busy = True
            try:
                self.epoch_file.write_text(
                    f"{PROFILE.name}:request:{job_id}:{time.time_ns()}\n",
                    encoding="utf-8",
                )
                result = self.generator.generate(sampling_params_kwargs=sampling_params)
                if result is None or isinstance(result, list):
                    raise RuntimeError(f"Expected one MiniMax-H3 result, got {result!r}")
                record = _result_record(result)
                output_path = Path(record["file_path"])
                if not output_path.is_file() or output_path.stat().st_size == 0:
                    raise RuntimeError(f"MiniMax-H3 produced no output: {output_path}")
                self.requests += 1
                response = {
                    "status": "completed",
                    "mode": self.mode,
                    "profile": PROFILE.name,
                    "seconds": seconds,
                    "seed": seed,
                    "size": output_path.stat().st_size,
                    **record,
                }
                request_metadata.write_text(
                    json.dumps(
                        {
                            "request": {
                                "job_id": job_id,
                                "prompt": prompt,
                                "image_path": str(image),
                                "seconds": seconds,
                                "seed": seed,
                            },
                            "response": response,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                print(f"[minimax-h3] completed {job_id}: {json.dumps(response)}")
                return response
            finally:
                self.busy = False

    def close(self) -> None:
        try:
            self.generator.shutdown()
        finally:
            self.epoch_file.unlink(missing_ok=True)
            self.service_file.unlink(missing_ok=True)


ENGINE: SolEngineH3Service | None = None


class Handler(BaseHTTPRequestHandler):
    server_version = "SolEngineMiniMaxH3/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[minimax-h3-http] {fmt % args}", flush=True)

    def send_json(self, value: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path.rstrip("/") != "/health":
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        self.send_json(ENGINE.health())

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/generate":
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1024 * 1024:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length))
            self.send_json(ENGINE.generate(payload))
        except (ValueError, json.JSONDecodeError) as error:
            self.send_json(
                {"status": "failed", "error": str(error)},
                HTTPStatus.BAD_REQUEST,
            )
        except Exception as error:
            traceback.print_exc()
            self.send_json(
                {"status": "failed", "error": str(error)},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shared-root", required=True, type=Path)
    parser.add_argument("--service-file", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--advertise-host", default="127.0.0.1")
    parser.add_argument("--port", default=30020, type=int)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    bound_port = int(server.server_address[1])
    global ENGINE
    try:
        ENGINE = SolEngineH3Service(
            model_path=args.model,
            output_dir=args.output,
            shared_root=args.shared_root,
            service_file=args.service_file,
            advertise_host=args.advertise_host,
            port=bound_port,
        )
    except BaseException:
        server.server_close()
        raise

    def request_shutdown(signum: int, _frame: Any) -> None:
        print(f"[minimax-h3] received signal {signum}; shutting down", flush=True)
        # BaseServer.shutdown() must run outside the serve_forever() thread.
        threading.Thread(target=server.shutdown, daemon=True).start()

    previous_handlers = {}
    for handled_signal in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[handled_signal] = signal.getsignal(handled_signal)
        signal.signal(handled_signal, request_shutdown)
    print(
        f"[minimax-h3] ready on http://{args.advertise_host}:{bound_port} "
        f"mode={ENGINE.mode} profile={PROFILE.name}",
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        ENGINE.close()
        for handled_signal, previous_handler in previous_handlers.items():
            signal.signal(handled_signal, previous_handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
