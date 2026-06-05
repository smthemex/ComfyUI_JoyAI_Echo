#!/usr/bin/env python3
"""Standalone multishot inference (release mode - single merged checkpoint)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import torch
import torchaudio
import yaml

from ..ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ..ltx_distillation.inference.bidirectional_pipeline import BidirectionalAVInferencePipeline
from ..ltx_distillation.inference.memory_bidirectional_pipeline import BidirectionalMemoryAVInferencePipeline
from ..ltx_distillation.inference.memory_multishot import (
    PairedAudioVideoMemoryBank,
    audio_waveform_stats,
    build_paired_audio_memory_kwargs,
    load_multishot_prompts,
    video_uint8_to_pil_frames,
)
from ..ltx_distillation.models.ltx_wrapper import create_ltx2_wrapper
from ..ltx_distillation.models.text_encoder_wrapper import create_text_encoder_wrapper
from ..ltx_distillation.models.vae_wrapper import create_vae_wrappers
from ..ltx_distillation.utils import (
    add_noise,
    compute_latent_shapes,
    concat_shot_audios,
    concat_shot_videos,
    decode_benchmark_sample,
    encode_memory_frames_batch,
    save_memory_bank_frames,
    write_benchmark_media,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "inference.yaml"


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_path(path_str: str, repo_root: Path) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = repo_root / p
    return p.resolve()


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected a comma-separated integer list")
    return [int(item) for item in items]


def _parse_float_list(value: str) -> list[float]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected a comma-separated float list")
    return [float(item) for item in items]


def _build_config(config_path: Path, cli_overrides: dict[str, Any]) -> argparse.Namespace:
    """Load YAML config and apply CLI overrides on top."""
    cfg = _load_yaml_config(config_path)

    paths_cfg = cfg.get("paths", {})
    video_cfg = cfg.get("video", {})
    denoising_cfg = cfg.get("denoising", {})
    memory_cfg = cfg.get("memory", {})
    audio_cfg = cfg.get("audio_memory", {})
    inference_cfg = cfg.get("inference", {})

    defaults = argparse.Namespace(
        # paths
        checkpoint=str(_resolve_path(paths_cfg.get("checkpoint", "checkpoints/echo-longvideo-release.safetensors"), REPO_ROOT)),
        gemma_path=str(_resolve_path(paths_cfg.get("gemma_path", "checkpoints/gemma-3-12b"), REPO_ROOT)),
        prompts_file=None,
        output_dir=None,
        # video
        num_frames=video_cfg.get("num_frames", 241),
        video_height=video_cfg.get("height", 736),
        video_width=video_cfg.get("width", 1280),
        video_fps=video_cfg.get("fps", 25),
        seed=video_cfg.get("seed", 12345),
        # denoising
        denoising_steps=denoising_cfg.get("steps", [1000, 993, 984, 973, 959, 942, 918, 885, 836, 755, 591, 500, 400, 300, 200, 100, 0]),
        denoising_sigmas=denoising_cfg.get("sigmas", [1.0, 0.99256, 0.983633, 0.972721, 0.959082, 0.941546, 0.918165, 0.885432, 0.836333, 0.754505, 0.590859, 0.50, 0.40, 0.30, 0.20, 0.10, 0.0]),
        # memory
        memory_max_size=memory_cfg.get("max_size", 7),
        num_fix_frames=memory_cfg.get("num_fix_frames", 3),
        memory_downscale_factor=memory_cfg.get("downscale_factor", 1),
        memory_position_mode=memory_cfg.get("position_mode", "reference"),
        memory_lora_strength=memory_cfg.get("lora_strength", 1.0),
        memory_lora_generator=memory_cfg.get("lora_generator", True),
        memory_lora_path=memory_cfg.get("lora_path", "") or None,
        save_mode=memory_cfg.get("save_mode", "random_every_shot_frame"),
        video_memory_frame_selection_mode=memory_cfg.get("frame_selection_mode", "center"),
        video_memory_clip_num_frames=memory_cfg.get("clip_num_frames", 9),
        # audio memory
        enable_audio_memory=audio_cfg.get("enable", True),
        audio_memory_window_size=audio_cfg.get("window_size", 96),
        audio_memory_window_selection_mode=audio_cfg.get("window_selection_mode", "max_response"),
        audio_memory_sample_rate=audio_cfg.get("sample_rate", 16000),
        audio_memory_mel_bins=audio_cfg.get("mel_bins", 128),
        audio_memory_mel_hop_length=audio_cfg.get("mel_hop_length", 160),
        audio_memory_n_fft=audio_cfg.get("n_fft", 1024),
        audio_memory_downsample_factor=audio_cfg.get("downsample_factor", 4),
        audio_memory_is_causal=audio_cfg.get("is_causal", True),
        # inference
        device=inference_cfg.get("device", "cuda"),
        dtype=inference_cfg.get("dtype", "bfloat16"),
        v2a_grad_scale=inference_cfg.get("v2a_grad_scale", 2.0),
        # misc
        prompt_max_chars=None,
    )

    for key, value in cli_overrides.items():
        if value is not None:
            setattr(defaults, key, value)

    return defaults


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multishot inference with merged release checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--gemma-path", type=str, default=None)
    parser.add_argument("--prompts-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default=None, choices=["bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--video-height", type=int, default=None)
    parser.add_argument("--video-width", type=int, default=None)
    parser.add_argument("--video-fps", type=int, default=None)
    parser.add_argument("--save-mode", type=str, default=None)
    parser.add_argument("--memory-max-size", type=int, default=None)
    parser.add_argument("--num-fix-frames", type=int, default=None)
    parser.add_argument("--prompt-max-chars", type=int, default=None)
    parser.add_argument("--enable-audio-memory", type=str_to_bool, default=None)
    parser.add_argument("--denoising-steps", type=_parse_int_list, default=None)
    parser.add_argument("--denoising-sigmas", type=_parse_float_list, default=None)
    parser.add_argument("--memory-downscale-factor", type=int, default=None)
    parser.add_argument("--v2a-grad-scale", type=float, default=None)
    parser.add_argument("--memory-position-mode", type=str, default=None)
    parser.add_argument("--audio-memory-window-size", type=int, default=None)
    parser.add_argument("--audio-memory-window-selection-mode", type=str, default=None)
    parser.add_argument("--video-memory-frame-selection-mode", type=str, default=None)
    parser.add_argument("--video-memory-clip-num-frames", type=int, default=None)
    parser.add_argument("--audio-memory-sample-rate", type=int, default=None)
    parser.add_argument("--audio-memory-mel-bins", type=int, default=None)
    parser.add_argument("--audio-memory-mel-hop-length", type=int, default=None)
    parser.add_argument("--audio-memory-n-fft", type=int, default=None)
    parser.add_argument("--audio-memory-downsample-factor", type=int, default=None)
    parser.add_argument("--audio-memory-is-causal", type=str_to_bool, default=None)

    raw_args = parser.parse_args()

    cli_overrides = {k: v for k, v in vars(raw_args).items() if v is not None and k != "config"}

    config_path = Path(raw_args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    return _build_config(config_path, cli_overrides)


def main() -> None:
    args = parse_args()
    if len(args.denoising_steps) != len(args.denoising_sigmas):
        raise ValueError("denoising steps and sigmas must have the same length")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    gemma_path = Path(args.gemma_path).expanduser().resolve()
    prompts_file = Path(args.prompts_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not gemma_path.exists():
        raise FileNotFoundError(f"Gemma path not found: {gemma_path}")

    prompts = load_multishot_prompts(prompts_file, prompt_max_chars=args.prompt_max_chars)
    if not prompts:
        raise ValueError(f"No prompts found in {prompts_file}")

    print(f"[Inference] prompts_file={prompts_file}", flush=True)
    print(f"[Inference] num_prompts={len(prompts)}", flush=True)
    print(f"[Inference] checkpoint={checkpoint}", flush=True)
    print(
        f"[Inference] frames={args.num_frames} size={args.video_height}x{args.video_width} "
        f"fps={args.video_fps} enable_audio_memory={bool(args.enable_audio_memory)}",
        flush=True,
    )

    loras: tuple[LoraPathStrengthAndSDOps, ...] = ()
    if args.memory_lora_path and args.memory_lora_generator:
        loras = (
            LoraPathStrengthAndSDOps(
                str(Path(args.memory_lora_path).expanduser()),
                float(args.memory_lora_strength),
                LTXV_LORA_COMFY_RENAMING_MAP,
            ),
        )

    # ------------------------------------------------------------------
    # Stage 1: load text encoder, encode all prompts, release encoder.
    # ------------------------------------------------------------------
    print(f"[Stage 1] Loading text encoder...", flush=True)
    text_encoder = create_text_encoder_wrapper(
        checkpoint_path=str(checkpoint),
        gemma_path=str(gemma_path),
        device=device,
        dtype=dtype,
    )
    text_encoder.eval()

    print(f"[Stage 1] Encoding {len(prompts)} prompts...", flush=True)
    cached_conds: list[dict[str, Any]] = []
    for prompt in prompts:
        cond = text_encoder([prompt])
        cached_conds.append(
            {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in cond.items()}
        )
        del cond

    del text_encoder
    import gc
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[Stage 1] Text encoder released.", flush=True)

    # ------------------------------------------------------------------
    # Stage 2: load generator + VAEs.
    # ------------------------------------------------------------------
    print(f"[Stage 2] Loading generator + VAEs...", flush=True)
    generator = create_ltx2_wrapper(
        checkpoint_path=str(checkpoint),
        gemma_path=str(gemma_path),
        device=device,
        dtype=dtype,
        video_height=int(args.video_height),
        video_width=int(args.video_width),
        loras=loras,
    )
    generator.eval()

    video_vae, audio_vae = create_vae_wrappers(
        checkpoint_path=str(checkpoint),
        device=device,
        dtype=dtype,
        with_video_encoder=True,
        with_audio_encoder=True,
        decoder_device=device,
    )
    video_vae.eval()
    audio_vae.eval()

    denoising_sigmas = torch.tensor(list(args.denoising_sigmas), device=device, dtype=torch.float32)
    base_pipeline = BidirectionalAVInferencePipeline(
        generator=generator,
        add_noise_fn=add_noise,
        denoising_sigmas=denoising_sigmas,
    )
    memory_pipeline = BidirectionalMemoryAVInferencePipeline(
        generator=generator,
        add_noise_fn=add_noise,
        denoising_sigmas=denoising_sigmas,
        memory_downscale_factor=int(args.memory_downscale_factor),
    )

    audio_sample_rate = audio_vae.get_output_sample_rate() or 24000
    video_shape, audio_shape = compute_latent_shapes(
        num_frames=int(args.num_frames),
        video_height=int(args.video_height),
        video_width=int(args.video_width),
        batch_size=1,
        video_fps=float(args.video_fps),
    )

    memory_bank = PairedAudioVideoMemoryBank(
        max_size=int(args.memory_max_size),
        save_mode=str(args.save_mode),
        num_fix_frames=int(args.num_fix_frames),
    )
    print(f"[Stage 2] Generator + VAEs ready.", flush=True)


    shot_paths: list[Path] = []
    shot_audios: list[torch.Tensor] = []
    metadata: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "prompts_file": str(prompts_file),
        "output_dir": str(output_dir),
        "denoising_steps": [int(x) for x in args.denoising_steps],
        "denoising_sigmas": [float(x) for x in denoising_sigmas.detach().cpu().tolist()],
        "num_prompts": len(prompts),
        "save_mode": str(args.save_mode),
        "memory_max_size": int(args.memory_max_size),
        "num_fix_frames": int(args.num_fix_frames),
        "enable_audio_memory": bool(args.enable_audio_memory),
        "shots": [],
    }

    for shot_idx, prompt in enumerate(prompts):
        conditional_dict = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in cached_conds[shot_idx].items()
        }
        prompt_seed = int(args.seed) + shot_idx
        memory_size_before = len(memory_bank)

        print(
            f"[Inference] shot={shot_idx + 1}/{len(prompts)} "
            f"memory_size_before={memory_size_before} seed={prompt_seed}",
            flush=True,
        )

        memory_video = None
        memory_audio_kwargs: dict[str, Any] = {}

        with torch.random.fork_rng(devices=[device]):
            torch.manual_seed(prompt_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(prompt_seed)

            if len(memory_bank) > 0:
                memory_video = encode_memory_frames_batch(
                    video_vae=video_vae,
                    batch_memory_frames=[memory_bank.get_memory_frames()],
                    target_h=int(args.video_height),
                    target_w=int(args.video_width),
                    device=device,
                    dtype=dtype,
                )
                memory_audio_kwargs = build_paired_audio_memory_kwargs(
                    memory_bank,
                    enable_audio_memory=bool(args.enable_audio_memory),
                    v2a_grad_scale=float(args.v2a_grad_scale),
                    memory_position_mode=str(args.memory_position_mode),
                )

                video_latent, audio_latent = memory_pipeline.generate(
                    video_shape=tuple(video_shape),
                    audio_shape=tuple(audio_shape),
                    conditional_dict=conditional_dict,
                    memory_video=memory_video,
                    seed=prompt_seed,
                    **memory_audio_kwargs,
                )
            else:
                video_latent, audio_latent = base_pipeline.generate(
                    video_shape=tuple(video_shape),
                    audio_shape=tuple(audio_shape),
                    conditional_dict=conditional_dict,
                    seed=prompt_seed,
                )

        audio_memory_latent = (
            audio_latent.detach().cpu().contiguous() if (args.enable_audio_memory and audio_latent is not None) else None
        )
        video_uint8, audio_waveform = decode_benchmark_sample(video_vae, audio_vae, video_latent, audio_latent)
        memory_frames_for_bank = video_uint8_to_pil_frames(video_uint8)

        new_memory_metadata: dict[str, Any] = {}
        if audio_memory_latent is not None:
            new_memory_metadata = memory_bank.save_memory_slot(
                memory_frames_for_bank,
                audio_memory_latent,
                audio_window_size=int(args.audio_memory_window_size),
                video_clip_num_frames=int(args.video_memory_clip_num_frames),
                audio_waveform=audio_waveform,
                audio_sample_rate=int(args.audio_memory_sample_rate),
                video_fps=float(args.video_fps),
                audio_window_selection_mode=str(args.audio_memory_window_selection_mode),
                video_frame_selection_mode=str(args.video_memory_frame_selection_mode),
                audio_memory_mel_bins=int(args.audio_memory_mel_bins),
                audio_memory_mel_hop_length=int(args.audio_memory_mel_hop_length),
                audio_memory_n_fft=int(args.audio_memory_n_fft),
                audio_memory_downsample_factor=int(args.audio_memory_downsample_factor),
                audio_memory_is_causal=bool(args.audio_memory_is_causal),
            )

        save_memory_bank_frames(
            memory_bank.get_memory_frames(),
            output_dir / "memory_bank" / f"shot_{shot_idx:03d}",
        )

        shot_path = output_dir / f"shot_{shot_idx:03d}.mp4"
        write_result = write_benchmark_media(
            output_path=shot_path,
            video_uint8=video_uint8,
            audio_waveform=audio_waveform,
            fps=int(args.video_fps),
            audio_sr=int(audio_sample_rate),
        )
        shot_paths.append(shot_path)
        if audio_waveform is not None:
            shot_audios.append(audio_waveform.cpu())

        metadata["shots"].append(
            {
                "shot_idx": int(shot_idx),
                "prompt": prompt,
                "output_path": str(shot_path),
                "memory_size_before": int(memory_size_before),
                "memory_size_after": int(len(memory_bank)),
                "new_memory_entry": new_memory_metadata,
                "audio_latent_shape": list(audio_latent.shape) if audio_latent is not None else None,
                "wrote_audio_in_mp4": bool(write_result["wrote_audio_in_mp4"]),
                "wrote_sidecar_wav": bool(write_result["wrote_sidecar_wav"]),
                "audio_stats": write_result["audio_stats"],
                "memory_entries": memory_bank.get_memory_metadata(),
            }
        )

        del conditional_dict, video_latent, audio_latent, video_uint8, audio_waveform
        del audio_memory_latent, memory_frames_for_bank, memory_video, memory_audio_kwargs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    combined_path = output_dir / "combined_shots.mp4"
    concat_shot_videos(shot_paths, combined_path)
    combined_audio = concat_shot_audios(shot_audios)
    combined_audio_path = None
    if combined_audio is not None:
        combined_audio_path = output_dir / "combined_shots.wav"
        torchaudio.save(str(combined_audio_path), combined_audio, sample_rate=int(audio_sample_rate))

    metadata["combined_path"] = str(combined_path)
    metadata["combined_audio_path"] = str(combined_audio_path) if combined_audio_path is not None else None
    metadata["combined_audio_stats"] = audio_waveform_stats(combined_audio)
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Inference] done -> {combined_path}", flush=True)


if __name__ == "__main__":
    main()
